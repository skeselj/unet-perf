"""
Module to support accessing semantic segmentation data.
"""

import hashlib
import logging
import os
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from constants import DEFAULT_SEED

logger = logging.getLogger(__name__)

# One datapoint: (image, mask).
#   image: (H, W, 3) shaped uint8 values.
#   mask: (H, W) shaped uint8 values.
Datapoint = tuple[np.ndarray, np.ndarray]
DataIter = Iterator[Datapoint]

TensorDatapoint = tuple[torch.Tensor, torch.Tensor]
TensorIter = Iterator[TensorDatapoint]

DEFAULT_PREFETCH_DEPTH = int(os.environ.get("DATA_PREFETCH_DEPTH", "1"))
DEFAULT_WORKER_COUNT = int(os.environ.get("DATA_WORKER_COUNT", "1"))


def prefetch(
    data_iter: DataIter, depth: int = DEFAULT_PREFETCH_DEPTH
) -> DataIter:
    """
    Yield from `data_iter`, loading up to `depth` datapoint sets ahead.
    """

    if depth < 1:
        raise ValueError(f"{depth=} must be at least 1.")

    loaded: queue.Queue = queue.Queue(maxsize=depth)
    stop_event = threading.Event()
    done_indicator = object()

    def load() -> None:
        try:
            for batch in data_iter:
                while not stop_event.is_set():
                    try:
                        loaded.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue

                if stop_event.is_set():
                    return
        except Exception as exc:  # noqa: BLE001
            loaded.put(exc)
        finally:
            if not stop_event.is_set():
                loaded.put(done_indicator)

    thread = threading.Thread(target=load, daemon=True, name="data-prefetch")
    thread.start()

    try:
        while True:
            batch = loaded.get()

            if batch is done_indicator:
                return
            if isinstance(batch, Exception):
                raise batch

            yield batch
    finally:
        stop_event.set()


def pin(data_iter: DataIter) -> TensorIter:
    """
    Yield from `data_iter` as tensors, in page-locked memory if possible.

    Page-locked memory can be read by the GPU's DMA engine directly.
    """

    is_pinnable = torch.cuda.is_available() and (
        os.environ.get("PIN_DATA", "0") != "0"
    )

    for image_batch, mask_batch in data_iter:
        images = torch.from_numpy(image_batch)
        masks = torch.from_numpy(mask_batch)

        if is_pinnable:
            images = images.pin_memory()
            masks = masks.pin_memory()

        yield images, masks


class DataBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_dataset(self, batch_size: int) -> DataIter:
        raise NotImplementedError


class CarvanaData(DataBase):
    """
    Class to support accessing Carvana semantic segmentation dataset.
    """

    ORIGINAL_HEIGHT = 1280
    ORIGINAL_WIDTH = 1918  # Aspect ratio is about 1.5.
    NUM_CLASSES = 2

    @staticmethod
    def _hash_car_id(car_id: str) -> str:
        return hashlib.sha256(car_id.encode()).hexdigest()

    @staticmethod
    def _flatten_car_id_to_view_ids(
        car_id_to_view_ids: dict[str, list[str]],
    ) -> list[tuple[str, str]]:
        """
        Flatten a (car_id --> [view_id]) map to a [(car_id, view_id)] list.
        """

        return [
            (car_id, view_id)
            for car_id, view_ids in car_id_to_view_ids.items()
            for view_id in view_ids
        ]

    @staticmethod
    def _is_image_path(path: str) -> bool:
        _, extension = os.path.splitext(path)
        return extension == ".jpg"

    def _get_image_path(self, car_id: str, view_id: str) -> str:
        return os.path.join(self.image_dir, f"{car_id}_{view_id}.jpg")

    def _get_mask_path(self, car_id: str, view_id: str) -> str:
        return os.path.join(self.mask_dir, f"{car_id}_{view_id}_mask.gif")

    def __init__(
        self, data_dir: str, val_frac: float = 0.2, seed: int = DEFAULT_SEED
    ):
        super().__init__()

        self.image_dir = os.path.join(data_dir, "train")
        self.mask_dir = os.path.join(data_dir, "train_masks")

        self.car_id_to_view_ids: dict[str, list[str]] = {}
        for file_name in sorted(os.listdir(self.image_dir)):
            if not self._is_image_path(file_name):
                continue
            stem, _ = os.path.splitext(file_name)
            car_id, view_id = stem.split("_")
            self.car_id_to_view_ids.setdefault(car_id, []).append(view_id)

        if not 0 <= val_frac <= 1:
            raise ValueError(f"{val_frac=} must be in [0, 1].")

        car_ids_by_hash = sorted(self.car_id_to_view_ids, key=self._hash_car_id)
        val_car_count = round(val_frac * len(car_ids_by_hash))
        val_car_ids = set(car_ids_by_hash[:val_car_count])

        self.train_car_id_to_view_ids: dict[str, list[str]] = {
            car_id: view_ids
            for car_id, view_ids in self.car_id_to_view_ids.items()
            if car_id not in val_car_ids
        }
        self.val_car_id_to_view_ids: dict[str, list[str]] = {
            car_id: view_ids
            for car_id, view_ids in self.car_id_to_view_ids.items()
            if car_id in val_car_ids
        }

        self.all_car_view_ids = self._flatten_car_id_to_view_ids(
            self.car_id_to_view_ids
        )
        assert self.all_car_view_ids, "No data found"
        self.train_car_view_ids = self._flatten_car_id_to_view_ids(
            self.train_car_id_to_view_ids
        )
        self.val_car_view_ids = self._flatten_car_id_to_view_ids(
            self.val_car_id_to_view_ids
        )

        missing_masks = [
            (car_id, view_id)
            for car_id, view_id in self.all_car_view_ids
            if not os.path.exists(self._get_mask_path(car_id, view_id))
        ]
        assert not missing_masks, "Found images without masks"

        rng_seed, train_rng_seed, val_rng_seed = np.random.SeedSequence(
            seed
        ).spawn(3)
        self.rng = np.random.default_rng(rng_seed)
        self.train_rng = np.random.default_rng(train_rng_seed)
        self.val_rng = np.random.default_rng(val_rng_seed)

        # fmt: off
        logger.info(
            "\n".join([
                f"Loaded Carvana dataset from ({self.image_dir!r}, {self.mask_dir!r}).",
                f"\tThere are {len(self.all_car_view_ids):,} datapoints ({len(self.train_car_view_ids):,} train, {len(self.val_car_view_ids):,} val).",
                f"\tThere are {len(self.car_id_to_view_ids):,} cars ({len(self.train_car_id_to_view_ids):,} train, {len(self.val_car_id_to_view_ids):,} val)."
            ])
        )
        # fmt: on

    def _load_datapoint(self, car_id: str, view_id: str) -> Datapoint:
        with Image.open(self._get_image_path(car_id, view_id)) as image:
            image_array = np.asarray(image.convert("RGB"))
        with Image.open(self._get_mask_path(car_id, view_id)) as mask:
            mask_array = (np.asarray(mask) > 0).astype(np.uint8)

        return image_array, mask_array

    def _get_dataset(
        self,
        car_view_ids: list[tuple[str, str]],
        batch_size: int,
        rng: np.random.Generator,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> DataIter:
        """
        Return an iterator over `car_view_ids` in a random order.
        """

        if worker_count < 1:
            raise ValueError(f"{worker_count=} must be at least 1.")

        order = rng.permutation(len(car_view_ids))

        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="data-load"
        ) as pool:
            for start in range(0, len(order), batch_size):
                datapoints = pool.map(
                    lambda index: self._load_datapoint(*car_view_ids[index]),
                    order[start : start + batch_size],
                )
                image_batch, mask_batch = zip(*datapoints, strict=True)

                yield np.stack(image_batch), np.stack(mask_batch)

    def get_dataset(
        self, batch_size: int, worker_count: int = DEFAULT_WORKER_COUNT
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Return an iterator over all datapoints."""
        return self._get_dataset(
            self.all_car_view_ids, batch_size, self.rng, worker_count
        )

    def get_train_dataset(
        self, batch_size: int, worker_count: int = DEFAULT_WORKER_COUNT
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Return an iterator over train datapoints."""
        return self._get_dataset(
            self.train_car_view_ids, batch_size, self.train_rng, worker_count
        )

    def get_val_dataset(
        self, batch_size: int, worker_count: int = DEFAULT_WORKER_COUNT
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Return an iterator over val datapoints."""
        return self._get_dataset(
            self.val_car_view_ids, batch_size, self.val_rng, worker_count
        )
