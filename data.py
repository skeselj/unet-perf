"""
Module to support accessing semantic segmentation datasets.
"""

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from PIL import Image

from constants import DEFAULT_SEED

logger = logging.getLogger(__name__)

# Iterator over (images, masks) batches.
DataIter = Iterator[tuple[np.ndarray, np.ndarray]]


class DataBase(ABC):
    """
    Base class for sources of semantic segmentation datapoints.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_dataset(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Return an iterator over the dataset, in `batch_size`-sized batches.

        Yields
        ------
            image_batch: (B, H, W, 3) uint8 values.
            mask_batch: (B, H, W) uint8 values.
        """

        raise NotImplementedError


class CarvanaData(DataBase):
    """
    Class to support accessing Carvana semantic segmentation datapoints.
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
        Flatten (car_id --> [view_id]) to [(car_id, view_id)].
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
        self, data_dir: str, holdout_frac: float = 0.2, seed: int = DEFAULT_SEED
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

        if not 0 <= holdout_frac <= 1:
            raise ValueError(f"{holdout_frac=} must be in [0, 1].")

        car_ids_by_hash = sorted(self.car_id_to_view_ids, key=self._hash_car_id)
        val_car_count = round(holdout_frac * len(car_ids_by_hash))
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
        self.train_car_view_ids = self._flatten_car_id_to_view_ids(
            self.train_car_id_to_view_ids
        )
        self.val_car_view_ids = self._flatten_car_id_to_view_ids(
            self.val_car_id_to_view_ids
        )

        assert self.all_car_view_ids

        missing_masks = [
            (car_id, view_id)
            for car_id, view_id in self.all_car_view_ids
            if not os.path.exists(self._get_mask_path(car_id, view_id))
        ]
        assert not missing_masks, f"Images without masks: {missing_masks[:5]}"

        rng_seed, train_rng_seed, val_rng_seed = np.random.SeedSequence(seed).spawn(3)
        self.rng = np.random.default_rng(rng_seed)
        self.train_rng = np.random.default_rng(train_rng_seed)
        self.val_rng = np.random.default_rng(val_rng_seed)

        logger.info(
            f"Loaded Carvana dataset at ({self.image_dir!r}, {self.mask_dir!r}).\n"
            f"There are {len(self.all_car_view_ids):,} datapoints "
            f"({len(self.train_car_view_ids):,} train, "
            f"{len(self.val_car_view_ids):,} val).\n"
            f"There are {len(self.car_id_to_view_ids):,} cars "
            f"({len(self.train_car_id_to_view_ids):,} train, "
            f"{len(self.val_car_id_to_view_ids):,} val)."
        )

    def _get_dataset(
        self,
        car_view_ids: list[tuple[str, str]],
        batch_size: int,
        rng: np.random.Generator,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Return an iterator over `car_view_ids` in a random order, drawn from `rng`.

        Yields
        ------
            image_batch: (B, H, W, 3) uint8 values.
            mask_batch: (B, H, W) uint8 values in {0, 1}.
        """

        order = rng.permutation(len(car_view_ids))

        for start in range(0, len(order), batch_size):
            image_batch, mask_batch = [], []

            for index in order[start : start + batch_size]:
                car_id, view_id = car_view_ids[index]
                with Image.open(self._get_image_path(car_id, view_id)) as image:
                    image_batch.append(np.asarray(image.convert("RGB")))
                with Image.open(self._get_mask_path(car_id, view_id)) as mask:
                    mask_batch.append((np.asarray(mask) > 0).astype(np.uint8))

            yield np.stack(image_batch), np.stack(mask_batch)

    def get_dataset(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return self._get_dataset(self.all_car_view_ids, batch_size, self.rng)

    def get_train_dataset(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return self._get_dataset(self.train_car_view_ids, batch_size, self.train_rng)

    def get_val_dataset(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return self._get_dataset(self.val_car_view_ids, batch_size, self.val_rng)
