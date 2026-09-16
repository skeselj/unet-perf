"""
Module to support running inference with U-Net models.
"""

import glob
import itertools
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F
from profiler import profiler, torch_profile

from constants import DEFAULT_SEED, MAX_PIXEL_INT_VALUE
from data import CarvanaData, DataIter, TensorIter, pin, prefetch
from model import UNet
from training import DEFAULT_CHECKPOINT_FILE_NAME, DEFAULT_LOG_EVERY_N

standard_logger = logging.getLogger(__name__)


def _get_latest_checkpoint_path(log_root_dir: str = "./logs/runs") -> str:
    paths = glob.glob(
        os.path.join(log_root_dir, "*", DEFAULT_CHECKPOINT_FILE_NAME)
    )
    if not paths:
        raise FileNotFoundError(f"No checkpoints under {log_root_dir!r}.")

    return max(paths, key=os.path.getmtime)


class Logger:
    """
    Class to support logging inference state.
    """

    def log_metrics(
        self,
        datapoints_seen: int,
        total_datapoint_count: int,
        loss: float,
        accuracy: float,
    ) -> None:
        """
        Log core inference metrics.
        """

        # fmt: off
        standard_logger.info(
            "\n".join(
                [
                    "inference logs:",
                    f"\tdatapoints seen: {datapoints_seen:,}/{total_datapoint_count:,}",
                    f"\tloss: {loss:.4f}",
                    f"\tpixel accuracy: {accuracy:.2%}",
                ]
            )
        )
        # fmt: on


class Evaluator:
    """
    Class to support running inference for a model w.r.t. a dataset.
    """

    def __init__(self, data_iter: DataIter):
        self.data_iter = data_iter

        self.logger = Logger()

    def _iterate_data(self, datapoint_count: int) -> TensorIter:
        """
        Yield batches until `datapoint_count` datapoints yielded.
        """

        if datapoint_count <= 0:
            return

        remaining = datapoint_count
        iterator = prefetch(pin(self.data_iter))

        while True:
            with profiler.phase("load raw data"):
                batch = next(iterator, None)

            if batch is None:
                break

            image_batch, mask_batch = batch
            image_batch = image_batch[:remaining]
            mask_batch = mask_batch[:remaining]

            remaining -= len(image_batch)
            yield image_batch, mask_batch
            if remaining == 0:
                return

        raise ValueError(
            f"Data ran out after {(datapoint_count - remaining):,} datapoints. "
            f"Requested {datapoint_count:,}."
        )

    @staticmethod
    def _to_tensors(
        image_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert numpy batches to model-ready tensors on device.
        """

        # uint8 (B, H, W, 3) -> `dtype` (B, 3, H, W) in [0, 1].
        images = image_batch.to(device, non_blocking=True)
        images = images.permute(0, 3, 1, 2).to(dtype) / MAX_PIXEL_INT_VALUE

        # uint8 (B, H, W) -> int64 (B, H, W).
        masks = mask_batch.to(device, non_blocking=True).long()

        return images, masks

    @torch.no_grad()
    def infer(
        self,
        model: UNet,
        datapoint_count: int,
        log_every_n_datapoints: int = DEFAULT_LOG_EVERY_N,
    ) -> None:
        """
        Infer with `model` on `datapont_count` sampled datapoints.

        Metrics are logged over all datapoints seen so far: after the 1st
        batch, every `log_every_n_datapoints`, and after the last batch.
        """

        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype

        datapoints_seen = 0
        loss_sum = torch.zeros((), device=device)
        correct_pixel_count = torch.zeros((), device=device)
        total_pixel_count = torch.zeros((), device=device)
        datapoint_index_for_next_log = 0

        with torch_profile() as end_torch_profile_step:
            for image_batch, mask_batch in self._iterate_data(datapoint_count):
                # Standard step.
                with profiler.phase("infer on batch", on_gpu=True):
                    images, masks = self._to_tensors(
                        image_batch, mask_batch, device, dtype
                    )
                    logits = model(images)
                    # Compute loss always in fp32.
                    loss = F.cross_entropy(logits.float(), masks)
                    predictions = logits.argmax(dim=1)

                    datapoints_seen += len(images)
                    loss_sum += loss * len(images)
                    correct_pixel_count += (predictions == masks).sum()
                    total_pixel_count += masks.numel()

                end_torch_profile_step()

                if not (
                    datapoints_seen > datapoint_index_for_next_log
                    or datapoints_seen == datapoint_count
                ):
                    continue

                # Logging step.
                with profiler.phase("log for batch"):
                    self.logger.log_metrics(
                        datapoints_seen=datapoints_seen,
                        total_datapoint_count=datapoint_count,
                        loss=(loss_sum / datapoints_seen).item(),
                        accuracy=(
                            correct_pixel_count / total_pixel_count
                        ).item(),
                    )

                datapoint_index_for_next_log = (
                    math.ceil(datapoints_seen / log_every_n_datapoints)
                    * log_every_n_datapoints
                )


def infer_with_unet_on_carvana(
    datapoint_count: int,
    batch_size: int,
    log_every_n_datapoints: int = DEFAULT_LOG_EVERY_N,
    checkpoint_path: str | None = None,
    data_dir: str = "./data/carvana",
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """
    Run a checkpointed UNet over Carvana val datapoints, and log metrics.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard_logger.info(f"device: {device}")

    if checkpoint_path is None:
        checkpoint_path = _get_latest_checkpoint_path()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_config") is None:
        raise ValueError(f"{checkpoint_path!r} holds no model config")

    standard_logger.info(f"Loaded model from {checkpoint_path!r}")

    data = CarvanaData(data_dir=data_dir, seed=seed)
    data_iter = itertools.chain.from_iterable(
        data.get_val_dataset(batch_size=batch_size) for _ in itertools.count()
    )

    with profiler.session():
        with profiler.phase("define model", on_gpu=True):
            dtype = checkpoint.get("dtype", torch.float32)
            model = UNet(**checkpoint["model_config"]).to(
                device=device, dtype=dtype
            )  # Cast before loading.
            model.load_state_dict(checkpoint["model"])
            model.eval()

            if os.environ.get("COMPILE_MODEL", "0") != "0":
                model.compile()

        evaluator = Evaluator(data_iter=data_iter)
        evaluator.infer(
            model=model,
            datapoint_count=datapoint_count,
            log_every_n_datapoints=log_every_n_datapoints,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    if len(sys.argv) > 2:
        sys.exit(f"usage: python {sys.argv[0]} [checkpoint_path]")
    infer_with_unet_on_carvana(
        datapoint_count=1500,
        batch_size=12,
        checkpoint_path=sys.argv[1] if len(sys.argv) == 2 else None,
    )
