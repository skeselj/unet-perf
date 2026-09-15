"""
Module to support training a U-Net model.
"""

import itertools
import logging
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from constants import DEFAULT_SEED, MAX_PIXEL_INT_VALUE
from data import CarvanaData, DataIter
from model import DEFAULT_BASE_CHANNEL_COUNT, DEFAULT_LEVEL_COUNT, UNet

standard_logger = logging.getLogger(__name__)


DEFAULT_LOG_EVERY_N = 1000
DEFAULT_LOG_IMAGE_COUNT = 2
DEFAULT_LOG_IMAGE_DOWNSCALE = 4


class Logger:
    """
    Class to support logging training-related state.
    """

    def __init__(self, log_dir: str | None):
        self.writer = SummaryWriter(log_dir) if log_dir is not None else None

    @staticmethod
    def _shrink(x: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
        """
        Resize (N, C, H, W) `x` to (N, C, height, width).
        """

        if mode == "nearest":
            return F.interpolate(x, size=(height, width), mode="nearest")
        return F.interpolate(x, size=(height, width), mode=mode, antialias=True)

    @staticmethod
    def _classes_to_rgb(
        labels: torch.Tensor, class_count: int, height: int, width: int
    ) -> torch.Tensor:
        """
        Convert (N, H, W) labels to a (N, 3, height, width) grayscale image.

        Class 0 is black, the last class is white.
        """

        small_labels = Logger._shrink(
            labels[:, None].float(), height, width, mode="nearest"
        )
        gray = small_labels / (class_count - 1)
        return gray.expand(-1, 3, -1, -1)

    @staticmethod
    def _logits_to_rgb(
        class_logits: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Convert (N, H, W) one-class logits to a (N, 3, height, width) heatmap.

        Min-max normalized per datapoint; lowest logit is black, highest logit
        is white.
        """

        small_class_logits = Logger._shrink(
            class_logits[:, None], height, width, mode="bilinear"
        )

        low = small_class_logits.amin(dim=(-2, -1), keepdim=True)
        high = small_class_logits.amax(dim=(-2, -1), keepdim=True)
        small_class_logits = (small_class_logits - low) / (high - low).clamp_min(1e-12)

        return small_class_logits.expand(-1, 3, -1, -1)

    def log_train_metrics(
        self,
        loss: float,
        learning_rate: float,
        datapoints_seen: int,
        total_datapoint_count: int,
        datapoints_per_second: float,
    ) -> None:
        """
        Log core training metrics.
        """

        standard_logger.info(
            "  ".join(
                [
                    f"train loss: {loss:.4f}",
                    f"learning rate: {learning_rate:.2e}",
                    f"datapoints: {datapoints_seen:,}/{total_datapoint_count:,}",
                    f"datapoints/s: {datapoints_per_second:.1f}",
                ]
            )
        )

        if self.writer is not None:
            for name, value in [
                ("train/loss", loss),
                ("train/learning_rate", learning_rate),
                ("train/datapoints_per_second", datapoints_per_second),
            ]:
                self.writer.add_scalar(name, value, datapoints_seen)

    def log_val_metrics(
        self,
        loss: float,
        datapoints_seen: int,
        total_datapoint_count: int,
    ) -> None:
        """
        Log core validation metrics.
        """

        standard_logger.info(
            "  ".join(
                [
                    f"val loss: {loss:.4f}",
                    f"datapoints: {datapoints_seen:,}/{total_datapoint_count:,}",
                ]
            )
        )

        if self.writer is not None:
            self.writer.add_scalar("val/loss", loss, datapoints_seen)

    @torch.no_grad()
    def log_images(
        self,
        log_base_name: str,
        images: torch.Tensor,
        masks: torch.Tensor,
        logits: torch.Tensor,
        datapoints_seen: int,
        images_to_log: int,
        image_downscale: int = DEFAULT_LOG_IMAGE_DOWNSCALE,
    ) -> None:
        """
        Log images, true & predicted classes, and logits.
        """

        if self.writer is None or images_to_log <= 0:
            return

        images = images[:images_to_log]
        masks = masks[:images_to_log]
        logits = logits[:images_to_log].float()

        predictions = logits.argmax(dim=1)

        class_count = logits.shape[1]
        height = images.shape[-2] // image_downscale
        width = images.shape[-1] // image_downscale

        # Part 1: images.
        # Each element of `panels` has shape (N, 3, height, width).
        panels = [
            self._shrink(images, height, width, mode="bilinear").clamp(0, 1),
            self._classes_to_rgb(masks, class_count, height, width),
            self._classes_to_rgb(predictions, class_count, height, width),
            *(
                self._logits_to_rgb(logits[:, c], height, width)
                for c in range(class_count)
            ),
        ]
        # `rows` has shape (N, 3, height, panel_count * width).
        rows = torch.cat(panels, dim=-1)
        # `grid` has shape (3, N * height, panel_count * width).
        grid = torch.cat(list(rows), dim=-2)
        self.writer.add_image(f"{log_base_name}/samples", grid.cpu(), datapoints_seen)

        # Part 2: histograms.
        small_logits = self._shrink(logits, height, width, mode="bilinear")
        for c in range(class_count):
            self.writer.add_histogram(
                f"{log_base_name}/logits_class_{c}",
                small_logits[:, c].cpu(),
                datapoints_seen,
            )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class Trainer:
    """
    Class to support making training updates to a model w.r.t. some data.
    """

    def __init__(
        self,
        train_data_iter: DataIter,
        val_data_iter: DataIter,
        log_dir: str | None,
    ):
        self.train_data_iter = train_data_iter
        self.val_data_iter = val_data_iter
        self.log_dir = log_dir

        self.logger = Logger(log_dir=log_dir)

    @staticmethod
    def _iterate_data(data_iter: DataIter, datapoint_count: int) -> DataIter:
        """
        Yield batches from `data_iter` until `datapoint_count` datapoints yielded.
        """

        if datapoint_count <= 0:
            return

        remaining = datapoint_count

        for image_batch, mask_batch in data_iter:
            image_batch = image_batch[:remaining]
            mask_batch = mask_batch[:remaining]
            remaining -= len(image_batch)
            yield image_batch, mask_batch

            if remaining == 0:
                return

        raise ValueError(
            f"Data ran out after {datapoint_count - remaining:,} of "
            f"{datapoint_count:,} datapoints."
        )

    def _iterate_train_data(self, datapoint_count: int) -> DataIter:
        return self._iterate_data(self.train_data_iter, datapoint_count)

    def _iterate_val_data(self, datapoint_count: int) -> DataIter:
        return self._iterate_data(self.val_data_iter, datapoint_count)

    @staticmethod
    def _to_tensors(
        image_batch: np.ndarray, mask_batch: np.ndarray, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert numpy batch to model-ready tensors on device.

        int images                  float images
        uint8 (B, H, W, 3)      ->  float (B, 3, H, W) in [0, 1].

        int masks                   int labels
        uint8 (B, H, W)         ->  int64 (B, H, W)
        """

        images = torch.from_numpy(image_batch).to(device)
        images = images.permute(0, 3, 1, 2).float() / MAX_PIXEL_INT_VALUE

        masks = torch.from_numpy(mask_batch).to(device).long()

        return images, masks

    def _infer_on_train_batch(
        self,
        model: nn.Module,
        image_batch: np.ndarray,
        mask_batch: np.ndarray,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate `model` on a train batch, and update it with `optimizer`.
        """

        # images: (B, 3, H, W) floats in [0, 1].
        # masks: (B, H, W) labels.
        # logits: (B, N_{classes}, H, W) class scores.
        images, masks = self._to_tensors(image_batch, mask_batch, device)
        logits = model(images)

        loss = F.cross_entropy(logits, masks)  # Averaged.

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return images, masks, logits.detach(), loss.detach()

    @torch.no_grad()
    def _infer_on_val_batch(
        self,
        model: nn.Module,
        image_batch: np.ndarray,
        mask_batch: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate `model` on a val batch, without updating it.
        """

        images, masks = self._to_tensors(image_batch, mask_batch, device)
        logits = model(images)

        loss = F.cross_entropy(logits, masks)  # Averaged.

        return images, masks, logits, loss

    def _save_checkpoint(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Save model & optimizer state to `self.log_dir`/checkpoint.pt, if set.
        """

        if self.log_dir is None:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "checkpoint.pt")

        temporary_path = f"{path}.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            temporary_path,
        )
        os.replace(temporary_path, path)

        standard_logger.info(f"Saved checkpoint to {path!r}.")

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        datapoint_count: int,
        log_datapoints_seen_count: int,
        log_total_datapoints_count: int,
        log_image_count: int = DEFAULT_LOG_IMAGE_COUNT,
    ) -> float | None:
        """
        Validate `model` w.r.t. `datapoint_count` sampled datapoints.
        """

        device = next(model.parameters()).device

        was_training = model.training
        model.eval()

        datapoints_seen = 0
        loss_sum = torch.zeros((), device=device)

        try:
            for image_batch, mask_batch in self._iterate_val_data(datapoint_count):
                is_logging_step = datapoints_seen == 0

                # Standard step.
                images, masks, logits, loss = self._infer_on_val_batch(
                    model, image_batch, mask_batch, device
                )

                loss_sum += loss * len(images)
                datapoints_seen += len(images)

                if not is_logging_step:
                    continue

                # Logging step.
                self.logger.log_images(
                    log_base_name="val",
                    images=images,
                    masks=masks,
                    logits=logits,
                    datapoints_seen=log_datapoints_seen_count,
                    images_to_log=log_image_count,
                )
        finally:
            model.train(was_training)

        if datapoints_seen == 0:
            return None

        mean_loss = (loss_sum / datapoints_seen).item()
        self.logger.log_val_metrics(
            loss=mean_loss,
            datapoints_seen=log_datapoints_seen_count,
            total_datapoint_count=log_total_datapoints_count,
        )
        return mean_loss

    def train(
        self,
        model: nn.Module,
        train_datapoint_count: int,
        val_datapoint_count: int,
        optimizer: torch.optim.Optimizer,
        log_every_n_datapoints: int = DEFAULT_LOG_EVERY_N,
        log_image_count: int = DEFAULT_LOG_IMAGE_COUNT,
    ) -> None:
        """
        Train `model` w.r.t. `train_datapoint_count` sampled datapoints.
        """

        device = next(model.parameters()).device
        model.train()

        datapoints_seen = 0

        loss_sum_since_last_log = torch.zeros((), device=device)
        time_of_last_log = time.perf_counter()
        datapoints_seen_at_last_log = 0
        datapoint_index_for_next_log = 0

        try:
            for image_batch, mask_batch in self._iterate_train_data(
                train_datapoint_count
            ):
                # Standard step.
                images, masks, logits, loss = self._infer_on_train_batch(
                    model, image_batch, mask_batch, optimizer, device
                )

                datapoints_seen += len(images)
                loss_sum_since_last_log += loss * len(images)

                if not (
                    datapoints_seen > datapoint_index_for_next_log
                    or datapoints_seen == train_datapoint_count
                ):
                    continue

                # Logging & validation step.
                datapoints_since_last_log = (
                    datapoints_seen - datapoints_seen_at_last_log
                )
                mean_loss_since_last_log = (
                    loss_sum_since_last_log / datapoints_since_last_log
                ).item()
                time_since_last_log = time.perf_counter() - time_of_last_log

                self.logger.log_train_metrics(
                    loss=mean_loss_since_last_log,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    datapoints_seen=datapoints_seen,
                    total_datapoint_count=train_datapoint_count,
                    datapoints_per_second=(
                        datapoints_since_last_log / time_since_last_log
                    ),
                )
                self.logger.log_images(
                    log_base_name="train",
                    images=images,
                    masks=masks,
                    logits=logits,
                    datapoints_seen=datapoints_seen,
                    images_to_log=log_image_count,
                )

                self.validate(
                    model=model,
                    datapoint_count=val_datapoint_count,
                    log_datapoints_seen_count=datapoints_seen,
                    log_total_datapoints_count=train_datapoint_count,
                    log_image_count=log_image_count,
                )

                loss_sum_since_last_log.zero_()
                time_of_last_log = time.perf_counter()
                datapoints_seen_at_last_log = datapoints_seen
                datapoint_index_for_next_log = (
                    math.ceil(datapoints_seen / log_every_n_datapoints)
                    * log_every_n_datapoints
                )
        finally:
            self.logger.close()
            self._save_checkpoint(model, optimizer)


def train_unet_from_scratch_on_carvana(
    train_datapoint_count: int,
    val_datapoint_count: int,
    batch_size: int,
    log_every_n_datapoints: int,
    base_height: int = 256,
    base_width: int = 384,
    base_channel_count: int = DEFAULT_BASE_CHANNEL_COUNT,
    level_count: int = DEFAULT_LEVEL_COUNT,
    learning_rate: float = 1e-4,
    data_dir: str = "./data/carvana",
    log_dir: str | None = None,
    seed: int = DEFAULT_SEED,
) -> UNet:
    """
    Train a freshly initialized UNet on the Carvana dataset.
    """

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard_logger.info(f"device: {device}")

    model = UNet(
        base_height=base_height,
        base_width=base_width,
        output_channel_count=CarvanaData.NUM_CLASSES,
        base_channel_count=base_channel_count,
        level_count=level_count,
    ).to(device)

    data = CarvanaData(data_dir=data_dir, seed=seed)
    train_data_iter = itertools.chain.from_iterable(
        data.get_train_dataset(batch_size=batch_size) for _ in itertools.count()
    )
    val_data_iter = itertools.chain.from_iterable(
        data.get_val_dataset(batch_size=batch_size) for _ in itertools.count()
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        train_data_iter=train_data_iter,
        val_data_iter=val_data_iter,
        log_dir=log_dir,
    )
    trainer.train(
        model=model,
        train_datapoint_count=train_datapoint_count,
        val_datapoint_count=val_datapoint_count,
        optimizer=optimizer,
        log_every_n_datapoints=log_every_n_datapoints,
    )

    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    start_time = datetime.now().astimezone()
    train_unet_from_scratch_on_carvana(
        train_datapoint_count=20000,   # About 5 epochs.
        val_datapoint_count=100,
        batch_size=6,
        log_every_n_datapoints=500,
        log_dir=os.path.join("logs/runs", f"unet_carvana_{start_time:%Y%m%d_%H%M%S}"),
    )
