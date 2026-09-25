"""
Module to support training U-Net models.
"""

import itertools
import logging
import math
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from profiler import profiler, torch_profile
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from constants import DEFAULT_SEED, MAX_PIXEL_INT_VALUE
from data import CarvanaData, DataIter, TensorIter, pin, prefetch
from model import UNet

standard_logger = logging.getLogger(__name__)


DEFAULT_LOG_EVERY_N = 500
DEFAULT_LOG_IMAGE_COUNT = 2
DEFAULT_LOG_IMAGE_DOWNSCALE = 4

DEFAULT_CHECKPOINT_FILE_NAME = "checkpoint.pt"


class Logger:
    """
    Class to support logging training state.
    """

    def __init__(self, log_dir: str | None):
        self.writer = SummaryWriter(log_dir) if log_dir is not None else None

    @staticmethod
    def _shrink(
        x: torch.Tensor, height: int, width: int, mode: str
    ) -> torch.Tensor:
        """
        Resize a (N, C, H, W) tensor to (N, C, height, width).
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

        grayscale = small_labels / (class_count - 1)

        return grayscale.expand(-1, 3, -1, -1)

    @staticmethod
    def _logits_to_rgb(
        class_logits: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Convert (N, H, W) single-class logits to a (N, 3, height, width)
        grayscale image.

        Lowest logit is black, highest logit is white.
        """

        small_class_logits = Logger._shrink(
            class_logits[:, None], height, width, mode="bilinear"
        )

        low = small_class_logits.amin(dim=(-2, -1), keepdim=True)
        high = small_class_logits.amax(dim=(-2, -1), keepdim=True)
        grayscale = (small_class_logits - low) / (high - low).clamp_min(1e-12)

        return grayscale.expand(-1, 3, -1, -1)

    def log_train_metrics(
        self,
        datapoints_seen: int,
        total_datapoint_count: int,
        datapoints_per_second: float,
        loss: float,
        learning_rate: float,
    ) -> None:
        """
        Log core training metrics.
        """

        # fmt: off
        standard_logger.info(
            "\n".join(
                [
                    "train logs:",
                    f"\tdatapoints seen: {datapoints_seen:,}/{total_datapoint_count:,}",
                    f"\tdatapoints per second: {datapoints_per_second:.1f}",
                    f"\tloss: {loss:.4f}",
                    f"\tlearning rate: {learning_rate:.2e}",
                ]
            )
        )
        # fmt: on

        if self.writer is not None:
            for name, value in [
                ("train/datapoints_per_second", datapoints_per_second),
                ("train/loss", loss),
                ("train/learning_rate", learning_rate),
            ]:
                self.writer.add_scalar(name, value, datapoints_seen)

    def log_val_metrics(
        self,
        datapoints_seen: int,
        total_datapoint_count: int,
        loss: float,
    ) -> None:
        """
        Log core validation metrics.
        """

        # fmt: off
        standard_logger.info(
            "\n".join(
                [
                    "val logs:",
                    f"\t(train) datapoints seen: {datapoints_seen:,}/{total_datapoint_count:,}",
                    f"\tloss: {loss:.4f}",
                ]
            )
        )
        # fmt: on

        if self.writer is not None:
            self.writer.add_scalar("val/loss", loss, datapoints_seen)

    @torch.no_grad()
    def log_images(
        self,
        log_base_name: str,
        datapoints_seen: int,
        images: torch.Tensor,
        masks: torch.Tensor,
        logits: torch.Tensor,
        num_images_to_log: int,
        image_downscale: int = DEFAULT_LOG_IMAGE_DOWNSCALE,
    ) -> None:
        """
        Log images, true & predicted classes, and logits.
        """

        if self.writer is None or num_images_to_log <= 0:
            return

        images = images[:num_images_to_log]
        masks = masks[:num_images_to_log]
        logits = logits[:num_images_to_log].float()

        predictions = logits.argmax(dim=1)

        class_count = logits.shape[1]
        height = images.shape[-2] // image_downscale
        width = images.shape[-1] // image_downscale

        # Part 1: plain images.
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
        self.writer.add_image(
            f"{log_base_name}/samples", grid.cpu(), datapoints_seen
        )

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
    Class to support training a model w.r.t. a dataset.
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
    def _iterate_data(data_iter: DataIter, datapoint_count: int) -> TensorIter:
        """
        Yield batches from `data_iter` until `datapoint_count` datapoints yielded.
        """

        if datapoint_count <= 0:
            return

        remaining = datapoint_count
        iterator = prefetch(pin(data_iter))

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

    def _iterate_train_data(self, datapoint_count: int) -> TensorIter:
        return self._iterate_data(self.train_data_iter, datapoint_count)

    def _iterate_val_data(self, datapoint_count: int) -> TensorIter:
        return self._iterate_data(self.val_data_iter, datapoint_count)

    @staticmethod
    def _to_tensors(
        image_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert numpy batches to model-ready tensors on device.
        """

        # int images                  float images
        # uint8 (B, H, W, 3)      ->  float (B, 3, H, W) in [0, 1].
        images = image_batch.to(device, non_blocking=True)
        images = images.permute(0, 3, 1, 2).float() / MAX_PIXEL_INT_VALUE

        # int masks                   int labels
        # uint8 (B, H, W)         ->  int64 (B, H, W)
        masks = mask_batch.to(device, non_blocking=True).long()

        return images, masks

    def _infer_on_train_batch(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate `model` on a train batch, and update it with `optimizer`.
        """

        images, masks = self._to_tensors(image_batch, mask_batch, device)
        logits = model(images)
        loss = F.cross_entropy(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return images, masks, logits.detach(), loss.detach()

    @torch.no_grad()
    def _infer_on_val_batch(
        self,
        model: nn.Module,
        image_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate `model` on a val batch; do not update the model.
        """

        images, masks = self._to_tensors(image_batch, mask_batch, device)
        logits = model(images)
        loss = F.cross_entropy(logits, masks)

        return images, masks, logits, loss

    def _save_checkpoint(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Save model & optimizer state under `self.log_dir`, if it is set.
        """

        if self.log_dir is None:
            return

        with profiler.phase("save checkpoint"):
            self._save_checkpoint_to_log_dir(model, optimizer)

    def _save_checkpoint_to_log_dir(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, DEFAULT_CHECKPOINT_FILE_NAME)

        temporary_path = f"{path}.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": getattr(model, "config", None),
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
        log_train_datapoints_seen: int,
        log_train_total_datapoint_count: int,
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
            for image_batch, mask_batch in self._iterate_val_data(
                datapoint_count
            ):
                is_logging_step = datapoints_seen == 0

                # Standard step.
                with profiler.phase("infer on val batch", on_gpu=True):
                    images, masks, logits, loss = self._infer_on_val_batch(
                        model, image_batch, mask_batch, device
                    )

                    datapoints_seen += len(images)
                    loss_sum += loss * len(images)

                if not is_logging_step:
                    continue

                # Logging step.
                with profiler.phase("log for val batch"):
                    self.logger.log_images(
                        log_base_name="val",
                        datapoints_seen=log_train_datapoints_seen,
                        images=images,
                        masks=masks,
                        logits=logits,
                        num_images_to_log=log_image_count,
                    )
        finally:
            model.train(was_training)

        if datapoints_seen == 0:
            return None

        with profiler.phase("log for val batch"):
            mean_loss = (loss_sum / datapoints_seen).item()
            self.logger.log_val_metrics(
                datapoints_seen=log_train_datapoints_seen,
                total_datapoint_count=log_train_total_datapoint_count,
                loss=mean_loss,
            )
        return mean_loss

    def warm_up(
        self,
        model: nn.Module,
        batch_size: int,
        height: int = CarvanaData.ORIGINAL_HEIGHT,
        width: int = CarvanaData.ORIGINAL_WIDTH,
    ) -> None:
        """
        Run a train & a val step on a synthetic batch, without updating `model`.
        """

        device = next(model.parameters()).device
        was_training = model.training

        image_batch = torch.zeros(
            (batch_size, height, width, 3), dtype=torch.uint8
        )
        mask_batch = torch.zeros((batch_size, height, width), dtype=torch.uint8)
        images, masks = self._to_tensors(image_batch, mask_batch, device)

        try:
            # Train mode, with grad: forward & backward, but no optimizer step.
            model.train()
            loss = F.cross_entropy(model(images), masks)
            loss.backward()
            model.zero_grad(set_to_none=True)

            # Eval mode, without grad: a separate compiled graph.
            model.eval()
            with torch.no_grad():
                F.cross_entropy(model(images), masks)
        finally:
            model.train(was_training)

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
            with torch_profile() as end_torch_profile_step:
                for image_batch, mask_batch in self._iterate_train_data(
                    train_datapoint_count
                ):
                    # Standard step.
                    with profiler.phase("infer on train batch", on_gpu=True):
                        images, masks, logits, loss = (
                            self._infer_on_train_batch(
                                model,
                                image_batch,
                                mask_batch,
                                optimizer,
                                device,
                            )
                        )

                        datapoints_seen += len(images)
                        loss_sum_since_last_log += loss * len(images)

                    end_torch_profile_step()

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

                    with profiler.phase("log for train batch"):
                        self.logger.log_train_metrics(
                            datapoints_seen=datapoints_seen,
                            total_datapoint_count=train_datapoint_count,
                            datapoints_per_second=(
                                datapoints_since_last_log / time_since_last_log
                            ),
                            loss=mean_loss_since_last_log,
                            learning_rate=optimizer.param_groups[0]["lr"],
                        )
                        self.logger.log_images(
                            log_base_name="train",
                            datapoints_seen=datapoints_seen,
                            images=images,
                            masks=masks,
                            logits=logits,
                            num_images_to_log=log_image_count,
                        )

                    self.validate(
                        model=model,
                        datapoint_count=val_datapoint_count,
                        log_train_datapoints_seen=datapoints_seen,
                        log_train_total_datapoint_count=train_datapoint_count,
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
    base_channel_count: int = UNet.DEFAULT_BASE_CHANNEL_COUNT,
    level_count: int = UNet.DEFAULT_LEVEL_COUNT,
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

    data = CarvanaData(data_dir=data_dir, seed=seed)
    train_data_iter = itertools.chain.from_iterable(
        data.get_train_dataset(batch_size=batch_size) for _ in itertools.count()
    )
    val_data_iter = itertools.chain.from_iterable(
        data.get_val_dataset(batch_size=batch_size) for _ in itertools.count()
    )

    with profiler.session():
        with profiler.phase("define model", on_gpu=True):
            model = UNet(
                base_height=base_height,
                base_width=base_width,
                output_channel_count=CarvanaData.NUM_CLASSES,
                base_channel_count=base_channel_count,
                level_count=level_count,
            ).to(device)

            if os.environ.get("COMPILE_MODEL", "0") != "0":
                model.compile()

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        trainer = Trainer(
            train_data_iter=train_data_iter,
            val_data_iter=val_data_iter,
            log_dir=log_dir,
        )

        with profiler.phase("warm up", on_gpu=True):
            trainer.warm_up(model=model, batch_size=batch_size)

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
        train_datapoint_count=3000,  # About 3 epochs.
        val_datapoint_count=60,
        batch_size=6,
        log_every_n_datapoints=500,
        log_dir=os.path.join(
            "logs/runs", f"unet_carvana_{start_time:%Y%m%d_%H%M%S}"
        ),
    )
