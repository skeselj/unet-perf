"""
Module to support quantizing U-Net models.
"""

import logging
import os
import sys

import torch

from model import UNet

standard_logger = logging.getLogger(__name__)


def quantize_fp32_to_fp16(
    input_checkpoint_path: str, output_checkpoint_path: str
) -> None:
    """
    Quantize the fp32 U-Net checkpoint at `input_checkpoint_path` to fp16.
    """

    checkpoint = torch.load(input_checkpoint_path, map_location="cpu")
    if checkpoint.get("model_config") is None:
        raise ValueError(f"{input_checkpoint_path!r} holds no model config")

    checkpoint_dtype = checkpoint.get("dtype", torch.float32)
    tensor_dtypes = {
        tensor.dtype
        for tensor in checkpoint["model"].values()
        if tensor.is_floating_point()
    }
    if checkpoint_dtype != torch.float32 or tensor_dtypes != {torch.float32}:
        raise ValueError(
            f"{input_checkpoint_path!r} is not an fp32 model: "
            f"dtype {checkpoint_dtype}, tensor dtypes {tensor_dtypes}"
        )

    model = UNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.half()

    output_dir = os.path.dirname(output_checkpoint_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temporary_path = f"{output_checkpoint_path}.tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config,
            "dtype": torch.float16,
        },
        temporary_path,
    )
    os.replace(temporary_path, output_checkpoint_path)

    standard_logger.info(
        f"Quantized {input_checkpoint_path!r} to fp16 "
        f"at {output_checkpoint_path!r}."
    )


def convert_to_onnx(
    input_torch_checkpoint: str, output_onnx_checkpoint: str
) -> None:
    checkpoint = torch.load(input_torch_checkpoint, map_location="cpu")
    dtype = checkpoint.get("dtype", torch.float32)

    # Cast before loading.
    model = UNet(**checkpoint["model_config"]).to(dtype=dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    example_input = torch.randn(1, 3, 1280, 1918, dtype=dtype)
    torch.onnx.export(
        model, (example_input,), output_onnx_checkpoint, dynamo=True
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    if len(sys.argv) != 4:
        sys.exit(
            f"usage: python {sys.argv[0]} "
            "<input_path> <output_quantized_path> <output_onnx_path>"
        )
    quantize_fp32_to_fp16(sys.argv[1], sys.argv[2])
    convert_to_onnx(sys.argv[2], sys.argv[3])
