"""
Module to support defining U-Net models in PyTorch.
"""

import torch
import torch.nn.functional as F
from torch import nn

from constants import IMAGE_CHANNEL_COUNT

DEFAULT_BASE_CHANNEL_COUNT = 32
DEFAULT_LEVEL_COUNT = 5


def _fan_in(layer: nn.Conv2d | nn.ConvTranspose2d) -> int:
    """
    Number of inputs contributing to each output pixel of layer.
    """

    kernel_height, kernel_width = layer.kernel_size

    if isinstance(layer, nn.ConvTranspose2d):
        stride_height, stride_width = layer.stride
        return (
            layer.in_channels
            * (kernel_height // stride_height)
            * (kernel_width // stride_width)
        )
    return layer.in_channels * kernel_height * kernel_width


class _ConvOp(nn.Module):
    """
    The enhanced convolution operation used in this module.

    Two convolutional layers with a skip connection at the end.
    """

    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: int,
        conv_kernel_size: int,
        conv_padding: int,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=input_channel_count,
            out_channels=output_channel_count,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.conv_2 = nn.Conv2d(
            in_channels=output_channel_count,
            out_channels=output_channel_count,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters.
        """

        fan_in_1 = _fan_in(self.conv_1)
        nn.init.normal_(self.conv_1.weight, mean=0.0, std=(1 / fan_in_1) ** 0.5)
        nn.init.zeros_(self.conv_1.bias)

        fan_in_2 = _fan_in(self.conv_2)
        nn.init.normal_(self.conv_2.weight, mean=0.0, std=(1 / fan_in_2) ** 0.5)
        nn.init.zeros_(self.conv_2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations_1 = self.activation(self.conv_1(x))
        activations_2 = self.activation(self.conv_2(activations_1))
        return activations_1 + activations_2


class _DownSampleOp(nn.Module):
    """
    The downsample operation used in this module.

    A 2x2 max pool. Every 2x2 patch becomes its largest pixel.
    """

    def __init__(self):
        super().__init__()
        self.op = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class _UpSampleOp(nn.Module):
    """
    The upsample operation used in this module.

    A learned 2x2 deconvolution. Every pixel-sized feature becomes a (2x2)-
    sized feature.
    """

    def __init__(self, input_channel_count: int, output_channel_count: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(
            in_channels=input_channel_count,
            out_channels=output_channel_count,
            kernel_size=2,
            stride=2,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters.
        """

        fan_in = _fan_in(self.op)
        nn.init.normal_(self.op.weight, mean=0.0, std=(1 / fan_in) ** 0.5)
        nn.init.zeros_(self.op.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet(nn.Module):
    """
    Basic U-Net implementation for per-pixel classification.
    """

    CONV_KERNEL_SIZE = 3
    CONV_PADDING = CONV_KERNEL_SIZE // 2

    def __init__(
        self,
        *,
        base_height: int,
        base_width: int,
        output_channel_count: int,
        input_channel_count: int = IMAGE_CHANNEL_COUNT,
        base_channel_count: int = DEFAULT_BASE_CHANNEL_COUNT,
        level_count: int = DEFAULT_LEVEL_COUNT,
    ):
        """
        Construct the model.

        Parameters
        ----------
            base_height: input images are resized to this before processing.
            base_width: input images are resized to this before processing.
            output_channel_count: number of channels in model output, one per
                class, at most 256.
            input_channel_count: number of channels in model input.
            base_channel_count: number of channels in 1st level's output.
            level_count: number of levels of processing.
        """

        super().__init__()

        if not 2 <= output_channel_count <= 256:
            raise ValueError(
                f"{output_channel_count=} must be in [2, 256]. "
                "Class predictions will be returned as np.uint8 values."
            )

        downsample_factor = 2 ** (level_count - 1)
        if base_height % downsample_factor or base_width % downsample_factor:
            raise ValueError(
                f"{base_height=} x {base_width=} must both be divisible by "
                f"{downsample_factor=}"
            )

        self.base_height = base_height
        self.base_width = base_width
        self.level_count = level_count

        # Level to number of channels in its output feature map set.
        level_to_channel_count = {
            "-1": base_channel_count // 2,  # Not a real level.
            **{
                str(level): base_channel_count * 2**level
                for level in range(level_count)
            },
        }

        # Converts input image to feature map set processable by 1st level.
        self.input_bridge = nn.Conv2d(
            in_channels=input_channel_count,
            out_channels=level_to_channel_count["-1"],
            kernel_size=self.CONV_KERNEL_SIZE,
            padding=self.CONV_PADDING,
        )

        # An encoder-side convolution takes a feature map with C channels, and
        # returns one with 2*C channels.
        self.level_to_encoder_conv_op = nn.ModuleDict(
            {
                str(level): _ConvOp(
                    input_channel_count=level_to_channel_count[str(level - 1)],
                    output_channel_count=level_to_channel_count[str(level)],
                    conv_kernel_size=self.CONV_KERNEL_SIZE,
                    conv_padding=self.CONV_PADDING,
                )
                for level in range(level_count)
            }
        )
        # Down-sampling happens at the end of most encoder-side levels.
        self.level_to_downsample_op = nn.ModuleDict(
            {str(level): _DownSampleOp() for level in range(level_count - 1)}
            | {str(level_count - 1): None}
        )

        # A decoder-side convolution takes a feature map with C channels, and
        # returns one with C//2 channels.
        self.level_to_decoder_conv_op = nn.ModuleDict(
            {
                str(level): (
                    _ConvOp(
                        input_channel_count=(2 * level_to_channel_count[str(level)]),
                        output_channel_count=level_to_channel_count[str(level)],
                        conv_kernel_size=self.CONV_KERNEL_SIZE,
                        conv_padding=self.CONV_PADDING,
                    )
                )
                for level in range(level_count - 1)
            }
            | {str(level_count - 1): None}
        )
        # Up-sampling happens before most decoder-side levels.
        self.level_to_upsample_op = nn.ModuleDict(
            {
                str(level): (
                    _UpSampleOp(
                        input_channel_count=level_to_channel_count[str(level + 1)],
                        output_channel_count=level_to_channel_count[str(level)],
                    )
                )
                for level in range(level_count - 1)
            }
            | {str(level_count - 1): None}
        )

        # Converts final feature maps into output score images.
        self.output_bridge = nn.Conv2d(
            in_channels=level_to_channel_count["0"],
            out_channels=output_channel_count,
            kernel_size=self.CONV_KERNEL_SIZE,
            padding=self.CONV_PADDING,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters owned directly by this module.
        """

        fan_in = _fan_in(self.input_bridge)
        nn.init.normal_(self.input_bridge.weight, mean=0.0, std=(1 / fan_in) ** 0.5)
        nn.init.zeros_(self.input_bridge.bias)

        fan_in = _fan_in(self.output_bridge)
        nn.init.normal_(
            self.output_bridge.weight, mean=0.0, std=0.01 * (1 / fan_in) ** 0.5
        )
        nn.init.zeros_(self.output_bridge.bias)

    @staticmethod
    def _resize(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if x.shape[-2:] == (height, width):
            return x
        return F.interpolate(
            input=x,
            size=(height, width),
            mode="bilinear",
            antialias=True,  # Consider the whole receptive field.
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass on x, a (B, C, base_H, base_W).
        """

        x = self.input_bridge(x)

        # Encoder-side, from high level to low.
        level_to_skip = {}
        for level in range(self.level_count):
            x = self.level_to_encoder_conv_op[str(level)](x)
            level_to_skip[str(level)] = x

            downsample_op = self.level_to_downsample_op[str(level)]
            if downsample_op is not None:
                x = downsample_op(x)

        # Decoder-side, from low level to high.
        for level in reversed(range(self.level_count)):
            upsample_op = self.level_to_upsample_op[str(level)]
            if upsample_op is not None:
                x = upsample_op(x)

            decoder_conv_op = self.level_to_decoder_conv_op[str(level)]
            if decoder_conv_op is not None:
                x = decoder_conv_op(torch.cat([x, level_to_skip[str(level)]], dim=1))

        return self.output_bridge(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass on x, a (B, C, H, W) tensor of any H and W.

        Returns a (B, C_out, H, W) tensor of scores.
        """

        original_height, original_width = x.shape[-2:]

        x = self._resize(x, self.base_height, self.base_width)
        x = self._forward(x)
        return self._resize(x, original_height, original_width)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for x, a (B, C, H, W) tensor of any H and W.

        Returns a (B, 1, H, W) uint8 tensor of class labels.
        """

        return self(x).argmax(dim=-3, keepdim=True).to(torch.uint8)
