from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class LightweightUNet(nn.Module):
    """Lightweight UNet for dense multi-trait regression built on SMP."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 31,
        n_traits: int | None = None,
        encoder_name: str = "resnet18",
        encoder_weights: str | None = None,
        decoder_channels: tuple[int, int, int, int, int] | list[int] = (
            128,
            64,
            32,
            16,
            8,
        ),
        decoder_use_norm: bool | str = "batchnorm",
        decoder_attention_type: str | None = None,
    ) -> None:
        super().__init__()

        if n_traits is not None:
            out_channels = int(n_traits)

        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}.")

        decoder_channels = tuple(int(channel) for channel in decoder_channels)
        if len(decoder_channels) != 5:
            raise ValueError(
                f"decoder_channels must have 5 entries for a 5-stage UNet, got {len(decoder_channels)}."
            )

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights=encoder_weights,
            decoder_use_norm=decoder_use_norm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_traits = out_channels
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.decoder_channels = decoder_channels
        self.decoder_use_norm = decoder_use_norm
        self.decoder_attention_type = decoder_attention_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input of shape (B, C, H, W), got {tuple(x.shape)}"
            )
        return self.model(x)
