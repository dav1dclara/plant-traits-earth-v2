import torch
import torch.nn as nn
from typing import Literal


class PadarianInspiredPatchCNN(nn.Module):
    """
    Padarian-inspired patch CNN baseline.

    Adjustments vs your current code:
      - dropout default = 0.3 (closer to Padarian Table 1)
      - optional head_type:
          * "single": one shared head Linear(hidden -> n_traits)
          * "multi" : n_traits separate heads Linear(hidden -> 1)

    Architecture:
      Conv(3x3,16) -> ReLU -> MaxPool(2) -> Dropout(p)
      Conv(3x3,32) -> ReLU
      Flatten -> LazyLinear(hidden) -> ReLU -> Dropout(p)
      Head(s) -> (B, n_traits)
    """

    def __init__(
        self,
        in_channels: int,
        n_traits: int = 31,
        hidden_dim: int = 64,
        dropout_p: float = 0.3,
        head_type: Literal["single", "multi"] = "single",
    ):
        super().__init__()
        self.n_traits = n_traits
        self.head_type = head_type

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        if head_type == "single":
            self.head = nn.LazyLinear(n_traits)
            self.heads = None
        elif head_type == "multi":
            self.head = None
            self.heads = nn.ModuleList([nn.LazyLinear(1) for _ in range(n_traits)])
        else:
            raise ValueError(
                f"Unknown head_type='{head_type}'. Use 'single' or 'multi'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x to be (B,C,H,W), got {tuple(x.shape)}")

        h = self.backbone(x)  # (B, hidden_dim)

        if self.head_type == "single":
            return self.head(h)  # (B, n_traits)

        return torch.cat([head(h) for head in self.heads], dim=1)  # (B, n_traits)
