from __future__ import annotations

import torch
import torch.nn as nn

from models.diffusion import ImageEncoder, SpatialPairImageEncoder


class DirectStateRegressor(nn.Module):
    """Direct image -> normalized Blender state baseline.

    This intentionally removes diffusion/noising/sampling from the problem while
    reusing the same image encoder families. It is a diagnostic baseline for
    testing whether image conditioning can predict the scene state at all.
    """

    def __init__(
        self,
        state_dim: int,
        image_channels: int,
        encoder_type: str = "spatial_pair",
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.image_channels = int(image_channels)
        self.encoder_type = str(encoder_type)

        if self.encoder_type == "legacy":
            self.image_encoder = ImageEncoder(cond_dim, in_channels=self.image_channels)
        elif self.encoder_type == "spatial_pair":
            if self.image_channels != 6:
                raise ValueError(
                    "encoder_type='spatial_pair' requires 6 input channels (two RGB branches)"
                )
            self.image_encoder = SpatialPairImageEncoder(cond_dim)
        else:
            raise ValueError(
                f"Unknown encoder_type={self.encoder_type!r}; expected legacy or spatial_pair"
            )

        # Keep this close to the diffusion denoiser capacity, but remove x_t and t.
        self.head = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, self.state_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(image)
        return self.head(features)
