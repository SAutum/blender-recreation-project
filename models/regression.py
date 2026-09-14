from __future__ import annotations

import torch
import torch.nn as nn

from models.diffusion import ImageEncoder, SpatialPairImageEncoder


class AttentionPairImageEncoder(nn.Module):
    """Shared-CNN paired-view encoder with spatial self-attention.

    Each RGB view is first converted into an 8x8 feature grid by the same CNN.
    The two grids are then flattened into spatial tokens, augmented with learned
    positional and view embeddings, and processed jointly by a Transformer.
    A CLS token summarizes the full mono/stereo pair for state regression.
    """

    def __init__(
        self,
        out_dim: int = 384,
        model_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1536,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.model_dim = int(model_dim)
        self.grid_size = 8
        self.tokens_per_view = self.grid_size * self.grid_size

        def gn(channels: int) -> nn.GroupNorm:
            groups = 8 if channels >= 8 else 1
            return nn.GroupNorm(groups, channels)

        # Shared feature extractor. For 128x128 input this naturally reaches 8x8;
        # AdaptiveAvgPool keeps the token count fixed if image size changes later.
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            gn(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            gn(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            gn(256),
            nn.SiLU(),
            nn.Conv2d(256, model_dim, kernel_size=3, stride=2, padding=1),
            gn(model_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size)),
        )

        # One position table is shared between views; view embedding tells the
        # Transformer whether a token came from left/first or right/second view.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.tokens_per_view, model_dim)
        )
        self.view_embedding = nn.Parameter(torch.zeros(1, 2, 1, model_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.cls_position = nn.Parameter(torch.zeros(1, 1, model_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.projection = (
            nn.Identity() if out_dim == model_dim else nn.Linear(model_dim, out_dim)
        )

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_position, std=0.02)

    def _tokens(self, image: torch.Tensor) -> torch.Tensor:
        features = self.shared_cnn(image)
        return features.flatten(2).transpose(1, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 6:
            raise ValueError(
                "AttentionPairImageEncoder expects Bx6xHxW input (two RGB views)"
            )

        left = image[:, :3]
        right = image[:, 3:6]
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)

        left_tokens = (
            left_tokens
            + self.position_embedding
            + self.view_embedding[:, 0]
        )
        right_tokens = (
            right_tokens
            + self.position_embedding
            + self.view_embedding[:, 1]
        )

        tokens = torch.cat([left_tokens, right_tokens], dim=1)
        cls = self.cls_token.expand(image.shape[0], -1, -1) + self.cls_position
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.transformer(tokens)
        return self.projection(encoded[:, 0])


class DirectStateRegressor(nn.Module):
    """Direct image -> normalized Blender state baseline."""

    def __init__(
        self,
        state_dim: int,
        image_channels: int,
        encoder_type: str = "spatial_pair",
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.image_channels = int(image_channels)
        self.encoder_type = str(encoder_type)

        if self.encoder_type == "legacy":
            feature_dim = 256 if cond_dim is None else int(cond_dim)
            self.image_encoder = ImageEncoder(feature_dim, in_channels=self.image_channels)
        elif self.encoder_type == "spatial_pair":
            if self.image_channels != 6:
                raise ValueError(
                    "encoder_type='spatial_pair' requires 6 input channels (two RGB branches)"
                )
            feature_dim = 256 if cond_dim is None else int(cond_dim)
            self.image_encoder = SpatialPairImageEncoder(feature_dim)
        elif self.encoder_type == "attention_pair":
            if self.image_channels != 6:
                raise ValueError(
                    "encoder_type='attention_pair' requires 6 input channels (two RGB branches)"
                )
            feature_dim = 384 if cond_dim is None else int(cond_dim)
            self.image_encoder = AttentionPairImageEncoder(out_dim=feature_dim)
        else:
            raise ValueError(
                f"Unknown encoder_type={self.encoder_type!r}; expected legacy, spatial_pair, or attention_pair"
            )

        # The attention variant already has much more encoder capacity. Keep one
        # common regression head so the main architectural change stays localized
        # to how image information is extracted and related spatially.
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
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
