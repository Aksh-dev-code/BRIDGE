"""Transformer classifier for sign keypoint sequences.

Ported as-is from the notebook (Step 4) so old checkpoints keep loading.
"""

import math

import torch
import torch.nn as nn

from src import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SignTransformer(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, d_model=256, nhead=8, num_layers=4,
                 num_classes=262, max_len=config.MAX_FRAMES, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x[:, 0, :])


def build_model(num_classes, **overrides):
    """Create a SignTransformer from the default config plus overrides."""
    cfg = {**config.MODEL_CONFIG, "num_classes": num_classes, **overrides}
    return SignTransformer(**cfg)


def save_bundle(model, class_names, path, num_classes=None, **cfg_overrides):
    """Save weights + class names + architecture config in one file (notebook 'save' step)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {**config.MODEL_CONFIG,
           "num_classes": num_classes if num_classes is not None else len(class_names),
           **cfg_overrides}
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": list(class_names),
        "config": cfg,
    }, path)
    return path


def load_bundle(path=None, map_location="cpu"):
    """Load a saved bundle -> (model in eval mode, class_names, config)."""
    path = path or config.MODEL_BUNDLE
    bundle = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(bundle, dict) or "model_state_dict" not in bundle:
        raise ValueError(
            f"{path} is not a full bundle (no 'model_state_dict'). "
            "It looks like a bare state_dict — retrain, or run `train` to regenerate it."
        )

    cfg = dict(bundle["config"])
    class_names = list(bundle["class_names"])
    model = SignTransformer(**cfg)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(map_location).eval()
    return model, class_names, cfg
