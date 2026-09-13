"""EfficientNet-B3 + Dropout(0.3) + Linear head. Verbatim from every
reported ABS run's EfficientNetBasic."""
import torch.nn as nn
from torchvision import models


class EfficientNetBasic(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.base     = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT)
        feat_dim      = self.base.classifier[1].in_features
        self.base.classifier = nn.Identity()
        self.dropout  = nn.Dropout(0.3)
        self.fc       = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_features=False):
        features = self.base(x)
        features = self.dropout(features)
        logits   = self.fc(features)
        if return_features:
            return logits, features
        return logits

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
