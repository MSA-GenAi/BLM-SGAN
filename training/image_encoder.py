"""DAMSM image encoder (AttnGAN): Inception-v3 backbone (frozen) + two trainable 256-d heads.

Same parameter names and forward pass as DAMSM.CNN_ENCODER, so AttnGAN's image_encoder200.pth loads
into it (useful as an initialisation). The backbone always stays in eval mode, so its BatchNorm
statistics never drift (AttnGAN let them update while the weights were frozen).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

BACKBONE = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
            "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e",
            "Mixed_7a", "Mixed_7b", "Mixed_7c"]


class ImageEncoder(nn.Module):
    def __init__(self, emb_dim: int = 256, imagenet_weights: bool = True):
        super().__init__()
        from torchvision.models import Inception_V3_Weights
        net = torchvision.models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if imagenet_weights else None,
            aux_logits=True, init_weights=False, transform_input=False)
        for name in BACKBONE:
            setattr(self, name, getattr(net, name))
        for p in self.parameters():
            p.requires_grad = False
        self.emb_features = nn.Conv2d(768, emb_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.emb_cnn_code = nn.Linear(2048, emb_dim)
        nn.init.uniform_(self.emb_features.weight, -0.1, 0.1)
        nn.init.uniform_(self.emb_cnn_code.weight, -0.1, 0.1)

    def train(self, mode: bool = True):
        super().train(mode)
        for name in BACKBONE:
            getattr(self, name).eval()
        return self

    def features(self, x: torch.Tensor):
        """x: B x 3 x H x W in [-1, 1] -> (B x 768 x 17 x 17 regions, B x 2048 global)."""
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        regions = x
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return regions, x

    def heads(self, regions: torch.Tensor, pooled: torch.Tensor):
        return self.emb_features(regions), self.emb_cnn_code(pooled)

    def forward(self, x: torch.Tensor):
        """-> (B x emb_dim x 17 x 17 region features, B x emb_dim global code)."""
        return self.heads(*self.features(x))

    @classmethod
    def load(cls, path, map_location="cpu") -> "ImageEncoder":
        sd = torch.load(path, map_location=map_location, weights_only=True)
        sd = sd.get("state_dict", sd)
        enc = cls(emb_dim=sd["emb_cnn_code.weight"].shape[0], imagenet_weights=False)
        enc.load_state_dict(sd, strict=True)
        return enc
