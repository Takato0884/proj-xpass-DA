import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights
import open_clip

num_bins = 7

_BACKBONE_OUT_DIM = {
    'resnet50': 2048,
    'i3d': 512,
    'vit_b_16': 768,
    'clip_rn50': 1024,
    'clip_vit_b16': 512,
}


class EarthMoverDistance(nn.Module):
    def __init__(self, dim=-1):
        super(EarthMoverDistance, self).__init__()
        self.dim = dim

    def forward(self, x, y):
        """
        Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
        """
        cdf_x = torch.cumsum(x, dim=self.dim)
        cdf_y = torch.cumsum(y, dim=self.dim)
        emd = torch.norm(cdf_x - cdf_y, p=2, dim=self.dim)
        return emd

earth_mover_distance = EarthMoverDistance()


class NIMA(nn.Module):
    """NIMA model with selectable backbone.

    Backbones supported:
    - 'resnet50'    (2D, expects `[B, C, H, W]`)
    - 'i3d'         (3D, expects `[B, C, T, H, W]`)
    - 'vit_b_16'    (2D, expects `[B, C, H, W]`)
    - 'clip_rn50'   (2D, expects `[B, C, H, W]`)
    - 'clip_vit_b16'(2D, expects `[B, C, H, W]`)
    """
    def __init__(self, num_bins_aesthetic, backbone='resnet50', dropout=0.0, feat_dim=256):
        super(NIMA, self).__init__()

        self.backbone_name = backbone
        self.is_3d = False

        if backbone == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            backbone_out_features = self.backbone.fc.in_features  # 2048
            self.backbone.fc = nn.Identity()

        elif backbone == 'i3d':
            self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            backbone_out_features = self.backbone.fc.in_features  # 512
            self.backbone.fc = nn.Identity()
            self.is_3d = True

        elif backbone == 'vit_b_16':
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            backbone_out_features = self.backbone.heads.head.in_features  # 768
            self.backbone.heads = nn.Identity()

        elif backbone == 'clip_rn50':
            clip_model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
            self.backbone = clip_model.visual.float()
            backbone_out_features = 1024

        elif backbone == 'clip_vit_b16':
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            self.backbone = clip_model.visual.float()
            backbone_out_features = 512

        else:
            raise ValueError(f"Backbone '{backbone}' is not supported. "
                             f"Choose from: resnet50, i3d, vit_b_16, clip_rn50, clip_vit_b16")

        self.feat_dim = feat_dim

        def _drop():
            return nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.feat_proj = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            _drop(),
            nn.Linear(512, feat_dim),
        )
        self.fc_aesthetic = nn.Linear(feat_dim, num_bins_aesthetic)

    def freeze_backbone(self):
        """Freeze entire backbone. Only fc_aesthetic remains trainable."""
        backbone = self.backbone
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

    def _set_frozen_modules_eval(self):
        """Ensure frozen backbone stays in eval mode during model.train()."""
        backbone = self.backbone
        if not any(p.requires_grad for p in backbone.parameters()):
            backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._set_frozen_modules_eval()
        return self

    def forward(self, x, return_feat=False):
        # For 2D backbone expect `[B, C, H, W]`.
        # For 3D backbone expect `[B, C, T, H, W]`.
        raw_feat = self.backbone(x)
        domain_feat = self.feat_proj(raw_feat)
        aesthetic_logits = self.fc_aesthetic(domain_feat)
        if return_feat:
            return aesthetic_logits, domain_feat, raw_feat
        return aesthetic_logits


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.saved_tensors[0].item()
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def discover_folds(root_dir, version_prefix):
    """Discover all fold directories for a given version prefix (e.g., 'v3' -> ['v3_fold1', ...])."""
    split_dir = os.path.join(root_dir, 'split')
    folds = sorted([
        d for d in os.listdir(split_dir)
        if d.startswith(f'{version_prefix}_fold') and os.path.isdir(os.path.join(split_dir, d))
    ])
    return folds


# ─── Domain Adaptation Shared Utilities ───────────────────────────────────────

def parse_dann_target(dann_target: str) -> str:
    """'DANN-fashion' → 'fashion'."""
    if dann_target and dann_target.startswith('DANN-'):
        return dann_target[5:]
    return dann_target


def get_da_lambda(epoch: int, total_steps: int, gamma: float = 10.0) -> float:
    """λ schedule from Ganin et al. (2016). λ-scheduled DA 手法で共通利用。"""
    p = min(epoch / max(total_steps, 1), 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
