import os
import wandb
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import copy
import pandas as pd
from collections import defaultdict
import json
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights
import open_clip
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from torch.amp import autocast, GradScaler

from .utils.argflags import parse_arguments, model_dir, wandb_tags
from .datasets import load_data, collate_fn, multi_domain_collate_fn
from .datamodules import build_global_encoders

class NIMA(nn.Module):
    def __init__(self, num_bins_aesthetic, backbone):
        super(NIMA, self).__init__()
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

        self.fc_aesthetic = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_bins_aesthetic)
        )

    def forward(self, images):
        # images expected shape: [B, C, H, W]
        x = self.backbone(images)  # Pass images through the selected backbone
        aesthetic_logits = self.fc_aesthetic(x)
        return aesthetic_logits
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SharedMLP(nn.Module):
    """
    MLP with shared lower layers and genre-specific upper layers.
    """
    def __init__(self, input_size, shared_hidden_size, genre_hidden_size, output_size, genres):
        super(SharedMLP, self).__init__()
        self.genres = genres
        # Shared lower layers (common across all genres)
        self.shared_fc1 = nn.Linear(input_size, shared_hidden_size)
        self.shared_fc2 = nn.Linear(shared_hidden_size, shared_hidden_size)
        # Genre-specific upper layers
        self.genre_fc1_dict = nn.ModuleDict()
        self.genre_fc2_dict = nn.ModuleDict()
        for genre in self.genres:
            self.genre_fc1_dict[genre] = nn.Linear(shared_hidden_size, genre_hidden_size)
            self.genre_fc2_dict[genre] = nn.Linear(genre_hidden_size, output_size)
    
    def forward(self, x, genre):
        # Shared lower layers
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        # Genre-specific upper layers
        x = F.relu(self.genre_fc1_dict[genre](x))
        x = self.genre_fc2_dict[genre](x)
        return x
    
class PIAA_MIR_CrossDomain(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, genres, backbone_dict, hidden_size=1024, dropout=None, use_uncertainty_weighting=False):
        super(PIAA_MIR_CrossDomain, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.genres = genres  # list of genre names, e.g., ['art', 'fashion', 'scenery']
        self.register_buffer('scale', torch.arange(0, num_bins).float())
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Uncertainty weighting parameters (log variance for each genre)
        # Loss = (1 / (2 * sigma^2)) * L + log(sigma)
        # We parameterize as log_var = log(sigma^2) for numerical stability
        if self.use_uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                genre: nn.Parameter(torch.zeros(1))
                for genre in genres
            })
        
        # Genre-specific NIMA backbones with genre-specific backbone types
        self.nima_dict = nn.ModuleDict()
        for genre in genres:
            backbone_type = backbone_dict.get(genre, 'resnet50')
            self.nima_dict[genre] = NIMA(num_bins, backbone_type)
        
        # Interaction MLPs with shared lower layers + genre-specific upper layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout) if dropout is not None and dropout > 0 else None
        
        # SharedMLP1: for interaction features (image_attr * personal_traits)
        # input: (num_attr * num_pt), shared_hidden: hidden_size, genre_hidden: hidden_size // 2, output: 1
        self.mlp1 = SharedMLP(
            input_size=(num_attr * num_pt),
            shared_hidden_size=hidden_size,
            genre_hidden_size=hidden_size // 2,
            output_size=1,
            genres=genres)

        # mlp2: genre-specific (no shared part)
        # input: num_bins, hidden: 32, output: 1
        self.mlp2_dict = nn.ModuleDict()
        for genre in genres:
            self.mlp2_dict[genre] = MLP(num_bins, 16, 1)

    def freeze_backbone(self):
        """
        Freeze entire backbone of all NIMA models.
        Only fc_aesthetic remains trainable (plus mlp1/mlp2_dict outside NIMA).
        """
        for genre, nima in self.nima_dict.items():
            backbone = nima.backbone
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()

    def _set_frozen_modules_eval(self):
        """Ensure frozen backbone modules stay in eval mode during model.train()."""
        for genre, nima in self.nima_dict.items():
            backbone = nima.backbone
            if not any(p.requires_grad for p in backbone.parameters()):
                backbone.eval()

    def train(self, mode=True):
        """Override train() to keep frozen backbone modules in eval mode."""
        super().train(mode)
        if mode:
            self._set_frozen_modules_eval()
        return self

    def forward(self, images, personal_traits, image_attributes, genre):
        """
        Forward pass for a specific genre.
        Args:
            images: [B, C, H, W]
            personal_traits: [B, num_pt]
            image_attributes: [B, num_attr]
            genre: str, the genre for this batch
        """
        # Use genre-specific NIMA
        logit = self.nima_dict[genre](images)
        prob = F.softmax(logit, dim=1)

        # Interaction map calculation
        A_ij = image_attributes.unsqueeze(2) * personal_traits.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        
        # Use SharedMLP1 (shared lower, genre-specific upper) and mlp2_dict (genre-specific only)
        interaction_outputs = self.mlp1(I_ij, genre)
        direct_outputs = self.mlp2_dict[genre](prob * self.scale)
        output = interaction_outputs + direct_outputs
        return output