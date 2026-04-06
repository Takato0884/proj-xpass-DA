import os
import random as pyrandom
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
from torch.utils.data import DataLoader, Sampler
import torch.nn as nn
from torch.amp import autocast, GradScaler

from .argflags import parse_arguments, model_dir, wandb_tags
from .data import load_data, collate_fn, build_global_encoders
from .train_common import NIMA, discover_folds, num_bins, _BACKBONE_OUT_DIM

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.drop is not None:
            x = self.drop(x)
        return self.fc2(x)

class InternalInteraction(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super(InternalInteraction, self).__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, attribute_embeddings):
        batch_size, num_attributes, _ = attribute_embeddings.shape

        # Compute all pairwise element-wise products at once via broadcasting
        # [B, num_attr, 1, dim] * [B, 1, num_attr, dim] -> [B, num_attr, num_attr, dim]
        combined = attribute_embeddings.unsqueeze(2) * attribute_embeddings.unsqueeze(1)

        # Apply MLP to all pairs in a single forward pass
        out = self.mlp(combined.view(batch_size, num_attributes * num_attributes, self.input_dim))

        # Sum over the i dimension
        aggregated_interactions = out.view(batch_size, num_attributes, num_attributes, self.input_dim).sum(dim=1)
        return aggregated_interactions

class ExternalInteraction(nn.Module):
    def __init__(self):
        super(ExternalInteraction, self).__init__()

    def forward(self, user_attributes, image_attributes):
        # user_attributes and image_attributes have shapes [B, num_attr]
        # Compute the outer product: the result will have shape [B, num_attr_user, num_attr_img]
        interaction_results = user_attributes.unsqueeze(2) * image_attributes.unsqueeze(1)

        aggregated_interactions_user = torch.sum(interaction_results, dim=2)
        aggregated_interactions_img = torch.sum(interaction_results, dim=1)
        return aggregated_interactions_user, aggregated_interactions_img

class Interfusion_GRU(nn.Module):
    def __init__(self, input_dim=64):
        super(Interfusion_GRU, self).__init__()
        self.gru = nn.GRUCell(input_dim, input_dim)

    def forward(self, initial_node, internal_interaction, external_interaction):
        num_attr = initial_node.shape[1]
        results = []
        for i in range(num_attr):
            fused_node = self.gru(initial_node[:, i], None)
            fused_node = self.gru(internal_interaction[:, i], fused_node)
            fused_node = self.gru(external_interaction[:, i], fused_node)
            results.append(fused_node)
        results = torch.stack(results, dim=1)
        return results

class PIAA_ICI_CrossDomain(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, genres, backbone_dict, input_dim=64, hidden_size=256, dropout=None, use_backbone_image=False):
        super(PIAA_ICI_CrossDomain, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.genres = genres  # list of genre names, e.g., ['art']
        self.register_buffer('scale', torch.arange(0, num_bins).float())
        self.input_dim = input_dim

        # Genre-specific NIMA backbones with genre-specific backbone types
        self.nima_dict = nn.ModuleDict()
        for genre in genres:
            backbone_type = backbone_dict.get(genre, 'resnet50')
            self.nima_dict[genre] = NIMA(num_bins, backbone_type, dropout=dropout if dropout else 0.0)

        # Internal and External Interaction Modules (shared across genres)
        self.internal_interaction_img = InternalInteraction(input_dim=input_dim, hidden_dim=hidden_size, dropout=dropout if dropout else 0.0)
        self.internal_interaction_user = InternalInteraction(input_dim=input_dim, hidden_dim=hidden_size, dropout=dropout if dropout else 0.0)
        self.external_interaction = ExternalInteraction()

        # Interfusion Module (shared across genres)
        self.interfusion_img = Interfusion_GRU(input_dim=input_dim)
        self.interfusion_user = Interfusion_GRU(input_dim=input_dim)

        # MLPs (shared across genres)
        _dropout = dropout if dropout else 0.0
        self.node_attr_user = MLP(num_pt, hidden_size, num_attr*input_dim, dropout=_dropout)
        node_attr_img_input_dim = num_attr + input_dim if use_backbone_image else num_attr
        self.node_attr_img = MLP(node_attr_img_input_dim, hidden_size, num_attr*input_dim, dropout=_dropout)

        # MLP for attr_corr
        self.attr_corr = MLP(input_dim, hidden_size, 1, dropout=_dropout)

        # Genre-specific mlp_dist
        self.mlp_dist_dict = nn.ModuleDict()
        for genre in genres:
            self.mlp_dist_dict[genre] = MLP(num_bins, hidden_size, 1, dropout=_dropout)

        # Backbone image projection (optional)
        self.use_backbone_image = use_backbone_image
        if use_backbone_image:
            self.backbone_image_proj = nn.ModuleDict()
            for genre in genres:
                backbone_type = backbone_dict.get(genre, 'resnet50')
                in_dim = _BACKBONE_OUT_DIM[backbone_type]
                self.backbone_image_proj[genre] = nn.Sequential(
                        nn.Linear(in_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, input_dim),
                    )


    def freeze_backbone(self):
        """
        Freeze entire backbone of all NIMA models.
        Only fc_aesthetic remains trainable (plus ICI interaction modules outside NIMA).
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
        if self.use_backbone_image:
            logit, backbone_feat = self.nima_dict[genre](images, return_feat=True)
        else:
            logit = self.nima_dict[genre](images)
        prob = F.softmax(logit, dim=1)

        # Build attribute nodes from provided image_attributes and user traits
        n_attr = image_attributes.shape[1]
        if self.use_backbone_image:
            img_feat = self.backbone_image_proj[genre](backbone_feat)  # [B, input_dim]
            img_input = torch.cat([image_attributes, img_feat], dim=1)  # [B, num_attr + input_dim]
            attr_img = self.node_attr_img(img_input).view(-1, n_attr, self.input_dim)
        else:
            attr_img = self.node_attr_img(image_attributes).view(-1, n_attr, self.input_dim)
        attr_user = self.node_attr_user(personal_traits).view(-1, n_attr, self.input_dim)

        # Internal Interaction (among image attributes)
        internal_img = self.internal_interaction_img(attr_img)
        internal_user = self.internal_interaction_user(attr_user)
        # External Interaction (between user and image attributes)
        aggregated_interactions_user, aggregated_interactions_img = self.external_interaction(attr_user, attr_img)

        # Interfusion to combine interactions and initial attributes
        fused_features_img = self.interfusion_img(attr_img, internal_img, aggregated_interactions_img)
        fused_features_user = self.interfusion_user(attr_user, internal_user, aggregated_interactions_user)

        # Final prediction
        I_ij = torch.sum(fused_features_img, dim=1, keepdim=False) + torch.sum(fused_features_user, dim=1, keepdim=False)
        interaction_outputs = self.attr_corr(I_ij)
        direct_outputs = self.mlp_dist_dict[genre](prob * self.scale)
        output = interaction_outputs + direct_outputs
        return output

_SAMPLES_PER_USER = 32


class UserGroupedBatchSampler(Sampler):
    """
    Each batch contains n_users_per_batch = batch_size // _SAMPLES_PER_USER users,
    each contributing _SAMPLES_PER_USER samples (without replacement within batch).
    Users are sampled with replacement across batches; no cross-batch deduplication.
    """
    def __init__(self, dataset, batch_size):
        self.user_to_indices = defaultdict(list)
        for i, uid in enumerate(dataset.data['user_id'].values):
            self.user_to_indices[uid].append(i)
        self.users = list(self.user_to_indices.keys())
        self.n_users_per_batch = max(1, batch_size // _SAMPLES_PER_USER)
        self.steps = max(1, len(dataset) // batch_size)

    def __iter__(self):
        for _ in range(self.steps):
            selected_users = pyrandom.choices(self.users, k=self.n_users_per_batch)
            batch = []
            for uid in selected_users:
                indices = self.user_to_indices[uid]
                k = min(_SAMPLES_PER_USER, len(indices))
                batch += pyrandom.sample(indices, k)
            yield batch

    def __len__(self):
        return self.steps


def _normalize_user_ids(user_ids, device):
    """Convert various user_id formats from collate_fn to a 1D long tensor."""
    if isinstance(user_ids, torch.Tensor):
        return user_ids.view(-1).long().to(device)
    flat = [int(u.item()) if isinstance(u, torch.Tensor) else int(u) for u in user_ids]
    return torch.tensor(flat, dtype=torch.long, device=device)


def _collect_user_ids(user_ids) -> list:
    """Convert various user_id formats from collate_fn to a flat list of ints."""
    if isinstance(user_ids, torch.Tensor):
        return user_ids.view(-1).cpu().numpy().tolist()
    result = []
    for u in user_ids:
        result.append(int(u.item()) if isinstance(u, torch.Tensor) else int(u))
    return result


def ccc_loss_per_user(preds, targets, user_ids):
    """
    Compute (1 - CCC) per user, return the mean across users.
    preds/targets: [B] or [B,1]; user_ids: 1D long tensor on same device.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    losses = []
    for uid in user_ids.unique():
        mask = user_ids == uid
        if mask.sum() < 2:
            continue
        p = preds[mask]
        t = targets[mask]
        mu_p, mu_t = p.mean(), t.mean()
        var_p = p.var(unbiased=False)
        var_t = t.var(unbiased=False)
        cov = ((p - mu_p) * (t - mu_t)).mean()
        ccc = 2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8)
        losses.append(1.0 - ccc)
    if not losses:
        return preds.sum() * 0.0  # differentiable zero
    return torch.stack(losses).mean()


def compute_loss(preds, targets, user_ids, args):
    """
    Unified loss dispatcher.
    loss_type: 'mse' | 'rmse' | 'ccc' | 'ccc+rmse'
    For 'ccc+rmse': loss = ccc_weight * (1-CCC) + RMSE
    """
    loss_type = args.loss_type
    if loss_type == 'mse':
        return F.mse_loss(preds, targets)
    if loss_type == 'rmse':
        return torch.sqrt(F.mse_loss(preds, targets) + 1e-8)
    if loss_type == 'ccc':
        return ccc_loss_per_user(preds, targets, user_ids)
    if loss_type == 'ccc+rmse':
        rmse = torch.sqrt(F.mse_loss(preds, targets) + 1e-8)
        ccc_l = ccc_loss_per_user(preds, targets, user_ids)
        return args.ccc_weight * ccc_l + rmse
    raise ValueError(f"Unknown loss_type: {loss_type}")


def train(model, dataloader, optimizer, scaler, device, args, genre, epoch: int = None):
    """
    Train function for a single genre.
    Args:
        model: PIAA_ICI_CrossDomain model
        dataloader: DataLoader with collate_fn
        optimizer: optimizer
        scaler: GradScaler
        device: device
        args: arguments
        genre: genre name (str)
        epoch: current epoch number
    Returns:
        train_loss: average training loss
    """
    model.train()
    running_loss = 0.0
    total_batches = 0

    desc = f"Epoch {epoch} [Train]" if epoch is not None else "Train"
    progress_bar = tqdm(total=len(dataloader), leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample in dataloader:
        optimizer.zero_grad()
        images = sample['image'].to(device)
        sample_pt = sample['traits'].float().to(device)
        sample_attr = sample['QIP'].float().to(device)
        aesthetic_scores = sample['Aesthetic'].to(device).view(-1, 1)
        user_ids = _normalize_user_ids(sample['user_id'], device) if 'user_id' in sample else None

        with autocast('cuda'):
            score_pred = model(images, sample_pt, sample_attr, genre)
            loss = compute_loss(score_pred, aesthetic_scores, user_ids, args)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        total_batches += 1
        progress_bar.update(1)
        progress_bar.set_postfix({args.loss_type: loss.item()})

    progress_bar.close()
    return running_loss / total_batches if total_batches > 0 else 0.0

def evaluate(model, dataloaders_dict, device, epoch: int = None, phase_name: str = "Val"):
    """
    Evaluate function that processes all genres separately.
    Args:
        model: PIAA_ICI_CrossDomain model
        dataloaders_dict: dict of {genre: dataloader}
        device: device
        epoch: current epoch number
        phase_name: name of the phase (Val/Test)
    Returns:
        genre_metrics: dict of {genre: {'srocc': float, 'mae': float, 'ndcg@10': float, 'ccc': float}}
        total_mae_loss: average MAE loss across all genres
    """
    model.eval()

    desc = f"Epoch {epoch} [{phase_name}]" if epoch is not None else phase_name

    # Calculate total number of batches
    total_expected_batches = sum(len(loader) for loader in dataloaders_dict.values())
    progress_bar = tqdm(total=total_expected_batches, leave=True, desc=desc, position=1, ncols=120, colour="#fffb00", ascii="-=")

    # Store predictions and targets per genre
    genre_predictions = defaultdict(list)
    genre_targets = defaultdict(list)
    genre_user_ids = defaultdict(list)
    genre_mae_losses = {}
    genre_batch_counts = {}

    with torch.no_grad():
        for genre, dataloader in dataloaders_dict.items():
            running_mae = 0.0
            batch_count = 0

            for sample in dataloader:
                images = sample['image'].to(device)
                target = sample['Aesthetic'].to(device).view(-1, 1)
                sample_pt = sample['traits'].float().to(device)
                sample_attr = sample['QIP'].float().to(device)

                # Collect user ids for per-user SROCC
                if 'user_id' in sample:
                    genre_user_ids[genre].extend(_collect_user_ids(sample['user_id']))

                with autocast('cuda'):
                    outputs = model(images, sample_pt, sample_attr, genre)
                outputs = outputs.view(-1, 1)
                genre_predictions[genre].append(outputs.view(-1).cpu().numpy())
                genre_targets[genre].append(target.view(-1).cpu().numpy())
                mae = F.l1_loss(outputs, target)
                running_mae += mae.item()
                batch_count += 1
                progress_bar.update(1)
                progress_bar.set_postfix({'MAE': mae.item(), 'genre': genre})

            genre_mae_losses[genre] = running_mae
            genre_batch_counts[genre] = batch_count

    progress_bar.close()

    # Calculate genre-specific metrics
    genre_metrics = {}
    total_mae_loss = 0.0

    for genre in dataloaders_dict.keys():
        if len(genre_predictions[genre]) == 0:
            continue

        # Concatenate predictions and targets for this genre
        predicted_scores = np.concatenate(genre_predictions[genre], axis=0)
        true_scores = np.concatenate(genre_targets[genre], axis=0)
        user_ids = genre_user_ids[genre]
        if len(user_ids) == 0:
            raise ValueError(
                f"No user_id found for genre '{genre}'. "
                f"The dataset must contain 'user_id' field for per-user SROCC computation."
            )

        # Compute per-user SROCC, NDCG@10, and CCC for this genre
        unique_user_ids = np.unique(user_ids)
        sroccs = []
        ndcgs = []
        cccs = []
        for uid in unique_user_ids:
            uid_mask = (np.array(user_ids) == uid)
            n_samples = np.sum(uid_mask)
            if n_samples <= 1:
                raise ValueError(
                    f"User {uid} in genre '{genre}' has only {n_samples} sample(s). "
                    f"At least 2 samples are required for SROCC computation."
                )
            uid_pred = predicted_scores[uid_mask]
            uid_true = true_scores[uid_mask]
            uid_srocc, _ = spearmanr(uid_pred, uid_true)
            sroccs.append(uid_srocc)
            uid_ndcg = ndcg_score([uid_true], [uid_pred], k=10)
            ndcgs.append(uid_ndcg)
            mu_p, mu_t = uid_pred.mean(), uid_true.mean()
            var_p, var_t = uid_pred.var(), uid_true.var()
            cov = ((uid_pred - mu_p) * (uid_true - mu_t)).mean()
            uid_ccc = 2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8)
            cccs.append(float(uid_ccc))
        genre_srocc = np.mean(sroccs) if len(sroccs) > 0 else 0.0
        genre_ndcg = np.mean(ndcgs) if len(ndcgs) > 0 else 0.0
        genre_ccc = np.mean(cccs) if len(cccs) > 0 else 0.0

        # Calculate average MAE for this genre
        genre_mae = genre_mae_losses[genre] / genre_batch_counts[genre] if genre_batch_counts[genre] > 0 else 0.0

        genre_metrics[genre] = {
            'srocc': genre_srocc,
            'mae': genre_mae,
            'ndcg@10': genre_ndcg,
            'ccc': genre_ccc,
        }

        # Add to total MAE loss (sum across genres)
        total_mae_loss += genre_mae_losses[genre]

    # Average total MAE by total batch count
    total_batch_count = sum(genre_batch_counts.values())
    total_mae_loss = total_mae_loss / total_batch_count if total_batch_count > 0 else 0.0

    return genre_metrics, total_mae_loss

def evaluate_cross_domain(model, eval_dataloaders_dict, device, source_genres):
    """
    Cross-domain evaluation: evaluate target domain data using source domain heads.
    For single source head, use that head's output directly.
    For multiple source heads, average the outputs from all heads.

    Args:
        model: PIAA_ICI_CrossDomain model
        eval_dataloaders_dict: dict of {target_genre: dataloader} (target domain test data)
        device: device
        source_genres: list of source genre names (trained heads to use)
    Returns:
        cross_domain_results: dict of {target_genre: {
            'average': {'srocc': float, 'ndcg@10': float},
            'per_head': {source_genre: {'srocc': float, 'ndcg@10': float}},
            'per_user': {uid: {'srocc': float, 'ndcg@10': float}},
            'per_user_per_head': {uid: {source_genre: {'srocc': float, 'ndcg@10': float}}}
        }}
    """
    model.eval()
    cross_domain_results = {}

    for target_genre, dataloader in eval_dataloaders_dict.items():
        print(f"\n[Cross-Domain] Evaluating {target_genre} data with heads: {source_genres}")

        # Collect predictions per source head, targets, and user_ids
        head_predictions = {sg: [] for sg in source_genres}
        all_targets = []
        all_user_ids = []

        desc = f"Cross-Domain [{target_genre}]"
        progress_bar = tqdm(total=len(dataloader), leave=True, desc=desc, ncols=120, colour="#ff8800", ascii="-=")

        with torch.no_grad():
            for sample in dataloader:
                images = sample['image'].to(device)
                target = sample['Aesthetic'].to(device).view(-1, 1)
                sample_pt = sample['traits'].float().to(device)
                sample_attr = sample['QIP'].float().to(device)

                # Collect user ids
                if 'user_id' in sample:
                    all_user_ids.extend(_collect_user_ids(sample['user_id']))

                all_targets.append(target.view(-1).cpu().numpy())

                # Forward through each source head
                for sg in source_genres:
                    with autocast('cuda'):
                        outputs = model(images, sample_pt, sample_attr, sg)
                    head_predictions[sg].append(outputs.view(-1).cpu().numpy())

                progress_bar.update(1)

        progress_bar.close()

        # Concatenate
        true_scores = np.concatenate(all_targets, axis=0)
        user_ids = np.array(all_user_ids)
        unique_user_ids = np.unique(user_ids)

        per_head_preds = {}
        for sg in source_genres:
            per_head_preds[sg] = np.concatenate(head_predictions[sg], axis=0)

        # Averaged predictions across all source heads
        avg_preds = np.mean([per_head_preds[sg] for sg in source_genres], axis=0)

        # Per-user metrics for averaged predictions
        avg_sroccs, avg_ndcgs = [], []
        per_user_results = {}
        for uid in unique_user_ids:
            uid_mask = (user_ids == uid)
            n_samples = np.sum(uid_mask)
            if n_samples <= 1:
                raise ValueError(
                    f"User {uid} in target genre '{target_genre}' has only {n_samples} sample(s). "
                    f"At least 2 samples are required for SROCC computation."
                )
            uid_srocc, _ = spearmanr(avg_preds[uid_mask], true_scores[uid_mask])
            uid_ndcg = ndcg_score([true_scores[uid_mask]], [avg_preds[uid_mask]], k=10)
            avg_sroccs.append(uid_srocc)
            avg_ndcgs.append(uid_ndcg)
            per_user_results[str(int(uid))] = {
                'srocc': float(uid_srocc),
                'ndcg@10': float(uid_ndcg)
            }

        # Per-head per-user metrics
        per_head_metrics = {}
        per_user_per_head = {}
        for sg in source_genres:
            sg_sroccs, sg_ndcgs = [], []
            for uid in unique_user_ids:
                uid_mask = (user_ids == uid)
                n_samples = np.sum(uid_mask)
                if n_samples <= 1:
                    raise ValueError(
                        f"User {uid} in target genre '{target_genre}' has only {n_samples} sample(s). "
                        f"At least 2 samples are required for SROCC computation."
                    )
                uid_srocc, _ = spearmanr(per_head_preds[sg][uid_mask], true_scores[uid_mask])
                uid_ndcg = ndcg_score([true_scores[uid_mask]], [per_head_preds[sg][uid_mask]], k=10)
                sg_sroccs.append(uid_srocc)
                sg_ndcgs.append(uid_ndcg)
                uid_str = str(int(uid))
                if uid_str not in per_user_per_head:
                    per_user_per_head[uid_str] = {}
                per_user_per_head[uid_str][sg] = {
                    'srocc': float(uid_srocc),
                    'ndcg@10': float(uid_ndcg)
                }
            per_head_metrics[sg] = {
                'srocc': float(np.mean(sg_sroccs)) if sg_sroccs else 0.0,
                'ndcg@10': float(np.mean(sg_ndcgs)) if sg_ndcgs else 0.0,
            }

        cross_domain_results[target_genre] = {
            'source_heads': source_genres,
            'method': 'average',
            'average': {
                'srocc': float(np.mean(avg_sroccs)) if avg_sroccs else 0.0,
                'ndcg@10': float(np.mean(avg_ndcgs)) if avg_ndcgs else 0.0,
            },
            'per_head': per_head_metrics,
            'per_user': per_user_results,
            'per_user_per_head': per_user_per_head,
        }

        print(f"[Cross-Domain] {target_genre}: avg SROCC={cross_domain_results[target_genre]['average']['srocc']:.4f}, "
              f"avg NDCG@10={cross_domain_results[target_genre]['average']['ndcg@10']:.4f}")
        for sg, m in per_head_metrics.items():
            print(f"  head [{sg}]: SROCC={m['srocc']:.4f}, NDCG@10={m['ndcg@10']:.4f}")

    return cross_domain_results


def inference_finetune(datasets_dict, args, device, dirname, experiment_name, backbone_dict, eval_datasets_dict=None):
    """
    Inference for all users across all genres after finetuning.
    Args:
        datasets_dict: dict of {genre: {'train': ds, 'val': ds, 'test': ds}}
        args: arguments
        device: device
        dirname: directory name for saving models
        experiment_name: experiment name
        backbone_dict: dict of {genre: backbone_type}
        eval_datasets_dict: optional dict of {target_genre: {'test': dataset}} for cross-domain evaluation
    """
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre_str = '-'.join(genres)
    all_user_ids = set()
    genre_srocc_list = defaultdict(list)
    genre_mae_list = defaultdict(list)
    genre_ndcg_list = defaultdict(list)
    genre_ccc_list = defaultdict(list)
    for genre in genres:
        all_user_ids.update(datasets_dict[genre]['test'].data['user_id'].values)

    model_name_base = experiment_name

    results = {}
    cd_per_target = defaultdict(lambda: {'per_user': {}, 'per_user_per_head': {},
                                          'head_sroccs': defaultdict(list), 'head_ndcgs': defaultdict(list),
                                          'avg_sroccs': [], 'avg_ndcgs': []}) if eval_datasets_dict is not None else None

    for uid in sorted(list(all_user_ids)):
        print(f"Running inference for user {uid} using saved best model...")
        best_model_path = os.path.join(dirname, f'{genre_str}_ICI_user_{uid}_{model_name_base}_finetune.pth')
        model_user = PIAA_ICI_CrossDomain(num_bins, num_attr, num_pt, genres, backbone_dict, dropout=args.dropout, use_backbone_image=args.use_backbone_image).to(device)
        try:
            model_user.load_state_dict(torch.load(best_model_path))
        except Exception as e:
            print(f"Warning: best model not found for user {uid} at {best_model_path}, skipping. Error: {e}")
            results[uid] = (np.nan, np.nan)
            continue

        # In-domain evaluation
        test_loaders_dict = {}
        total_test_samples = 0
        for genre in genres:
            user_test_ds = copy.copy(datasets_dict[genre]['test'])
            user_test_ds.data = datasets_dict[genre]['test'].data[datasets_dict[genre]['test'].data['user_id'] == uid].reset_index(drop=True)
            if len(user_test_ds) > 0:
                test_loaders_dict[genre] = DataLoader(user_test_ds, batch_size=batch_size, shuffle=False,
                                                       num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
                total_test_samples += len(user_test_ds)
        if total_test_samples == 0:
            print(f"No test samples for user {uid}, skipping.")
            results[uid] = ({}, np.nan)
        else:
            genre_metrics, total_mae = evaluate(model_user, test_loaders_dict, device)
            for genre, metrics in genre_metrics.items():
                genre_srocc_list[genre].append(metrics['srocc'])
                genre_mae_list[genre].append(metrics['mae'])
                genre_ndcg_list[genre].append(metrics['ndcg@10'])
                genre_ccc_list[genre].append(metrics['ccc'])
            if args.is_log:
                for genre, metrics in genre_metrics.items():
                    wandb.log({
                        f"{genre}/Test SROCC user_{uid}": metrics['srocc'],
                        f"{genre}/Test NDCG@10 user_{uid}": metrics['ndcg@10'],
                    }, commit=False)
                wandb.log({}, commit=True)
            results[uid] = (genre_metrics, total_mae)

        # Cross-domain evaluation
        if eval_datasets_dict is not None:
            eval_loaders_dict = {}
            for target_genre, ds_dict in eval_datasets_dict.items():
                user_eval_ds = copy.copy(ds_dict['test'])
                user_eval_ds.data = ds_dict['test'].data[ds_dict['test'].data['user_id'] == uid].reset_index(drop=True)
                if len(user_eval_ds) > 0:
                    eval_loaders_dict[target_genre] = DataLoader(
                        user_eval_ds, batch_size=batch_size, shuffle=False,
                        num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
            if len(eval_loaders_dict) > 0:
                cd_results = evaluate_cross_domain(model_user, eval_loaders_dict, device, genres)
                for tg, tg_result in cd_results.items():
                    for uid_str, metrics in tg_result['per_user'].items():
                        cd_per_target[tg]['per_user'][uid_str] = metrics
                        cd_per_target[tg]['avg_sroccs'].append(metrics['srocc'])
                        cd_per_target[tg]['avg_ndcgs'].append(metrics['ndcg@10'])
                    for uid_str, head_metrics in tg_result['per_user_per_head'].items():
                        cd_per_target[tg]['per_user_per_head'][uid_str] = head_metrics
                        for sg, m in head_metrics.items():
                            cd_per_target[tg]['head_sroccs'][sg].append(m['srocc'])
                            cd_per_target[tg]['head_ndcgs'][sg].append(m['ndcg@10'])

    # Calculate genre-specific averages
    genre_avg_metrics = {}
    for genre in genres:
        if genre in genre_srocc_list and len(genre_srocc_list[genre]) > 0:
            genre_avg_metrics[genre] = {
                'srocc': np.mean(genre_srocc_list[genre]),
                'mae': np.mean(genre_mae_list[genre]),
                'ndcg@10': np.mean(genre_ndcg_list[genre]),
                'ccc': np.mean(genre_ccc_list[genre]),
            }

    if args.is_log:
        log_dict = {}
        for genre, metrics in genre_avg_metrics.items():
            log_dict[f"{genre}/Avg. Test SROCC"] = metrics['srocc']
            log_dict[f"{genre}/Avg. Test MAE"] = metrics['mae']
            log_dict[f"{genre}/Avg. Test NDCG@10"] = metrics['ndcg@10']
        wandb.log(log_dict, commit=True)

    cross_domain_results = {}
    if cd_per_target is not None:
        for tg, data in cd_per_target.items():
            cross_domain_results[tg] = {
                'source_heads': genres,
                'method': 'average',
                'average': {
                    'srocc': float(np.mean(data['avg_sroccs'])) if data['avg_sroccs'] else 0.0,
                    'ndcg@10': float(np.mean(data['avg_ndcgs'])) if data['avg_ndcgs'] else 0.0,
                },
                'per_head': {
                    sg: {
                        'srocc': float(np.mean(data['head_sroccs'][sg])) if data['head_sroccs'][sg] else 0.0,
                        'ndcg@10': float(np.mean(data['head_ndcgs'][sg])) if data['head_ndcgs'][sg] else 0.0,
                    }
                    for sg in genres
                },
                'per_user': data['per_user'],
                'per_user_per_head': data['per_user_per_head'],
            }

    # Save test performance to JSON
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'exp', args.dataset_ver, genre_str)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare per-user results
    per_user_results = {}
    for uid, (genre_metrics_user, total_mae) in results.items():
        if isinstance(genre_metrics_user, dict):
            per_user_results[str(uid)] = {
                genre: {'srocc': float(metrics['srocc']), 'ndcg@10': float(metrics['ndcg@10'])}
                for genre, metrics in genre_metrics_user.items()
            }

    result_data = {
        'experiment_name': model_name_base,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'PIAA_finetune',
        'genres': genres,
        'average_metrics': {
            genre: {'srocc': float(metrics['srocc']), 'ndcg@10': float(metrics['ndcg@10'])}
            for genre, metrics in genre_avg_metrics.items()
        },
        'per_user_metrics': per_user_results
    }
    if cross_domain_results:
        result_data['cross_domain_metrics'] = cross_domain_results

    # Remove trailing mode suffix to avoid duplication (e.g., "name_finetune_finetune.json")
    base_name = model_name_base.removesuffix('_finetune')
    json_filename = f"{genre_str}_ICI_{base_name}_finetune.json"
    json_path = os.path.join(save_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"Test results saved to {json_path}")

    return results

def trainer_finetune(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict):
    """
    Finetune trainer for a single genre.
    Args:
        datasets_dict: dict of {genre: {'train': ds, 'val': ds, 'test': ds}}
        args: arguments
        device: device
        dirname: directory name for saving models
        experiment_name: experiment name
        backbone_dict: dict of {genre: backbone_type}
        pretrained_model_dict: dict of {genre: pretrained_model_path}
    """
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"Training for user {uid}...")

        user_train_ds = copy.copy(datasets_dict[genre]['train'])
        user_train_ds.data = datasets_dict[genre]['train'].data[datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)

        user_val_ds = copy.copy(datasets_dict[genre]['val'])
        user_val_ds.data = datasets_dict[genre]['val'].data[datasets_dict[genre]['val'].data['user_id'] == uid].reset_index(drop=True)

        total_train_samples = len(user_train_ds)
        total_val_samples = len(user_val_ds)
        print(f"User {uid}: train {total_train_samples} val {total_val_samples} samples")
        if total_train_samples == 0 or total_val_samples == 0:
            print(f"Skipping user {uid}: insufficient data")
            continue

        train_loader = DataLoader(user_train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        val_loaders_dict = {genre: DataLoader(user_val_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}

        model_user = PIAA_ICI_CrossDomain(num_bins, num_attr, num_pt, genres, backbone_dict, dropout=args.dropout, use_backbone_image=args.use_backbone_image).to(device)
        pretrained_path = pretrained_model_dict[genre]
        if pretrained_path is not None and os.path.exists(pretrained_path):
            try:
                state = torch.load(pretrained_path)
                model_user.load_state_dict(state)
                print(f"Loaded PIAA_ICI_CrossDomain weights from {pretrained_path}")
            except Exception as e:
                raise RuntimeError(f"Error: Failed to load model weights from {pretrained_path}: {e}")
        else:
            raise FileNotFoundError(f"Error: Pretrained model file not found: {pretrained_path}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            print(f"[Freeze] Backbone frozen: {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        optimizer_user = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)
        scheduler_user = optim.lr_scheduler.ReduceLROnPlateau(optimizer_user, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

        best_val_ccc = -float('inf')
        patience = 0
        best_model_path = os.path.join(dirname, f'{genre_str}_ICI_user_{uid}_{experiment_name}_finetune.pth')

        scaler = GradScaler('cuda')
        for epoch in range(args.num_epochs):
            train_loss = train(model_user, train_loader, optimizer_user, scaler, device, args, genre, epoch=epoch)
            genre_metrics, val_mae = evaluate(model_user, val_loaders_dict, device, epoch=epoch, phase_name="Val")

            val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

            if args.is_log:
                log_dict = {"epoch": epoch}
                if genre in genre_metrics:
                    log_dict[f"{genre}/Val MAE user_{uid}"] = genre_metrics[genre]['mae']
                    log_dict[f"{genre}/Val SROCC user_{uid}"] = genre_metrics[genre]['srocc']
                    log_dict[f"{genre}/Val NDCG@10 user_{uid}"] = genre_metrics[genre]['ndcg@10']
                    log_dict[f"{genre}/Val CCC user_{uid}"] = genre_metrics[genre]['ccc']
                wandb.log(log_dict, commit=True)

            scheduler_user.step(val_ccc)

            if val_ccc > best_val_ccc:
                best_val_ccc = val_ccc
                patience = 0
                torch.save(model_user.state_dict(), best_model_path)
            else:
                patience += 1
                if patience >= args.max_patience_epochs:
                    print(f"User {uid}: early stopping at epoch {epoch}")
                    break

num_attr = None  # Determined dynamically from dataset
num_pt = None    # Determined dynamically from dataset

def trainer_pretrain(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict):
    """
    Pretrain trainer for a single genre.
    Trains on GIAA data with NIMA initialization, uses val GIAA data for early stopping.
    Args:
        datasets_dict: dict of {genre: {'train': ds, 'val': ds, 'test': ds}} (GIAA data)
        args: arguments
        device: device
        dirname: directory name for saving models
        experiment_name: experiment name
        backbone_dict: dict of {genre: backbone_type}
        pretrained_model_dict: dict of {genre: nima_pretrained_model_path}
    Returns:
        best_model_path: path to saved best model
        best_state_dict: state dict if args.no_save_model, else None
    """
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    if args.user_grouped_batch:
        batch_sampler = UserGroupedBatchSampler(datasets_dict[genre]['train'], batch_size)
        train_loader = DataLoader(datasets_dict[genre]['train'], batch_sampler=batch_sampler,
                                  num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(datasets_dict[genre]['train'], batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    val_loaders_dict = {genre: DataLoader(datasets_dict[genre]['val'], batch_size=batch_size, shuffle=False,
                                          num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}

    model = PIAA_ICI_CrossDomain(num_bins, num_attr, num_pt, genres, backbone_dict, dropout=args.dropout, use_backbone_image=args.use_backbone_image).to(device)

    # Load genre-specific NIMA weights
    pretrained_path = pretrained_model_dict[genre]
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Error: Pretrained NIMA model file not found: {pretrained_path}")
    try:
        state = torch.load(pretrained_path)
        model.nima_dict[genre].load_state_dict(state)
        print(f"Loaded NIMA weights for {genre} from {pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"Error: Failed to load NIMA weights for {genre} from {pretrained_path}: {e}")

    model.freeze_backbone()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"[Freeze] Backbone frozen: {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

    best_val_ccc = -float('inf')
    patience = 0
    best_model_path = os.path.join(dirname, f'{genre_str}_ICI_{experiment_name}_pretrain.pth')
    best_state_dict = None

    scaler = GradScaler('cuda')
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, scaler, device, args, genre, epoch=epoch)
        genre_metrics, val_mae = evaluate(model, val_loaders_dict, device, epoch=epoch, phase_name="Val")

        val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

        if args.is_log:
            log_dict = {"epoch": epoch}
            log_dict[f"{genre}/Train {args.loss_type.upper()}"] = train_loss
            if genre in genre_metrics:
                log_dict[f"{genre}/Val MAE"] = genre_metrics[genre]['mae']
                log_dict[f"{genre}/Val SROCC"] = genre_metrics[genre]['srocc']
                log_dict[f"{genre}/Val NDCG@10"] = genre_metrics[genre]['ndcg@10']
                log_dict[f"{genre}/Val CCC"] = genre_metrics[genre]['ccc']
            wandb.log(log_dict, commit=True)

        scheduler.step(val_ccc)

        if val_ccc > best_val_ccc:
            best_val_ccc = val_ccc
            patience = 0
            if args.no_save_model:
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"Pretrain: early stopping at epoch {epoch}")
                break

    return best_model_path, best_state_dict

def evaluate_pretrain_on_val_piaa(datasets_dict_user, args, device, backbone_dict, best_model_path, model_state_dict=None):
    """
    Pretrain後のベストモデルでval_piaa_datasetを使い、
    ユーザーごとにSROCC/NDCGを算出し、wandbに100回分ログする。
    """
    batch_size = args.batch_size
    genres = list(datasets_dict_user.keys())
    genre = genres[0]

    model = PIAA_ICI_CrossDomain(num_bins, num_attr, num_pt, genres, backbone_dict, dropout=args.dropout, use_backbone_image=args.use_backbone_image).to(device)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(torch.load(best_model_path))

    all_user_ids = set(datasets_dict_user[genre]['val'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    user_metrics = {}
    for uid in unique_user_ids:
        user_val_ds = copy.copy(datasets_dict_user[genre]['val'])
        user_val_ds.data = datasets_dict_user[genre]['val'].data[datasets_dict_user[genre]['val'].data['user_id'] == uid].reset_index(drop=True)
        if len(user_val_ds) > 0:
            val_loaders_dict = {genre: DataLoader(user_val_ds, batch_size=batch_size, shuffle=False,
                                                   num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}
            genre_metrics, _ = evaluate(model, val_loaders_dict, device)
            user_metrics[uid] = genre_metrics

    print(f"Pretrain val PIAA evaluation done: {len(user_metrics)} users evaluated")

def inference_pretrain(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict, best_model_path, eval_datasets_dict=None, model_state_dict=None):
    """
    Per-user evaluation after pretraining, separated from training.
    Args:
        datasets_dict: dict of {genre: {'train': ds, 'val': ds, 'test': ds}} (PIAA data)
        eval_datasets_dict: optional dict of {target_genre: {'test': dataset}} for cross-domain evaluation
        model_state_dict: if provided, load from this state dict instead of best_model_path
    """
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    model = PIAA_ICI_CrossDomain(num_bins, num_attr, num_pt, genres, backbone_dict, dropout=args.dropout, use_backbone_image=args.use_backbone_image).to(device)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(torch.load(best_model_path))

    all_user_ids = set(datasets_dict[genre]['test'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    genre_srocc_list = defaultdict(list)
    genre_mae_list = defaultdict(list)
    genre_ndcg_list = defaultdict(list)
    genre_ccc_list = defaultdict(list)
    per_user_results = {}

    for uid in unique_user_ids:
        user_test_ds = copy.copy(datasets_dict[genre]['test'])
        user_test_ds.data = datasets_dict[genre]['test'].data[datasets_dict[genre]['test'].data['user_id'] == uid].reset_index(drop=True)
        if len(user_test_ds) > 0:
            test_loaders_dict = {genre: DataLoader(user_test_ds, batch_size=batch_size, shuffle=False,
                                                    num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}
            genre_metrics, total_mae = evaluate(model, test_loaders_dict, device)
            for g, metrics in genre_metrics.items():
                genre_srocc_list[g].append(metrics['srocc'])
                genre_mae_list[g].append(metrics['mae'])
                genre_ndcg_list[g].append(metrics['ndcg@10'])
                genre_ccc_list[g].append(metrics['ccc'])
            if args.is_log:
                log_dict = {}
                for g, metrics in genre_metrics.items():
                    log_dict[f"{g}/Test SROCC user_{uid}"] = metrics['srocc']
                    log_dict[f"{g}/Test NDCG@10 user_{uid}"] = metrics['ndcg@10']
                wandb.log(log_dict, commit=True)
            per_user_results[str(uid)] = {
                g: {'srocc': float(metrics['srocc']), 'ndcg@10': float(metrics['ndcg@10']), 'mae': float(metrics['mae']), 'ccc': float(metrics['ccc'])}
                for g, metrics in genre_metrics.items()
            }

    genre_avg_metrics = {}
    for g in genres:
        if g in genre_srocc_list and len(genre_srocc_list[g]) > 0:
            genre_avg_metrics[g] = {
                'srocc': np.mean(genre_srocc_list[g]),
                'mae': np.mean(genre_mae_list[g]),
                'ndcg@10': np.mean(genre_ndcg_list[g]),
                'ccc': np.mean(genre_ccc_list[g]),
            }

    if args.is_log:
        log_dict = {}
        for g, metrics in genre_avg_metrics.items():
            log_dict[f"{g}/Avg. Test SROCC"] = metrics['srocc']
            log_dict[f"{g}/Avg. Test MAE"] = metrics['mae']
            log_dict[f"{g}/Avg. Test NDCG@10"] = metrics['ndcg@10']
        wandb.log(log_dict, commit=True)

    # Cross-domain evaluation
    cross_domain_results = {}
    if eval_datasets_dict is not None:
        eval_loaders_dict = {}
        for target_genre, ds_dict in eval_datasets_dict.items():
            eval_loaders_dict[target_genre] = DataLoader(
                ds_dict['test'], batch_size=batch_size, shuffle=False,
                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        cross_domain_results = evaluate_cross_domain(model, eval_loaders_dict, device, genres)

    # Save test performance to JSON
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'exp', args.dataset_ver, genre_str)
    os.makedirs(save_dir, exist_ok=True)

    model_basename = os.path.splitext(os.path.basename(best_model_path))[0]
    result_data = {
        'experiment_name': model_basename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'PIAA_pretrain',
        'genres': genres,
        'average_metrics': {
            g: {'srocc': float(metrics['srocc']), 'ndcg@10': float(metrics['ndcg@10']), 'mae': float(metrics['mae']), 'ccc': float(metrics['ccc'])}
            for g, metrics in genre_avg_metrics.items()
        },
        'per_user_metrics': per_user_results
    }
    if cross_domain_results:
        result_data['cross_domain_metrics'] = cross_domain_results

    base_name = model_basename.removesuffix('_pretrain')
    json_filename = f"{base_name}_pretrain.json"
    json_path = os.path.join(save_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"Test results saved to {json_path}")

    return None

def discover_pretrained_models(dataset_ver, genre, piaa_mode='PIAA_finetune'):
    """Auto-discover pretrained model files.

    For PIAA_pretrain: finds *NIMA*.pth in models_pth/{dataset_ver}/{genre}/
    For PIAA_finetune: finds *_pretrain.pth in models_pth/{dataset_ver}/{genre}/
    """
    genre_dir = os.path.join('models_pth', dataset_ver, genre)
    if not os.path.isdir(genre_dir):
        raise FileNotFoundError(f"Directory not found: {genre_dir}")

    if piaa_mode == 'PIAA_pretrain':
        nima_files = [f for f in os.listdir(genre_dir) if 'NIMA' in f and f.endswith('.pth')]
        if len(nima_files) == 1:
            return {genre: os.path.join(genre_dir, nima_files[0])}
        elif len(nima_files) > 1:
            raise ValueError(f"Multiple NIMA pth files found in {genre_dir}: {nima_files}. Please specify --pretrained_model explicitly.")
        else:
            raise FileNotFoundError(f"No NIMA pth file found in {genre_dir}")
    else:
        ici_files = [f for f in os.listdir(genre_dir) if f.endswith('_pretrain.pth')]
        if len(ici_files) == 1:
            return {genre: os.path.join(genre_dir, ici_files[0])}
        elif len(ici_files) > 1:
            raise ValueError(f"Multiple pretrain pth files found in {genre_dir}: {ici_files}. Please specify --pretrained_model explicitly.")
        else:
            raise FileNotFoundError(f"No pretrain pth file found in {genre_dir}")

def run_main(args):
    global num_pt, num_attr

    genre = args.genre.strip()
    if ',' in genre:
        raise ValueError(
            f"Multi-domain training has been removed. "
            f"Specify a single genre (e.g., --genre art), got: '{genre}'"
        )
    genres = [genre]
    print(f"Training with genre: {genre}")

    # Create backbone dictionary
    backbone_dict = {}
    if genre == 'scenery' and args.use_video:
        backbone_dict[genre] = 'i3d'
    else:
        backbone_dict[genre] = args.backbone
    print(f"Backbone: {backbone_dict[genre]}")

    # Auto-discover pretrained models
    pretrained_model_dict = discover_pretrained_models(args.dataset_ver, genre, args.piaa_mode)
    print(f"Auto-discovered pretrained models: {pretrained_model_dict}")

    if args.is_log:
        tags = [args.piaa_mode]
        tags += wandb_tags(args)
        if args.use_video:
            tags.append("use_video")
        wandb.init(
                   project=f"XPASS",
                   notes=f"ICI",
                   tags=tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "genre": genre,
            "backbone": backbone_dict[genre],
            "use_backbone_image": args.use_backbone_image
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''

    print(args)

    # Build global encoders once before loading dataset
    global_trait_encoders, global_age_bins = build_global_encoders(args.root_dir)

    # Load datasets
    _, train_piaa_dataset, train_giaa_dataset, _, val_piaa_dataset, val_giaa_dataset, test_piaa_dataset = load_data(
        args, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)

    datasets_dict = {genre: {'train': train_giaa_dataset, 'val': val_giaa_dataset, 'test': test_piaa_dataset}}
    datasets_dict_user = {genre: {'train': train_piaa_dataset, 'val': val_piaa_dataset, 'test': test_piaa_dataset}}

    # Dynamically determine num_pt and num_attr from the dataset
    _sample = train_giaa_dataset[0]
    num_pt = len(_sample['traits'])
    num_attr = len(_sample['QIP'])
    print(f"Detected num_pt={num_pt}, num_attr={num_attr} from dataset")

    # Load cross-domain evaluation datasets if --use_cross_eval is enabled
    ALL_GENRES = ['art', 'fashion', 'scenery']
    eval_datasets_dict = None
    if args.use_cross_eval:
        eval_genres = [g for g in ALL_GENRES if g != genre]
        if not eval_genres:
            print("Warning: No other genres available for cross-domain evaluation.")
        else:
            print(f"Cross-domain evaluation targets: {eval_genres}")
            eval_datasets_dict = {}
            for eval_genre in eval_genres:
                args_copy = copy.deepcopy(args)
                args_copy.genre = eval_genre
                if eval_genre == 'scenery' and args.use_video:
                    args_copy.backbone = 'i3d'
                else:
                    args_copy.backbone = args.backbone
                _, _, _, _, _, _, test_piaa_dataset_eval = load_data(args_copy, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
                eval_datasets_dict[eval_genre] = {'test': test_piaa_dataset_eval}
                print(f"Loaded {len(test_piaa_dataset_eval)} test samples for cross-domain eval genre '{eval_genre}'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirname = os.path.join(model_dir(args), genre)
    os.makedirs(dirname, exist_ok=True)

    if args.piaa_mode == 'PIAA_pretrain':
        best_model_path, best_state_dict = trainer_pretrain(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict)
        evaluate_pretrain_on_val_piaa(datasets_dict_user, args, device, backbone_dict, best_model_path, model_state_dict=best_state_dict)
        inference_pretrain(datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict, best_model_path, eval_datasets_dict=eval_datasets_dict, model_state_dict=best_state_dict)
    elif args.piaa_mode == 'PIAA_finetune':
        trainer_finetune(datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict)
        inference_finetune(datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, eval_datasets_dict=eval_datasets_dict)
        for pth_file in [f for f in os.listdir(dirname) if f.endswith('_finetune.pth')]:
            os.remove(os.path.join(dirname, pth_file))
        print(f"Deleted temporary finetune model files from {dirname}")
    else:
        raise ValueError(f"Error: --piaa_mode must be 'PIAA_pretrain' or 'PIAA_finetune', got: {args.piaa_mode}")

    if args.is_log:
        wandb.finish()

if __name__ == '__main__':
    args = parse_arguments()

    if args.dataset_ver.endswith('_all'):
        version_prefix = args.dataset_ver[:-4]  # e.g., 'v3_all' -> 'v3'
        folds = discover_folds(args.root_dir, version_prefix)
        if not folds:
            raise ValueError(f"No fold directories found for version '{version_prefix}' in {os.path.join(args.root_dir, 'split')}")
        print(f"Running all {len(folds)} folds sequentially: {folds}")
        for i, fold in enumerate(folds):
            if i + 1 < args.start_fold:
                print(f"Skipping fold {i+1}/{len(folds)}: {fold} (start_fold={args.start_fold})")
                continue
            print(f"\n{'='*60}")
            print(f"  Fold {i+1}/{len(folds)}: {fold}")
            print(f"{'='*60}\n")
            args_fold = copy.deepcopy(args)
            args_fold.dataset_ver = fold
            run_main(args_fold)
    else:
        # Search for fold structure in models_pth
        models_base = 'models_pth'
        fold_dirs = sorted([
            d for d in os.listdir(models_base)
            if d.startswith(f'{args.dataset_ver}_fold') and os.path.isdir(os.path.join(models_base, d))
        ]) if os.path.exists(models_base) else []

        if fold_dirs:
            print(f"models_pth/{args.dataset_ver}/ not found. Running fold structure: {fold_dirs}")
            for i, fold in enumerate(fold_dirs):
                if i + 1 < args.start_fold:
                    print(f"Skipping fold {i+1}/{len(fold_dirs)}: {fold} (start_fold={args.start_fold})")
                    continue
                print(f"\n{'='*60}")
                print(f"  Fold {i+1}/{len(fold_dirs)}: {fold}")
                print(f"{'='*60}\n")
                args_fold = copy.deepcopy(args)
                args_fold.dataset_ver = fold
                run_main(args_fold)
        else:
            run_main(args)
