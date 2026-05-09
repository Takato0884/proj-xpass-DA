"""UGA-Feature (Uncertainty-Guided Alignment) for PIAA / ICI.

Reference: Nejjar et al., "Uncertainty-Guided Alignment for Unsupervised Domain
Adaptation in Regression", Reliability Engineering and System Safety 2026.
Official repo: https://github.com/ismailnejjar/UGA

Implementation scope (per documents/design/UGAFeat_設計書.md):
  - PIAA pretrain / finetune only (GIAA not implemented).
  - ICI model only (MIR raises NotImplementedError).
  - DER head replaces attr_corr (Linear(64, 4) → (μ, log_v, log_α, log_β)).
  - MMD aligns I_ij (64-dim) features (multi-bandwidth RBF).
  - C-Mixup is enabled by default (toggle via --ugafeat_use_cmixup).
  - λ for alignment loss is fixed (no schedule).
"""

import os
import copy
import math

import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..train_common import build_piaa_model, num_bins, PIAA_ICI_CrossDomain
from ..data import collate_fn
from ..evaluate import evaluate_piaa


# ── MMD loss (multi-bandwidth RBF) ────────────────────────────────────────────

class MMD_loss(nn.Module):
    """Multi-bandwidth RBF Maximum Mean Discrepancy.

    Bandwidths follow the official implementation: median heuristic base, then a
    geometric ladder of `num_kernels` widths with multiplier `mul`.
    """

    def __init__(self, num_kernels: int = 5, mul: float = 2.0, fix_sigma: float = None):
        super().__init__()
        self.num_kernels = num_kernels
        self.mul = mul
        self.fix_sigma = fix_sigma

    def _gaussian_kernel_matrix(self, source, target):
        n_s = source.size(0)
        n_t = target.size(0)
        total = torch.cat([source, target], dim=0)
        n = total.size(0)
        diff = total.unsqueeze(0) - total.unsqueeze(1)  # (n, n, d)
        L2 = (diff * diff).sum(dim=2)                    # (n, n)

        if self.fix_sigma is not None:
            base = float(self.fix_sigma)
        else:
            denom = float(n * n - n)
            base = float(L2.detach().sum().item()) / max(denom, 1.0)
            if base <= 0.0:
                base = 1.0

        K = torch.zeros_like(L2)
        for i in range(self.num_kernels):
            bw = base * (self.mul ** (i - self.num_kernels // 2))
            K = K + torch.exp(-L2 / max(bw, 1e-8))
        return K, n_s, n_t

    def forward(self, source, target):
        K, n_s, n_t = self._gaussian_kernel_matrix(source, target)
        K_ss = K[:n_s, :n_s].mean()
        K_tt = K[n_s:, n_s:].mean()
        K_st = K[:n_s, n_s:].mean()
        return K_ss + K_tt - 2.0 * K_st


# ── Deep Evidential Regression loss ───────────────────────────────────────────

def der_loss_components(y, gamma, v, alpha, beta):
    """Return (NLL, Reg) per-sample-mean for Deep Evidential Regression.

    y, gamma, v, alpha, beta all share shape (B, 1) (or broadcastable).
    α > 1 is required (caller ensures via softplus + 1).
    """
    pi = torch.tensor(math.pi, device=y.device, dtype=y.dtype)
    twoBlambda = 2.0 * beta * (1.0 + v)
    nll = (
        0.5 * torch.log(pi / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log((y - gamma) ** 2 * v + twoBlambda)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    )
    reg = (y - gamma).abs() * (2.0 * v + alpha)
    return nll.mean(), reg.mean()


# ── C-Mixup utilities (KDE-based label-aware mixing) ──────────────────────────

def _gaussian_kde_weights(query, pool, bandwidth):
    """Return row-normalized Gaussian weights of `pool` w.r.t. each `query`.

    query: (n_q, 1) numpy
    pool:  (n_p, 1) numpy
    Returns (n_q, n_p) row-normalized weight matrix.
    """
    diff = (query - pool.T) / max(float(bandwidth), 1e-8)
    w = np.exp(-0.5 * diff * diff)
    s = w.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return w / s


def get_batch_kde_mixup_idx(Y_query, Y_pool, bandwidth=0.2):
    """For each entry in Y_query, sample one index in Y_pool weighted by KDE proximity."""
    q = np.asarray(Y_query).reshape(-1, 1)
    p = np.asarray(Y_pool).reshape(-1, 1)
    weights = _gaussian_kde_weights(q, p, bandwidth)
    n_p = p.shape[0]
    n_q = q.shape[0]
    idx = np.empty(n_q, dtype=np.int64)
    for i in range(n_q):
        idx[i] = np.random.choice(n_p, p=weights[i])
    return idx


def get_batch_kde_mixup_batch(Y1, Y2, bandwidth=0.2):
    """Concatenate Y1 and Y2 to form Y; sample idx2 (one per Y1 entry) from Y."""
    y1 = Y1.detach().cpu().numpy().reshape(-1) if isinstance(Y1, torch.Tensor) else np.asarray(Y1).reshape(-1)
    y2 = Y2.detach().cpu().numpy().reshape(-1) if isinstance(Y2, torch.Tensor) else np.asarray(Y2).reshape(-1)
    pool = np.concatenate([y1, y2], axis=0)
    return get_batch_kde_mixup_idx(y1, pool, bandwidth=bandwidth)


# ── PIAA UGAFeat wrapper (ICI only) ───────────────────────────────────────────

class PIAA_UGAFeat(nn.Module):
    """Wraps a PIAA_ICI_CrossDomain instance, replacing attr_corr with a DER head.

    forward() default returns the final score: μ·(K-1) + 1 + direct_score
    (compatible with evaluate_piaa).

    forward(..., return_feat=True) returns (score, I_ij, (μ, ν, α, β)).
    """

    def __init__(self, model: PIAA_ICI_CrossDomain):
        super().__init__()
        if not isinstance(model, PIAA_ICI_CrossDomain):
            raise NotImplementedError(
                "UGAFeat supports the ICI architecture only (PIAA_ICI_CrossDomain)."
            )
        self._model = model
        self.der_head = nn.Linear(model.input_dim, 4)
        nn.init.xavier_uniform_(self.der_head.weight)
        nn.init.zeros_(self.der_head.bias)

    # ── Train-mode propagation (preserve frozen-eval semantics) ──────────────

    def train(self, mode: bool = True):
        super().train(mode)
        # Delegate to the base PIAA model so frozen submodules stay in eval mode.
        if mode:
            self._model._set_frozen_modules_eval()
        return self

    # ── DER head ──────────────────────────────────────────────────────────────

    def _split_der(self, raw):
        mu_logit = raw[:, 0:1]
        v_raw = raw[:, 1:2]
        a_raw = raw[:, 2:3]
        b_raw = raw[:, 3:4]
        mu = torch.sigmoid(mu_logit)
        v = F.softplus(v_raw) + 1e-5
        alpha = F.softplus(a_raw) + 1.0 + 1e-5
        beta = F.softplus(b_raw) + 1e-5
        return mu, v, alpha, beta

    def head_from_feat(self, I_ij):
        return self._split_der(self.der_head(I_ij))

    # ── Feature extraction (replicates PIAA_ICI forward up to I_ij) ──────────

    def forward_feat(self, images, personal_traits, image_attributes, genre):
        """Return (I_ij, direct_outputs). direct_outputs comes from the frozen NIMA."""
        m = self._model
        logit, _, raw_feat = m.nima_dict[genre](images, return_feat=True)
        prob = F.softmax(logit, dim=1)

        n_attr = image_attributes.shape[1]
        img_feat = m.backbone_image_proj[genre](raw_feat)
        img_input = torch.cat([image_attributes, img_feat], dim=1)
        attr_img = m.node_attr_img(img_input).view(-1, n_attr, m.input_dim)
        attr_user = m.node_attr_user(personal_traits).view(-1, n_attr, m.input_dim)

        internal_img = m.internal_interaction_img(attr_img)
        internal_user = m.internal_interaction_user(attr_user)
        agg_user, agg_img = m.external_interaction(attr_user, attr_img)

        fused_img = m.interfusion_img(attr_img, internal_img, agg_img)
        fused_user = m.interfusion_user(attr_user, internal_user, agg_user)
        I_ij = torch.sum(fused_img, dim=1) + torch.sum(fused_user, dim=1)

        bins = torch.arange(1, m.num_bins + 1, dtype=prob.dtype, device=prob.device).unsqueeze(0)
        direct_outputs = (prob * bins).sum(dim=1, keepdim=True)
        return I_ij, direct_outputs

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, images, personal_traits, image_attributes, genre, return_feat=False):
        m = self._model
        I_ij, direct_outputs = self.forward_feat(images, personal_traits, image_attributes, genre)
        mu, v, alpha, beta = self.head_from_feat(I_ij)

        interaction_outputs = mu * (m.num_bins - 1) + 1.0
        score = interaction_outputs + direct_outputs

        m._last_interaction_mean = interaction_outputs.detach().abs().mean().item()
        m._last_direct_mean = direct_outputs.detach().abs().mean().item()

        if return_feat:
            return score, I_ij, (mu, v, alpha, beta)
        return score


def build_ugafeat_wrapper(num_bins_, num_attr, num_pt, genres, backbone_dict, args, device=None):
    """Build a PIAA_UGAFeat wrapper around a fresh base PIAA_ICI model."""
    if getattr(args, 'model_type', 'ICI') != 'ICI':
        raise NotImplementedError("UGAFeat supports ICI only")
    base = build_piaa_model(num_bins_, num_attr, num_pt, genres, backbone_dict, args)
    if device is not None:
        base = base.to(device)
    wrapper = PIAA_UGAFeat(base)
    if device is not None:
        wrapper = wrapper.to(device)
    return wrapper


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_one_epoch_piaa(uga_model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                           mmd_fn, epoch=None, phase='pretrain'):
    uga_model.train()

    coef_evi = float(getattr(args, 'ugafeat_lambda_evi', 1.0))
    use_cmixup = bool(getattr(args, 'ugafeat_use_cmixup', True))
    bandwidth = float(getattr(args, 'ugafeat_kde_bandwidth', 0.2))
    lam_align = float(getattr(args, 'ugafeat_lambda_align', 1.0))

    running = {
        'L_evi': 0.0, 'L_nll': 0.0, 'L_reg': 0.0,
        'L_mmd_st': 0.0, 'L_mmd_smix': 0.0,
        'mu': 0.0, 'v': 0.0, 'alpha': 0.0, 'beta': 0.0,
    }
    n_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [UGAFEAT-{phase}]" if epoch is not None else f"Train UGAFEAT-{phase}"
    progress = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120,
                    colour="#00ff00", ascii="-=")

    for sample_src in progress:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        images_src = sample_src['image'].to(device)
        y_src_norm = sample_src['Aesthetic'].to(device).view(-1, 1)  # already in [0, 1]
        pt_src = sample_src['traits'].float().to(device)
        attr_src = sample_src['QIP'].float().to(device)

        images_tgt = sample_tgt['image'].to(device)
        pt_tgt = sample_tgt['traits'].float().to(device)
        attr_tgt = sample_tgt['QIP'].float().to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            I_ij_src, _direct_src = uga_model.forward_feat(images_src, pt_src, attr_src, genre)
            I_ij_tgt, _direct_tgt = uga_model.forward_feat(images_tgt, pt_tgt, attr_tgt, genre)
            mu_s, v_s, a_s, b_s = uga_model.head_from_feat(I_ij_src)
            mu_t, _v_t, _a_t, _b_t = uga_model.head_from_feat(I_ij_tgt)
            L_nll, L_reg = der_loss_components(y_src_norm, mu_s, v_s, a_s, b_s)
            L_evi = L_nll + coef_evi * L_reg

        # Compute MMD in FP32 (outside autocast) for numerical stability.
        I_ij_src_fp = I_ij_src.float()
        I_ij_tgt_fp = I_ij_tgt.float()
        L_mmd_st = mmd_fn(I_ij_src_fp, I_ij_tgt_fp)

        L_mmd_smix = torch.zeros((), device=device)
        if use_cmixup:
            with torch.no_grad():
                Y1 = mu_t.detach().view(-1)
                Y2 = y_src_norm.detach().view(-1)
            idx2_np = get_batch_kde_mixup_batch(Y1, Y2, bandwidth=bandwidth)
            idx2 = torch.as_tensor(idx2_np, dtype=torch.long, device=device)
            lam_mix = float(np.random.beta(2.0, 2.0))
            feat_concat = torch.cat([I_ij_tgt_fp, I_ij_src_fp], dim=0)
            feat_mix = I_ij_tgt_fp * lam_mix + feat_concat[idx2] * (1.0 - lam_mix)
            # Pass feat_mix through head per design 5.2 step 6 (kept for parity, not used in loss).
            with autocast('cuda'):
                _ = uga_model.head_from_feat(feat_mix.to(I_ij_src.dtype))
            L_mmd_smix = mmd_fn(I_ij_src_fp, feat_mix)

        L_align = L_mmd_st + L_mmd_smix
        loss = L_evi + lam_align * L_align

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running['L_evi'] += float(L_evi.item())
        running['L_nll'] += float(L_nll.item())
        running['L_reg'] += float(L_reg.item())
        running['L_mmd_st'] += float(L_mmd_st.item())
        running['L_mmd_smix'] += float(L_mmd_smix.item())
        running['mu'] += float(mu_s.detach().mean().item())
        running['v'] += float(v_s.detach().mean().item())
        running['alpha'] += float(a_s.detach().mean().item())
        running['beta'] += float(b_s.detach().mean().item())
        n_batches += 1

        progress.set_postfix({
            'L_evi':  f"{L_evi.item():.4f}",
            'MMD':    f"{L_mmd_st.item():.4f}",
            'MMDmix': f"{float(L_mmd_smix.item()):.4f}",
        })

    n = max(n_batches, 1)
    return {k: v / n for k, v in running.items()}


# ── Pretrain ──────────────────────────────────────────────────────────────────

def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    """UGAFEAT pretrain trainer for PIAA (ICI only).

    Source: GIAA-style train data with DER loss + MMD on I_ij.
    Target: target genre GIAA-style train data (labels unused).
    Early stopping: source val CCC.
    Returns: (best_model_path, best_state_dict)
    """
    if getattr(args, 'model_type', 'ICI') != 'ICI':
        raise NotImplementedError("UGAFEAT pretrain supports the ICI model only")

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = domain_tag if domain_tag else genre
    tgt_genre_name = domain_tag.split('2')[1] if domain_tag and '2' in domain_tag else 'target'

    src_loader = DataLoader(datasets_dict[genre]['train'], batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    tgt_loader = DataLoader(tgt_train_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    val_loaders_dict = {genre: DataLoader(datasets_dict[genre]['val'], batch_size=batch_size, shuffle=False,
                                          num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}
    tgt_val_loaders_dict = {genre: DataLoader(tgt_val_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}

    base_model = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)

    pretrained_path = pretrained_model_dict[genre]
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained NIMA model not found: {pretrained_path}")
    try:
        state = torch.load(pretrained_path)
        base_model.nima_dict[genre].load_state_dict(state)
        print(f"Loaded NIMA weights for {genre} from {pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load NIMA weights for {genre}: {e}")

    base_model.freeze_backbone()
    uga_model = PIAA_UGAFeat(base_model).to(device)

    total_params = sum(p.numel() for p in uga_model.parameters())
    trainable_params = sum(p.numel() for p in uga_model.parameters() if p.requires_grad)
    print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, uga_model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor,
                                                     patience=args.lr_patience)
    mmd_fn = MMD_loss(num_kernels=int(getattr(args, 'ugafeat_mmd_num', 5)),
                      mul=float(getattr(args, 'ugafeat_mmd_mul', 2.0)))

    best_val_ccc = -float('inf')
    patience = 0
    _uga_run = experiment_name.removeprefix('UGAFEAT_')
    best_model_path = os.path.join(dirname, f'{genre_str}_UGAFEAT_{args.model_type}_{_uga_run}_pretrain.pth')
    best_state_dict = None
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        m = _train_one_epoch_piaa(
            uga_model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
            mmd_fn, epoch=epoch, phase='pretrain')

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train L_evi":      m['L_evi'],
                f"{genre}/Train L_nll":      m['L_nll'],
                f"{genre}/Train L_reg":      m['L_reg'],
                f"{genre}/Train L_mmd_st":   m['L_mmd_st'],
                f"{genre}/Train L_mmd_smix": m['L_mmd_smix'],
                f"{genre}/Train mu_mean":    m['mu'],
                f"{genre}/Train v_mean":     m['v'],
                f"{genre}/Train alpha_mean": m['alpha'],
                f"{genre}/Train beta_mean":  m['beta'],
            }, commit=False)

        genre_metrics, _ = evaluate_piaa(uga_model, val_loaders_dict, device, epoch=epoch, phase_name="Val")
        val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

        tgt_genre_metrics, _ = evaluate_piaa(uga_model, tgt_val_loaders_dict, device, epoch=epoch, phase_name="Val (tgt)")

        if args.is_log:
            log_dict = {"epoch": epoch}
            if genre in genre_metrics:
                log_dict[f"{genre}/Val MAE"]     = genre_metrics[genre]['mae']
                log_dict[f"{genre}/Val SROCC"]   = genre_metrics[genre]['srocc']
                log_dict[f"{genre}/Val NDCG@10"] = genre_metrics[genre]['ndcg@10']
                log_dict[f"{genre}/Val CCC"]     = genre_metrics[genre]['ccc']
            if hasattr(uga_model._model, '_eval_component_stats') and genre in uga_model._model._eval_component_stats:
                cs = uga_model._model._eval_component_stats[genre]
                log_dict[f"{genre}/Val interaction_mean"]  = cs['interaction_mean']
                log_dict[f"{genre}/Val direct_mean"]       = cs['direct_mean']
                log_dict[f"{genre}/Val interaction_ratio"] = cs['ratio']
            if genre in tgt_genre_metrics:
                tgt_m = tgt_genre_metrics[genre]
                log_dict[f"{tgt_genre_name}/Val MAE"]     = tgt_m['mae']
                log_dict[f"{tgt_genre_name}/Val SROCC"]   = tgt_m['srocc']
                log_dict[f"{tgt_genre_name}/Val NDCG@10"] = tgt_m['ndcg@10']
                log_dict[f"{tgt_genre_name}/Val CCC"]     = tgt_m['ccc']
            wandb.log(log_dict, commit=True)

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_ccc)
        cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr < prev_lr:
            tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (epoch {epoch}) <<<")

        if val_ccc > best_val_ccc:
            best_val_ccc = val_ccc
            patience = 0
            if args.no_save_model:
                best_state_dict = copy.deepcopy(uga_model.state_dict())
            else:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(uga_model.state_dict(), best_model_path)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"UGAFEAT Pretrain: early stopping at epoch {epoch}")
                break

    return best_model_path, best_state_dict


# ── Finetune ──────────────────────────────────────────────────────────────────

def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, ugafeat_target_genre=None):
    """UGAFEAT finetune trainer for PIAA (ICI only). Per-user training."""
    if getattr(args, 'model_type', 'ICI') != 'ICI':
        raise NotImplementedError("UGAFEAT finetune supports the ICI model only")

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"UGAFEAT finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)
        user_val_src = copy.copy(datasets_dict[genre]['val'])
        user_val_src.data = datasets_dict[genre]['val'].data[
            datasets_dict[genre]['val'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{ugafeat_target_genre}' train_piaa_dataset.")
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        tgt_val_mask = tgt_val_piaa_dataset.data['user_id'] == uid
        if tgt_val_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{ugafeat_target_genre}' val_piaa_dataset.")
        user_val_tgt = copy.copy(tgt_val_piaa_dataset)
        user_val_tgt.data = tgt_val_piaa_dataset.data[tgt_val_mask].reset_index(drop=True)

        total_train_src = len(user_train_src)
        total_train_tgt = len(user_train_tgt)
        total_val_src = len(user_val_src)
        print(f"User {uid}: train src={total_train_src}, train tgt={total_train_tgt}, val src={total_val_src}")
        if total_train_src < batch_size or total_train_tgt < batch_size or total_val_src == 0:
            print(f"Skipping user {uid}: need >={batch_size} per split")
            continue

        src_loader = DataLoader(user_train_src, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        tgt_loader = DataLoader(user_train_tgt, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        val_src_loaders = {genre: DataLoader(user_val_src, batch_size=batch_size, shuffle=False,
                                              num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}
        val_tgt_loaders = {genre: DataLoader(user_val_tgt, batch_size=batch_size, shuffle=False,
                                              num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}

        base_user = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
        uga_user = PIAA_UGAFeat(base_user).to(device)

        pretrained_path = pretrained_model_dict[genre]
        if pretrained_path is None or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"UGAFEAT pretrained model not found: {pretrained_path}")
        try:
            state = torch.load(pretrained_path)
            incompatible = uga_user.load_state_dict(state, strict=False)
            if incompatible.missing_keys:
                print(f"[load_state_dict] Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"[load_state_dict] Unexpected keys: {incompatible.unexpected_keys}")
            print(f"Loaded UGAFEAT pretrain weights from {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {pretrained_path}: {e}")

        base_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in uga_user.parameters())
            trainable_params = sum(p.numel() for p in uga_user.parameters() if p.requires_grad)
            print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, uga_user.parameters()), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor,
                                                         patience=args.lr_patience)
        mmd_fn = MMD_loss(num_kernels=int(getattr(args, 'ugafeat_mmd_num', 5)),
                          mul=float(getattr(args, 'ugafeat_mmd_mul', 2.0)))

        best_val_ccc = -float('inf')
        patience = 0
        best_model_path = os.path.join(
            dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            m = _train_one_epoch_piaa(
                uga_user, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                mmd_fn, epoch=epoch, phase='finetune')

            genre_metrics, _ = evaluate_piaa(uga_user, val_src_loaders, device, epoch=epoch, phase_name="Val (src)")
            val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

            tgt_genre_metrics, _ = evaluate_piaa(uga_user, val_tgt_loaders, device, epoch=epoch, phase_name="Val (tgt)")

            if args.is_log:
                log_dict = {"epoch": epoch}
                log_dict[f"{genre}/Train L_evi user_{uid}"]      = m['L_evi']
                log_dict[f"{genre}/Train L_nll user_{uid}"]      = m['L_nll']
                log_dict[f"{genre}/Train L_reg user_{uid}"]      = m['L_reg']
                log_dict[f"{genre}/Train L_mmd_st user_{uid}"]   = m['L_mmd_st']
                log_dict[f"{genre}/Train L_mmd_smix user_{uid}"] = m['L_mmd_smix']
                log_dict[f"{genre}/Train mu_mean user_{uid}"]    = m['mu']
                log_dict[f"{genre}/Train v_mean user_{uid}"]     = m['v']
                log_dict[f"{genre}/Train alpha_mean user_{uid}"] = m['alpha']
                log_dict[f"{genre}/Train beta_mean user_{uid}"]  = m['beta']
                if genre in genre_metrics:
                    log_dict[f"{genre}/Val MAE user_{uid}"]   = genre_metrics[genre]['mae']
                    log_dict[f"{genre}/Val SROCC user_{uid}"] = genre_metrics[genre]['srocc']
                    log_dict[f"{genre}/Val CCC user_{uid}"]   = genre_metrics[genre]['ccc']
                if genre in tgt_genre_metrics:
                    tgt_m = tgt_genre_metrics[genre]
                    log_dict[f"{ugafeat_target_genre}/Val MAE user_{uid}"]   = tgt_m['mae']
                    log_dict[f"{ugafeat_target_genre}/Val SROCC user_{uid}"] = tgt_m['srocc']
                    log_dict[f"{ugafeat_target_genre}/Val CCC user_{uid}"]   = tgt_m['ccc']
                wandb.log(log_dict, commit=True)

            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_ccc)
            cur_lr = optimizer.param_groups[0]['lr']
            if cur_lr < prev_lr:
                tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (user {uid}, epoch {epoch}) <<<")

            if val_ccc > best_val_ccc:
                best_val_ccc = val_ccc
                patience = 0
                torch.save(uga_user.state_dict(), best_model_path)
            else:
                patience += 1
                if patience >= args.max_patience_epochs:
                    print(f"User {uid}: early stopping at epoch {epoch}")
                    break
