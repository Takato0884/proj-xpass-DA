"""Adversarial-Learned Loss for Domain Adaptation (ALDA; Chen et al., AAAI 2020).

Implementation scope (per documents/design/ALDA_設計書.md §6):
  - GIAA: EMD on source + L_T (corrected target loss) + L_Adv (noise-correcting
    domain discrimination) + L_Reg (source classification regularizer).
  - PIAA pretrain / finetune: MSE on source + L_T + L_Adv + L_Reg, with
    p_t = gaussian_soft_label(score_tgt * 6 + 1, σ) mapping [0, 1] → bin [1, 7].
  - PIAA: ICI model only (MIR raises NotImplementedError, matching CDAN/UGAFEAT).
  - ALDADiscriminator: 3-layer MLP (in_dim → 256 → 128 → K=7, no Dropout, no internal sigmoid).
  - GRL with shared `get_da_lambda` schedule (γ=10, --da_schedule_epochs).
  - Loss weights mirror DANN/CDAN: GIAA `L_y + λ·L_T + L_Adv + L_Reg`,
    PIAA `L_y + 0.1·(λ·L_T + L_Adv + L_Reg)`.
"""

import os
import copy

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..train_common import (
    earth_mover_distance, GradientReversalLayer, get_da_lambda,
    build_piaa_model, num_bins, parse_da_method)
from ..data import collate_fn
from ..evaluate import evaluate, evaluate_piaa
from .cdan import gaussian_soft_label


# ── ALDA modules ──────────────────────────────────────────────────────────────

class ALDADiscriminator(nn.Module):
    """3-layer MLP, K-dim output. Sigmoid applied externally to obtain ξ ∈ [0,1]^K.

    Structure mirrors `train_common.DomainDiscriminator` (Linear→ReLU→Linear→ReLU→Linear,
    hidden 256→128) extended to K-dim output. No Dropout, no internal Sigmoid (see
    design §6.6 案A).
    """
    def __init__(self, in_dim: int, K: int = num_bins, hidden_dim: int = 256):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, K),
        )

    def forward(self, x):
        return self.net(x)


def _compute_corrected_label(xi: torch.Tensor, y_onehot: torch.Tensor, K: int = num_bins) -> torch.Tensor:
    """Compute corrected label vector c = η · ŷ for class-wise uniform η (design §3.3).

    Given ξ ∈ [0, 1]^K (per-class noise vector) and one-hot label ŷ ∈ {0, 1}^K, returns
        c_k = ξ_{ŷ_idx}                 if k = ŷ_idx
              (1 - ξ_{ŷ_idx}) / (K - 1) otherwise

    Vectorized using y_onehot as a soft mask. Differentiable w.r.t. ξ (and y_onehot).
    """
    xi_at_y = (xi * y_onehot).sum(dim=1, keepdim=True)  # (B, 1) = ξ at the true/pseudo class
    return xi_at_y * y_onehot + (1.0 - xi_at_y) / (K - 1) * (1.0 - y_onehot)


def _opposite_distribution(y_onehot: torch.Tensor, K: int = num_bins) -> torch.Tensor:
    """u_k = 0 if k = ŷ_idx, else 1/(K-1).  Used as target for L_Adv on target samples."""
    return (1.0 - y_onehot) / (K - 1)


def _piaa_score_to_bin(score: torch.Tensor) -> torch.Tensor:
    """Map PIAA score in [0, 1] (min-max normalized, see data.py:752) to bin index in [1, 7].

    `gaussian_soft_label` assumes bin centers at integers [1..K]. The PIAA model output is
    in [0, 1], so we expand by *6 (range [0, 6]) and shift by +1 (range [1, 7]).
    """
    return score * (num_bins - 1) + 1.0


def _piaa_score_to_class_idx(score: torch.Tensor) -> torch.Tensor:
    """Map PIAA score in [0, 1] to discrete class index in [0, 6] for one-hot encoding."""
    return (score * (num_bins - 1)).round().long().clamp(0, num_bins - 1)


# ── GIAA ──────────────────────────────────────────────────────────────────────

def setup(model, args, device):
    """Create ALDA GIAA components: ALDADiscriminator (K=7), GRL, optimizer_disc."""
    discriminator = ALDADiscriminator(model.feat_dim, K=num_bins).to(device)
    grl = GradientReversalLayer()
    optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)
    return {
        'discriminator': discriminator,
        'grl': grl,
        'optimizer_disc': optimizer_disc,
    }


def _train_one_epoch(model, src_loader, tgt_loader, optimizer, scaler, device, args,
                     discriminator, grl, optimizer_disc,
                     epoch=None, global_step=0, alda_total_steps=50):
    model.train()
    discriminator.train()
    running_L_y = running_L_T = running_L_Adv = running_L_Reg = 0.0
    running_disc_acc_tgt = 0.0
    running_max_pt = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    threshold = float(getattr(args, 'alda_threshold', 0.0))

    lambda_ = get_da_lambda(global_step, alda_total_steps, getattr(args, 'da_gamma', 10.0))
    desc = f"Epoch {epoch} [ALDA λ={lambda_:.3f}]" if epoch is not None else "Train ALDA"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        lambda_ = get_da_lambda(global_step, alda_total_steps, getattr(args, 'da_gamma', 10.0))

        images_src = sample_src['image'].to(device)
        hist_src = sample_src['Aesthetic'].to(device)  # (B, 7) histogram
        images_tgt = sample_tgt['image'].to(device)

        optimizer.zero_grad()
        optimizer_disc.zero_grad()
        with autocast('cuda'):
            logit_src, domain_feat_src, _ = model(images_src, return_feat=True)
            prob_src = F.softmax(logit_src, dim=1)
            L_y = earth_mover_distance(prob_src, hist_src).mean()

            logit_tgt, domain_feat_tgt, _ = model(images_tgt, return_feat=True)
            p_t = F.softmax(logit_tgt, dim=1)  # (B_tgt, K)

            # one-hot labels: source = argmax(hist), target = argmax(p_t)
            y_s_idx = hist_src.argmax(dim=1)
            y_s_onehot = F.one_hot(y_s_idx, num_classes=num_bins).float()
            y_t_idx = p_t.argmax(dim=1)
            y_t_onehot = F.one_hot(y_t_idx, num_classes=num_bins).float()

            n_src = domain_feat_src.size(0)
            feat_all = torch.cat([domain_feat_src, domain_feat_tgt], dim=0)

            # ── L_Adv: xi from GRL-wrapped features (gradient → D and G via GRL) ──
            xi_adv = torch.sigmoid(discriminator(grl(feat_all, lambda_)))
            label_all_onehot = torch.cat([y_s_onehot, y_t_onehot], dim=0)
            u_tgt = _opposite_distribution(y_t_onehot)

            # BCE is unsafe under autocast (fp16 log loses precision). Cast to fp32 and
            # disable autocast just for the BCE call, mirroring DEEPCORAL/DAREGRAM patterns.
            with autocast('cuda', enabled=False):
                c_adv = _compute_corrected_label(xi_adv.float(), label_all_onehot.float())
                c_adv_clamped = c_adv.clamp(min=1e-7, max=1.0 - 1e-7)
                L_Adv_src = F.binary_cross_entropy(c_adv_clamped[:n_src], y_s_onehot.float())
                L_Adv_tgt = F.binary_cross_entropy(c_adv_clamped[n_src:], u_tgt.float())
                L_Adv = L_Adv_src + L_Adv_tgt

            # ── L_T: detach ξ on the target side so c does not back-propagate to D/G ──
            xi_LT = xi_adv[n_src:].detach()
            c_LT = _compute_corrected_label(xi_LT, y_t_onehot)
            L_T_per = 1.0 - (c_LT * p_t).sum(dim=1)  # (B_tgt,)
            max_pt = p_t.max(dim=1).values
            if threshold > 0.0:
                mask = max_pt > threshold
                if mask.any():
                    L_T = L_T_per[mask].mean()
                else:
                    L_T = torch.zeros((), device=device, dtype=L_T_per.dtype)
            else:
                L_T = L_T_per.mean()

            # ── L_Reg: D-only branch on detached source feature ──
            L_Reg = F.cross_entropy(discriminator(domain_feat_src.detach()), y_s_idx)

            loss = L_y + lambda_ * L_T + L_Adv + L_Reg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(optimizer_disc)
        scaler.update()

        # discriminator accuracy on target. Run inside autocast so the Linear matmul
        # gets matching dtypes (domain_feat_tgt is fp16 from the training forward).
        with torch.no_grad(), autocast('cuda'):
            xi_tgt_eval = torch.sigmoid(discriminator(domain_feat_tgt))
            pred_tgt_idx = xi_tgt_eval.argmin(dim=1)  # opposite distribution → ξ should be small at ŷ
            disc_acc_tgt = (pred_tgt_idx == y_t_idx).float().mean().item()

        max_pt_batch = max_pt.float().mean().item()
        running_L_y += L_y.item()
        running_L_T += L_T.item()
        running_L_Adv += L_Adv.item()
        running_L_Reg += L_Reg.item()
        running_disc_acc_tgt += disc_acc_tgt
        running_max_pt += max_pt_batch
        total_batches += 1
        global_step += 1
        progress_bar.set_postfix({
            'L_y': f'{L_y.item():.4f}',
            'L_T': f'{L_T.item():.4f}',
            'L_Adv': f'{L_Adv.item():.4f}',
            'L_Reg': f'{L_Reg.item():.4f}',
            'max_pt': f'{max_pt_batch:.3f}',
            'λ': f'{lambda_:.3f}',
        })

    n = max(total_batches, 1)
    return {
        'train_emd': running_L_y / n,
        'L_T': running_L_T / n,
        'L_Adv': running_L_Adv / n,
        'L_Reg': running_L_Reg / n,
        'disc_acc_tgt': running_disc_acc_tgt / n,
        'max_pt': running_max_pt / n,
        'global_step': global_step,
    }


def trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname, components,
            tgt_val_loader=None, tgt_genre=None):
    """GIAA trainer for ALDA. Early stopping on source val EMD (DANN/CDAN-compatible)."""
    src_train_loader, val_loader, _ = src_dataloaders
    discriminator = components['discriminator']
    grl = components['grl']
    optimizer_disc = components['optimizer_disc']

    if tgt_loader is None:
        raise ValueError("ALDA GIAA requires a target loader (use --da_method ALDA-<target>).")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)

    steps_per_epoch = len(src_train_loader)
    alda_total_steps = getattr(args, 'da_schedule_epochs', 50) * steps_per_epoch

    best_val_emd = float('inf')
    patience = 0
    global_step = 0
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        metrics = _train_one_epoch(
            model, src_train_loader, tgt_loader, optimizer, scaler, device, args,
            discriminator=discriminator, grl=grl, optimizer_disc=optimizer_disc,
            epoch=epoch, global_step=global_step, alda_total_steps=alda_total_steps)
        global_step = metrics['global_step']
        lambda_ = get_da_lambda(global_step, alda_total_steps, getattr(args, 'da_gamma', 10.0))

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA": metrics['train_emd'],
                f"{args.genre}/Train L_T": metrics['L_T'],
                f"{args.genre}/Train L_Adv": metrics['L_Adv'],
                f"{args.genre}/Train L_Reg": metrics['L_Reg'],
                f"{args.genre}/Train Disc Acc (tgt)": metrics['disc_acc_tgt'],
                f"{args.genre}/Train max(p_t)": metrics['max_pt'],
                f"{args.genre}/ALDA lambda": lambda_,
            }, commit=False)

        val_emd, val_srocc, _, val_mse, _, _, val_ccc = evaluate(
            model, val_loader, device, epoch=epoch, phase_name="Val")
        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Val EMD GIAA": val_emd,
                f"{args.genre}/Val SROCC GIAA": val_srocc,
                f"{args.genre}/Val MSE GIAA": val_mse,
                f"{args.genre}/Val CCC GIAA": val_ccc,
            }, commit=tgt_val_loader is None)

        if tgt_val_loader is not None:
            tgt_val_emd, tgt_val_srocc, _, tgt_val_mse, _, _, tgt_val_ccc = evaluate(
                model, tgt_val_loader, device, epoch=epoch, phase_name=f"Val [{tgt_genre}]")
            if args.is_log:
                wandb.log({
                    "epoch": epoch,
                    f"{tgt_genre}/Val EMD GIAA": tgt_val_emd,
                    f"{tgt_genre}/Val SROCC GIAA": tgt_val_srocc,
                    f"{tgt_genre}/Val MSE GIAA": tgt_val_mse,
                    f"{tgt_genre}/Val CCC GIAA": tgt_val_ccc,
                }, commit=True)

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_emd)
        cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr < prev_lr:
            tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (epoch {epoch}) <<<")

        if val_emd < best_val_emd:
            best_val_emd = val_emd
            patience = 0
            os.makedirs(os.path.dirname(best_modelname), exist_ok=True)
            torch.save(model.state_dict(), best_modelname)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"ALDA: early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_modelname))


# ── PIAA ──────────────────────────────────────────────────────────────────────

def _train_one_epoch_piaa(model, src_loader, tgt_loader, discriminator, grl,
                          optimizer, optimizer_disc, scaler, device, args, genre,
                          sigma, epoch=None, global_step=0, alda_total_steps=50,
                          desc_suffix=""):
    """PIAA 1 epoch: MSE + L_T + L_Adv + L_Reg.

    PIAA score is in [0, 1] (data.py:752). Map to bin index [1, 7] before
    gaussian_soft_label, and to class index [0, 6] for one-hot encoding.
    """
    model.train()
    discriminator.train()
    running_L_y = running_L_T = running_L_Adv = running_L_Reg = 0.0
    running_disc_acc_tgt = 0.0
    running_max_pt = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    threshold = float(getattr(args, 'alda_threshold', 0.0))

    lambda_ = get_da_lambda(global_step, alda_total_steps, getattr(args, 'da_gamma', 10.0))
    desc = f"Epoch {epoch} [ALDA{desc_suffix} λ={lambda_:.3f}]" if epoch is not None else f"Train ALDA{desc_suffix}"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        lambda_ = get_da_lambda(global_step, alda_total_steps, getattr(args, 'da_gamma', 10.0))

        images_src = sample_src['image'].to(device)
        aesthetic_src = sample_src['Aesthetic'].to(device).view(-1, 1)  # [0, 1]
        pt_src = sample_src['traits'].float().to(device)
        attr_src = sample_src['QIP'].float().to(device)

        images_tgt = sample_tgt['image'].to(device)
        pt_tgt = sample_tgt['traits'].float().to(device)
        attr_tgt = sample_tgt['QIP'].float().to(device)

        optimizer.zero_grad()
        optimizer_disc.zero_grad()

        with autocast('cuda'):
            score_src, I_ij_src = model(images_src, pt_src, attr_src, genre, return_feat=True)
            L_y = F.mse_loss(score_src, aesthetic_src)

            score_tgt, I_ij_tgt = model(images_tgt, pt_tgt, attr_tgt, genre, return_feat=True)

            # source one-hot from ground-truth aesthetic (already discrete: aesthetic ∈ {0, 1/6, ..., 1})
            y_s_idx = _piaa_score_to_class_idx(aesthetic_src.view(-1))
            y_s_onehot = F.one_hot(y_s_idx, num_classes=num_bins).float()

            # target p_t: gaussian soft label centered at bin-index of score_tgt
            score_tgt_bin = _piaa_score_to_bin(score_tgt.view(-1))  # [0, 1] → [1, 7]
            p_t = gaussian_soft_label(score_tgt_bin, sigma)
            y_t_idx = p_t.argmax(dim=1)
            y_t_onehot = F.one_hot(y_t_idx, num_classes=num_bins).float()

            n_src = I_ij_src.size(0)
            feat_all = torch.cat([I_ij_src, I_ij_tgt], dim=0)

            # ── L_Adv ──
            xi_adv = torch.sigmoid(discriminator(grl(feat_all, lambda_)))
            label_all_onehot = torch.cat([y_s_onehot, y_t_onehot], dim=0)
            u_tgt = _opposite_distribution(y_t_onehot)

            # BCE is unsafe under autocast (fp16 log loses precision). Cast to fp32 and
            # disable autocast just for the BCE call, mirroring DEEPCORAL/DAREGRAM patterns.
            with autocast('cuda', enabled=False):
                c_adv = _compute_corrected_label(xi_adv.float(), label_all_onehot.float())
                c_adv_clamped = c_adv.clamp(min=1e-7, max=1.0 - 1e-7)
                L_Adv_src = F.binary_cross_entropy(c_adv_clamped[:n_src], y_s_onehot.float())
                L_Adv_tgt = F.binary_cross_entropy(c_adv_clamped[n_src:], u_tgt.float())
                L_Adv = L_Adv_src + L_Adv_tgt

            # ── L_T ──
            xi_LT = xi_adv[n_src:].detach()
            c_LT = _compute_corrected_label(xi_LT, y_t_onehot)
            L_T_per = 1.0 - (c_LT * p_t).sum(dim=1)
            max_pt = p_t.max(dim=1).values
            if threshold > 0.0:
                mask = max_pt > threshold
                if mask.any():
                    L_T = L_T_per[mask].mean()
                else:
                    L_T = torch.zeros((), device=device, dtype=L_T_per.dtype)
            else:
                L_T = L_T_per.mean()

            # ── L_Reg ──
            L_Reg = F.cross_entropy(discriminator(I_ij_src.detach()), y_s_idx)

            # PIAA loss weighting (mirror DANN/CDAN PIAA: 0.1× DA-related terms)
            loss = L_y + 0.1 * (lambda_ * L_T + L_Adv + L_Reg)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(optimizer_disc)
        scaler.update()

        # I_ij_tgt is fp16 from the training forward; wrap in autocast so the
        # Linear matmul inside discriminator gets matching dtypes.
        with torch.no_grad(), autocast('cuda'):
            xi_tgt_eval = torch.sigmoid(discriminator(I_ij_tgt))
            pred_tgt_idx = xi_tgt_eval.argmin(dim=1)
            disc_acc_tgt = (pred_tgt_idx == y_t_idx).float().mean().item()

        max_pt_batch = max_pt.float().mean().item()
        running_L_y += L_y.item()
        running_L_T += L_T.item()
        running_L_Adv += L_Adv.item()
        running_L_Reg += L_Reg.item()
        running_disc_acc_tgt += disc_acc_tgt
        running_max_pt += max_pt_batch
        total_batches += 1
        global_step += 1
        progress_bar.set_postfix({
            'L_y': f'{L_y.item():.4f}',
            'L_T': f'{L_T.item():.4f}',
            'L_Adv': f'{L_Adv.item():.4f}',
            'L_Reg': f'{L_Reg.item():.4f}',
            'max_pt': f'{max_pt_batch:.3f}',
            'λ': f'{lambda_:.3f}',
        })

    n = max(total_batches, 1)
    return (running_L_y / n, running_L_T / n, running_L_Adv / n,
            running_L_Reg / n, running_disc_acc_tgt / n, running_max_pt / n, global_step)


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    """ALDA pretrain trainer for PIAA (ICI). Early stopping on source val CCC."""
    if getattr(args, 'model_type', 'ICI') != 'ICI':
        raise NotImplementedError("ALDA pretrain supports the ICI model only")

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = domain_tag if domain_tag else genre
    alda_target_genre = parse_da_method(getattr(args, 'da_method', None))[1]
    sigma = float(getattr(args, 'alda_sigma', 1.0))

    src_loader = DataLoader(datasets_dict[genre]['train'], batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    tgt_loader = DataLoader(tgt_train_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    val_loaders_dict = {genre: DataLoader(datasets_dict[genre]['val'], batch_size=batch_size, shuffle=False,
                                          num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}
    tgt_val_loaders_dict = {genre: DataLoader(tgt_val_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}

    model = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)

    pretrained_path = pretrained_model_dict[genre]
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained NIMA model not found: {pretrained_path}")
    try:
        state = torch.load(pretrained_path)
        model.nima_dict[genre].load_state_dict(state)
        print(f"Loaded NIMA weights for {genre} from {pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load NIMA weights for {genre}: {e}")

    model.freeze_backbone()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    d_f = model.input_dim
    discriminator = ALDADiscriminator(d_f, K=num_bins).to(device)
    grl = GradientReversalLayer()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

    steps_per_epoch = len(src_loader)
    alda_total_steps = getattr(args, 'da_schedule_epochs', 50) * steps_per_epoch

    best_val_ccc = -float('inf')
    patience = 0
    global_step = 0
    _alda_run = experiment_name.removeprefix('ALDA_')
    best_model_path = os.path.join(dirname, f'{genre_str}_ALDA_{args.model_type}_{_alda_run}_pretrain.pth')
    best_state_dict = None

    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        L_y, L_T, L_Adv, L_Reg, disc_acc_tgt, max_pt, global_step = _train_one_epoch_piaa(
            model, src_loader, tgt_loader, discriminator, grl,
            optimizer, optimizer_disc, scaler, device, args, genre, sigma,
            epoch=epoch, global_step=global_step, alda_total_steps=alda_total_steps,
            desc_suffix=" pretrain")
        lambda_ = get_da_lambda(global_step, alda_total_steps, getattr(args, 'da_gamma', 10.0))

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss": L_y,
                f"{genre}/Train L_T": L_T,
                f"{genre}/Train L_Adv": L_Adv,
                f"{genre}/Train L_Reg": L_Reg,
                f"{genre}/Train Disc Acc (tgt)": disc_acc_tgt,
                f"{genre}/Train max(p_t)": max_pt,
                f"{genre}/ALDA lambda": lambda_,
            }, commit=False)

        genre_metrics, _ = evaluate_piaa(model, val_loaders_dict, device, epoch=epoch, phase_name="Val")
        val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

        tgt_genre_metrics, _ = evaluate_piaa(model, tgt_val_loaders_dict, device, epoch=epoch, phase_name="Val (tgt)")

        if args.is_log:
            log_dict = {"epoch": epoch}
            if genre in genre_metrics:
                log_dict[f"{genre}/Val MAE"] = genre_metrics[genre]['mae']
                log_dict[f"{genre}/Val SROCC"] = genre_metrics[genre]['srocc']
                log_dict[f"{genre}/Val NDCG@10"] = genre_metrics[genre]['ndcg@10']
                log_dict[f"{genre}/Val CCC"] = genre_metrics[genre]['ccc']
            if hasattr(model, '_eval_component_stats') and genre in model._eval_component_stats:
                cs = model._eval_component_stats[genre]
                log_dict[f"{genre}/Val interaction_mean"] = cs['interaction_mean']
                log_dict[f"{genre}/Val direct_mean"] = cs['direct_mean']
                log_dict[f"{genre}/Val interaction_ratio"] = cs['ratio']
            if genre in tgt_genre_metrics:
                tgt_m = tgt_genre_metrics[genre]
                log_dict[f"{alda_target_genre}/Val MAE"] = tgt_m['mae']
                log_dict[f"{alda_target_genre}/Val SROCC"] = tgt_m['srocc']
                log_dict[f"{alda_target_genre}/Val NDCG@10"] = tgt_m['ndcg@10']
                log_dict[f"{alda_target_genre}/Val CCC"] = tgt_m['ccc']
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
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"ALDA Pretrain: early stopping at epoch {epoch}")
                break

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, alda_target_genre=None):
    """ALDA finetune trainer for PIAA (ICI). Per-user training, early stopping on source val CCC."""
    if getattr(args, 'model_type', 'ICI') != 'ICI':
        raise NotImplementedError("ALDA finetune supports the ICI model only")

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre
    sigma = float(getattr(args, 'alda_sigma', 1.0))

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"ALDA finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)
        user_val_src = copy.copy(datasets_dict[genre]['val'])
        user_val_src.data = datasets_dict[genre]['val'].data[
            datasets_dict[genre]['val'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{alda_target_genre}' train_piaa_dataset. "
                f"All finetune users must exist in the target genre."
            )
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        tgt_val_mask = tgt_val_piaa_dataset.data['user_id'] == uid
        if tgt_val_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{alda_target_genre}' val_piaa_dataset. "
                f"All finetune users must exist in the target genre."
            )
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

        model_user = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
        pretrained_path = pretrained_model_dict[genre]
        if pretrained_path is None or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"ALDA pretrained model not found: {pretrained_path}")
        try:
            state = torch.load(pretrained_path)
            incompatible = model_user.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                print(f"[load_state_dict] Ignored unexpected keys: {incompatible.unexpected_keys}")
            print(f"Loaded ALDA pretrain weights from {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {pretrained_path}: {e}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            print(f"[Freeze] Backbone frozen: {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        d_f = model_user.input_dim
        discriminator = ALDADiscriminator(d_f, K=num_bins).to(device)
        grl = GradientReversalLayer()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)
        optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

        steps_per_epoch = len(src_loader)
        alda_total_steps = getattr(args, 'da_schedule_epochs', 50) * steps_per_epoch

        best_val_ccc = -float('inf')
        patience = 0
        global_step = 0
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        # epoch 0 前に pretrain 重みを保存しておく:
        # val_ccc が一度も改善しない (NaN 等) ユーザーでも .pth が必ず存在し、
        # inference 時の "best model not found" によるユーザー欠損を防ぐ。
        torch.save(model_user.state_dict(), best_model_path)

        for epoch in range(args.num_epochs):
            L_y, L_T, L_Adv, L_Reg, disc_acc_tgt, max_pt, global_step = _train_one_epoch_piaa(
                model_user, src_loader, tgt_loader, discriminator, grl,
                optimizer, optimizer_disc, scaler, device, args, genre, sigma,
                epoch=epoch, global_step=global_step, alda_total_steps=alda_total_steps,
                desc_suffix=" finetune")
            lambda_ = get_da_lambda(global_step, alda_total_steps, getattr(args, 'da_gamma', 10.0))

            genre_metrics, _ = evaluate_piaa(model_user, val_src_loaders, device, epoch=epoch, phase_name="Val (src)")
            val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

            tgt_genre_metrics, _ = evaluate_piaa(model_user, val_tgt_loaders, device, epoch=epoch, phase_name="Val (tgt)")

            if args.is_log:
                log_dict = {"epoch": epoch}
                log_dict[f"{genre}/Train Loss user_{uid}"] = L_y
                log_dict[f"{genre}/Train L_T user_{uid}"] = L_T
                log_dict[f"{genre}/Train L_Adv user_{uid}"] = L_Adv
                log_dict[f"{genre}/Train L_Reg user_{uid}"] = L_Reg
                log_dict[f"{genre}/Train Disc Acc (tgt) user_{uid}"] = disc_acc_tgt
                log_dict[f"{genre}/Train max(p_t) user_{uid}"] = max_pt
                log_dict[f"{genre}/ALDA lambda user_{uid}"] = lambda_
                if genre in genre_metrics:
                    log_dict[f"{genre}/Val MAE user_{uid}"] = genre_metrics[genre]['mae']
                    log_dict[f"{genre}/Val SROCC user_{uid}"] = genre_metrics[genre]['srocc']
                    log_dict[f"{genre}/Val CCC user_{uid}"] = genre_metrics[genre]['ccc']
                if genre in tgt_genre_metrics:
                    tgt_m = tgt_genre_metrics[genre]
                    log_dict[f"{alda_target_genre}/Val MAE user_{uid}"] = tgt_m['mae']
                    log_dict[f"{alda_target_genre}/Val SROCC user_{uid}"] = tgt_m['srocc']
                    log_dict[f"{alda_target_genre}/Val CCC user_{uid}"] = tgt_m['ccc']
                wandb.log(log_dict, commit=True)

            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_ccc)
            cur_lr = optimizer.param_groups[0]['lr']
            if cur_lr < prev_lr:
                tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (user {uid}, epoch {epoch}) <<<")

            if val_ccc > best_val_ccc:
                best_val_ccc = val_ccc
                patience = 0
                torch.save(model_user.state_dict(), best_model_path)
            else:
                patience += 1
                if patience >= args.max_patience_epochs:
                    print(f"User {uid}: early stopping at epoch {epoch}")
                    break
