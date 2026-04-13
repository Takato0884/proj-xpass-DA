import os

import numpy as np
import ot
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from ..train_common import earth_mover_distance
from ..evaluate import evaluate


def setup(model, args, device):
    """DeepJDOT requires no extra components beyond the base model."""
    return {}


def _emd_matrix(y_s, y_t):
    """Broadcast EMD between all (n_s, n_t) distribution pairs.

    Uses CDF-based L2 norm (same formula as EarthMoverDistance in train_common).

    Args:
        y_s: (n_s, num_bins) source distributions
        y_t: (n_t, num_bins) target distributions

    Returns:
        (n_s, n_t) EMD matrix
    """
    cdf_s = torch.cumsum(y_s, dim=-1).unsqueeze(1)   # (n_s, 1, num_bins)
    cdf_t = torch.cumsum(y_t, dim=-1).unsqueeze(0)   # (1, n_t, num_bins)
    return torch.norm(cdf_s - cdf_t, p=2, dim=-1)    # (n_s, n_t)


def _train_one_epoch(model, src_loader, tgt_loader, optimizer, scaler, device, args,
                     epoch=None, global_step=0):
    model.train()
    alpha = getattr(args, 'djdot_alpha', 0.001)
    lambda_t = getattr(args, 'djdot_lambda_t', 0.0001)

    running_L_s = running_L_feat = running_L_label = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [DJDOT]" if epoch is not None else "Train DJDOT"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120,
                        colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        images_src = sample_src['image'].to(device)
        hist_src   = sample_src['Aesthetic'].to(device)
        images_tgt = sample_tgt['image'].to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            # ── Forward pass ──────────────────────────────────────────────────
            logit_src, z_s, _ = model(images_src, return_feat=True)
            prob_src = F.softmax(logit_src, dim=1)

            logit_tgt, z_t, _ = model(images_tgt, return_feat=True)
            pred_t = F.softmax(logit_tgt, dim=1)

            # Source EMD loss (① in objective)
            L_s = earth_mover_distance(prob_src, hist_src).mean()

            # ── Step 1: solve OT to get γ (detached, CPU) ─────────────────────
            with torch.no_grad():
                feat_dist_d = torch.cdist(z_s.float(), z_t.float(), p=2).pow(2)  # (n_s, n_t)
                label_cost_d = _emd_matrix(hist_src.float(), pred_t.float())      # (n_s, n_t)
                C = (alpha * feat_dist_d + lambda_t * label_cost_d).cpu().numpy().astype(np.float64)
                n_s, n_t = C.shape
                a = np.ones(n_s, dtype=np.float64) / n_s
                b = np.ones(n_t, dtype=np.float64) / n_t
                gamma = torch.from_numpy(
                    ot.emd(a, b, C)               # network simplex LP
                ).to(dtype=z_s.dtype, device=device)

            # ── Step 2: compute alignment losses with gradients ────────────────
            # γ is detached; gradients flow through z_s, z_t, and pred_t
            feat_dist  = torch.cdist(z_s, z_t, p=2).pow(2)   # (n_s, n_t)
            label_cost = _emd_matrix(hist_src, pred_t)         # (n_s, n_t)

            L_feat  = (gamma * feat_dist).sum()   # ② feature alignment
            L_label = (gamma * label_cost).sum()  # ③ label alignment

            loss = L_s + alpha * L_feat + lambda_t * L_label

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_L_s     += L_s.item()
        running_L_feat  += L_feat.item()
        running_L_label += L_label.item()
        total_batches   += 1
        global_step     += 1

        progress_bar.set_postfix({
            'L_s':     f'{L_s.item():.4f}',
            'L_feat':  f'{L_feat.item():.4f}',
            'L_label': f'{L_label.item():.4f}',
        })

    n = max(total_batches, 1)
    return {
        'train_emd':   running_L_s     / n,
        'feat_loss':   running_L_feat  / n,
        'label_loss':  running_L_label / n,
        'global_step': global_step,
    }


def trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname, components,
            tgt_val_loader=None, tgt_genre=None):
    src_train_loader, val_loader, _ = src_dataloaders

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)

    best_val_emd = float('inf')
    patience     = 0
    global_step  = 0
    scaler       = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        metrics = _train_one_epoch(
            model, src_train_loader, tgt_loader, optimizer, scaler, device, args,
            epoch=epoch, global_step=global_step)
        global_step = metrics['global_step']

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA":        metrics['train_emd'],
                f"{args.genre}/Train Feature Loss":    metrics['feat_loss'],
                f"{args.genre}/Train Label Loss":      metrics['label_loss'],
            }, commit=False)

        val_emd, val_srocc, _, val_mse, _, _, _ = evaluate(
            model, val_loader, device, epoch=epoch, phase_name="Val")
        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Val EMD GIAA":   val_emd,
                f"{args.genre}/Val SROCC GIAA": val_srocc,
                f"{args.genre}/Val MSE GIAA":   val_mse,
            }, commit=tgt_val_loader is None)

        if tgt_val_loader is not None:
            tgt_val_emd, tgt_val_srocc, _, tgt_val_mse, _, _, _ = evaluate(
                model, tgt_val_loader, device, epoch=epoch, phase_name=f"Val [{tgt_genre}]")
            if args.is_log:
                wandb.log({
                    "epoch": epoch,
                    f"{tgt_genre}/Val EMD GIAA":   tgt_val_emd,
                    f"{tgt_genre}/Val SROCC GIAA": tgt_val_srocc,
                    f"{tgt_genre}/Val MSE GIAA":   tgt_val_mse,
                }, commit=True)

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_emd)
        cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr < prev_lr:
            tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (epoch {epoch}) <<<")

        # Early stopping based on source val EMD (per design doc §8.5)
        if val_emd < best_val_emd:
            best_val_emd = val_emd
            patience = 0
            os.makedirs(os.path.dirname(best_modelname), exist_ok=True)
            torch.save(model.state_dict(), best_modelname)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_modelname))
