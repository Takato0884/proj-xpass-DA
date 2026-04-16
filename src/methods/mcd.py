import os

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from ..train_common import NIMA, earth_mover_distance
from ..evaluate import evaluate


# ── MCD Model ─────────────────────────────────────────────────────────────────

class NIMA_MCD(nn.Module):
    """MCD wrapper around NIMA.

    G  = backbone + feat_proj  (shared references to the base NIMA instance)
    F1 = fc_aesthetic           (shared reference to the base NIMA instance)
    F2 = independent linear head (same shape as F1, freshly initialized)

    Training NIMA_MCD updates the underlying NIMA parameters in-place,
    so the base model passed to setup() always reflects the current state.
    """

    def __init__(self, nima: NIMA):
        super().__init__()
        self._nima = nima  # registered as submodule; shares G and F1 parameters
        self.f2 = nn.Linear(nima.feat_dim, nima.fc_aesthetic.out_features)
        nn.init.xavier_uniform_(self.f2.weight)
        nn.init.zeros_(self.f2.bias)

    # ── Forward helpers ───────────────────────────────────────────────────────

    def forward_feat(self, x):
        """G(x) = backbone(x) → feat_proj."""
        raw = self._nima.backbone(x)
        return self._nima.feat_proj(raw)

    def forward_f1(self, feat):
        return self._nima.fc_aesthetic(feat)

    def forward_f2(self, feat):
        return self.f2(feat)

    def forward(self, x, return_feat=False):
        """Default forward using F1 only (compatible with evaluate())."""
        feat = self.forward_feat(x)
        logit = self.forward_f1(feat)
        if return_feat:
            return logit, feat, None
        return logit

    # ── Parameter groups ─────────────────────────────────────────────────────

    def g_parameters(self):
        """Trainable parameters of G (feat_proj only; backbone is frozen)."""
        return self._nima.feat_proj.parameters()

    def f_parameters(self):
        """Parameters of F1 (fc_aesthetic) and F2 combined."""
        return list(self._nima.fc_aesthetic.parameters()) + list(self.f2.parameters())


# ── L1-EMD (Discrepancy) ──────────────────────────────────────────────────────

def _emd_l1(p, q):
    """Per-sample L1 Earth Mover's Distance via cumulative-sum L1 distance.

    EMD(p, q) = sum_k |CDF_p(k) - CDF_q(k)|

    This is the L1 variant recommended by the MCD paper for discrepancy
    calculation; it differs from the L2 variant used in train_common.py.

    Args:
        p, q: (batch, K) probability distributions (should sum to 1 along dim=-1)

    Returns:
        (batch,) L1-EMD values
    """
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return (cdf_p - cdf_q).abs().sum(dim=-1)


# ── setup / trainer ───────────────────────────────────────────────────────────

def setup(model, args, device):
    """Create MCD-specific components: NIMA_MCD wrapper and dual optimizers.

    Returns a components dict with:
      - 'mcd_model':   NIMA_MCD wrapping the base model
      - 'optimizer_G': AdamW for G (feat_proj)
      - 'optimizer_F': AdamW for F1+F2 (fc_aesthetic + f2)
    """
    mcd_model = NIMA_MCD(model).to(device)
    optimizer_G = optim.AdamW(mcd_model.g_parameters(), lr=args.lr)
    optimizer_F = optim.AdamW(mcd_model.f_parameters(), lr=args.lr)
    return {'mcd_model': mcd_model, 'optimizer_G': optimizer_G, 'optimizer_F': optimizer_F}


def _train_one_epoch(mcd_model, model, src_loader, tgt_loader,
                     optimizer_G, optimizer_F, scaler, device, args,
                     epoch=None):
    mcd_model.train()

    lambda_ = getattr(args, 'mcd_lambda', 1.0)
    n_steps  = getattr(args, 'mcd_n_steps', 4)

    running_L_s = running_L_adv_B = running_L_adv_C = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [MCD]" if epoch is not None else "Train MCD"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0,
                        ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        images_src = sample_src['image'].to(device)
        hist_src   = sample_src['Aesthetic'].to(device)
        images_tgt = sample_tgt['image'].to(device)

        # ── Step A: min_{G, F1, F2} L_s ──────────────────────────────────────
        # Update both G and F on source classification loss.
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        with autocast('cuda'):
            feat_src   = mcd_model.forward_feat(images_src)
            prob1_src  = F.softmax(mcd_model.forward_f1(feat_src), dim=1)
            prob2_src  = F.softmax(mcd_model.forward_f2(feat_src), dim=1)
            L_s_A = (earth_mover_distance(prob1_src, hist_src).mean() +
                     earth_mover_distance(prob2_src, hist_src).mean()) / 2
        scaler.scale(L_s_A).backward()
        scaler.step(optimizer_G)
        scaler.step(optimizer_F)
        scaler.update()

        # ── Step B: min_{F1, F2} L_s - lambda * L_adv  (G fixed) ─────────────
        # Maximize discrepancy on target; G is frozen via torch.no_grad().
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        with autocast('cuda'):
            with torch.no_grad():
                feat_src_b = mcd_model.forward_feat(images_src)
                feat_tgt_b = mcd_model.forward_feat(images_tgt)
            prob1_src_b = F.softmax(mcd_model.forward_f1(feat_src_b), dim=1)
            prob2_src_b = F.softmax(mcd_model.forward_f2(feat_src_b), dim=1)
            L_s_B = (earth_mover_distance(prob1_src_b, hist_src).mean() +
                     earth_mover_distance(prob2_src_b, hist_src).mean()) / 2
            prob1_tgt_b = F.softmax(mcd_model.forward_f1(feat_tgt_b), dim=1)
            prob2_tgt_b = F.softmax(mcd_model.forward_f2(feat_tgt_b), dim=1)
            L_adv_B = _emd_l1(prob1_tgt_b, prob2_tgt_b).mean()
            loss_B  = L_s_B - lambda_ * L_adv_B
        scaler.scale(loss_B).backward()
        scaler.step(optimizer_F)
        scaler.update()

        # ── Step C: min_{G} L_adv  (F1, F2 fixed), repeated n_steps times ─────
        # Minimize discrepancy by pulling target features toward source support.
        # optimizer_F is never stepped here, so F1/F2 remain unchanged.
        L_adv_C_val = 0.0
        for _ in range(n_steps):
            optimizer_G.zero_grad()
            with autocast('cuda'):
                feat_tgt_c  = mcd_model.forward_feat(images_tgt)
                prob1_tgt_c = F.softmax(mcd_model.forward_f1(feat_tgt_c), dim=1)
                prob2_tgt_c = F.softmax(mcd_model.forward_f2(feat_tgt_c), dim=1)
                L_adv_C = _emd_l1(prob1_tgt_c, prob2_tgt_c).mean()
            scaler.scale(L_adv_C).backward()
            scaler.step(optimizer_G)
            scaler.update()
            L_adv_C_val = L_adv_C.item()

        running_L_s     += L_s_A.item()
        running_L_adv_B += L_adv_B.item()
        running_L_adv_C += L_adv_C_val
        total_batches   += 1

        progress_bar.set_postfix({
            'L_s':     f'{L_s_A.item():.4f}',
            'L_adv_B': f'{L_adv_B.item():.4f}',
            'L_adv_C': f'{L_adv_C_val:.4f}',
        })

    n = max(total_batches, 1)
    return {
        'train_emd': running_L_s     / n,
        'L_adv_B':   running_L_adv_B / n,
        'L_adv_C':   running_L_adv_C / n,
    }


def trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname, components,
            tgt_val_loader=None, tgt_genre=None):
    """MCD training loop.

    Args:
        model:      Base NIMA model (G+F1); same object as components['mcd_model']._nima.
        optimizer:  Not used (MCD uses its own dual optimizers from components).
        components: Dict returned by setup(); contains 'mcd_model', 'optimizer_G', 'optimizer_F'.
    """
    src_train_loader, val_loader, _ = src_dataloaders
    mcd_model   = components['mcd_model']
    optimizer_G = components['optimizer_G']
    optimizer_F = components['optimizer_F']

    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)
    scheduler_F = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_F, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)

    best_val_emd = float('inf')
    patience     = 0
    scaler       = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        metrics = _train_one_epoch(
            mcd_model, model, src_train_loader, tgt_loader,
            optimizer_G, optimizer_F, scaler, device, args,
            epoch=epoch)

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA":  metrics['train_emd'],
                f"{args.genre}/Train L_adv_B":   metrics['L_adv_B'],
                f"{args.genre}/Train L_adv_C":   metrics['L_adv_C'],
            }, commit=False)

        # ── Source validation (early-stopping criterion) ──────────────────────
        val_emd, val_srocc, _, val_mse, _, _, _ = evaluate(
            mcd_model, val_loader, device, epoch=epoch, phase_name="Val")
        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Val EMD GIAA":   val_emd,
                f"{args.genre}/Val SROCC GIAA": val_srocc,
                f"{args.genre}/Val MSE GIAA":   val_mse,
            }, commit=tgt_val_loader is None)

        # ── Target validation (monitoring only) ───────────────────────────────
        if tgt_val_loader is not None:
            tgt_val_emd, tgt_val_srocc, _, tgt_val_mse, _, _, _ = evaluate(
                mcd_model, tgt_val_loader, device, epoch=epoch,
                phase_name=f"Val [{tgt_genre}]")
            if args.is_log:
                wandb.log({
                    "epoch": epoch,
                    f"{tgt_genre}/Val EMD GIAA":   tgt_val_emd,
                    f"{tgt_genre}/Val SROCC GIAA": tgt_val_srocc,
                    f"{tgt_genre}/Val MSE GIAA":   tgt_val_mse,
                }, commit=True)

        # ── LR scheduling ─────────────────────────────────────────────────────
        prev_lr = optimizer_G.param_groups[0]['lr']
        scheduler_G.step(val_emd)
        scheduler_F.step(val_emd)
        cur_lr = optimizer_G.param_groups[0]['lr']
        if cur_lr < prev_lr:
            tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (epoch {epoch}) <<<")

        # ── Early stopping (based on source val EMD) ──────────────────────────
        if val_emd < best_val_emd:
            best_val_emd = val_emd
            patience = 0
            os.makedirs(os.path.dirname(best_modelname), exist_ok=True)
            # Save only the base NIMA state (G + F1); F2 is not needed at inference.
            torch.save(model.state_dict(), best_modelname)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best weights into the base model (shared with mcd_model._nima).
    model.load_state_dict(torch.load(best_modelname))
