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

from ..train_common import NIMA, earth_mover_distance, build_piaa_model, num_bins, parse_da_method
from ..evaluate import evaluate, evaluate_piaa
from ..data import collate_fn


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


# ─── PIAA MCD ─────────────────────────────────────────────────────────────────

class PIAA_MCD(nn.Module):
    """MCD wrapper around PIAA_ICI_CrossDomain.

    G  = backbone_image_proj + all interaction modules (up to I_ij)
    F1 = attr_corr  (existing nn.Linear(input_dim, 1), shared reference)
    F2 = independent nn.Linear(input_dim, 1), freshly initialized
    """

    def __init__(self, model):
        super().__init__()
        self._model = model
        self.f2 = nn.Linear(model.input_dim, 1)
        nn.init.xavier_uniform_(self.f2.weight)
        nn.init.zeros_(self.f2.bias)

    def forward(self, images, pt, attrs, genre, return_feat=False):
        """Default forward using F1 (attr_corr); compatible with evaluate_piaa."""
        return self._model(images, pt, attrs, genre, return_feat=return_feat)

    def forward_feat(self, images, pt, attrs, genre):
        """Returns I_ij (G output). The full score computed internally is discarded."""
        _, I_ij = self._model(images, pt, attrs, genre, return_feat=True)
        return I_ij

    def forward_f1(self, I_ij):
        return self._model.attr_corr(I_ij)

    def forward_f2(self, I_ij):
        return self.f2(I_ij)

    def g_parameters(self):
        """Trainable G params: all trainable params except attr_corr (F1)."""
        f1_ids = {id(p) for p in self._model.attr_corr.parameters()}
        return [p for p in self._model.parameters() if p.requires_grad and id(p) not in f1_ids]

    def f_parameters(self):
        """F1 (attr_corr) + F2 parameters."""
        return list(self._model.attr_corr.parameters()) + list(self.f2.parameters())


def _train_one_epoch_piaa(mcd_model, src_loader, tgt_loader,
                           optimizer_G, optimizer_F, scaler, device, args, genre,
                           epoch=None, phase='pretrain'):
    mcd_model.train()
    lambda_ = getattr(args, 'mcd_lambda', 1.0)
    n_steps  = getattr(args, 'mcd_n_steps', 4)

    running_L_s = running_L_adv_B = running_L_adv_C = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [MCD-PIAA {phase}]" if epoch is not None else f"Train MCD-PIAA {phase}"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0,
                        ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        images_src    = sample_src['image'].to(device)
        aesthetic_src = sample_src['Aesthetic'].to(device).view(-1, 1)
        pt_src        = sample_src['traits'].float().to(device)
        attr_src      = sample_src['QIP'].float().to(device)

        images_tgt = sample_tgt['image'].to(device)
        pt_tgt     = sample_tgt['traits'].float().to(device)
        attr_tgt   = sample_tgt['QIP'].float().to(device)

        # ── Step A: min_{G, F1, F2} L_s ──────────────────────────────────────
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        with autocast('cuda'):
            score_f1_src, I_ij_src = mcd_model(images_src, pt_src, attr_src, genre, return_feat=True)
            # direct = frozen NIMA contribution (score_f1 - attr_corr(I_ij))
            with torch.no_grad():
                f1_int_const = mcd_model.forward_f1(I_ij_src)
            direct_src = score_f1_src.detach() - f1_int_const
            score_f2_src = mcd_model.forward_f2(I_ij_src) + direct_src
            L_s_A = (F.mse_loss(score_f1_src, aesthetic_src) +
                     F.mse_loss(score_f2_src, aesthetic_src)) / 2
        scaler.scale(L_s_A).backward()
        scaler.step(optimizer_G)
        scaler.step(optimizer_F)
        scaler.update()

        # ── Step B: min_{F1, F2} L_s - lambda * L_adv  (G fixed) ─────────────
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        with autocast('cuda'):
            with torch.no_grad():
                score_tmp_b, I_ij_src_b = mcd_model(images_src, pt_src, attr_src, genre, return_feat=True)
                f1_tmp_b   = mcd_model.forward_f1(I_ij_src_b)
                direct_src_b = score_tmp_b - f1_tmp_b  # constant (frozen NIMA)
                I_ij_tgt_b = mcd_model.forward_feat(images_tgt, pt_tgt, attr_tgt, genre)

            # I_ij_src_b / I_ij_tgt_b are detached → grad flows through F only
            f1_int_b = mcd_model.forward_f1(I_ij_src_b)
            f2_int_b = mcd_model.forward_f2(I_ij_src_b)
            score_f1_b = f1_int_b + direct_src_b
            score_f2_b = f2_int_b + direct_src_b
            L_s_B = (F.mse_loss(score_f1_b, aesthetic_src) +
                     F.mse_loss(score_f2_b, aesthetic_src)) / 2

            f1_tgt_b = mcd_model.forward_f1(I_ij_tgt_b)
            f2_tgt_b = mcd_model.forward_f2(I_ij_tgt_b)
            L_adv_B = (f1_tgt_b - f2_tgt_b).abs().mean()
            loss_B  = L_s_B - lambda_ * L_adv_B
        scaler.scale(loss_B).backward()
        scaler.step(optimizer_F)
        scaler.update()

        # ── Step C: min_{G} L_adv  (F1, F2 fixed), repeated n_steps times ─────
        L_adv_C_val = 0.0
        for _ in range(n_steps):
            optimizer_G.zero_grad()
            with autocast('cuda'):
                I_ij_tgt_c = mcd_model.forward_feat(images_tgt, pt_tgt, attr_tgt, genre)
                f1_tgt_c = mcd_model.forward_f1(I_ij_tgt_c)
                f2_tgt_c = mcd_model.forward_f2(I_ij_tgt_c)
                L_adv_C = (f1_tgt_c - f2_tgt_c).abs().mean()
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
    return running_L_s / n, running_L_adv_B / n, running_L_adv_C / n


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    """MCD pretrain trainer for PIAA（ICI のみ）.
    ソース: train_giaa_dataset でタスク学習 + I_ij レベルの MCD 適応.
    早期停止: ソース val CCC.
    Returns:
        best_model_path, best_state_dict
    """
    if args.model_type != 'ICI':
        raise NotImplementedError("MCD pretrain は ICI モデルのみサポートしています")

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
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    mcd_model  = PIAA_MCD(model).to(device)
    optimizer_G = optim.AdamW(mcd_model.g_parameters(), lr=args.lr)
    optimizer_F = optim.AdamW(mcd_model.f_parameters(), lr=args.lr)
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)
    scheduler_F = optim.lr_scheduler.ReduceLROnPlateau(optimizer_F, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

    best_val_ccc = -float('inf')
    patience = 0
    _mcd_run = experiment_name.removeprefix('MCD_')
    best_model_path = os.path.join(dirname, f'{genre_str}_MCD_{args.model_type}_{_mcd_run}_pretrain.pth')
    best_state_dict = None
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        L_s, L_adv_B, L_adv_C = _train_one_epoch_piaa(
            mcd_model, src_loader, tgt_loader,
            optimizer_G, optimizer_F, scaler, device, args, genre,
            epoch=epoch, phase='pretrain')

        if args.is_log:
            lambda_ = getattr(args, 'mcd_lambda', 1.0)
            adv_ratio = lambda_ * L_adv_B / L_s if L_s > 0 else 0.0
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss":           L_s,
                f"{genre}/Train L_adv_B":        L_adv_B,
                f"{genre}/Train L_adv_C":        L_adv_C,
                f"{genre}/Train adv_ratio":      adv_ratio,
            }, commit=False)

        genre_metrics, _ = evaluate_piaa(model, val_loaders_dict, device, epoch=epoch, phase_name="Val")
        val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

        tgt_genre_metrics, _ = evaluate_piaa(model, tgt_val_loaders_dict, device, epoch=epoch, phase_name="Val (tgt)")

        if args.is_log:
            log_dict = {"epoch": epoch}
            if genre in genre_metrics:
                log_dict[f"{genre}/Val MAE"]    = genre_metrics[genre]['mae']
                log_dict[f"{genre}/Val SROCC"]  = genre_metrics[genre]['srocc']
                log_dict[f"{genre}/Val NDCG@10"] = genre_metrics[genre]['ndcg@10']
                log_dict[f"{genre}/Val CCC"]    = genre_metrics[genre]['ccc']
            if hasattr(model, '_eval_component_stats') and genre in model._eval_component_stats:
                cs = model._eval_component_stats[genre]
                log_dict[f"{genre}/Val interaction_mean"] = cs['interaction_mean']
                log_dict[f"{genre}/Val direct_mean"]      = cs['direct_mean']
                log_dict[f"{genre}/Val interaction_ratio"] = cs['ratio']
            if genre in tgt_genre_metrics:
                tgt_m = tgt_genre_metrics[genre]
                log_dict[f"{tgt_genre_name}/Val MAE"]    = tgt_m['mae']
                log_dict[f"{tgt_genre_name}/Val SROCC"]  = tgt_m['srocc']
                log_dict[f"{tgt_genre_name}/Val NDCG@10"] = tgt_m['ndcg@10']
                log_dict[f"{tgt_genre_name}/Val CCC"]    = tgt_m['ccc']
            wandb.log(log_dict, commit=True)

        prev_lr = optimizer_G.param_groups[0]['lr']
        scheduler_G.step(val_ccc)
        scheduler_F.step(val_ccc)
        cur_lr = optimizer_G.param_groups[0]['lr']
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
                print(f"MCD Pretrain: early stopping at epoch {epoch}")
                break

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, mcd_target_genre=None):
    """MCD finetune trainer for PIAA（ICI のみ）.
    ユーザーごとに:
      - ソース: 該当ユーザーの train_piaa_dataset でタスク学習 + I_ij レベルの MCD 適応
      - ターゲット: 同ユーザーの target genre train_piaa_dataset（ラベル不使用）
      - 早期停止: ソース val CCC
      - ターゲット val: 同ユーザーの val_piaa_dataset で観察のみ
    同ユーザーがターゲットに存在しない場合はエラー。
    """
    if args.model_type != 'ICI':
        raise NotImplementedError("MCD finetune は ICI モデルのみサポートしています")

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids    = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"MCD finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)
        user_val_src = copy.copy(datasets_dict[genre]['val'])
        user_val_src.data = datasets_dict[genre]['val'].data[
            datasets_dict[genre]['val'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{mcd_target_genre}' train_piaa_dataset.")
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        tgt_val_mask = tgt_val_piaa_dataset.data['user_id'] == uid
        if tgt_val_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{mcd_target_genre}' val_piaa_dataset.")
        user_val_tgt = copy.copy(tgt_val_piaa_dataset)
        user_val_tgt.data = tgt_val_piaa_dataset.data[tgt_val_mask].reset_index(drop=True)

        total_train_src = len(user_train_src)
        total_train_tgt = len(user_train_tgt)
        total_val_src   = len(user_val_src)
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
            raise FileNotFoundError(f"MCD pretrained model not found: {pretrained_path}")
        try:
            state = torch.load(pretrained_path)
            incompatible = model_user.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                print(f"[load_state_dict] Ignored unexpected keys: {incompatible.unexpected_keys}")
            print(f"Loaded MCD pretrain weights from {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {pretrained_path}: {e}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params     = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        mcd_model   = PIAA_MCD(model_user).to(device)
        optimizer_G = optim.AdamW(mcd_model.g_parameters(), lr=args.lr)
        optimizer_F = optim.AdamW(mcd_model.f_parameters(), lr=args.lr)
        scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

        best_val_ccc = -float('inf')
        patience     = 0
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            L_s, L_adv_B, L_adv_C = _train_one_epoch_piaa(
                mcd_model, src_loader, tgt_loader,
                optimizer_G, optimizer_F, scaler, device, args, genre,
                epoch=epoch, phase='finetune')

            genre_metrics, _ = evaluate_piaa(model_user, val_src_loaders, device, epoch=epoch, phase_name="Val (src)")
            val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

            tgt_genre_metrics, _ = evaluate_piaa(model_user, val_tgt_loaders, device, epoch=epoch, phase_name="Val (tgt)")

            if args.is_log:
                log_dict = {"epoch": epoch}
                lambda_ = getattr(args, 'mcd_lambda', 1.0)
                adv_ratio = lambda_ * L_adv_B / L_s if L_s > 0 else 0.0
                log_dict[f"{genre}/Train Loss user_{uid}"]       = L_s
                log_dict[f"{genre}/Train L_adv_B user_{uid}"]    = L_adv_B
                log_dict[f"{genre}/Train L_adv_C user_{uid}"]    = L_adv_C
                log_dict[f"{genre}/Train adv_ratio user_{uid}"]  = adv_ratio
                if genre in genre_metrics:
                    log_dict[f"{genre}/Val MAE user_{uid}"] = genre_metrics[genre]['mae']
                    log_dict[f"{genre}/Val CCC user_{uid}"] = genre_metrics[genre]['ccc']
                if genre in tgt_genre_metrics:
                    tgt_m = tgt_genre_metrics[genre]
                    log_dict[f"{mcd_target_genre}/Val MAE user_{uid}"] = tgt_m['mae']
                    log_dict[f"{mcd_target_genre}/Val CCC user_{uid}"] = tgt_m['ccc']
                wandb.log(log_dict, commit=True)

            prev_lr = optimizer_G.param_groups[0]['lr']
            scheduler.step(val_ccc)
            cur_lr = optimizer_G.param_groups[0]['lr']
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
