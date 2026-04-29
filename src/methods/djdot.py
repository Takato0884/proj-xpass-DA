import os

import numpy as np
import ot
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import copy
from torch.utils.data import DataLoader

from ..train_common import earth_mover_distance, build_piaa_model, num_bins
from ..data import collate_fn
from ..evaluate import evaluate, evaluate_piaa


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

    alpha = getattr(args, 'djdot_alpha', 0.001)
    lambda_t = getattr(args, 'djdot_lambda_t', 0.0001)

    best_val_emd = float('inf')
    patience     = 0
    global_step  = 0
    scaler       = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        metrics = _train_one_epoch(
            model, src_train_loader, tgt_loader, optimizer, scaler, device, args,
            epoch=epoch, global_step=global_step)
        global_step = metrics['global_step']

        total_loss = metrics['train_emd'] + alpha * metrics['feat_loss'] + lambda_t * metrics['label_loss']
        L_s_ratio = metrics['train_emd'] / total_loss if total_loss > 0 else 0.0
        align_total = alpha * metrics['feat_loss'] + lambda_t * metrics['label_loss']
        L_feat_ratio = (alpha * metrics['feat_loss']) / align_total if align_total > 0 else 0.0

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA":        metrics['train_emd'],
                f"{args.genre}/Train Feature Loss":    metrics['feat_loss'],
                f"{args.genre}/Train Label Loss":      metrics['label_loss'],
                f"{args.genre}/Train L_s Ratio":       L_s_ratio,
                f"{args.genre}/Train L_feat Ratio":    L_feat_ratio,
            }, commit=False)

        val_emd, val_srocc, _, val_mse, _, _, val_ccc = evaluate(
            model, val_loader, device, epoch=epoch, phase_name="Val")
        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Val EMD GIAA":   val_emd,
                f"{args.genre}/Val SROCC GIAA": val_srocc,
                f"{args.genre}/Val MSE GIAA":   val_mse,
                f"{args.genre}/Val CCC GIAA":   val_ccc,
            }, commit=tgt_val_loader is None)

        if tgt_val_loader is not None:
            tgt_val_emd, tgt_val_srocc, _, tgt_val_mse, _, _, tgt_val_ccc = evaluate(
                model, tgt_val_loader, device, epoch=epoch, phase_name=f"Val [{tgt_genre}]")
            if args.is_log:
                wandb.log({
                    "epoch": epoch,
                    f"{tgt_genre}/Val EMD GIAA":   tgt_val_emd,
                    f"{tgt_genre}/Val SROCC GIAA": tgt_val_srocc,
                    f"{tgt_genre}/Val MSE GIAA":   tgt_val_mse,
                    f"{tgt_genre}/Val CCC GIAA":   tgt_val_ccc,
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


# ─── PIAA ─────────────────────────────────────────────────────────────────────

def _train_one_epoch_piaa(model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                           epoch=None, global_step=0, desc_suffix=""):
    """DeepJDOT 1エポック学習（pretrain / finetune 共通）。
    ラベルコスト: (score_s_i - score_t_j)^2（スカラー回帰）
    特徴量: I_ij（model.forward return_feat=True で取得、ICI/MIR 共通）
    """
    from torch.amp import autocast
    model.train()
    alpha = getattr(args, 'djdot_alpha', 0.001)
    lambda_t = getattr(args, 'djdot_lambda_t', 0.0001)

    running_L_y = running_L_feat = running_L_label = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [DJDOT{desc_suffix}]" if epoch is not None else f"Train DJDOT{desc_suffix}"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        images_src = sample_src['image'].to(device)
        aesthetic_src = sample_src['Aesthetic'].to(device).view(-1, 1)
        pt_src = sample_src['traits'].float().to(device)
        attr_src = sample_src['QIP'].float().to(device)

        images_tgt = sample_tgt['image'].to(device)
        pt_tgt = sample_tgt['traits'].float().to(device)
        attr_tgt = sample_tgt['QIP'].float().to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            score_src, I_ij_src = model(images_src, pt_src, attr_src, genre, return_feat=True)
            L_y = F.mse_loss(score_src, aesthetic_src)

            score_tgt, I_ij_tgt = model(images_tgt, pt_tgt, attr_tgt, genre, return_feat=True)

            with torch.no_grad():
                feat_dist_d = torch.cdist(I_ij_src.float(), I_ij_tgt.float(), p=2).pow(2)
                label_cost_d = (score_src.float() - score_tgt.float().T).pow(2)
                C = (alpha * feat_dist_d + lambda_t * label_cost_d).cpu().numpy().astype(np.float64)
                n_s, n_t = C.shape
                a = np.ones(n_s, dtype=np.float64) / n_s
                b = np.ones(n_t, dtype=np.float64) / n_t
                gamma = torch.from_numpy(
                    ot.emd(a, b, C)
                ).to(dtype=I_ij_src.dtype, device=device)

            feat_dist = torch.cdist(I_ij_src, I_ij_tgt, p=2).pow(2)
            label_cost = (score_src - score_tgt.T).pow(2)

            L_feat = (gamma * feat_dist).sum()
            L_label = (gamma * label_cost).sum()

            loss = L_y + alpha * L_feat + lambda_t * L_label

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_L_y += L_y.item()
        running_L_feat += L_feat.item()
        running_L_label += L_label.item()
        total_batches += 1
        global_step += 1
        progress_bar.set_postfix({
            'L_y': f'{L_y.item():.4f}',
            'L_feat': f'{L_feat.item():.4f}',
            'L_label': f'{L_label.item():.4f}',
        })

    n = max(total_batches, 1)
    return running_L_y / n, running_L_feat / n, running_L_label / n, global_step


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    """DeepJDOT pretrain trainer for PIAA（ICI / MIR 対応）。
    ソース: train_giaa_dataset でタスク学習 + I_ij レベルの OT 整合。
    ターゲット: tgt_train_dataset（ターゲットジャンルの GIAA data、ラベル不使用）。
    早期停止: ソース val CCC。
    Returns:
        best_model_path, best_state_dict
    """
    import wandb

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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

    best_val_ccc = -float('inf')
    patience = 0
    global_step = 0
    _djdot_run = experiment_name.removeprefix('DJDOT_')
    best_model_path = os.path.join(dirname, f'{genre_str}_DJDOT_{args.model_type}_{_djdot_run}_pretrain.pth')
    best_state_dict = None
    scaler = GradScaler('cuda')

    alpha = getattr(args, 'djdot_alpha', 0.001)
    lambda_t = getattr(args, 'djdot_lambda_t', 0.0001)

    for epoch in range(args.num_epochs):
        L_y, L_feat, L_label, global_step = _train_one_epoch_piaa(
            model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
            epoch=epoch, global_step=global_step, desc_suffix=" pretrain")

        total_loss = L_y + alpha * L_feat + lambda_t * L_label
        L_y_ratio = L_y / total_loss if total_loss > 0 else 0.0
        align_total = alpha * L_feat + lambda_t * L_label
        L_feat_ratio = (alpha * L_feat) / align_total if align_total > 0 else 0.0

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss": L_y,
                f"{genre}/Train Feature Loss": L_feat,
                f"{genre}/Train Label Loss": L_label,
                f"{genre}/Train L_y Ratio": L_y_ratio,
                f"{genre}/Train L_feat Ratio": L_feat_ratio,
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
                log_dict[f"{tgt_genre_name}/Val MAE"] = tgt_m['mae']
                log_dict[f"{tgt_genre_name}/Val SROCC"] = tgt_m['srocc']
                log_dict[f"{tgt_genre_name}/Val NDCG@10"] = tgt_m['ndcg@10']
                log_dict[f"{tgt_genre_name}/Val CCC"] = tgt_m['ccc']
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
                torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"DJDOT Pretrain: early stopping at epoch {epoch}")
                break

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, djdot_target_genre=None):
    """DeepJDOT finetune trainer for PIAA（ICI / MIR 対応）。
    ユーザーごとに：
      - ソース: 該当ユーザーの train_piaa_dataset でタスク学習 + I_ij レベルの OT 整合
      - ターゲット: 同ユーザーの target genre train_piaa_dataset（ラベル不使用）
      - 早期停止: ソース val CCC
      - ターゲット val: 同ユーザーの val_piaa_dataset で観察のみ
    同ユーザーがターゲットに存在しない場合はエラー。
    """
    import wandb

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"DJDOT finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)
        user_val_src = copy.copy(datasets_dict[genre]['val'])
        user_val_src.data = datasets_dict[genre]['val'].data[
            datasets_dict[genre]['val'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{djdot_target_genre}' train_piaa_dataset. "
                f"All finetune users must exist in the target genre."
            )
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        tgt_val_mask = tgt_val_piaa_dataset.data['user_id'] == uid
        if tgt_val_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{djdot_target_genre}' val_piaa_dataset. "
                f"All finetune users must exist in the target genre."
            )
        user_val_tgt = copy.copy(tgt_val_piaa_dataset)
        user_val_tgt.data = tgt_val_piaa_dataset.data[tgt_val_mask].reset_index(drop=True)

        total_train_src = len(user_train_src)
        total_train_tgt = len(user_train_tgt)
        total_val_src = len(user_val_src)
        print(f"User {uid}: train src={total_train_src}, train tgt={total_train_tgt}, val src={total_val_src}")
        if total_train_src < batch_size or total_train_tgt < batch_size or total_val_src == 0:
            print(f"Skipping user {uid}: need >={batch_size} samples per domain (val>0)")
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
            raise FileNotFoundError(f"DJDOT pretrained model not found: {pretrained_path}")
        try:
            state = torch.load(pretrained_path)
            incompatible = model_user.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                print(f"[load_state_dict] Ignored unexpected keys: {incompatible.unexpected_keys}")
            print(f"Loaded DJDOT pretrain weights from {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {pretrained_path}: {e}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            print(f"[Freeze] {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

        alpha = getattr(args, 'djdot_alpha', 0.001)
        lambda_t = getattr(args, 'djdot_lambda_t', 0.0001)

        best_val_ccc = -float('inf')
        patience = 0
        global_step = 0
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            L_y, L_feat, L_label, global_step = _train_one_epoch_piaa(
                model_user, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                epoch=epoch, global_step=global_step, desc_suffix=" finetune")

            total_loss = L_y + alpha * L_feat + lambda_t * L_label
            L_y_ratio = L_y / total_loss if total_loss > 0 else 0.0
            align_total = alpha * L_feat + lambda_t * L_label
            L_feat_ratio = (alpha * L_feat) / align_total if align_total > 0 else 0.0

            genre_metrics, _ = evaluate_piaa(model_user, val_src_loaders, device, epoch=epoch, phase_name="Val (src)")
            val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

            tgt_genre_metrics, _ = evaluate_piaa(model_user, val_tgt_loaders, device, epoch=epoch, phase_name="Val (tgt)")

            if args.is_log:
                log_dict = {"epoch": epoch}
                log_dict[f"{genre}/Train Loss user_{uid}"] = L_y
                log_dict[f"{genre}/Train Feature Loss user_{uid}"] = L_feat
                log_dict[f"{genre}/Train Label Loss user_{uid}"] = L_label
                log_dict[f"{genre}/Train L_y Ratio user_{uid}"] = L_y_ratio
                log_dict[f"{genre}/Train L_feat Ratio user_{uid}"] = L_feat_ratio
                if genre in genre_metrics:
                    log_dict[f"{genre}/Val MAE user_{uid}"] = genre_metrics[genre]['mae']
                    log_dict[f"{genre}/Val SROCC user_{uid}"] = genre_metrics[genre]['srocc']
                    log_dict[f"{genre}/Val CCC user_{uid}"] = genre_metrics[genre]['ccc']
                if genre in tgt_genre_metrics:
                    tgt_m = tgt_genre_metrics[genre]
                    log_dict[f"{djdot_target_genre}/Val MAE user_{uid}"] = tgt_m['mae']
                    log_dict[f"{djdot_target_genre}/Val SROCC user_{uid}"] = tgt_m['srocc']
                    log_dict[f"{djdot_target_genre}/Val CCC user_{uid}"] = tgt_m['ccc']
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
