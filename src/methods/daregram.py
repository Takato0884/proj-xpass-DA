import os
import copy

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..train_common import build_piaa_model, num_bins
from ..data import collate_fn
from ..evaluate import evaluate_piaa


# ── DARE-GRAM loss ────────────────────────────────────────────────────────────

def _daregram_losses(Z_s, Z_t, T=0.95):
    """Compute DARE-GRAM angle (L_cos) and scale (L_scale) losses.

    Args:
        Z_s, Z_t: (B, p) feature matrices (same p, possibly different B).
        T: cumulative eigenvalue threshold for truncation.

    Returns:
        L_cos, L_scale (scalar tensors on same device/dtype as inputs).
    """
    # Compute in float32 for SVD numerical stability.
    Zs = Z_s.float()
    Zt = Z_t.float()

    G_s = Zs.t() @ Zs  # (p, p)
    G_t = Zt.t() @ Zt  # (p, p)

    U_s, S_s, Vh_s = torch.linalg.svd(G_s, full_matrices=False)
    U_t, S_t, Vh_t = torch.linalg.svd(G_t, full_matrices=False)

    # Smallest k whose cumulative eigenvalue ratio reaches T (per source and target).
    def _select_k(S):
        total = S.sum().clamp_min(1e-12)
        cum = torch.cumsum(S, dim=0) / total
        k = int((cum < T).sum().item()) + 1
        return max(1, min(k, S.numel()))

    k_s = _select_k(S_s)
    k_t = _select_k(S_t)
    k = max(k_s, k_t)

    # Pseudo-inverse (truncated to top-k singular values).
    eps = 1e-8
    inv_s = torch.zeros_like(S_s)
    inv_s[:k] = 1.0 / (S_s[:k] + eps)
    G_s_pinv = (U_s[:, :k] * inv_s[:k].unsqueeze(0)) @ Vh_s[:k]

    inv_t = torch.zeros_like(S_t)
    inv_t[:k] = 1.0 / (S_t[:k] + eps)
    G_t_pinv = (U_t[:, :k] * inv_t[:k].unsqueeze(0)) @ Vh_t[:k]

    # Column-wise cosine similarity between pseudo-inverses.
    cos_sim = F.cosine_similarity(G_s_pinv, G_t_pinv, dim=0)  # (p,)
    L_cos = (1.0 - cos_sim).abs().sum()

    L_scale = torch.norm(S_s[:k] - S_t[:k], p=2)

    return L_cos, L_scale, k


# ── PIAA DARE-GRAM ────────────────────────────────────────────────────────────

def _train_one_epoch_piaa(model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                           epoch=None, desc_suffix=""):
    """DARE-GRAM 1エポック学習（pretrain / finetune 共通）。

    ソース損失: MSE(interaction_score + direct_score, y_s)
    ドメイン整合損失: L_cos + L_scale, z = I_ij（ICI専用）
    """
    model.train()
    alpha_cos = getattr(args, 'daregram_alpha_cos', 0.1)
    gamma_scale = getattr(args, 'daregram_gamma_scale', 0.1)
    T = getattr(args, 'daregram_T', 0.95)

    running_L_y = running_L_cos = running_L_scale = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [DAREGRAM{desc_suffix}]" if epoch is not None else f"Train DAREGRAM{desc_suffix}"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120,
                        colour="#00ff00", ascii="-=")

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
            _, I_ij_tgt = model(images_tgt, pt_tgt, attr_tgt, genre, return_feat=True)

            L_y = F.mse_loss(score_src, aesthetic_src)

        # SVD is done in float32 outside autocast for numerical stability.
        L_cos, L_scale, _ = _daregram_losses(I_ij_src, I_ij_tgt, T=T)

        loss = L_y + alpha_cos * L_cos + gamma_scale * L_scale

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_L_cos = alpha_cos * L_cos
        weighted_L_scale = gamma_scale * L_scale

        running_L_y += L_y.item()
        running_L_cos += weighted_L_cos.item()
        running_L_scale += weighted_L_scale.item()
        total_batches += 1

        progress_bar.set_postfix({
            'L_y':     f'{L_y.item():.4f}',
            'L_cos':   f'{weighted_L_cos.item():.4f}',
            'L_scale': f'{weighted_L_scale.item():.4f}',
        })

    n = max(total_batches, 1)
    return running_L_y / n, running_L_cos / n, running_L_scale / n


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    """DARE-GRAM pretrain trainer for PIAA（ICI のみ）。
    ソース: train_giaa_dataset でタスク学習 + I_ij レベルの DARE-GRAM 整合。
    ターゲット: tgt_train_dataset（ターゲットジャンルの GIAA data、ラベル不使用）。
    早期停止: ソース val CCC。
    Returns:
        best_model_path, best_state_dict
    """
    if args.model_type != 'ICI':
        raise NotImplementedError("DAREGRAM pretrain は ICI モデルのみサポートしています")

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
    _dg_run = experiment_name.removeprefix('DAREGRAM_')
    best_model_path = os.path.join(dirname, f'{genre_str}_DAREGRAM_{args.model_type}_{_dg_run}_pretrain.pth')
    best_state_dict = None
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        L_y, L_cos, L_scale = _train_one_epoch_piaa(
            model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
            epoch=epoch, desc_suffix=" pretrain")

        if args.is_log:
            ratio_y_da = L_y / (L_y + L_scale + L_cos) if (L_y + L_scale + L_cos) > 0 else 0.0
            ratio_cos_scale = L_cos / (L_scale + L_cos) if (L_scale + L_cos) > 0 else 0.0
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss":    L_y,
                f"{genre}/Train L_cos":   L_cos,
                f"{genre}/Train L_scale": L_scale,
                f"{genre}/Train ratio L_y/(L_y+L_scale+L_cos)": ratio_y_da,
                f"{genre}/Train ratio L_cos/(L_scale+L_cos)":   ratio_cos_scale,
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
                print(f"DAREGRAM Pretrain: early stopping at epoch {epoch}")
                break

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, daregram_target_genre=None):
    """DARE-GRAM finetune trainer for PIAA（ICI のみ）。
    ユーザーごとに：
      - ソース: 該当ユーザーの train_piaa_dataset でタスク学習 + I_ij レベルの DARE-GRAM 整合
      - ターゲット: 同ユーザーの target genre train_piaa_dataset（ラベル不使用）
      - 早期停止: ソース val CCC
      - ターゲット val: 同ユーザーの val_piaa_dataset で観察のみ
    同ユーザーがターゲットに存在しない場合はエラー。
    """
    if args.model_type != 'ICI':
        raise NotImplementedError("DAREGRAM finetune は ICI モデルのみサポートしています")

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"DAREGRAM finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)
        user_val_src = copy.copy(datasets_dict[genre]['val'])
        user_val_src.data = datasets_dict[genre]['val'].data[
            datasets_dict[genre]['val'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{daregram_target_genre}' train_piaa_dataset.")
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        tgt_val_mask = tgt_val_piaa_dataset.data['user_id'] == uid
        if tgt_val_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{daregram_target_genre}' val_piaa_dataset.")
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
            raise FileNotFoundError(f"DAREGRAM pretrained model not found: {pretrained_path}")
        try:
            state = torch.load(pretrained_path)
            incompatible = model_user.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                print(f"[load_state_dict] Ignored unexpected keys: {incompatible.unexpected_keys}")
            print(f"Loaded DAREGRAM pretrain weights from {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {pretrained_path}: {e}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            print(f"[Freeze] {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

        best_val_ccc = -float('inf')
        patience = 0
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            L_y, L_cos, L_scale = _train_one_epoch_piaa(
                model_user, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                epoch=epoch, desc_suffix=" finetune")

            genre_metrics, _ = evaluate_piaa(model_user, val_src_loaders, device, epoch=epoch, phase_name="Val (src)")
            val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

            tgt_genre_metrics, _ = evaluate_piaa(model_user, val_tgt_loaders, device, epoch=epoch, phase_name="Val (tgt)")

            if args.is_log:
                ratio_y_da = L_y / (L_y + L_scale + L_cos) if (L_y + L_scale + L_cos) > 0 else 0.0
                ratio_cos_scale = L_cos / (L_scale + L_cos) if (L_scale + L_cos) > 0 else 0.0
                log_dict = {"epoch": epoch}
                log_dict[f"{genre}/Train Loss user_{uid}"]    = L_y
                log_dict[f"{genre}/Train L_cos user_{uid}"]   = L_cos
                log_dict[f"{genre}/Train L_scale user_{uid}"] = L_scale
                log_dict[f"{genre}/Train ratio L_y/(L_y+L_scale+L_cos) user_{uid}"] = ratio_y_da
                log_dict[f"{genre}/Train ratio L_cos/(L_scale+L_cos) user_{uid}"]   = ratio_cos_scale
                if genre in genre_metrics:
                    log_dict[f"{genre}/Val MAE user_{uid}"] = genre_metrics[genre]['mae']
                    log_dict[f"{genre}/Val SROCC user_{uid}"] = genre_metrics[genre]['srocc']
                    log_dict[f"{genre}/Val CCC user_{uid}"] = genre_metrics[genre]['ccc']
                if genre in tgt_genre_metrics:
                    tgt_m = tgt_genre_metrics[genre]
                    log_dict[f"{daregram_target_genre}/Val MAE user_{uid}"] = tgt_m['mae']
                    log_dict[f"{daregram_target_genre}/Val SROCC user_{uid}"] = tgt_m['srocc']
                    log_dict[f"{daregram_target_genre}/Val CCC user_{uid}"] = tgt_m['ccc']
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
