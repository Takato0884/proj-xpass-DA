"""Deep CORAL (Sun & Saenko, 2016) Feature Alignment.

Implementation scope (per documents/design/DeepCORAL_設計書.md §4):
  - GIAA: EMD on source + CORAL on `domain_feat` (256-dim, NIMA mid-feature).
  - PIAA pretrain / finetune: MSE on source + CORAL on `I_ij` (64-dim).
  - PIAA: ICI model only (MIR raises NotImplementedError, matching DAREGRAM/UGAFEAT).
  - λ for CORAL is fixed via --coral_lambda (no schedule).
  - CORAL is computed in fp32 outside autocast for numerical stability.
"""

import os
import copy

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..train_common import earth_mover_distance, build_piaa_model, num_bins
from ..data import collate_fn
from ..evaluate import evaluate, evaluate_piaa


# ── CORAL loss ────────────────────────────────────────────────────────────────

def _coral_loss(Z_s, Z_t):
    """Deep CORAL loss: ||C_s - C_t||_F^2 / (4 d^2).

    Args:
        Z_s, Z_t: (n_s, d) and (n_t, d) feature matrices (same d, possibly different n).
    Returns:
        Scalar tensor (autograd-tracked). Computed in fp32 for stability.
    """
    Zs = Z_s.float()
    Zt = Z_t.float()
    n_s, d = Zs.shape
    n_t = Zt.shape[0]
    assert n_s >= 2 and n_t >= 2, f"CORAL requires n>=2 (got n_s={n_s}, n_t={n_t})"

    mean_s = Zs.mean(dim=0, keepdim=True)
    mean_t = Zt.mean(dim=0, keepdim=True)
    Cs = (Zs - mean_s).t() @ (Zs - mean_s) / (n_s - 1)
    Ct = (Zt - mean_t).t() @ (Zt - mean_t) / (n_t - 1)

    return ((Cs - Ct) ** 2).sum() / (4.0 * d * d)


# ── GIAA ──────────────────────────────────────────────────────────────────────

def setup(model, args, device):
    """No extra components for DEEPCORAL (CORAL is computed in-loop)."""
    return {}


def _train_one_epoch(model, src_loader, tgt_loader, optimizer, scaler, device, args, epoch=None):
    """GIAA 1 epoch: EMD on source + CORAL on `domain_feat`."""
    model.train()
    coral_lambda = float(getattr(args, 'coral_lambda', 1.0))
    running_L_y = running_L_coral = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [DEEPCORAL]" if epoch is not None else "Train DEEPCORAL"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120,
                        colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        images_src = sample_src['image'].to(device)
        hist_src = sample_src['Aesthetic'].to(device)
        images_tgt = sample_tgt['image'].to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            logit_src, domain_feat_src, _ = model(images_src, return_feat=True)
            prob_src = F.softmax(logit_src, dim=1)
            L_y = earth_mover_distance(prob_src, hist_src).mean()

            _, domain_feat_tgt, _ = model(images_tgt, return_feat=True)

        L_coral = _coral_loss(domain_feat_src, domain_feat_tgt)
        loss = L_y + coral_lambda * L_coral

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_L_coral = coral_lambda * L_coral
        running_L_y += L_y.item()
        running_L_coral += weighted_L_coral.item()
        total_batches += 1
        progress_bar.set_postfix({
            'L_y':     f'{L_y.item():.4f}',
            'L_coral': f'{weighted_L_coral.item():.4f}',
        })

    n = max(total_batches, 1)
    return {
        'train_emd':    running_L_y / n,
        'train_coral':  running_L_coral / n,
    }


def trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname, components,
            tgt_val_loader=None, tgt_genre=None):
    """GIAA trainer for DEEPCORAL. Early stopping on source val EMD."""
    src_train_loader, val_loader, _ = src_dataloaders

    if tgt_loader is None:
        raise ValueError("DEEPCORAL GIAA requires a target loader (use --da_method DEEPCORAL-<target>).")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)

    best_val_emd = float('inf')
    patience = 0
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        metrics = _train_one_epoch(
            model, src_train_loader, tgt_loader, optimizer, scaler, device, args, epoch=epoch)

        if args.is_log:
            ratio = (metrics['train_emd'] / (metrics['train_emd'] + metrics['train_coral'])
                     if (metrics['train_emd'] + metrics['train_coral']) > 0 else 0.0)
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA":   metrics['train_emd'],
                f"{args.genre}/Train L_coral":    metrics['train_coral'],
                f"{args.genre}/Train ratio L_y/(L_y+L_coral)": ratio,
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

        if val_emd < best_val_emd:
            best_val_emd = val_emd
            patience = 0
            os.makedirs(os.path.dirname(best_modelname), exist_ok=True)
            torch.save(model.state_dict(), best_modelname)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"DEEPCORAL: early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_modelname))


# ── PIAA ──────────────────────────────────────────────────────────────────────

def _train_one_epoch_piaa(model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                          epoch=None, desc_suffix=""):
    """PIAA 1 epoch: MSE on source + CORAL on `I_ij` (DAREGRAM-style)."""
    model.train()
    coral_lambda = float(getattr(args, 'coral_lambda', 1.0))
    running_L_y = running_L_coral = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    desc = f"Epoch {epoch} [DEEPCORAL{desc_suffix}]" if epoch is not None else f"Train DEEPCORAL{desc_suffix}"
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

        L_coral = _coral_loss(I_ij_src, I_ij_tgt)
        loss = L_y + coral_lambda * L_coral

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_L_coral = coral_lambda * L_coral
        running_L_y += L_y.item()
        running_L_coral += weighted_L_coral.item()
        total_batches += 1
        progress_bar.set_postfix({
            'L_y':     f'{L_y.item():.4f}',
            'L_coral': f'{weighted_L_coral.item():.4f}',
        })

    n = max(total_batches, 1)
    return running_L_y / n, running_L_coral / n


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    """DEEPCORAL pretrain trainer for PIAA (ICI). Early stopping on source val CCC."""
    if getattr(args, 'model_type', 'ICI') != 'ICI':
        raise NotImplementedError("DEEPCORAL pretrain supports the ICI model only")

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
    _dc_run = experiment_name.removeprefix('DEEPCORAL_')
    best_model_path = os.path.join(dirname, f'{genre_str}_DEEPCORAL_{args.model_type}_{_dc_run}_pretrain.pth')
    best_state_dict = None
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        L_y, L_coral = _train_one_epoch_piaa(
            model, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
            epoch=epoch, desc_suffix=" pretrain")

        if args.is_log:
            ratio = L_y / (L_y + L_coral) if (L_y + L_coral) > 0 else 0.0
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss":    L_y,
                f"{genre}/Train L_coral": L_coral,
                f"{genre}/Train ratio L_y/(L_y+L_coral)": ratio,
            }, commit=False)

        genre_metrics, _ = evaluate_piaa(model, val_loaders_dict, device, epoch=epoch, phase_name="Val")
        val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

        tgt_genre_metrics, _ = evaluate_piaa(model, tgt_val_loaders_dict, device, epoch=epoch, phase_name="Val (tgt)")

        if args.is_log:
            log_dict = {"epoch": epoch}
            if genre in genre_metrics:
                log_dict[f"{genre}/Val MAE"]     = genre_metrics[genre]['mae']
                log_dict[f"{genre}/Val SROCC"]   = genre_metrics[genre]['srocc']
                log_dict[f"{genre}/Val NDCG@10"] = genre_metrics[genre]['ndcg@10']
                log_dict[f"{genre}/Val CCC"]     = genre_metrics[genre]['ccc']
            if hasattr(model, '_eval_component_stats') and genre in model._eval_component_stats:
                cs = model._eval_component_stats[genre]
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
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"DEEPCORAL Pretrain: early stopping at epoch {epoch}")
                break

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, deepcoral_target_genre=None):
    """DEEPCORAL finetune trainer for PIAA (ICI). Per-user training."""
    if getattr(args, 'model_type', 'ICI') != 'ICI':
        raise NotImplementedError("DEEPCORAL finetune supports the ICI model only")

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"DEEPCORAL finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[
            datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)
        user_val_src = copy.copy(datasets_dict[genre]['val'])
        user_val_src.data = datasets_dict[genre]['val'].data[
            datasets_dict[genre]['val'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{deepcoral_target_genre}' train_piaa_dataset.")
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        tgt_val_mask = tgt_val_piaa_dataset.data['user_id'] == uid
        if tgt_val_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{deepcoral_target_genre}' val_piaa_dataset.")
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
            raise FileNotFoundError(f"DEEPCORAL pretrained model not found: {pretrained_path}")
        try:
            state = torch.load(pretrained_path)
            incompatible = model_user.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                print(f"[load_state_dict] Ignored unexpected keys: {incompatible.unexpected_keys}")
            print(f"Loaded DEEPCORAL pretrain weights from {pretrained_path}")
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
        best_model_path = os.path.join(
            dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            L_y, L_coral = _train_one_epoch_piaa(
                model_user, src_loader, tgt_loader, optimizer, scaler, device, args, genre,
                epoch=epoch, desc_suffix=" finetune")

            genre_metrics, _ = evaluate_piaa(model_user, val_src_loaders, device, epoch=epoch, phase_name="Val (src)")
            val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

            tgt_genre_metrics, _ = evaluate_piaa(model_user, val_tgt_loaders, device, epoch=epoch, phase_name="Val (tgt)")

            if args.is_log:
                ratio = L_y / (L_y + L_coral) if (L_y + L_coral) > 0 else 0.0
                log_dict = {"epoch": epoch}
                log_dict[f"{genre}/Train Loss user_{uid}"]    = L_y
                log_dict[f"{genre}/Train L_coral user_{uid}"] = L_coral
                log_dict[f"{genre}/Train ratio L_y/(L_y+L_coral) user_{uid}"] = ratio
                if genre in genre_metrics:
                    log_dict[f"{genre}/Val MAE user_{uid}"]   = genre_metrics[genre]['mae']
                    log_dict[f"{genre}/Val SROCC user_{uid}"] = genre_metrics[genre]['srocc']
                    log_dict[f"{genre}/Val CCC user_{uid}"]   = genre_metrics[genre]['ccc']
                if genre in tgt_genre_metrics:
                    tgt_m = tgt_genre_metrics[genre]
                    log_dict[f"{deepcoral_target_genre}/Val MAE user_{uid}"]   = tgt_m['mae']
                    log_dict[f"{deepcoral_target_genre}/Val SROCC user_{uid}"] = tgt_m['srocc']
                    log_dict[f"{deepcoral_target_genre}/Val CCC user_{uid}"]   = tgt_m['ccc']
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
