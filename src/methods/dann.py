import os

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import copy
from torch.utils.data import DataLoader

from ..train_common import (
    earth_mover_distance, GradientReversalLayer, DomainDiscriminator, get_da_lambda,
    build_piaa_model, num_bins, parse_da_method)
from ..data import collate_fn
from ..evaluate import evaluate, evaluate_piaa


def setup(model, args, device):
    """Create DANN-specific components: discriminator, GRL, and discriminator optimizer."""
    discriminator = DomainDiscriminator(model.feat_dim).to(device)
    grl = GradientReversalLayer()
    optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)
    return {'discriminator': discriminator, 'grl': grl, 'optimizer_disc': optimizer_disc}


def _train_one_epoch(model, src_loader, tgt_loader, optimizer, scaler, device, args,
                     discriminator, grl, optimizer_disc,
                     epoch=None, global_step=0, dann_total_steps=50):
    model.train()
    discriminator.train()
    running_L_y = running_L_d = running_L_d_tgt = running_disc_acc_tgt = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))
    desc = f"Epoch {epoch} [DANN λ={lambda_:.3f}]" if epoch is not None else "Train DANN"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))

        images_src = sample_src['image'].to(device)
        hist_src   = sample_src['Aesthetic'].to(device)
        images_tgt = sample_tgt['image'].to(device)

        optimizer.zero_grad()
        optimizer_disc.zero_grad()
        with autocast('cuda'):
            logit_src, domain_feat_src, _ = model(images_src, return_feat=True)
            prob_src = F.softmax(logit_src, dim=1)
            L_y = earth_mover_distance(prob_src, hist_src).mean()

            _, domain_feat_tgt, _ = model(images_tgt, return_feat=True)
            feat_all = torch.cat([domain_feat_src, domain_feat_tgt], dim=0)
            domain_labels = torch.cat([
                torch.zeros(domain_feat_src.size(0), 1),
                torch.ones(domain_feat_tgt.size(0), 1),
            ], dim=0).to(device)
            domain_logit = discriminator(grl(feat_all, lambda_))
            L_d = F.binary_cross_entropy_with_logits(domain_logit, domain_labels)
            loss = L_y + L_d

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(optimizer_disc)
        scaler.update()

        n_src = domain_feat_src.size(0)
        logit_tgt = domain_logit[n_src:]
        label_tgt = domain_labels[n_src:]
        with torch.no_grad():
            L_d_tgt = F.binary_cross_entropy_with_logits(logit_tgt, label_tgt).item()
            disc_acc_tgt = ((torch.sigmoid(logit_tgt) > 0.5).float() == label_tgt).float().mean().item()

        running_L_y += L_y.item()
        running_L_d += L_d.item()
        running_L_d_tgt += L_d_tgt
        running_disc_acc_tgt += disc_acc_tgt
        total_batches += 1
        global_step += 1
        progress_bar.set_postfix({
            'L_y': f'{L_y.item():.4f}', 'L_d': f'{L_d.item():.4f}',
            'L_d_tgt': f'{L_d_tgt:.4f}', 'acc_tgt': f'{disc_acc_tgt:.3f}',
            'λ': f'{lambda_:.3f}',
        })

    n = max(total_batches, 1)
    return {
        'train_emd': running_L_y / n,
        'domain_loss': running_L_d / n,
        'domain_loss_tgt': running_L_d_tgt / n,
        'disc_acc_tgt': running_disc_acc_tgt / n,
        'global_step': global_step,
    }


def trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname, components,
            tgt_val_loader=None, tgt_genre=None):
    src_train_loader, val_loader, _ = src_dataloaders
    discriminator = components['discriminator']
    grl = components['grl']
    optimizer_disc = components['optimizer_disc']

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)

    steps_per_epoch = len(src_train_loader)
    dann_total_steps = getattr(args, 'dann_epochs', 50) * steps_per_epoch

    best_val_emd = float('inf')
    patience = 0
    global_step = 0
    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        metrics = _train_one_epoch(
            model, src_train_loader, tgt_loader, optimizer, scaler, device, args,
            discriminator=discriminator, grl=grl, optimizer_disc=optimizer_disc,
            epoch=epoch, global_step=global_step, dann_total_steps=dann_total_steps)
        global_step = metrics['global_step']
        lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA": metrics['train_emd'],
                f"{args.genre}/Train Domain Loss": metrics['domain_loss'],
                f"{args.genre}/Train Domain Loss (tgt)": metrics['domain_loss_tgt'],
                f"{args.genre}/Train Disc Acc (tgt)": metrics['disc_acc_tgt'],
                f"{args.genre}/DANN lambda": lambda_,
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
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_modelname))


# ─── PIAA ─────────────────────────────────────────────────────────────────────

def _train_one_epoch_pretrain_piaa(model, src_loader, tgt_loader, discriminator, grl,
                                    optimizer, optimizer_disc, scaler, device, args, genre,
                                    epoch=None, global_step=0, dann_total_steps=50):
    """
    DANN の 1 エポック学習（PIAA pretrain レベル）。
    ソース: タスク損失（MSE）＋ドメイン識別損失（I_ij に GRL）
    ターゲット: ドメイン識別損失のみ（ラベル不使用）
    Returns:
        (avg_task_loss, avg_domain_loss, avg_domain_loss_tgt, avg_disc_acc_tgt, updated_global_step)
    """
    from torch.amp import autocast
    model.train()
    discriminator.train()
    running_L_y = running_L_d = running_L_d_tgt = running_disc_acc_tgt = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))
    desc = f"Epoch {epoch} [DANN λ={lambda_:.3f}]" if epoch is not None else "Train DANN"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))

        images_src = sample_src['image'].to(device)
        aesthetic_src = sample_src['Aesthetic'].to(device).view(-1, 1)
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

            _, I_ij_tgt = model(images_tgt, pt_tgt, attr_tgt, genre, return_feat=True)
            feat_all = torch.cat([I_ij_src, I_ij_tgt], dim=0)
            domain_labels = torch.cat([
                torch.zeros(I_ij_src.size(0), 1),
                torch.ones(I_ij_tgt.size(0), 1),
            ], dim=0).to(device)
            domain_logit = discriminator(grl(feat_all, lambda_))
            L_d = F.binary_cross_entropy_with_logits(domain_logit, domain_labels)
            loss = L_y + 0.1 * L_d

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(optimizer_disc)
        scaler.update()

        n_src = I_ij_src.size(0)
        logit_tgt = domain_logit[n_src:]
        label_tgt = domain_labels[n_src:]
        with torch.no_grad():
            L_d_tgt = F.binary_cross_entropy_with_logits(logit_tgt, label_tgt).item()
            disc_acc_tgt = ((torch.sigmoid(logit_tgt) > 0.5).float() == label_tgt).float().mean().item()

        running_L_y += L_y.item()
        running_L_d += L_d.item()
        running_L_d_tgt += L_d_tgt
        running_disc_acc_tgt += disc_acc_tgt
        total_batches += 1
        global_step += 1
        progress_bar.set_postfix({
            'L_y': f'{L_y.item():.4f}', 'L_d': f'{L_d.item():.4f}',
            'L_d_tgt': f'{L_d_tgt:.4f}', 'acc_tgt': f'{disc_acc_tgt:.3f}',
            'λ': f'{lambda_:.3f}',
        })

    n = max(total_batches, 1)
    return running_L_y / n, running_L_d / n, running_L_d_tgt / n, running_disc_acc_tgt / n, global_step


def _train_one_epoch_finetune_piaa(model, src_loader, tgt_loader, discriminator, grl,
                                    optimizer, optimizer_disc, scaler, device, args, genre,
                                    epoch=None, global_step=0, dann_total_steps=50):
    """
    DANN の 1 エポック学習（PIAA finetune レベル）。
    ソース: タスク損失（MSE）＋ドメイン識別損失（I_ij に GRL）
    ターゲット: ドメイン識別損失のみ（ラベル不使用）
    Returns:
        (avg_task_loss, avg_domain_loss, avg_domain_loss_tgt, avg_disc_acc_tgt, updated_global_step)
    """
    from torch.amp import autocast
    model.train()
    discriminator.train()
    running_L_y = running_L_d = running_L_d_tgt = running_disc_acc_tgt = 0.0
    total_batches = 0
    tgt_iter = iter(tgt_loader)

    lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))
    desc = f"Epoch {epoch} [DANN finetune λ={lambda_:.3f}]" if epoch is not None else "Train DANN finetune"
    progress_bar = tqdm(src_loader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")

    for sample_src in progress_bar:
        try:
            sample_tgt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            sample_tgt = next(tgt_iter)

        lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))

        images_src = sample_src['image'].to(device)
        aesthetic_src = sample_src['Aesthetic'].to(device).view(-1, 1)
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

            _, I_ij_tgt = model(images_tgt, pt_tgt, attr_tgt, genre, return_feat=True)
            feat_all = torch.cat([I_ij_src, I_ij_tgt], dim=0)
            domain_labels = torch.cat([
                torch.zeros(I_ij_src.size(0), 1),
                torch.ones(I_ij_tgt.size(0), 1),
            ], dim=0).to(device)
            domain_logit = discriminator(grl(feat_all, lambda_))
            L_d = F.binary_cross_entropy_with_logits(domain_logit, domain_labels)
            loss = L_y + 0.1 * L_d

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(optimizer_disc)
        scaler.update()

        n_src = I_ij_src.size(0)
        logit_tgt = domain_logit[n_src:]
        label_tgt = domain_labels[n_src:]
        with torch.no_grad():
            L_d_tgt = F.binary_cross_entropy_with_logits(logit_tgt, label_tgt).item()
            disc_acc_tgt = ((torch.sigmoid(logit_tgt) > 0.5).float() == label_tgt).float().mean().item()

        running_L_y += L_y.item()
        running_L_d += L_d.item()
        running_L_d_tgt += L_d_tgt
        running_disc_acc_tgt += disc_acc_tgt
        total_batches += 1
        global_step += 1
        progress_bar.set_postfix({
            'L_y': f'{L_y.item():.4f}', 'L_d': f'{L_d.item():.4f}',
            'L_d_tgt': f'{L_d_tgt:.4f}', 'acc_tgt': f'{disc_acc_tgt:.3f}',
            'λ': f'{lambda_:.3f}',
        })

    n = max(total_batches, 1)
    return running_L_y / n, running_L_d / n, running_L_d_tgt / n, running_disc_acc_tgt / n, global_step


def trainer_pretrain(datasets_dict, tgt_train_dataset, tgt_val_dataset, args, device, dirname,
                     experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                     domain_tag=None):
    """
    DANN pretrain trainer for PIAA（ICI）。
    ソースの train_giaa_dataset でタスク学習 + I_ij レベルのドメイン適応。
    Returns:
        best_model_path, best_state_dict
    """
    import wandb
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = domain_tag if domain_tag else genre

    src_loader = DataLoader(datasets_dict[genre]['train'], batch_size=batch_size, shuffle=True,
                            num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    tgt_loader = DataLoader(tgt_train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=args.num_workers, timeout=300, collate_fn=collate_fn, drop_last=True)
    val_loaders_dict = {genre: DataLoader(datasets_dict[genre]['val'], batch_size=batch_size, shuffle=False,
                                          num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}
    dann_target_genre = parse_da_method(getattr(args, 'da_method', None))[1]
    tgt_val_loader = DataLoader(tgt_val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    tgt_val_loaders_dict = {genre: tgt_val_loader}

    model = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)

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
    print(f"[Freeze] Backbone frozen: {total_params - trainable_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    discriminator = DomainDiscriminator(model.input_dim).to(device)
    grl = GradientReversalLayer()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

    steps_per_epoch = len(src_loader)
    dann_total_steps = getattr(args, 'dann_epochs', 50) * steps_per_epoch

    best_val_ccc = -float('inf')
    patience = 0
    global_step = 0
    _dann_run = experiment_name.removeprefix('DANN_')
    best_model_path = os.path.join(dirname, f'{genre_str}_DANN_{args.model_type}_{_dann_run}_pretrain.pth')
    best_state_dict = None

    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        L_y, L_d, L_d_tgt, disc_acc_tgt, global_step = _train_one_epoch_pretrain_piaa(
            model, src_loader, tgt_loader, discriminator, grl, optimizer, optimizer_disc, scaler,
            device, args, genre, epoch=epoch, global_step=global_step, dann_total_steps=dann_total_steps)
        lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{genre}/Train Loss": L_y,
                f"{genre}/Train Domain Loss": L_d,
                f"{genre}/Train Domain Loss (tgt)": L_d_tgt,
                f"{genre}/Train Disc Acc (tgt)": disc_acc_tgt,
                f"{genre}/DANN lambda": lambda_,
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
                log_dict[f"{dann_target_genre}/Val MAE"] = tgt_m['mae']
                log_dict[f"{dann_target_genre}/Val SROCC"] = tgt_m['srocc']
                log_dict[f"{dann_target_genre}/Val NDCG@10"] = tgt_m['ndcg@10']
                log_dict[f"{dann_target_genre}/Val CCC"] = tgt_m['ccc']
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
                print(f"DANN Pretrain: early stopping at epoch {epoch}")
                break

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                     args, device, dirname, experiment_name, backbone_dict,
                     pretrained_model_dict, num_attr, num_pt, dann_target_genre=None):
    """
    DANN finetune trainer for PIAA（ICI）。
    ユーザーごとに：
      - ソース: 該当ユーザーの source genre train_piaa_dataset でタスク学習 + I_ij レベルのドメイン適応
      - ターゲット: 同ユーザーの target genre train_piaa_dataset（ラベル不使用）
      - early stopping: ソース val CCC のみ
      - ターゲット val: 同ユーザーの val_piaa_dataset で観察のみ
    """
    import wandb
    if args.model_type != 'ICI':
        raise NotImplementedError("DANN finetune は ICI モデルのみサポートしています")

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    for uid in unique_user_ids:
        print(f"DANN finetune for user {uid}...")

        user_train_src = copy.copy(datasets_dict[genre]['train'])
        user_train_src.data = datasets_dict[genre]['train'].data[datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)
        user_val_src = copy.copy(datasets_dict[genre]['val'])
        user_val_src.data = datasets_dict[genre]['val'].data[datasets_dict[genre]['val'].data['user_id'] == uid].reset_index(drop=True)

        tgt_train_mask = tgt_train_piaa_dataset.data['user_id'] == uid
        if tgt_train_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{dann_target_genre}' train_piaa_dataset. "
                f"All finetune users must exist in the target genre."
            )
        user_train_tgt = copy.copy(tgt_train_piaa_dataset)
        user_train_tgt.data = tgt_train_piaa_dataset.data[tgt_train_mask].reset_index(drop=True)

        tgt_val_mask = tgt_val_piaa_dataset.data['user_id'] == uid
        if tgt_val_mask.sum() == 0:
            raise ValueError(
                f"User {uid} not found in target genre '{dann_target_genre}' val_piaa_dataset. "
                f"All finetune users must exist in the target genre."
            )
        user_val_tgt = copy.copy(tgt_val_piaa_dataset)
        user_val_tgt.data = tgt_val_piaa_dataset.data[tgt_val_mask].reset_index(drop=True)

        total_train_src = len(user_train_src)
        total_train_tgt = len(user_train_tgt)
        total_val_src = len(user_val_src)
        print(f"User {uid}: train src={total_train_src}, train tgt={total_train_tgt}, val src={total_val_src}")
        if total_train_src < batch_size or total_train_tgt < batch_size or total_val_src == 0:
            print(f"Skipping user {uid}: src train={total_train_src}, tgt train={total_train_tgt} (need >={batch_size}), val src={total_val_src}")
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
            raise FileNotFoundError(f"DANN pretrained model not found: {pretrained_path}")
        try:
            state = torch.load(pretrained_path)
            incompatible = model_user.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                print(f"[load_state_dict] Ignored unexpected keys: {incompatible.unexpected_keys}")
            print(f"Loaded DANN pretrain weights from {pretrained_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {pretrained_path}: {e}")

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            print(f"[Freeze] Backbone frozen: {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        discriminator = DomainDiscriminator(model_user.input_dim).to(device)
        grl = GradientReversalLayer()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)
        optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

        steps_per_epoch = len(src_loader)
        dann_total_steps = getattr(args, 'dann_epochs', 50) * steps_per_epoch

        best_val_ccc = -float('inf')
        patience = 0
        global_step = 0
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')
        scaler = GradScaler('cuda')

        for epoch in range(args.num_epochs):
            L_y, L_d, L_d_tgt, disc_acc_tgt, global_step = _train_one_epoch_finetune_piaa(
                model_user, src_loader, tgt_loader, discriminator, grl,
                optimizer, optimizer_disc, scaler, device, args, genre,
                epoch=epoch, global_step=global_step, dann_total_steps=dann_total_steps)
            lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))

            genre_metrics, _ = evaluate_piaa(model_user, val_src_loaders, device, epoch=epoch, phase_name="Val (src)")
            val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

            tgt_genre_metrics, _ = evaluate_piaa(model_user, val_tgt_loaders, device, epoch=epoch, phase_name="Val (tgt)")

            if args.is_log:
                log_dict = {"epoch": epoch}
                log_dict[f"{genre}/Train Loss user_{uid}"] = L_y
                log_dict[f"{genre}/Train Domain Loss user_{uid}"] = L_d
                log_dict[f"{genre}/Train Domain Loss (tgt) user_{uid}"] = L_d_tgt
                log_dict[f"{genre}/Train Disc Acc (tgt) user_{uid}"] = disc_acc_tgt
                log_dict[f"{genre}/DANN lambda user_{uid}"] = lambda_
                if genre in genre_metrics:
                    log_dict[f"{genre}/Val MAE user_{uid}"] = genre_metrics[genre]['mae']
                    log_dict[f"{genre}/Val SROCC user_{uid}"] = genre_metrics[genre]['srocc']
                    log_dict[f"{genre}/Val CCC user_{uid}"] = genre_metrics[genre]['ccc']
                if genre in tgt_genre_metrics:
                    tgt_m = tgt_genre_metrics[genre]
                    log_dict[f"{dann_target_genre}/Val MAE user_{uid}"] = tgt_m['mae']
                    log_dict[f"{dann_target_genre}/Val SROCC user_{uid}"] = tgt_m['srocc']
                    log_dict[f"{dann_target_genre}/Val CCC user_{uid}"] = tgt_m['ccc']
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
