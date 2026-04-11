# import warnings
# from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
# warnings.simplefilter("error", UnsupportedFieldAttributeWarning)

import os
import wandb
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.amp import autocast, GradScaler
import copy

from .argflags import parse_arguments, model_dir, wandb_tags
from .data import load_data, load_data_giaa_only, collate_fn
from .train_common import (NIMA, EarthMoverDistance, earth_mover_distance, discover_folds, num_bins,
                            GradientReversalLayer, DomainDiscriminator, parse_dann_target, get_da_lambda)
from .inference import inference


def train(model, dataloader, optimizer, scaler, device, args, epoch: int = None):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    # Persist progress bar in terminal per epoch
    desc = f"Epoch {epoch} [Train]" if epoch is not None else "Train"
    progress_bar = tqdm(dataloader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")
    for sample in progress_bar:
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['Aesthetic'].to(device)
        
        optimizer.zero_grad()
        with autocast('cuda'):
            aesthetic_logits = model(images)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            loss_aesthetic = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
            total_loss = loss_aesthetic

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_total_emd_loss += total_loss.item()
        running_aesthetic_emd_loss += loss_aesthetic.item()

        progress_bar.set_postfix({
            'Train EMD': loss_aesthetic.item(),
        })
    
    epoch_emd_loss = running_aesthetic_emd_loss / len(dataloader)
    return epoch_emd_loss

def evaluate(model, dataloader, device, PIAA=False, epoch: int = None, phase_name: str = "Val"):
    model.eval()
    running_emd_loss = 0.0
    running_mse_loss = 0.0
    running_mae_loss = 0.0
    scale = torch.arange(0, num_bins).to(device)
    # Persist validation/test progress bar per epoch
    desc = f"Epoch {epoch} [{phase_name}]" if epoch is not None else phase_name
    progress_bar = tqdm(dataloader, leave=True, desc=desc, position=1, ncols=120, colour="#fffb00", ascii="-=")
    mean_pred = []
    mean_target = []
    user_id_list = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['Aesthetic'].to(device)
            if PIAA:
                user_id_list.extend(sample['user_id'])
            with autocast('cuda'):
                aesthetic_logits = model(images)
                prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
                loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()

            outputs_mean = torch.sum(prob_aesthetic * scale, dim=1, keepdim=True)
            if aesthetic_score_histogram.shape[-1] == 1:
                # PIAA: scalar score in [0, 1], scale to [0, num_bins-1]
                target_mean = aesthetic_score_histogram * (num_bins - 1)
            else:
                # GIAA: histogram, compute expected value
                target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
            mean_pred.append(outputs_mean.view(-1).cpu().numpy())
            mean_target.append(target_mean.view(-1).cpu().numpy())
            # Normalize to [0, 1] for scale-consistent comparison with PIAA
            outputs_mean_norm = outputs_mean / (num_bins - 1)
            target_mean_norm = target_mean / (num_bins - 1)
            # MSE
            mse = criterion_mse(outputs_mean_norm, target_mean_norm)
            running_mse_loss += mse.item()
            # MAE
            mae = F.l1_loss(outputs_mean_norm, target_mean_norm)
            running_mae_loss += mae.item()

            running_emd_loss += loss.item()
            progress_bar.set_postfix({'EMD': loss.item()})

    # Calculate SROCC
    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc_GIAA, _ = spearmanr(predicted_scores, true_scores)

    # Calculate CCC (overall)
    mu_p, mu_t = predicted_scores.mean(), true_scores.mean()
    var_p, var_t = predicted_scores.var(), true_scores.var()
    cov = ((predicted_scores - mu_p) * (true_scores - mu_t)).mean()
    ccc = float(2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8))

    srocc_PIAA = 0
    ndcg_PIAA = 0
    if PIAA:
        unique_user_ids = np.unique(user_id_list)
        sroccs = []
        ndcgs = []
        for uid in unique_user_ids:
            uid_mask = (user_id_list == uid)
            if np.sum(uid_mask) > 1:
                uid_srocc, _ = spearmanr(predicted_scores[uid_mask], true_scores[uid_mask])
                sroccs.append(uid_srocc)
                uid_ndcg = ndcg_score([true_scores[uid_mask]], [predicted_scores[uid_mask]], k=10)
                ndcgs.append(uid_ndcg)
        srocc_PIAA = np.mean(sroccs)
        ndcg_PIAA = np.mean(ndcgs) if len(ndcgs) > 0 else 0

    emd_loss = running_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    mae_loss = running_mae_loss / len(dataloader)
    return emd_loss, srocc_GIAA, srocc_PIAA, mse_loss, ndcg_PIAA, mae_loss, ccc

def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname,
            tgt_val_loader=None, tgt_genre=None):

    train_dataloader, val_giaa_dataloader, test_dataloader = dataloaders

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)

    scaler = GradScaler('cuda')

    # Training loop
    best_test_emd = 1000
    num_patience_epochs = 0
    for epoch in range(args.num_epochs):
        # Training
        train_emd_loss = train_fn(model, train_dataloader, optimizer, scaler, device, args, epoch=epoch)
        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA": train_emd_loss,
                       }, commit=False)

        # validation GIAA
        val_emd_loss, val_giaa_srocc, _, val_giaa_mse, _, _, _ = evaluate_fn(model, val_giaa_dataloader, device, epoch=epoch, phase_name="Val")
        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Val EMD GIAA": val_emd_loss,
                f"{args.genre}/Val SROCC GIAA": val_giaa_srocc,
                f"{args.genre}/Val MSE GIAA": val_giaa_mse,
            }, commit=tgt_val_loader is None)

        if tgt_val_loader is not None:
            tgt_val_emd, tgt_val_srocc, _, tgt_val_mse, _, _, _ = evaluate_fn(
                model, tgt_val_loader, device, epoch=epoch, phase_name=f"Val [{tgt_genre}]")
            if args.is_log:
                wandb.log({
                    "epoch": epoch,
                    f"{tgt_genre}/Val EMD GIAA": tgt_val_emd,
                    f"{tgt_genre}/Val SROCC GIAA": tgt_val_srocc,
                    f"{tgt_genre}/Val MSE GIAA": tgt_val_mse,
                }, commit=True)

        eval_emd = val_emd_loss
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(eval_emd)
        cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr < prev_lr:
            tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (epoch {epoch}) <<<")

        # Early stopping check
        if eval_emd < best_test_emd:
            best_test_emd = eval_emd
            num_patience_epochs = 0
            # Ensure directory exists
            os.makedirs(os.path.dirname(best_modelname), exist_ok=True)
            torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= args.max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(args.max_patience_epochs))
                break

    model.load_state_dict(torch.load(best_modelname))

criterion_mse = nn.MSELoss()


# ─── DANN for GIAA ────────────────────────────────────────────────────────────

def train_dann_giaa(model, src_loader, tgt_loader, discriminator, grl, optimizer, optimizer_disc, scaler, device, args,
                    epoch=None, global_step=0, dann_total_steps=50):
    """
    DANN の 1 エポック学習（GIAA レベル）。
    ソース: タスク損失（EMD）＋ドメイン識別損失
    ターゲット: ドメイン識別損失のみ（ラベル不使用）
    Returns:
        (avg_task_loss, avg_domain_loss, updated_global_step)
    """
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
            # タスク損失（ソースのみ）
            logit_src, domain_feat_src, _ = model(images_src, return_feat=True)
            prob_src = F.softmax(logit_src, dim=1)
            L_y = earth_mover_distance(prob_src, hist_src).mean()

            # ドメイン識別損失（ソース＋ターゲット）
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
        progress_bar.set_postfix({'L_y': f'{L_y.item():.4f}', 'L_d': f'{L_d.item():.4f}',
                                   'L_d_tgt': f'{L_d_tgt:.4f}', 'acc_tgt': f'{disc_acc_tgt:.3f}',
                                   'λ': f'{lambda_:.3f}'})

    n = max(total_batches, 1)
    return running_L_y / n, running_L_d / n, running_L_d_tgt / n, running_disc_acc_tgt / n, global_step


def trainer_dann_giaa(src_dataloaders, tgt_loader, model, discriminator, grl, optimizer, optimizer_disc, args, device, best_modelname,
                      tgt_val_loader=None):
    src_train_loader, val_loader, _ = src_dataloaders
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)

    steps_per_epoch = len(src_train_loader)
    dann_total_steps = getattr(args, 'dann_epochs', 50) * steps_per_epoch

    best_val_emd = float('inf')
    patience = 0
    global_step = 0

    target_genre = parse_dann_target(args.dann_target)

    scaler = GradScaler('cuda')

    for epoch in range(args.num_epochs):
        L_y, L_d, L_d_tgt, disc_acc_tgt, global_step = train_dann_giaa(
            model, src_train_loader, tgt_loader, discriminator, grl, optimizer, optimizer_disc, scaler, device, args,
            epoch=epoch, global_step=global_step, dann_total_steps=dann_total_steps)
        lambda_ = get_da_lambda(global_step, dann_total_steps, getattr(args, 'dann_gamma', 10.0))

        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Train EMD GIAA": L_y,
                f"{args.genre}/Train Domain Loss": L_d,
                f"{args.genre}/Train Domain Loss (tgt)": L_d_tgt,
                f"{args.genre}/Train Disc Acc (tgt)": disc_acc_tgt,
                f"{args.genre}/DANN lambda": lambda_,
            }, commit=False)

        val_emd, val_srocc, _, val_mse, _, _, _ = evaluate(model, val_loader, device, epoch=epoch, phase_name="Val")
        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Val EMD GIAA": val_emd,
                f"{args.genre}/Val SROCC GIAA": val_srocc,
                f"{args.genre}/Val MSE GIAA": val_mse,
            }, commit=tgt_val_loader is None)

        if tgt_val_loader is not None:
            tgt_val_emd, tgt_val_srocc, _, tgt_val_mse, _, _, _ = evaluate(
                model, tgt_val_loader, device, epoch=epoch, phase_name=f"Val [{target_genre}]")
            if args.is_log:
                wandb.log({
                    "epoch": epoch,
                    f"{target_genre}/Val EMD GIAA": tgt_val_emd,
                    f"{target_genre}/Val SROCC GIAA": tgt_val_srocc,
                    f"{target_genre}/Val MSE GIAA": tgt_val_mse,
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

# ──────────────────────────────────────────────────────────────────────────────


def run_main(args):
    is_v_giaa = (args.dataset_ver == 'v_giaa')
    batch_size = args.batch_size
    print(args)

    use_dann = bool(getattr(args, 'dann_target', None))
    dann_target_genre = parse_dann_target(args.dann_target) if use_dann else None
    # UDA 適用時はフォルダ/ファイル名に {source}2{target} を使用
    domain_tag = f'{args.genre}2{dann_target_genre}' if use_dann else args.genre

    if args.is_log:
        tags = ["GIAA"]
        tags += wandb_tags(args)
        if use_dann:
            tags.append(domain_tag)  # e.g. "art2fashion"
        wandb.init(
                   project=f"XPASS",
                   notes=f"NIMA",
                   tags = tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs
        }
        experiment_name = f"{domain_tag}_PAA({wandb.run.name})"
        model_basename = f'{domain_tag}_NIMA_{wandb.run.name}.pth'
    else:
        experiment_name = ''
        model_basename = f'{domain_tag}_NIMA_default.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the best test loss and the best model
    dirname = os.path.join(model_dir(args), domain_tag)
    best_modelname = os.path.join(dirname, model_basename)
    model = NIMA(num_bins, backbone=args.backbone, dropout=args.dropout).to(device)
    model.freeze_backbone()

    if use_dann:
        discriminator = DomainDiscriminator(model.feat_dim).to(device)
        grl = GradientReversalLayer()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        optimizer_disc = optim.AdamW(discriminator.parameters(), lr=args.lr * 10)
    else:
        discriminator = grl = optimizer_disc = None
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if is_v_giaa:
        # v_giaa: train/val/test are image-level GIAA splits; no PIAA, no cross-validation
        train_giaa_dataset, val_giaa_dataset, test_giaa_dataset = load_data_giaa_only(args)
        train_giaa_dataloader = DataLoader(train_giaa_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        dataloaders = (train_giaa_dataloader, val_giaa_dataloader, test_giaa_dataloader)

        if use_dann:
            target_genre = parse_dann_target(args.dann_target)
            args_tgt = copy.deepcopy(args)
            args_tgt.genre = target_genre
            tgt_dataset, tgt_val_dataset, _ = load_data_giaa_only(args_tgt)
            tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=args.num_workers, timeout=300, collate_fn=collate_fn,
                                    drop_last=True)
            tgt_val_loader = DataLoader(tgt_val_dataset, batch_size=batch_size, shuffle=False,
                                        num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
            trainer_dann_giaa(dataloaders, tgt_loader, model, discriminator, grl, optimizer, optimizer_disc, args, device, best_modelname,
                               tgt_val_loader=tgt_val_loader)
        else:
            eval_target = getattr(args, 'eval_target', None)
            if eval_target:
                args_tgt = copy.deepcopy(args)
                args_tgt.genre = eval_target
                _, tgt_val_dataset, _ = load_data_giaa_only(args_tgt)
                tgt_val_loader = DataLoader(tgt_val_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
            else:
                tgt_val_loader = None
            trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname,
                    tgt_val_loader=tgt_val_loader, tgt_genre=eval_target)

        # Evaluate on held-out GIAA test set
        # Ground truth: expected value of the per-image histogram (average of multiple raters)
        # Prediction:   expected value of the predicted softmax distribution
        test_emd, test_srocc, _, test_mse, _, test_mae, test_ccc = evaluate(model, test_giaa_dataloader, device, phase_name="Test")
        print(f"[{args.genre} GIAA Test] EMD: {test_emd:.4f}  SROCC: {test_srocc:.4f}  CCC: {test_ccc:.4f}  MSE: {test_mse:.4f}")
        if args.is_log:
            wandb.log({
                f"{args.genre}/Test EMD GIAA": test_emd,
                f"{args.genre}/Test SROCC GIAA": test_srocc,
                f"{args.genre}/Test CCC GIAA": test_ccc,
                f"{args.genre}/Test MSE GIAA": test_mse,
            })
    else:
        # Create dataloaders (PIAA + GIAA)
        train_giaa_dataset, train_piaa_dataset, _, val_giaa_dataset, val_piaa_dataset, _, test_piaa_dataset = load_data(args)
        train_giaa_dataloader = DataLoader(train_giaa_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        train_piaa_dataloader = DataLoader(train_piaa_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        val_piaa_dataloader = DataLoader(val_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        dataloaders = (train_giaa_dataloader, val_giaa_dataloader, test_piaa_dataloader)

        # Cross-domain evaluation on all other genres
        ALL_GENRES = ['art', 'fashion', 'scenery']
        eval_genres = [g for g in ALL_GENRES if g != args.genre]
        eval_datasets_dict = {}
        print(f"Cross-domain evaluation targets: {eval_genres}")
        for eval_genre in eval_genres:
            args_copy = copy.deepcopy(args)
            args_copy.genre = eval_genre
            _, _, _, _, _, _, eval_test_piaa_dataset = load_data(args_copy)
            eval_datasets_dict[eval_genre] = {'test': eval_test_piaa_dataset}
            print(f"Loaded {len(eval_test_piaa_dataset)} test samples for cross-domain eval genre '{eval_genre}'")

        if use_dann:
            target_genre = parse_dann_target(args.dann_target)
            args_tgt = copy.deepcopy(args)
            args_tgt.genre = target_genre
            tgt_giaa_dataset, _, _, tgt_val_giaa_dataset, _, _, _ = load_data(args_tgt)
            tgt_loader = DataLoader(tgt_giaa_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=args.num_workers, timeout=300, collate_fn=collate_fn,
                                    drop_last=True)
            tgt_val_loader = DataLoader(tgt_val_giaa_dataset, batch_size=batch_size, shuffle=False,
                                        num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
            trainer_dann_giaa(dataloaders, tgt_loader, model, discriminator, grl, optimizer, optimizer_disc, args, device, best_modelname,
                               tgt_val_loader=tgt_val_loader)
        else:
            eval_target = getattr(args, 'eval_target', None)
            if eval_target:
                args_tgt = copy.deepcopy(args)
                args_tgt.genre = eval_target
                _, _, _, tgt_val_giaa_dataset, _, _, _ = load_data(args_tgt)
                tgt_val_loader = DataLoader(tgt_val_giaa_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
            else:
                tgt_val_loader = None
            trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname,
                    tgt_val_loader=tgt_val_loader, tgt_genre=eval_target)
        test_piaa_srocc, test_piaa_mse = inference(train_piaa_dataset, val_piaa_dataset,
            test_piaa_dataset, args, device, model, eval_split="Val", experiment_name=experiment_name, model_path=best_modelname)
        test_piaa_srocc, test_piaa_mse = inference(train_piaa_dataset, val_piaa_dataset,
            test_piaa_dataset, args, device, model, eval_split="Test", experiment_name=experiment_name,
            model_path=best_modelname, eval_datasets_dict=eval_datasets_dict)

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
            print(f"\n{'='*60}")
            print(f"  Fold {i+1}/{len(folds)}: {fold}")
            print(f"{'='*60}\n")
            args_fold = copy.deepcopy(args)
            args_fold.dataset_ver = fold
            run_main(args_fold)
    else:
        run_main(args)