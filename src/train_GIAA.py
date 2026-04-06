# import warnings
# from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
# warnings.simplefilter("error", UnsupportedFieldAttributeWarning)

import os
import wandb
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import json
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.amp import autocast, GradScaler
import copy
import pandas as pd

from .argflags import parse_arguments, model_dir, wandb_tags
from .data import load_data, collate_fn
from .train_common import NIMA, EarthMoverDistance, earth_mover_distance, discover_folds, num_bins


def train(model, dataloader, optimizer, device, args, epoch: int = None):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    scaler = GradScaler('cuda')
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
            # MSE
            mse = criterion_mse(outputs_mean, target_mean)
            running_mse_loss += mse.item()
            # MAE
            mae = F.l1_loss(outputs_mean, target_mean)
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

def inference(train_dataset, val_dataset, test_dataset, args, device, model, eval_split, experiment_name='', model_path=None, eval_datasets_dict=None):
    """Per-user inference: load each user's best model and evaluate on a chosen split (val or test).

    eval_split: 'Test' or 'Val'
    model_path: Path to the loaded model (used for filename generation)
    eval_datasets_dict: optional dict of {target_genre: {'test': dataset}} for cross-domain evaluation
    Returns mean_user_srocc, mean_user_mse
    """
    batch_size = args.batch_size
    user_sroccs = []
    user_mses = []
    user_ndcgs = []
    user_maes = []
    user_cccs = []
    per_user_results = {}

    # derive unique user ids from the train dataset (same approach as PIAA)
    try:
        unique_user_ids = np.unique(train_dataset.data['user_id'].values)
    except Exception:
        unique_user_ids = []

    # choose which dataset to evaluate: validation or test
    if eval_split == 'Val':
        source_dataset = val_dataset
    else:
        source_dataset = test_dataset

    for uid in unique_user_ids:
        if pd.isna(uid):
            continue

        # prepare per-user test dataset (filter selected split)
        user_test_ds = copy.deepcopy(source_dataset)
        try:
            user_test_ds.data = user_test_ds.data[user_test_ds.data['user_id'] == uid]
        except Exception:
            # if dataset structure differs, skip
            print(f"Skipping user {uid}: could not filter test dataset for user_id")
            continue

        if len(user_test_ds) == 0:
            print(f"No test samples for user {uid}, skipping.")
            continue

        user_test_loader = DataLoader(user_test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        _, _, final_srocc, final_mse, final_ndcg, final_mae, final_ccc = evaluate(model, user_test_loader, device, PIAA=True)

        user_sroccs.append(final_srocc if final_srocc is not None else np.nan)
        user_mses.append(final_mse if final_mse is not None else np.nan)
        user_ndcgs.append(final_ndcg if final_ndcg is not None else np.nan)
        user_maes.append(final_mae if final_mae is not None else np.nan)
        user_cccs.append(final_ccc if final_ccc is not None else np.nan)
        per_user_results[str(uid)] = {
            'srocc': float(final_srocc) if final_srocc is not None else None,
            'ndcg@10': float(final_ndcg) if final_ndcg is not None else None,
            'mae': float(final_mae) if final_mae is not None else None,
            'ccc': float(final_ccc) if final_ccc is not None else None,
        }

        if args.is_log:
            if eval_split == 'Val':
                for epoch in range(100):
                    wandb.log({
                        "epoch": epoch,
                        f"{args.genre}/{eval_split} SROCC user_{uid}": final_srocc,
                        f"{args.genre}/{eval_split} NDCG@10 user_{uid}": final_ndcg,
                          }, commit=True)
            else:
                wandb.log({
                    f"{args.genre}/{eval_split} SROCC user_{uid}": final_srocc,
                    f"{args.genre}/{eval_split} NDCG@10 user_{uid}": final_ndcg,
                }, commit=True)

    # log user-average
    mean_user_srocc = np.mean(user_sroccs) if len(user_sroccs) > 0 else np.nan
    mean_user_mse = np.mean(user_mses) if len(user_mses) > 0 else np.nan
    mean_user_ndcg = np.mean(user_ndcgs) if len(user_ndcgs) > 0 else np.nan
    mean_user_mae = np.mean(user_maes) if len(user_maes) > 0 else np.nan
    mean_user_ccc = np.mean(user_cccs) if len(user_cccs) > 0 else np.nan

    if args.is_log and (eval_split == 'Test'):
        wandb.log({
            f"{args.genre}/Avg. {eval_split} SROCC": mean_user_srocc,
            f"{args.genre}/Avg. {eval_split} MSE": mean_user_mse,
            f"{args.genre}/Avg. {eval_split} NDCG@10": mean_user_ndcg,
        }, commit=True)

    # Cross-domain evaluation on PIAA test sets from other genres
    cross_domain_results = {}
    if eval_split == 'Test' and eval_datasets_dict is not None:
        for target_genre, ds_dict in eval_datasets_dict.items():
            print(f"\n[Cross-Domain] Evaluating {args.genre} GIAA model on {target_genre} PIAA test set...")
            target_test_ds = ds_dict['test']
            cd_user_sroccs = []
            cd_user_ndcgs = []
            cd_per_user = {}

            # Get user ids from target test dataset
            try:
                target_user_ids = np.unique(target_test_ds.data['user_id'].values)
            except Exception:
                target_user_ids = []

            for uid in target_user_ids:
                if pd.isna(uid):
                    continue
                user_cd_ds = copy.deepcopy(target_test_ds)
                user_cd_ds.data = user_cd_ds.data[user_cd_ds.data['user_id'] == uid]
                if len(user_cd_ds) == 0:
                    continue

                user_cd_loader = DataLoader(user_cd_ds, batch_size=batch_size, shuffle=False,
                                            num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
                _, _, cd_srocc, _, cd_ndcg, cd_mae, cd_ccc = evaluate(model, user_cd_loader, device, PIAA=True)

                if cd_srocc is not None and not np.isnan(cd_srocc):
                    cd_user_sroccs.append(cd_srocc)
                    cd_user_ndcgs.append(cd_ndcg)
                    cd_per_user[str(uid)] = {
                        'srocc': float(cd_srocc),
                        'ndcg@10': float(cd_ndcg) if cd_ndcg is not None else None,
                        'mae': float(cd_mae) if cd_mae is not None else None,
                        'ccc': float(cd_ccc) if cd_ccc is not None else None,
                    }

            cd_avg_srocc = float(np.mean(cd_user_sroccs)) if cd_user_sroccs else 0.0
            cd_avg_ndcg = float(np.mean(cd_user_ndcgs)) if cd_user_ndcgs else 0.0
            cd_maes = [v['mae'] for v in cd_per_user.values() if v.get('mae') is not None]
            cd_cccs = [v['ccc'] for v in cd_per_user.values() if v.get('ccc') is not None]
            cd_avg_mae = float(np.mean(cd_maes)) if cd_maes else 0.0
            cd_avg_ccc = float(np.mean(cd_cccs)) if cd_cccs else 0.0

            cross_domain_results[target_genre] = {
                'source_head': args.genre,
                'average': {
                    'srocc': cd_avg_srocc,
                    'ndcg@10': cd_avg_ndcg,
                    'mae': cd_avg_mae,
                    'ccc': cd_avg_ccc,
                },
                'per_user': cd_per_user,
            }

            print(f"[Cross-Domain] {args.genre} -> {target_genre}: "
                  f"avg SROCC={cd_avg_srocc:.4f}, avg NDCG@10={cd_avg_ndcg:.4f}, "
                  f"avg MAE={cd_avg_mae:.4f}, avg CCC={cd_avg_ccc:.4f}")

    # Save results to JSON file (only for Test split)
    if eval_split == 'Test':
        save_dir = os.path.join('/home/hayashi0884/proj-xpass-DA/reports/exp', args.dataset_ver, args.genre)
        os.makedirs(save_dir, exist_ok=True)

        # Use model_path basename for both experiment_name field and json filename
        if model_path:
            model_basename = os.path.splitext(os.path.basename(model_path))[0]
            json_filename = f"{model_basename}.json"
            display_name = model_basename
        else:
            json_filename = f"{experiment_name}.json"
            display_name = experiment_name

        # Prepare per_user_metrics in the same format as other models: {uid: {genre: {srocc, ndcg@10}}}
        per_user_metrics_formatted = {}
        for uid_str, metrics_val in per_user_results.items():
            per_user_metrics_formatted[uid_str] = {
                args.genre: metrics_val
            }

        result_data = {
            'experiment_name': display_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': 'GIAA',
            'genres': [args.genre],
            'average_metrics': {
                args.genre: {
                    'srocc': float(mean_user_srocc) if not np.isnan(mean_user_srocc) else None,
                    'ndcg@10': float(mean_user_ndcg) if not np.isnan(mean_user_ndcg) else None,
                    'mae': float(mean_user_mae) if not np.isnan(mean_user_mae) else None,
                    'ccc': float(mean_user_ccc) if not np.isnan(mean_user_ccc) else None,
                }
            },
            'per_user_metrics': per_user_metrics_formatted
        }
        if cross_domain_results:
            result_data['cross_domain_metrics'] = cross_domain_results

        json_path = os.path.join(save_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"Test results saved to {json_path}")

    return mean_user_srocc, mean_user_mse

def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname):

    train_dataloader, val_giaa_dataloader, test_dataloader = dataloaders

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)

    # Training loop
    best_test_emd = 1000
    num_patience_epochs = 0
    for epoch in range(args.num_epochs):
        # Training
        train_emd_loss = train_fn(model, train_dataloader, optimizer, device, args, epoch=epoch)
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
            }, commit=True)

        eval_emd = val_emd_loss
        scheduler.step(eval_emd)

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

def run_main(args):
    batch_size = args.batch_size
    print(args)

    if args.is_log:
        tags = ["GIAA"]
        tags += wandb_tags(args)
        wandb.init(
                   project=f"XPASS",
                   notes=f"NIMA",
                   tags = tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs
        }
        experiment_name = f"{args.genre}_PAA({wandb.run.name})"
        model_basename = f'{args.genre}_NIMA_{wandb.run.name}.pth'
    else:
        experiment_name = ''
        model_basename = f'{args.genre}_NIMA_default.pth'

    # Create dataloaders
    train_giaa_dataset, train_piaa_dataset, _, val_giaa_dataset, val_piaa_dataset, _, test_piaa_dataset = load_data(args)
    train_giaa_dataloader = DataLoader(train_giaa_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    train_piaa_dataloader = DataLoader(train_piaa_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    val_piaa_dataloader = DataLoader(val_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    dataloaders = (train_giaa_dataloader, val_giaa_dataloader, test_piaa_dataloader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the best test loss and the best model
    dirname = os.path.join(model_dir(args), args.genre)
    best_modelname = os.path.join(dirname, model_basename)

    # Load cross-domain evaluation datasets if --use_cross_eval is enabled
    ALL_GENRES = ['art', 'fashion', 'scenery']
    eval_datasets_dict = None
    if args.use_cross_eval:
        eval_genres = [g for g in ALL_GENRES if g != args.genre]
        if not eval_genres:
            print("Warning: No other genres for cross-domain evaluation.")
        else:
            print(f"Cross-domain evaluation targets: {eval_genres}")
            eval_datasets_dict = {}
            for eval_genre in eval_genres:
                args_copy = copy.deepcopy(args)
                args_copy.genre = eval_genre
                _, _, _, _, _, _, eval_test_piaa_dataset = load_data(args_copy)
                eval_datasets_dict[eval_genre] = {'test': eval_test_piaa_dataset}
                print(f"Loaded {len(eval_test_piaa_dataset)} test samples for cross-domain eval genre '{eval_genre}'")

    # Initialize the combined model
    model = NIMA(num_bins, backbone=args.backbone, dropout=args.dropout).to(device)

    model.freeze_backbone()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)
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