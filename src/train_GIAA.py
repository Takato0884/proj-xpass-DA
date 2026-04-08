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
from .train_common import NIMA, EarthMoverDistance, earth_mover_distance, discover_folds, num_bins
from .inference import inference


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

def run_main(args):
    is_v_giaa = (args.dataset_ver == 'v_giaa')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the best test loss and the best model
    dirname = os.path.join(model_dir(args), args.genre)
    best_modelname = os.path.join(dirname, model_basename)

    # Initialize the combined model
    model = NIMA(num_bins, backbone=args.backbone, dropout=args.dropout).to(device)
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if is_v_giaa:
        # v_giaa: train/val/test are image-level GIAA splits; no PIAA, no cross-validation
        train_giaa_dataset, val_giaa_dataset, test_giaa_dataset = load_data_giaa_only(args)
        train_giaa_dataloader = DataLoader(train_giaa_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        dataloaders = (train_giaa_dataloader, val_giaa_dataloader, test_giaa_dataloader)

        trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)

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