import os

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from ..train_common import earth_mover_distance
from ..evaluate import evaluate


def setup(model, args, device):
    """No extra components for source-only training."""
    return {}


def _train_one_epoch(model, dataloader, optimizer, scaler, device, args, epoch: int = None):
    model.train()
    running_emd_loss = 0.0
    desc = f"Epoch {epoch} [Train]" if epoch is not None else "Train"
    progress_bar = tqdm(dataloader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")
    for sample in progress_bar:
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['Aesthetic'].to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            aesthetic_logits = model(images)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_emd_loss += loss.item()
        progress_bar.set_postfix({'Train EMD': loss.item()})

    return running_emd_loss / len(dataloader)


def trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname, components,
            tgt_val_loader=None, tgt_genre=None):
    train_dataloader, val_dataloader, _ = src_dataloaders

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)
    scaler = GradScaler('cuda')

    best_val_emd = float('inf')
    patience = 0
    for epoch in range(args.num_epochs):
        train_emd = _train_one_epoch(model, train_dataloader, optimizer, scaler, device, args, epoch=epoch)
        if args.is_log:
            wandb.log({"epoch": epoch, f"{args.genre}/Train EMD GIAA": train_emd}, commit=False)

        val_emd, val_srocc, _, val_mse, _, _, _ = evaluate(
            model, val_dataloader, device, epoch=epoch, phase_name="Val")
        if args.is_log:
            wandb.log({
                "epoch": epoch,
                f"{args.genre}/Val EMD GIAA": val_emd,
                f"{args.genre}/Val SROCC GIAA": val_srocc,
                f"{args.genre}/Val MSE GIAA": val_mse,
            }, commit=tgt_val_loader is None)

        if tgt_val_loader is not None:
            tgt_val_emd, tgt_val_srocc, _, tgt_val_mse, _, _, _ = evaluate(
                model, tgt_val_loader, device, epoch=epoch, phase_name=f"Val [{tgt_genre}]")
            if args.is_log:
                wandb.log({
                    "epoch": epoch,
                    f"{tgt_genre}/Val EMD GIAA": tgt_val_emd,
                    f"{tgt_genre}/Val SROCC GIAA": tgt_val_srocc,
                    f"{tgt_genre}/Val MSE GIAA": tgt_val_mse,
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
                print(f"Validation loss has not decreased for {args.max_patience_epochs} epochs. Stopping training.")
                break

    model.load_state_dict(torch.load(best_modelname))
