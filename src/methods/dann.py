import os

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from ..train_common import (
    earth_mover_distance, GradientReversalLayer, DomainDiscriminator, get_da_lambda)
from ..evaluate import evaluate


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

        val_emd, val_srocc, _, val_mse, _, _, _ = evaluate(
            model, val_loader, device, epoch=epoch, phase_name="Val")
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
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_modelname))
