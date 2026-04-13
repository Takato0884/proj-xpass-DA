import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from .train_common import earth_mover_distance, num_bins

_criterion_mse = nn.MSELoss()


def evaluate(model, dataloader, device, PIAA=False, epoch: int = None, phase_name: str = "Val"):
    model.eval()
    running_emd_loss = 0.0
    running_mse_loss = 0.0
    running_mae_loss = 0.0
    scale = torch.arange(0, num_bins).to(device)
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
            running_mse_loss += _criterion_mse(outputs_mean_norm, target_mean_norm).item()
            running_mae_loss += F.l1_loss(outputs_mean_norm, target_mean_norm).item()
            running_emd_loss += loss.item()
            progress_bar.set_postfix({'EMD': loss.item()})

    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc_GIAA, _ = spearmanr(predicted_scores, true_scores)

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
