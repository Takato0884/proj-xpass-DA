import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from collections import defaultdict
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


# ─── PIAA Evaluation ──────────────────────────────────────────────────────────

def _collect_user_ids(user_ids) -> list:
    """Convert various user_id formats from collate_fn to a flat list of ints."""
    if isinstance(user_ids, torch.Tensor):
        return user_ids.view(-1).cpu().numpy().tolist()
    result = []
    for u in user_ids:
        result.append(int(u.item()) if isinstance(u, torch.Tensor) else int(u))
    return result


def evaluate_piaa(model, dataloaders_dict, device, epoch: int = None, phase_name: str = "Val"):
    """
    Evaluate PIAA model across all genres.
    Returns:
        genre_metrics: dict of {genre: {'srocc': float, 'mae': float, 'ndcg@10': float, 'ccc': float}}
        total_mae_loss: average MAE loss across all genres
    """
    model.eval()

    desc = f"Epoch {epoch} [{phase_name}]" if epoch is not None else phase_name
    total_expected_batches = sum(len(loader) for loader in dataloaders_dict.values())
    progress_bar = tqdm(total=total_expected_batches, leave=True, desc=desc, position=1, ncols=120, colour="#fffb00", ascii="-=")

    genre_predictions = defaultdict(list)
    genre_targets = defaultdict(list)
    genre_user_ids = defaultdict(list)
    genre_mae_losses = {}
    genre_batch_counts = {}

    component_interaction = defaultdict(float)
    component_direct = defaultdict(float)
    component_batch_counts = defaultdict(int)

    with torch.no_grad():
        for genre, dataloader in dataloaders_dict.items():
            running_mae = 0.0
            batch_count = 0

            for sample in dataloader:
                images = sample['image'].to(device)
                target = sample['Aesthetic'].to(device).view(-1, 1)
                sample_pt = sample['traits'].float().to(device)
                sample_attr = sample['QIP'].float().to(device)

                if 'user_id' in sample:
                    genre_user_ids[genre].extend(_collect_user_ids(sample['user_id']))

                with autocast('cuda'):
                    outputs = model(images, sample_pt, sample_attr, genre)
                outputs = outputs.view(-1, 1)
                genre_predictions[genre].append(outputs.view(-1).cpu().numpy())
                genre_targets[genre].append(target.view(-1).cpu().numpy())
                mae = F.l1_loss(outputs, target)
                running_mae += mae.item()
                component_interaction[genre] += getattr(model, '_last_interaction_mean', 0.0)
                component_direct[genre] += getattr(model, '_last_direct_mean', 0.0)
                component_batch_counts[genre] += 1
                batch_count += 1
                progress_bar.update(1)
                progress_bar.set_postfix({'MAE': mae.item(), 'genre': genre})

            genre_mae_losses[genre] = running_mae
            genre_batch_counts[genre] = batch_count

    model._eval_component_stats = {}
    for genre in dataloaders_dict.keys():
        n = max(component_batch_counts[genre], 1)
        i_mean = component_interaction[genre] / n
        d_mean = component_direct[genre] / n
        model._eval_component_stats[genre] = {
            'interaction_mean': i_mean,
            'direct_mean': d_mean,
            'ratio': i_mean / d_mean if d_mean > 0 else 0.0,
        }

    progress_bar.close()

    genre_metrics = {}
    total_mae_loss = 0.0

    for genre in dataloaders_dict.keys():
        if len(genre_predictions[genre]) == 0:
            continue

        predicted_scores = np.concatenate(genre_predictions[genre], axis=0)
        true_scores = np.concatenate(genre_targets[genre], axis=0)
        user_ids = genre_user_ids[genre]
        if len(user_ids) == 0:
            raise ValueError(
                f"No user_id found for genre '{genre}'. "
                f"The dataset must contain 'user_id' field for per-user SROCC computation."
            )

        unique_user_ids = np.unique(user_ids)
        sroccs = []
        ndcgs = []
        cccs = []
        for uid in unique_user_ids:
            uid_mask = (np.array(user_ids) == uid)
            n_samples = np.sum(uid_mask)
            if n_samples <= 1:
                raise ValueError(
                    f"User {uid} in genre '{genre}' has only {n_samples} sample(s). "
                    f"At least 2 samples are required for SROCC computation."
                )
            uid_pred = predicted_scores[uid_mask]
            uid_true = true_scores[uid_mask]
            uid_srocc, _ = spearmanr(uid_pred, uid_true)
            sroccs.append(uid_srocc)
            uid_ndcg = ndcg_score([uid_true], [uid_pred], k=10)
            ndcgs.append(uid_ndcg)
            mu_p, mu_t = uid_pred.mean(), uid_true.mean()
            var_p, var_t = uid_pred.var(), uid_true.var()
            cov = ((uid_pred - mu_p) * (uid_true - mu_t)).mean()
            uid_ccc = 2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8)
            cccs.append(float(uid_ccc))
        genre_srocc = np.mean(sroccs) if len(sroccs) > 0 else 0.0
        genre_ndcg = np.mean(ndcgs) if len(ndcgs) > 0 else 0.0
        genre_ccc = np.mean(cccs) if len(cccs) > 0 else 0.0

        genre_mae = genre_mae_losses[genre] / genre_batch_counts[genre] if genre_batch_counts[genre] > 0 else 0.0

        genre_metrics[genre] = {
            'srocc': genre_srocc,
            'mae': genre_mae,
            'ndcg@10': genre_ndcg,
            'ccc': genre_ccc,
        }
        total_mae_loss += genre_mae_losses[genre]

    total_batch_count = sum(genre_batch_counts.values())
    total_mae_loss = total_mae_loss / total_batch_count if total_batch_count > 0 else 0.0

    return genre_metrics, total_mae_loss


def evaluate_cross_domain(model, eval_dataloaders_dict, device, source_genres):
    """
    Cross-domain evaluation: evaluate target domain data using source domain heads.
    For single source head, use that head's output directly.
    For multiple source heads, average the outputs from all heads.

    Returns:
        cross_domain_results: dict of {target_genre: {
            'average': {'srocc': float, 'ndcg@10': float},
            'per_head': {source_genre: {'srocc': float, 'ndcg@10': float}},
            'per_user': {uid: {'srocc': float, 'ndcg@10': float}},
            'per_user_per_head': {uid: {source_genre: {'srocc': float, 'ndcg@10': float}}}
        }}
    """
    model.eval()
    cross_domain_results = {}

    for target_genre, dataloader in eval_dataloaders_dict.items():
        print(f"\n[Cross-Domain] Evaluating {target_genre} data with heads: {source_genres}")

        head_predictions = {sg: [] for sg in source_genres}
        all_targets = []
        all_user_ids = []

        desc = f"Cross-Domain [{target_genre}]"
        progress_bar = tqdm(total=len(dataloader), leave=True, desc=desc, ncols=120, colour="#ff8800", ascii="-=")

        with torch.no_grad():
            for sample in dataloader:
                images = sample['image'].to(device)
                target = sample['Aesthetic'].to(device).view(-1, 1)
                sample_pt = sample['traits'].float().to(device)
                sample_attr = sample['QIP'].float().to(device)

                if 'user_id' in sample:
                    all_user_ids.extend(_collect_user_ids(sample['user_id']))

                all_targets.append(target.view(-1).cpu().numpy())

                for sg in source_genres:
                    with autocast('cuda'):
                        outputs = model(images, sample_pt, sample_attr, sg)
                    head_predictions[sg].append(outputs.view(-1).cpu().numpy())

                progress_bar.update(1)

        progress_bar.close()

        true_scores = np.concatenate(all_targets, axis=0)
        user_ids = np.array(all_user_ids)
        unique_user_ids = np.unique(user_ids)

        per_head_preds = {}
        for sg in source_genres:
            per_head_preds[sg] = np.concatenate(head_predictions[sg], axis=0)

        avg_preds = np.mean([per_head_preds[sg] for sg in source_genres], axis=0)

        avg_sroccs, avg_ndcgs, avg_maes, avg_cccs = [], [], [], []
        per_user_results = {}
        for uid in unique_user_ids:
            uid_mask = (user_ids == uid)
            n_samples = np.sum(uid_mask)
            if n_samples <= 1:
                raise ValueError(
                    f"User {uid} in target genre '{target_genre}' has only {n_samples} sample(s). "
                    f"At least 2 samples are required for SROCC computation."
                )
            uid_pred = avg_preds[uid_mask]
            uid_true = true_scores[uid_mask]
            uid_srocc, _ = spearmanr(uid_pred, uid_true)
            uid_ndcg = ndcg_score([uid_true], [uid_pred], k=10)
            uid_mae = float(np.mean(np.abs(uid_pred - uid_true)))
            mu_p, mu_t = uid_pred.mean(), uid_true.mean()
            var_p, var_t = uid_pred.var(), uid_true.var()
            cov = ((uid_pred - mu_p) * (uid_true - mu_t)).mean()
            uid_ccc = float(2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8))
            avg_sroccs.append(uid_srocc)
            avg_ndcgs.append(uid_ndcg)
            avg_maes.append(uid_mae)
            avg_cccs.append(uid_ccc)
            per_user_results[str(int(uid))] = {
                'srocc': float(uid_srocc),
                'ndcg@10': float(uid_ndcg),
                'mae': uid_mae,
                'ccc': uid_ccc,
            }

        per_head_metrics = {}
        per_user_per_head = {}
        for sg in source_genres:
            sg_sroccs, sg_ndcgs, sg_maes, sg_cccs = [], [], [], []
            for uid in unique_user_ids:
                uid_mask = (user_ids == uid)
                n_samples = np.sum(uid_mask)
                if n_samples <= 1:
                    raise ValueError(
                        f"User {uid} in target genre '{target_genre}' has only {n_samples} sample(s). "
                        f"At least 2 samples are required for SROCC computation."
                    )
                uid_pred = per_head_preds[sg][uid_mask]
                uid_true = true_scores[uid_mask]
                uid_srocc, _ = spearmanr(uid_pred, uid_true)
                uid_ndcg = ndcg_score([uid_true], [uid_pred], k=10)
                uid_mae = float(np.mean(np.abs(uid_pred - uid_true)))
                mu_p, mu_t = uid_pred.mean(), uid_true.mean()
                var_p, var_t = uid_pred.var(), uid_true.var()
                cov = ((uid_pred - mu_p) * (uid_true - mu_t)).mean()
                uid_ccc = float(2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8))
                sg_sroccs.append(uid_srocc)
                sg_ndcgs.append(uid_ndcg)
                sg_maes.append(uid_mae)
                sg_cccs.append(uid_ccc)
                uid_str = str(int(uid))
                if uid_str not in per_user_per_head:
                    per_user_per_head[uid_str] = {}
                per_user_per_head[uid_str][sg] = {
                    'srocc': float(uid_srocc),
                    'ndcg@10': float(uid_ndcg),
                    'mae': uid_mae,
                    'ccc': uid_ccc,
                }
            per_head_metrics[sg] = {
                'srocc': float(np.mean(sg_sroccs)) if sg_sroccs else 0.0,
                'ndcg@10': float(np.mean(sg_ndcgs)) if sg_ndcgs else 0.0,
                'mae': float(np.mean(sg_maes)) if sg_maes else 0.0,
                'ccc': float(np.mean(sg_cccs)) if sg_cccs else 0.0,
            }

        cross_domain_results[target_genre] = {
            'source_heads': source_genres,
            'method': 'average',
            'average': {
                'srocc': float(np.mean(avg_sroccs)) if avg_sroccs else 0.0,
                'ndcg@10': float(np.mean(avg_ndcgs)) if avg_ndcgs else 0.0,
                'mae': float(np.mean(avg_maes)) if avg_maes else 0.0,
                'ccc': float(np.mean(avg_cccs)) if avg_cccs else 0.0,
            },
            'per_head': per_head_metrics,
            'per_user': per_user_results,
            'per_user_per_head': per_user_per_head,
        }

        print(f"[Cross-Domain] {target_genre}: avg SROCC={cross_domain_results[target_genre]['average']['srocc']:.4f}, "
              f"avg NDCG@10={cross_domain_results[target_genre]['average']['ndcg@10']:.4f}, "
              f"avg MAE={cross_domain_results[target_genre]['average']['mae']:.4f}, "
              f"avg CCC={cross_domain_results[target_genre]['average']['ccc']:.4f}")
        for sg, m in per_head_metrics.items():
            print(f"  head [{sg}]: SROCC={m['srocc']:.4f}, NDCG@10={m['ndcg@10']:.4f}, MAE={m['mae']:.4f}, CCC={m['ccc']:.4f}")

    return cross_domain_results
