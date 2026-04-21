import os
import wandb
import numpy as np
import copy
import pandas as pd
import json
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from .data import collate_fn
from .train_common import num_bins


def inference_giaa(test_dataset, args, device, model, model_path=None, eval_datasets_dict=None):
    """GIAA-only inference: evaluate on test_images_GIAA.txt and save results to JSON.

    eval_datasets_dict: optional dict of {target_genre: {'test': giaa_test_dataset}}
        for cross-domain evaluation using the GIAA head.
    """
    from .evaluate import evaluate
    from .data import collate_fn
    from .train_common import parse_da_method as _parse_da

    batch_size = args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)

    test_emd, test_srocc, _, test_mse, _, test_mae, test_ccc = evaluate(
        model, test_loader, device, phase_name="Test")

    print(f"[{args.genre} GIAA Test] EMD: {test_emd:.4f}  SROCC: {test_srocc:.4f}  "
          f"CCC: {test_ccc:.4f}  MSE: {test_mse:.4f}  MAE: {test_mae:.4f}")

    # Cross-domain evaluation on GIAA test sets from other genres
    cross_domain_results = {}
    if eval_datasets_dict is not None:
        for target_genre, ds_dict in eval_datasets_dict.items():
            print(f"\n[Cross-Domain] Evaluating {args.genre} GIAA model on {target_genre} GIAA test set...")
            target_test_ds = ds_dict['test']
            target_loader = DataLoader(target_test_ds, batch_size=batch_size, shuffle=False,
                                       num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
            cd_emd, cd_srocc, _, cd_mse, _, cd_mae, cd_ccc = evaluate(
                model, target_loader, device, phase_name="Test")
            cross_domain_results[target_genre] = {
                'source_head': args.genre,
                'average': {
                    'emd': float(cd_emd),
                    'srocc': float(cd_srocc),
                    'mae': float(cd_mae),
                    'ccc': float(cd_ccc),
                },
            }
            print(f"[Cross-Domain] {args.genre} -> {target_genre}: "
                  f"EMD={cd_emd:.4f}  SROCC={cd_srocc:.4f}  "
                  f"CCC={cd_ccc:.4f}  MSE={cd_mse:.4f}  MAE={cd_mae:.4f}")

    _method_name, _tgt_genre = _parse_da(getattr(args, 'da_method', None))
    _domain_tag = f'{args.genre}2{_tgt_genre}' if _tgt_genre else args.genre
    save_dir = os.path.join('/home/hayashi0884/proj-xpass-DA/reports/exp', args.dataset_ver, _domain_tag)
    os.makedirs(save_dir, exist_ok=True)

    if model_path:
        model_basename = os.path.splitext(os.path.basename(model_path))[0]
        json_filename = f"{model_basename}.json"
        display_name = model_basename
    else:
        json_filename = f"{args.genre}_giaa_default.json"
        display_name = f"{args.genre}_giaa_default"

    result_data = {
        'experiment_name': display_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'GIAA',
        'genres': [args.genre],
        'average_metrics': {
            args.genre: {
                'emd': float(test_emd),
                'srocc': float(test_srocc),
                'mae': float(test_mae),
                'ccc': float(test_ccc),
            }
        },
    }
    if cross_domain_results:
        result_data['cross_domain_metrics'] = cross_domain_results

    json_path = os.path.join(save_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"Test results saved to {json_path}")

    return test_emd, test_srocc, test_mse, test_mae, test_ccc


def inference(train_dataset, val_dataset, test_dataset, args, device, model, eval_split, experiment_name='', model_path=None, eval_datasets_dict=None):
    """Per-user inference: load each user's best model and evaluate on a chosen split (val or test).

    eval_split: 'Test' or 'Val'
    model_path: Path to the loaded model (used for filename generation)
    eval_datasets_dict: optional dict of {target_genre: {'test': dataset}} for cross-domain evaluation
    Returns mean_user_srocc, mean_user_mse
    """
    from .evaluate import evaluate

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
        from .train_common import parse_da_method as _parse_da
        _method_name, _tgt_genre = _parse_da(getattr(args, 'da_method', None))
        _method_tag = _method_name if _method_name else 'Only'
        _domain_tag = f'{args.genre}2{_tgt_genre}' if _tgt_genre else args.genre
        save_dir = os.path.join('/home/hayashi0884/proj-xpass-DA/reports/exp', args.dataset_ver,
                                _domain_tag)
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


def inference_finetune(datasets_dict, args, device, dirname, experiment_name, backbone_dict, eval_datasets_dict=None):
    """
    Inference for all users across all genres after finetuning.
    Args:
        datasets_dict: dict of {genre: {'train': ds, 'val': ds, 'test': ds}}
        args: arguments
        device: device
        dirname: directory name for saving models
        experiment_name: experiment name
        backbone_dict: dict of {genre: backbone_type}
        eval_datasets_dict: optional dict of {target_genre: {'test': dataset}} for cross-domain evaluation
    """
    from . import train_PIAA as _tp
    from .evaluate import evaluate_piaa as evaluate, evaluate_cross_domain
    from .train_common import build_piaa_model
    num_attr = _tp.num_attr
    num_pt = _tp.num_pt

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre_str = '-'.join(genres)

    if num_attr is None or num_pt is None:
        _sample = datasets_dict[genres[0]]['test'][0]
        num_attr = len(_sample['QIP'])
        num_pt = len(_sample['traits'])
        _tp.num_attr = num_attr
        _tp.num_pt = num_pt
    all_user_ids = set()
    genre_srocc_list = defaultdict(list)
    genre_mae_list = defaultdict(list)
    genre_ndcg_list = defaultdict(list)
    genre_ccc_list = defaultdict(list)
    for genre in genres:
        all_user_ids.update(datasets_dict[genre]['test'].data['user_id'].values)

    model_name_base = experiment_name

    results = {}
    cd_per_target = defaultdict(lambda: {'per_user': {}, 'per_user_per_head': {},
                                          'head_sroccs': defaultdict(list), 'head_ndcgs': defaultdict(list),
                                          'head_maes': defaultdict(list), 'head_cccs': defaultdict(list),
                                          'avg_sroccs': [], 'avg_ndcgs': [],
                                          'avg_maes': [], 'avg_cccs': []}) if eval_datasets_dict is not None else None

    for uid in sorted(list(all_user_ids)):
        print(f"Running inference for user {uid} using saved best model...")
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{model_name_base}_finetune.pth')
        model_user = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
        try:
            model_user.load_state_dict(torch.load(best_model_path))
        except Exception as e:
            print(f"Warning: best model not found for user {uid} at {best_model_path}, skipping. Error: {e}")
            results[uid] = (np.nan, np.nan)
            continue

        # In-domain evaluation
        test_loaders_dict = {}
        total_test_samples = 0
        for genre in genres:
            user_test_ds = copy.copy(datasets_dict[genre]['test'])
            user_test_ds.data = datasets_dict[genre]['test'].data[datasets_dict[genre]['test'].data['user_id'] == uid].reset_index(drop=True)
            if len(user_test_ds) > 0:
                test_loaders_dict[genre] = DataLoader(user_test_ds, batch_size=batch_size, shuffle=False,
                                                       num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
                total_test_samples += len(user_test_ds)
        if total_test_samples == 0:
            print(f"No test samples for user {uid}, skipping.")
            results[uid] = ({}, np.nan)
        else:
            genre_metrics, total_mae = evaluate(model_user, test_loaders_dict, device)
            for genre, metrics in genre_metrics.items():
                genre_srocc_list[genre].append(metrics['srocc'])
                genre_mae_list[genre].append(metrics['mae'])
                genre_ndcg_list[genre].append(metrics['ndcg@10'])
                genre_ccc_list[genre].append(metrics['ccc'])
            if args.is_log:
                for genre, metrics in genre_metrics.items():
                    wandb.log({
                        f"{genre}/Test SROCC user_{uid}": metrics['srocc'],
                        f"{genre}/Test MAE user_{uid}": metrics['mae'],
                        f"{genre}/Test NDCG@10 user_{uid}": metrics['ndcg@10'],
                        f"{genre}/Test CCC user_{uid}": metrics['ccc'],
                    }, commit=False)
                wandb.log({}, commit=True)
            results[uid] = (genre_metrics, total_mae)

        # Cross-domain evaluation
        if eval_datasets_dict is not None:
            eval_loaders_dict = {}
            for target_genre, ds_dict in eval_datasets_dict.items():
                user_eval_ds = copy.copy(ds_dict['test'])
                user_eval_ds.data = ds_dict['test'].data[ds_dict['test'].data['user_id'] == uid].reset_index(drop=True)
                if len(user_eval_ds) > 0:
                    eval_loaders_dict[target_genre] = DataLoader(
                        user_eval_ds, batch_size=batch_size, shuffle=False,
                        num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
            if len(eval_loaders_dict) > 0:
                cd_results = evaluate_cross_domain(model_user, eval_loaders_dict, device, genres)
                for tg, tg_result in cd_results.items():
                    for uid_str, metrics in tg_result['per_user'].items():
                        cd_per_target[tg]['per_user'][uid_str] = metrics
                        cd_per_target[tg]['avg_sroccs'].append(metrics['srocc'])
                        cd_per_target[tg]['avg_ndcgs'].append(metrics['ndcg@10'])
                        cd_per_target[tg]['avg_maes'].append(metrics['mae'])
                        cd_per_target[tg]['avg_cccs'].append(metrics['ccc'])
                    for uid_str, head_metrics in tg_result['per_user_per_head'].items():
                        cd_per_target[tg]['per_user_per_head'][uid_str] = head_metrics
                        for sg, m in head_metrics.items():
                            cd_per_target[tg]['head_sroccs'][sg].append(m['srocc'])
                            cd_per_target[tg]['head_ndcgs'][sg].append(m['ndcg@10'])
                            cd_per_target[tg]['head_maes'][sg].append(m['mae'])
                            cd_per_target[tg]['head_cccs'][sg].append(m['ccc'])

    # Calculate genre-specific averages
    genre_avg_metrics = {}
    for genre in genres:
        if genre in genre_srocc_list and len(genre_srocc_list[genre]) > 0:
            genre_avg_metrics[genre] = {
                'srocc': np.mean(genre_srocc_list[genre]),
                'mae': np.mean(genre_mae_list[genre]),
                'ndcg@10': np.mean(genre_ndcg_list[genre]),
                'ccc': np.mean(genre_ccc_list[genre]),
            }

    if args.is_log:
        log_dict = {}
        for genre, metrics in genre_avg_metrics.items():
            log_dict[f"{genre}/Avg. Test SROCC"] = metrics['srocc']
            log_dict[f"{genre}/Avg. Test MAE"] = metrics['mae']
            log_dict[f"{genre}/Avg. Test NDCG@10"] = metrics['ndcg@10']
            log_dict[f"{genre}/Avg. Test CCC"] = metrics['ccc']
        wandb.log(log_dict, commit=True)

    cross_domain_results = {}
    if cd_per_target is not None:
        for tg, data in cd_per_target.items():
            cross_domain_results[tg] = {
                'source_heads': genres,
                'method': 'average',
                'average': {
                    'srocc': float(np.mean(data['avg_sroccs'])) if data['avg_sroccs'] else 0.0,
                    'ndcg@10': float(np.mean(data['avg_ndcgs'])) if data['avg_ndcgs'] else 0.0,
                    'mae': float(np.mean(data['avg_maes'])) if data['avg_maes'] else 0.0,
                    'ccc': float(np.mean(data['avg_cccs'])) if data['avg_cccs'] else 0.0,
                },
                'per_head': {
                    sg: {
                        'srocc': float(np.mean(data['head_sroccs'][sg])) if data['head_sroccs'][sg] else 0.0,
                        'ndcg@10': float(np.mean(data['head_ndcgs'][sg])) if data['head_ndcgs'][sg] else 0.0,
                        'mae': float(np.mean(data['head_maes'][sg])) if data['head_maes'][sg] else 0.0,
                        'ccc': float(np.mean(data['head_cccs'][sg])) if data['head_cccs'][sg] else 0.0,
                    }
                    for sg in genres
                },
                'per_user': data['per_user'],
                'per_user_per_head': data['per_user_per_head'],
            }

    # Save test performance to JSON
    from .train_common import parse_da_method as _parse_da_method
    _method, _da_target = _parse_da_method(getattr(args, 'da_method', None))
    if _da_target:
        _folder = f'{genre_str}2{_da_target}'
        _prefix = f'{genre_str}2{_da_target}'
    else:
        _folder = genre_str
        _prefix = genre_str
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'exp', args.dataset_ver, _folder)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare per-user results
    per_user_results = {}
    for uid, (genre_metrics_user, total_mae) in results.items():
        if isinstance(genre_metrics_user, dict):
            per_user_results[str(uid)] = {
                genre: {'srocc': float(metrics['srocc']), 'mae': float(metrics['mae']), 'ndcg@10': float(metrics['ndcg@10']), 'ccc': float(metrics['ccc'])}
                for genre, metrics in genre_metrics_user.items()
            }

    result_data = {
        'experiment_name': model_name_base,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'PIAA_finetune',
        'genres': genres,
        'average_metrics': {
            genre: {'srocc': float(metrics['srocc']), 'mae': float(metrics['mae']), 'ndcg@10': float(metrics['ndcg@10']), 'ccc': float(metrics['ccc'])}
            for genre, metrics in genre_avg_metrics.items()
        },
        'per_user_metrics': per_user_results
    }
    if cross_domain_results:
        result_data['cross_domain_metrics'] = cross_domain_results

    # Remove trailing mode suffix to avoid duplication (e.g., "name_finetune_finetune.json")
    base_name = model_name_base.removesuffix('_finetune')
    # Use {UDA手法名}_{モデル} order when a DA method is used
    if _method and base_name.startswith(_method + '_'):
        _run_suffix = base_name[len(_method) + 1:]
        json_filename = f"{_prefix}_{_method}_{args.model_type}_{_run_suffix}_finetune.json"
    else:
        json_filename = f"{_prefix}_{args.model_type}_{base_name}_finetune.json"
    json_path = os.path.join(save_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"Test results saved to {json_path}")

    return results


def evaluate_pretrain_on_val_piaa(datasets_dict_user, args, device, backbone_dict, best_model_path, model_state_dict=None):
    """
    Pretrain後のベストモデルでval_piaa_datasetを使い、
    ユーザーごとにSROCC/NDCGを算出し、wandbに100回分ログする。
    """
    from . import train_PIAA as _tp
    from .evaluate import evaluate_piaa as evaluate
    from .train_common import build_piaa_model
    num_attr = _tp.num_attr
    num_pt = _tp.num_pt

    batch_size = args.batch_size
    genres = list(datasets_dict_user.keys())
    genre = genres[0]

    if num_attr is None or num_pt is None:
        _sample = datasets_dict_user[genre]['val'][0]
        num_attr = len(_sample['QIP'])
        num_pt = len(_sample['traits'])
        _tp.num_attr = num_attr
        _tp.num_pt = num_pt

    model = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(torch.load(best_model_path))

    all_user_ids = set(datasets_dict_user[genre]['val'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    user_metrics = {}
    for uid in unique_user_ids:
        user_val_ds = copy.copy(datasets_dict_user[genre]['val'])
        user_val_ds.data = datasets_dict_user[genre]['val'].data[datasets_dict_user[genre]['val'].data['user_id'] == uid].reset_index(drop=True)
        if len(user_val_ds) > 0:
            val_loaders_dict = {genre: DataLoader(user_val_ds, batch_size=batch_size, shuffle=False,
                                                   num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}
            genre_metrics, _ = evaluate(model, val_loaders_dict, device)
            user_metrics[uid] = genre_metrics

    print(f"Pretrain val PIAA evaluation done: {len(user_metrics)} users evaluated")


def inference_pretrain(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict, best_model_path, eval_datasets_dict=None, model_state_dict=None):
    """
    Per-user evaluation after pretraining, separated from training.
    Args:
        datasets_dict: dict of {genre: {'train': ds, 'val': ds, 'test': ds}} (PIAA data)
        eval_datasets_dict: optional dict of {target_genre: {'test': dataset}} for cross-domain evaluation
        model_state_dict: if provided, load from this state dict instead of best_model_path
    """
    from . import train_PIAA as _tp
    from .evaluate import evaluate_piaa as evaluate, evaluate_cross_domain
    from .train_common import build_piaa_model
    num_attr = _tp.num_attr
    num_pt = _tp.num_pt

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    if num_attr is None or num_pt is None:
        _sample = datasets_dict[genre]['train'][0]
        num_attr = len(_sample['QIP'])
        num_pt = len(_sample['traits'])
        _tp.num_attr = num_attr
        _tp.num_pt = num_pt

    model = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(torch.load(best_model_path))

    all_user_ids = set(datasets_dict[genre]['test'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    genre_srocc_list = defaultdict(list)
    genre_mae_list = defaultdict(list)
    genre_ndcg_list = defaultdict(list)
    genre_ccc_list = defaultdict(list)
    per_user_results = {}

    for uid in unique_user_ids:
        user_test_ds = copy.copy(datasets_dict[genre]['test'])
        user_test_ds.data = datasets_dict[genre]['test'].data[datasets_dict[genre]['test'].data['user_id'] == uid].reset_index(drop=True)
        if len(user_test_ds) > 0:
            test_loaders_dict = {genre: DataLoader(user_test_ds, batch_size=batch_size, shuffle=False,
                                                    num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)}
            genre_metrics, total_mae = evaluate(model, test_loaders_dict, device)
            for g, metrics in genre_metrics.items():
                genre_srocc_list[g].append(metrics['srocc'])
                genre_mae_list[g].append(metrics['mae'])
                genre_ndcg_list[g].append(metrics['ndcg@10'])
                genre_ccc_list[g].append(metrics['ccc'])
            if args.is_log:
                log_dict = {}
                for g, metrics in genre_metrics.items():
                    log_dict[f"{g}/Test SROCC user_{uid}"] = metrics['srocc']
                    log_dict[f"{g}/Test NDCG@10 user_{uid}"] = metrics['ndcg@10']
                wandb.log(log_dict, commit=True)
            per_user_results[str(uid)] = {
                g: {'srocc': float(metrics['srocc']), 'ndcg@10': float(metrics['ndcg@10']), 'mae': float(metrics['mae']), 'ccc': float(metrics['ccc'])}
                for g, metrics in genre_metrics.items()
            }

    genre_avg_metrics = {}
    for g in genres:
        if g in genre_srocc_list and len(genre_srocc_list[g]) > 0:
            genre_avg_metrics[g] = {
                'srocc': np.mean(genre_srocc_list[g]),
                'mae': np.mean(genre_mae_list[g]),
                'ndcg@10': np.mean(genre_ndcg_list[g]),
                'ccc': np.mean(genre_ccc_list[g]),
            }

    if args.is_log:
        log_dict = {}
        for g, metrics in genre_avg_metrics.items():
            log_dict[f"{g}/Avg. Test SROCC"] = metrics['srocc']
            log_dict[f"{g}/Avg. Test MAE"] = metrics['mae']
            log_dict[f"{g}/Avg. Test NDCG@10"] = metrics['ndcg@10']
        wandb.log(log_dict, commit=True)

    # Cross-domain evaluation
    cross_domain_results = {}
    if eval_datasets_dict is not None:
        eval_loaders_dict = {}
        for target_genre, ds_dict in eval_datasets_dict.items():
            eval_loaders_dict[target_genre] = DataLoader(
                ds_dict['test'], batch_size=batch_size, shuffle=False,
                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        cross_domain_results = evaluate_cross_domain(model, eval_loaders_dict, device, genres)

    # Save test performance to JSON
    # DA時は {source}2{target} ディレクトリに保存
    from .train_common import parse_da_method as _parse_da_method
    _method, _da_target = _parse_da_method(getattr(args, 'da_method', None))
    if _da_target:
        _folder = f'{genre_str}2{_da_target}'
    else:
        _folder = genre_str
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'exp', args.dataset_ver, _folder)
    os.makedirs(save_dir, exist_ok=True)

    model_basename = os.path.splitext(os.path.basename(best_model_path))[0]
    result_data = {
        'experiment_name': model_basename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'PIAA_pretrain',
        'genres': genres,
        'average_metrics': {
            g: {'srocc': float(metrics['srocc']), 'ndcg@10': float(metrics['ndcg@10']), 'mae': float(metrics['mae']), 'ccc': float(metrics['ccc'])}
            for g, metrics in genre_avg_metrics.items()
        },
        'per_user_metrics': per_user_results
    }
    if cross_domain_results:
        result_data['cross_domain_metrics'] = cross_domain_results

    base_name = model_basename.removesuffix('_pretrain')
    json_filename = f"{base_name}_pretrain.json"
    json_path = os.path.join(save_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"Test results saved to {json_path}")

    return None


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _detect_mode(filename: str) -> str:
    """Infer inference mode from a .pth filename.

    Returns one of: 'GIAA', 'PIAA_pretrain', or '' (unknown / not supported).
    """
    if 'NIMA' in filename:
        return 'GIAA'
    if filename.endswith('_pretrain.pth'):
        return 'PIAA_pretrain'
    return ''


def _build_args(genre: str, fold: str, cli):
    """Build a simple namespace mimicking the training args for a given fold."""
    import types
    args = types.SimpleNamespace(
        genre=genre,
        dataset_ver=fold,
        backbone=cli.backbone,
        use_video=(cli.backbone == 'i3d'),
        root_dir=cli.root_dir,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        dropout=cli.dropout,
        use_cross_eval=True,
        is_log=False,
        no_save_model=True,
        model_type=getattr(cli, 'model_type', 'ICI'),
    )
    return args


def _run_giaa(model_path: str, genre: str, fold: str, cli, device):
    """Run GIAA (NIMA) inference for a single model."""
    from .train_common import NIMA
    from .data import load_data

    args = _build_args(genre, fold, cli)

    # Check if result JSON already exists
    model_basename = os.path.splitext(os.path.basename(model_path))[0]
    save_dir = os.path.join('/home/hayashi0884/proj-xpass-DA/reports/exp', fold, genre)
    json_path = os.path.join(save_dir, f'{model_basename}.json')
    if os.path.exists(json_path) and not cli.force:
        print(f"  [skip] JSON already exists: {json_path}")
        return

    print(f"  Loading data for {fold}/{genre} ...")
    train_giaa_dataset, train_piaa_dataset, _, _, val_piaa_dataset, _, test_piaa_dataset = load_data(args)

    print(f"  Loading NIMA model from {model_path} ...")
    model = NIMA(num_bins, backbone=args.backbone, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"  Running GIAA inference (Test) ...")
    inference(train_piaa_dataset, val_piaa_dataset, test_piaa_dataset,
              args, device, model, eval_split='Test', model_path=model_path)


def _run_piaa_pretrain(model_path: str, genre: str, fold: str, cli, device):
    """Run PIAA_pretrain inference for a single model."""
    from . import train_PIAA as _tp
    from .data import load_data, build_global_encoders

    args = _build_args(genre, fold, cli)

    # Check if result JSON already exists
    model_basename = os.path.splitext(os.path.basename(model_path))[0]
    base_name = model_basename.removesuffix('_pretrain')
    save_dir = os.path.join('/home/hayashi0884/proj-xpass-DA/reports/exp', fold, genre)
    json_path = os.path.join(save_dir, f'{base_name}_pretrain.json')
    if os.path.exists(json_path) and not cli.force:
        print(f"  [skip] JSON already exists: {json_path}")
        return

    print(f"  Loading data for {fold}/{genre} ...")
    global_trait_encoders, global_age_bins = build_global_encoders(args.root_dir)
    _, train_piaa_dataset, train_giaa_dataset, _, val_piaa_dataset, val_giaa_dataset, test_piaa_dataset = load_data(
        args, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)

    sample = train_giaa_dataset[0]
    _tp.num_pt = len(sample['traits'])
    _tp.num_attr = len(sample['QIP'])
    print(f"  Detected num_pt={_tp.num_pt}, num_attr={_tp.num_attr}")

    backbone_dict = {genre: args.backbone}
    datasets_dict_user = {genre: {'train': train_piaa_dataset, 'val': val_piaa_dataset, 'test': test_piaa_dataset}}

    # Cross-domain evaluation on all other genres
    import copy as _copy
    ALL_GENRES = ['art', 'fashion', 'scenery']
    eval_genres = [g for g in ALL_GENRES if g != genre]
    eval_datasets_dict = {}
    print(f"  Cross-domain evaluation targets: {eval_genres}")
    for eval_genre in eval_genres:
        args_copy = _copy.deepcopy(args)
        args_copy.genre = eval_genre
        _, _, _, _, _, _, eval_test = load_data(
            args_copy, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
        eval_datasets_dict[eval_genre] = {'test': eval_test}

    print(f"  Loading model from {model_path} ...")
    dirname = os.path.dirname(model_path)

    print(f"  Running PIAA_pretrain inference ...")
    inference_pretrain(
        datasets_dict_user, args, device, dirname, '',
        backbone_dict, {genre: model_path}, model_path,
        eval_datasets_dict=eval_datasets_dict,
    )


if __name__ == '__main__':
    import argparse
    import fnmatch

    MODELS_DIR = '/home/hayashi0884/proj-xpass-DA/models_pth'

    parser = argparse.ArgumentParser(description='Standalone inference runner')
    parser.add_argument('--genre', type=str, required=True,
                        help='Target genre (art / fashion / scenery)')
    parser.add_argument('--pattern', type=str, required=True,
                        help='Glob pattern matched against .pth filename, '
                             'e.g. "*NIMA*", "*_pretrain*"')
    parser.add_argument('--root_dir', type=str, default='/home/hayashi0884/proj-xpass-DA/data')
    parser.add_argument('--backbone', type=str, default='clip_vit_b16',
                        choices=['resnet50', 'i3d', 'vit_b_16', 'clip_rn50', 'clip_vit_b16'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--force', action='store_true', default=False,
                        help='Re-run even if result JSON already exists')
    parser.add_argument('--model_type', type=str, default=None, choices=['ICI', 'MIR'],
                        help='Filter pretrain .pth files by model type (ICI or MIR). '
                             'If not specified, all matched pretrain files are run.')
    cli = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Discover all fold directories in models_pth/
    folds = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])
    print(f"Discovered folds: {folds}")

    for fold in folds:
        genre_dir = os.path.join(MODELS_DIR, fold, cli.genre)
        if not os.path.isdir(genre_dir):
            print(f"[{fold}] No directory for genre '{cli.genre}', skipping.")
            continue

        all_files = [f for f in os.listdir(genre_dir) if f.endswith('.pth')]
        matched = [f for f in all_files if fnmatch.fnmatch(f, cli.pattern)]
        if not matched:
            print(f"[{fold}] No .pth files matching '{cli.pattern}' in {genre_dir}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"[{fold}] Matched: {matched}")
        print(f"{'='*60}")

        # Group by mode
        giaa_files = []
        pretrain_files = []

        for fname in matched:
            mode = _detect_mode(fname)
            if mode == 'GIAA':
                giaa_files.append(fname)
            elif mode == 'PIAA_pretrain':
                pretrain_files.append(fname)
            else:
                print(f"  [warn] Unknown or unsupported mode for '{fname}', skipping.")

        for fname in giaa_files:
            model_path = os.path.join(genre_dir, fname)
            print(f"\n[{fold}] GIAA inference: {fname}")
            try:
                _run_giaa(model_path, cli.genre, fold, cli, device)
            except Exception as e:
                print(f"  [error] {e}")

        # Filter pretrain files by model_type if specified
        if cli.model_type is not None:
            filtered = [f for f in pretrain_files if f'_{cli.model_type}_' in f]
            if len(filtered) < len(pretrain_files):
                excluded = [f for f in pretrain_files if f not in filtered]
                print(f"  [filter] model_type={cli.model_type}: excluding {excluded}")
            pretrain_files = filtered

        for fname in pretrain_files:
            model_path = os.path.join(genre_dir, fname)
            print(f"\n[{fold}] PIAA_pretrain inference: {fname}")
            try:
                _run_piaa_pretrain(model_path, cli.genre, fold, cli, device)
            except Exception as e:
                print(f"  [error] {e}")

    print("\nDone.")
