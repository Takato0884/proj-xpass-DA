import os
import copy

import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .argflags import parse_arguments, model_dir, wandb_tags
from .data import load_data, load_data_giaa_only, collate_fn
from .train_common import NIMA, num_bins, parse_da_method
from .evaluate import evaluate
from .inference import inference

# Registry: method name → module path (relative to this package)
_DA_METHOD_MODULES = {
    'DANN':  '.methods.dann',
    'DJDOT': '.methods.djdot',
}


def _load_method(method_name):
    """Return the method module for the given name, or source_only if None."""
    import importlib
    if method_name and method_name in _DA_METHOD_MODULES:
        return importlib.import_module(_DA_METHOD_MODULES[method_name], package=__package__)
    from .methods import source_only
    return source_only


def run_main(args):
    is_v_giaa = (args.dataset_ver == 'v_giaa')
    batch_size = args.batch_size
    print(args)

    method_name, target_genre = parse_da_method(args.da_method)
    method = _load_method(method_name)
    use_da = method_name is not None
    domain_tag = f'{args.genre}2{target_genre}' if use_da else args.genre
    method_tag = method_name if method_name else 'Only'

    if args.is_log:
        tags = ["GIAA"] + wandb_tags(args)
        if use_da:
            tags += [method_name, domain_tag]
        wandb.init(project="XPASS", notes="NIMA", tags=tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs,
        }
        experiment_name = f"{domain_tag}_{method_tag}_PAA({wandb.run.name})"
        model_basename = f'{domain_tag}_{method_tag}_NIMA_{wandb.run.name}.pth'
    else:
        experiment_name = ''
        model_basename = f'{domain_tag}_{method_tag}_NIMA_default.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirname = os.path.join(model_dir(args), domain_tag)
    best_modelname = os.path.join(dirname, model_basename)

    model = NIMA(num_bins, backbone=args.backbone, dropout=args.dropout).to(device)
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    components = method.setup(model, args, device)

    if is_v_giaa:
        train_dataset, val_dataset, test_dataset = load_data_giaa_only(args)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        src_dataloaders = (train_loader, val_loader, test_loader)

        tgt_loader, tgt_val_loader = _build_target_loaders_giaa_only(
            args, target_genre, batch_size, use_da)

        method.trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname,
                       components, tgt_val_loader=tgt_val_loader, tgt_genre=target_genre)

        test_emd, test_srocc, _, test_mse, _, test_mae, test_ccc = evaluate(
            model, test_loader, device, phase_name="Test")
        print(f"[{args.genre} GIAA Test] EMD: {test_emd:.4f}  SROCC: {test_srocc:.4f}  "
              f"CCC: {test_ccc:.4f}  MSE: {test_mse:.4f}")
        if args.is_log:
            wandb.log({
                f"{args.genre}/Test EMD GIAA": test_emd,
                f"{args.genre}/Test SROCC GIAA": test_srocc,
                f"{args.genre}/Test CCC GIAA": test_ccc,
                f"{args.genre}/Test MSE GIAA": test_mse,
            })

    else:
        (train_giaa_dataset, train_piaa_dataset, _,
         val_giaa_dataset, val_piaa_dataset, _,
         test_piaa_dataset) = load_data(args)

        train_giaa_loader = DataLoader(train_giaa_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        val_giaa_loader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        train_piaa_loader = DataLoader(train_piaa_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        val_piaa_loader = DataLoader(val_piaa_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        test_piaa_loader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        src_dataloaders = (train_giaa_loader, val_giaa_loader, test_piaa_loader)

        tgt_loader, tgt_val_loader = _build_target_loaders_full(
            args, target_genre, batch_size, use_da)

        # Cross-domain evaluation targets
        ALL_GENRES = ['art', 'fashion', 'scenery']
        eval_genres = [g for g in ALL_GENRES if g != args.genre]
        eval_datasets_dict = {}
        print(f"Cross-domain evaluation targets: {eval_genres}")
        for eval_genre in eval_genres:
            args_copy = copy.deepcopy(args)
            args_copy.genre = eval_genre
            _, _, _, _, _, _, eval_test_piaa = load_data(args_copy)
            eval_datasets_dict[eval_genre] = {'test': eval_test_piaa}
            print(f"Loaded {len(eval_test_piaa)} test samples for cross-domain eval genre '{eval_genre}'")

        method.trainer(src_dataloaders, tgt_loader, model, optimizer, args, device, best_modelname,
                       components, tgt_val_loader=tgt_val_loader, tgt_genre=target_genre)

        inference(train_piaa_dataset, val_piaa_dataset, test_piaa_dataset,
                  args, device, model, eval_split="Val",
                  experiment_name=experiment_name, model_path=best_modelname)
        inference(train_piaa_dataset, val_piaa_dataset, test_piaa_dataset,
                  args, device, model, eval_split="Test",
                  experiment_name=experiment_name, model_path=best_modelname,
                  eval_datasets_dict=eval_datasets_dict)

    if args.is_log:
        wandb.finish()


# ─── helpers ──────────────────────────────────────────────────────────────────

def _build_target_loaders_giaa_only(args, target_genre, batch_size, use_da):
    """Return (tgt_train_loader, tgt_val_loader) for v_giaa mode."""
    if use_da:
        args_tgt = copy.deepcopy(args)
        args_tgt.genre = target_genre
        tgt_train, tgt_val, _ = load_data_giaa_only(args_tgt)
        tgt_loader = DataLoader(tgt_train, batch_size=batch_size, shuffle=True,
                                num_workers=args.num_workers, timeout=300,
                                collate_fn=collate_fn, drop_last=True)
        tgt_val_loader = DataLoader(tgt_val, batch_size=batch_size, shuffle=False,
                                    num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        return tgt_loader, tgt_val_loader

    eval_target = getattr(args, 'eval_target', None)
    if eval_target:
        args_tgt = copy.deepcopy(args)
        args_tgt.genre = eval_target
        _, tgt_val, _ = load_data_giaa_only(args_tgt)
        tgt_val_loader = DataLoader(tgt_val, batch_size=batch_size, shuffle=False,
                                    num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        return None, tgt_val_loader

    return None, None


def _build_target_loaders_full(args, target_genre, batch_size, use_da):
    """Return (tgt_train_loader, tgt_val_loader) for full (PIAA) mode."""
    if use_da:
        args_tgt = copy.deepcopy(args)
        args_tgt.genre = target_genre
        tgt_giaa, _, _, tgt_val_giaa, _, _, _ = load_data(args_tgt)
        tgt_loader = DataLoader(tgt_giaa, batch_size=batch_size, shuffle=True,
                                num_workers=args.num_workers, timeout=300,
                                collate_fn=collate_fn, drop_last=True)
        tgt_val_loader = DataLoader(tgt_val_giaa, batch_size=batch_size, shuffle=False,
                                    num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        return tgt_loader, tgt_val_loader

    eval_target = getattr(args, 'eval_target', None)
    if eval_target:
        args_tgt = copy.deepcopy(args)
        args_tgt.genre = eval_target
        _, _, _, tgt_val_giaa, _, _, _ = load_data(args_tgt)
        tgt_val_loader = DataLoader(tgt_val_giaa, batch_size=batch_size, shuffle=False,
                                    num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
        return None, tgt_val_loader

    return None, None


# ──────────────────────────────────────────────────────────────────────────────

from .train_common import discover_folds  # noqa: E402

if __name__ == '__main__':
    args = parse_arguments()

    if args.dataset_ver.endswith('_all'):
        version_prefix = args.dataset_ver[:-4]
        folds = discover_folds(args.root_dir, version_prefix)
        if not folds:
            raise ValueError(
                f"No fold directories found for version '{version_prefix}' in "
                f"{os.path.join(args.root_dir, 'split')}")
        print(f"Running all {len(folds)} folds sequentially: {folds}")
        for i, fold in enumerate(folds):
            print(f"\n{'='*60}\n  Fold {i+1}/{len(folds)}: {fold}\n{'='*60}\n")
            args_fold = copy.deepcopy(args)
            args_fold.dataset_ver = fold
            run_main(args_fold)
    else:
        run_main(args)
