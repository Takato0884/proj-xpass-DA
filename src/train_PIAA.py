import os
import copy
import importlib
import wandb
from torch.utils.data import DataLoader
import torch

from .argflags import parse_arguments, model_dir, wandb_tags
from .data import load_data, collate_fn, build_global_encoders
from .train_common import discover_folds, parse_da_method
from .evaluate import evaluate_cross_domain
from .inference import inference_finetune, evaluate_pretrain_on_val_piaa, inference_pretrain

_DA_METHOD_MODULES_PIAA = {
    'DANN':     '.methods.dann',
    'DJDOT':    '.methods.djdot',
    'MCD':      '.methods.mcd',
    'DAREGRAM': '.methods.daregram',
}

num_attr = None  # Determined dynamically from dataset
num_pt = None    # Determined dynamically from dataset


def discover_pretrained_models(dataset_ver, genre, piaa_mode='PIAA_finetune', model_type=None, domain_tag=None, da_method=None):
    """Auto-discover pretrained model files.

    For PIAA_pretrain:
      - DANN時 (domain_tag が genre と異なる): models_pth/{dataset_ver}/{domain_tag}/ から DA済みNIMAを探す
      - 通常時: models_pth/{dataset_ver}/{genre}/ から NIMAを探す
      - da_method が指定された場合: ファイル名に da_method を含むものに絞り込む
    For PIAA_finetune: finds *_pretrain.pth in models_pth/{dataset_ver}/{domain_tag or genre}/,
                       filtered by model_type (ICI or MIR) if specified.
    """
    search_dir_name = domain_tag if (domain_tag and domain_tag != genre) else genre
    genre_dir = os.path.join('models_pth', dataset_ver, search_dir_name)
    if not os.path.isdir(genre_dir):
        raise FileNotFoundError(f"Directory not found: {genre_dir}")

    if piaa_mode == 'PIAA_pretrain':
        nima_files = [f for f in os.listdir(genre_dir) if 'NIMA' in f and f.endswith('.pth')]
        if da_method and len(nima_files) > 1:
            filtered = [f for f in nima_files if f'_{da_method}_' in f]
            if filtered:
                nima_files = filtered
        if len(nima_files) == 1:
            return {genre: os.path.join(genre_dir, nima_files[0])}
        elif len(nima_files) > 1:
            raise ValueError(f"Multiple NIMA pth files found in {genre_dir}: {nima_files}. Please specify --pretrained_model explicitly.")
        else:
            raise FileNotFoundError(f"No NIMA pth file found in {genre_dir}")
    else:
        all_pretrain_files = [f for f in os.listdir(genre_dir) if f.endswith('_pretrain.pth')]
        if model_type is not None:
            pretrain_files = [f for f in all_pretrain_files if f'_{model_type}_' in f]
        else:
            pretrain_files = all_pretrain_files
        if da_method and len(pretrain_files) > 1:
            filtered = [f for f in pretrain_files if f'_{da_method}_' in f]
            if filtered:
                pretrain_files = filtered
        if len(pretrain_files) == 1:
            return {genre: os.path.join(genre_dir, pretrain_files[0])}
        elif len(pretrain_files) > 1:
            raise ValueError(f"Multiple pretrain pth files found in {genre_dir}: {pretrain_files}. Please specify --pretrained_model explicitly.")
        else:
            raise FileNotFoundError(f"No pretrain pth file found in {genre_dir}")


def run_main(args):
    global num_pt, num_attr

    genre = args.genre.strip()
    if ',' in genre:
        raise ValueError(
            f"Multi-domain training has been removed. "
            f"Specify a single genre (e.g., --genre art), got: '{genre}'"
        )
    genres = [genre]
    print(f"Training with genre: {genre}")

    backbone_dict = {}
    if genre == 'scenery' and args.use_video:
        backbone_dict[genre] = 'i3d'
    else:
        backbone_dict[genre] = args.backbone
    print(f"Backbone: {backbone_dict[genre]}")

    method_name, target_genre = parse_da_method(getattr(args, 'da_method', None))
    use_da = method_name is not None
    domain_tag = f'{genre}2{target_genre}' if use_da else genre

    # NIMA discovery: for PIAA_pretrain with DAREGRAM, the NIMA can be sourced
    # from any other DA method or source_only via --nima_da_method.
    nima_override = getattr(args, 'nima_da_method', None)
    if args.piaa_mode == 'PIAA_pretrain' and nima_override:
        if nima_override == 'source_only':
            nima_search_tag = genre
            nima_filter = None
        else:
            nima_search_tag = f'{genre}2{target_genre}' if target_genre else genre
            nima_filter = nima_override
    else:
        nima_search_tag = domain_tag
        nima_filter = method_name

    pretrained_model_dict = discover_pretrained_models(args.dataset_ver, genre, args.piaa_mode, getattr(args, 'model_type', None), domain_tag=nima_search_tag, da_method=nima_filter)
    print(f"Auto-discovered pretrained models: {pretrained_model_dict}")

    if args.is_log:
        tags = [args.piaa_mode]
        tags += wandb_tags(args)
        if args.use_video:
            tags.append("use_video")
        if use_da:
            tags.append(method_name)
            tags.append(domain_tag)
        wandb.init(
                   project=args.wandb_project,
                   notes=f"{args.model_type}",
                   tags=tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "genre": genre,
            "backbone": backbone_dict[genre]
        }
        method_tag = method_name if method_name else 'Only'
        experiment_name = f"{method_tag}_{wandb.run.name}"
    else:
        method_tag = method_name if method_name else 'Only'
        experiment_name = method_tag

    print(args)

    global_trait_encoders, global_age_bins = build_global_encoders(args.root_dir)

    _, train_piaa_dataset, train_giaa_dataset, _, val_piaa_dataset, val_giaa_dataset, test_piaa_dataset = load_data(
        args, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)

    datasets_dict = {genre: {'train': train_giaa_dataset, 'val': val_giaa_dataset, 'test': test_piaa_dataset}}
    datasets_dict_user = {genre: {'train': train_piaa_dataset, 'val': val_piaa_dataset, 'test': test_piaa_dataset}}

    _sample = train_giaa_dataset[0]
    num_pt = len(_sample['traits'])
    num_attr = len(_sample['QIP'])
    print(f"Detected num_pt={num_pt}, num_attr={num_attr} from dataset")

    ALL_GENRES = ['art', 'fashion', 'scenery']
    eval_genres = [g for g in ALL_GENRES if g != genre]
    eval_datasets_dict = {}
    print(f"Cross-domain evaluation targets: {eval_genres}")
    for eval_genre in eval_genres:
        args_copy = copy.deepcopy(args)
        args_copy.genre = eval_genre
        if eval_genre == 'scenery' and args.use_video:
            args_copy.backbone = 'i3d'
        else:
            args_copy.backbone = args.backbone
        _, _, _, _, _, _, test_piaa_dataset_eval = load_data(args_copy, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
        eval_datasets_dict[eval_genre] = {'test': test_piaa_dataset_eval}
        print(f"Loaded {len(test_piaa_dataset_eval)} test samples for cross-domain eval genre '{eval_genre}'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirname = os.path.join(model_dir(args), domain_tag)
    os.makedirs(dirname, exist_ok=True)

    if args.piaa_mode == 'PIAA_pretrain':
        if method_name in _DA_METHOD_MODULES_PIAA:
            mod = importlib.import_module(_DA_METHOD_MODULES_PIAA[method_name], package=__package__)
            args_tgt = copy.deepcopy(args)
            args_tgt.genre = target_genre
            _, _, tgt_train_giaa_dataset, _, _, tgt_val_giaa_dataset, _ = load_data(
                args_tgt, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
            best_model_path, best_state_dict = mod.trainer_pretrain(
                datasets_dict, tgt_train_giaa_dataset, tgt_val_giaa_dataset, args, device, dirname,
                experiment_name, backbone_dict, pretrained_model_dict, num_attr, num_pt,
                domain_tag=domain_tag)
        else:
            src_mod = importlib.import_module('.methods.source_only', package=__package__)
            eval_target = getattr(args, 'eval_target', None)
            if eval_target:
                args_tgt = copy.deepcopy(args)
                args_tgt.genre = eval_target
                _, _, tgt_giaa_dataset, _, _, tgt_val_giaa_dataset, _ = load_data(
                    args_tgt, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
                tgt_val_loader = DataLoader(tgt_val_giaa_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
            else:
                tgt_val_loader = None
            best_model_path, best_state_dict = src_mod.trainer_pretrain(
                datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                num_attr, num_pt, tgt_val_loader=tgt_val_loader, tgt_genre=eval_target)
        evaluate_pretrain_on_val_piaa(datasets_dict_user, args, device, backbone_dict, best_model_path, model_state_dict=best_state_dict)
        inference_pretrain(datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict, best_model_path, eval_datasets_dict=eval_datasets_dict, model_state_dict=best_state_dict)
    elif args.piaa_mode == 'PIAA_finetune':
        if method_name in _DA_METHOD_MODULES_PIAA:
            mod = importlib.import_module(_DA_METHOD_MODULES_PIAA[method_name], package=__package__)
            args_tgt = copy.deepcopy(args)
            args_tgt.genre = target_genre
            _, tgt_train_piaa_dataset, _, _, tgt_val_piaa_dataset, _, _ = load_data(
                args_tgt, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
            if method_name == 'DANN':
                mod.trainer_finetune(
                    datasets_dict_user, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                    args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                    num_attr, num_pt, dann_target_genre=target_genre)
            elif method_name == 'MCD':
                mod.trainer_finetune(
                    datasets_dict_user, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                    args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                    num_attr, num_pt, mcd_target_genre=target_genre)
            elif method_name == 'DAREGRAM':
                mod.trainer_finetune(
                    datasets_dict_user, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                    args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                    num_attr, num_pt, daregram_target_genre=target_genre)
            else:  # DJDOT
                mod.trainer_finetune(
                    datasets_dict_user, tgt_train_piaa_dataset, tgt_val_piaa_dataset,
                    args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                    num_attr, num_pt, djdot_target_genre=target_genre)
        else:
            src_mod = importlib.import_module('.methods.source_only', package=__package__)
            eval_target = getattr(args, 'eval_target', None)
            if eval_target:
                args_tgt = copy.deepcopy(args)
                args_tgt.genre = eval_target
                _, _, _, _, tgt_val_piaa_dataset, _, _ = load_data(
                    args_tgt, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)
            else:
                tgt_val_piaa_dataset = None
            src_mod.trainer_finetune(
                datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                num_attr, num_pt, tgt_val_piaa_dataset=tgt_val_piaa_dataset, tgt_genre=eval_target)
        inference_finetune(datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, eval_datasets_dict=eval_datasets_dict)
        for pth_file in [f for f in os.listdir(dirname) if f.endswith('_finetune.pth')]:
            os.remove(os.path.join(dirname, pth_file))
        print(f"Deleted temporary finetune model files from {dirname}")
    else:
        raise ValueError(f"Error: --piaa_mode must be 'PIAA_pretrain' or 'PIAA_finetune', got: {args.piaa_mode}")

    if args.is_log:
        wandb.finish()

if __name__ == '__main__':
    parser = parse_arguments(parse=False)
    parser.add_argument('--model_type', type=str, default='ICI', choices=['ICI', 'MIR'],
                        help='PIAA model architecture: ICI (Interaction-based) or MIR (MLP Interaction Regression)')
    import sys
    parser.set_defaults(lr=5e-6, batch_size=32, djdot_alpha=0.1, djdot_lambda_t=1)
    args = parser.parse_args()
    if args.piaa_mode == 'PIAA_pretrain':
        if not any(a.startswith('--mcd_lambda') for a in sys.argv[1:]):
            args.mcd_lambda = 0.1
    elif args.piaa_mode == 'PIAA_finetune':
        if not any(a.startswith('--lr') for a in sys.argv[1:]):
            args.lr = 1e-5
        if not any(a.startswith('--batch_size') for a in sys.argv[1:]):
            args.batch_size = 16

    ALL_GENRES = ['art', 'fashion', 'scenery']
    if args.genre == 'all':
        source_genres = list(ALL_GENRES)
        print(f"--genre all: running sources sequentially: {source_genres}")
    else:
        source_genres = [args.genre]
    base_da_method = args.da_method

    for source in source_genres:
        args_src = copy.deepcopy(args)
        args_src.genre = source
        if len(source_genres) > 1:
            print(f"\n{'@'*60}\n  Source genre: {source}\n{'@'*60}\n")

        if base_da_method and '-' not in base_da_method:
            target_genres = [g for g in ALL_GENRES if g != source]
            print(f"Bare --da_method '{base_da_method}': running targets sequentially: {target_genres}")
        else:
            target_genres = [None]

        for target in target_genres:
            args_outer = copy.deepcopy(args_src)
            if target is not None:
                args_outer.da_method = f'{base_da_method}-{target}'
                print(f"\n{'#'*60}\n  Target genre: {target}  (da_method={args_outer.da_method})\n{'#'*60}\n")

            if args_outer.dataset_ver.endswith('_all'):
                version_prefix = args_outer.dataset_ver[:-4]
                folds = discover_folds(args_outer.root_dir, version_prefix)
                if not folds:
                    raise ValueError(f"No fold directories found for version '{version_prefix}' in {os.path.join(args_outer.root_dir, 'split')}")
                print(f"Running all {len(folds)} folds sequentially: {folds}")
                for i, fold in enumerate(folds):
                    if i + 1 < args_outer.start_fold:
                        print(f"Skipping fold {i+1}/{len(folds)}: {fold} (start_fold={args_outer.start_fold})")
                        continue
                    print(f"\n{'='*60}")
                    print(f"  Fold {i+1}/{len(folds)}: {fold}")
                    print(f"{'='*60}\n")
                    args_fold = copy.deepcopy(args_outer)
                    args_fold.dataset_ver = fold
                    run_main(args_fold)
            else:
                models_base = 'models_pth'
                fold_dirs = sorted([
                    d for d in os.listdir(models_base)
                    if d.startswith(f'{args_outer.dataset_ver}_fold') and os.path.isdir(os.path.join(models_base, d))
                ]) if os.path.exists(models_base) else []

                if fold_dirs:
                    print(f"models_pth/{args_outer.dataset_ver}/ not found. Running fold structure: {fold_dirs}")
                    for i, fold in enumerate(fold_dirs):
                        if i + 1 < args_outer.start_fold:
                            print(f"Skipping fold {i+1}/{len(fold_dirs)}: {fold} (start_fold={args_outer.start_fold})")
                            continue
                        print(f"\n{'='*60}")
                        print(f"  Fold {i+1}/{len(fold_dirs)}: {fold}")
                        print(f"{'='*60}\n")
                        args_fold = copy.deepcopy(args_outer)
                        args_fold.dataset_ver = fold
                        run_main(args_fold)
                else:
                    run_main(args_outer)
