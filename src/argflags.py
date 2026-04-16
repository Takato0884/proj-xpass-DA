import argparse
import os


def parse_arguments(parse=True):
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data splitting')

    parser.add_argument('--num_workers', type=int, default=4)
    # Dataset version identifier (string)
    parser.add_argument('--dataset_ver', type=str, default='v1_all', help='Dataset version (e.g., v1) used to locate split files and tag outputs')
    parser.add_argument('--start_fold', type=int, default=1, help='Fold number to start from (1-indexed). Use to resume from a specific fold when dataset_ver ends with _all.')
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)
    parser.add_argument('--genre', type=str, required=True, help='Dataset genre (e.g., art, fashion, scenery)')

    parser.add_argument('--backbone', type=str, default='clip_vit_b16',
                        choices=['resnet50', 'i3d', 'vit_b_16', 'clip_rn50', 'clip_vit_b16'],
                        help='Backbone architecture for feature extraction')
    parser.add_argument('--use_video', action='store_true', help='Use video (I3D) for scenery genre instead of images (ResNet50)')
    parser.add_argument('--root_dir', type=str, default='/home/hayashi0884/proj-xpass-DA/data')
    parser.add_argument('--piaa_mode', type=str, default='PIAA_pretrain')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--no_save_model', action='store_true', default=False,
                        help='If set, keep best model in memory instead of saving to disk')

    # Domain Adaptation
    parser.add_argument('--da_method', type=str, default=None,
                        help='Domain adaptation method and target domain. '
                             'Format: METHOD-target (e.g., DANN-fashion, DJDOT-scenery). '
                             'Omit to run source-only training.')
    parser.add_argument('--eval_target', type=str, default=None,
                        help='Target genre to monitor during source-only training (e.g., fashion). '
                             'Records target val EMD without doing domain adaptation.')
    # DANN-specific hyperparameters
    parser.add_argument('--dann_epochs', type=int, default=50,
                        help='[DANN] λ schedule: number of epochs over which λ reaches ~1.0. '
                             'Converted internally to total_steps = dann_epochs × (data_size / batch_size).')
    parser.add_argument('--dann_gamma', type=float, default=10.0,
                        help='[DANN] λ schedule: sharpness of the sigmoid (Ganin et al.)')
    # DeepJDOT-specific hyperparameters
    parser.add_argument('--djdot_alpha', type=float, default=0.001,
                        help='[DJDOT] Weight for feature alignment term (L2 feature distance).')
    parser.add_argument('--djdot_lambda_t', type=float, default=0.0001,
                        help='[DJDOT] Weight for label alignment term (EMD label cost).')
    # MCD-specific hyperparameters
    parser.add_argument('--mcd_lambda', type=float, default=1.0,
                        help='[MCD] Weight for the discrepancy loss in Step B (lambda in L_s - lambda * L_adv).')
    parser.add_argument('--mcd_n_steps', type=int, default=4,
                        help='[MCD] Number of Step C (generator update) repetitions per batch.')

    if parse:
        args = parser.parse_args()
        # Auto-configure backbone based on use_video flag
        if args.use_video and args.backbone == 'resnet50':
            args.backbone = 'i3d'
        return args
    else:
        return parser

def model_dir(args):
    # Model directory scoped by dataset version instead of fold
    dirname = 'models_pth'
    dirname = os.path.join(dirname, f'{args.dataset_ver}')
    return dirname

def wandb_tags(args):
    tags = [
        f"dataset_version={args.dataset_ver}",
        f"genre={args.genre}",
        f"backbone={args.backbone}",
        f"learning_rate: {args.lr}",
        f"batch_size: {args.batch_size}"
        ]

    if hasattr(args, 'model_type'):
        tags += [f"model_type={args.model_type}"]
    if args.dropout > 0.:
        tags += [f"dropout={args.dropout}"]
    return tags
