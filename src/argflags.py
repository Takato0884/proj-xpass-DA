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

    parser.add_argument('--use_cross_eval', action='store_true', default=True,
                        help='Enable cross-domain evaluation on all genres not in --genre. '
                             'Available genres: art, fashion, scenery.')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--use_backbone_image', action='store_true', default=True,
                        help='Use image backbone features (projected to input_dim) as additional image landscape features in the interaction component')

    # Batching strategy
    parser.add_argument('--user_grouped_batch', action='store_true', default=False,
                        help='Pretrain: group each batch by user (n_users=batch_size//32, 32 samples/user). Default: standard random batching.')

    # Loss function
    parser.add_argument('--loss_type', type=str, default='rmse',
                        choices=['rmse', 'ccc', 'ccc+rmse'],
                        help='Training loss: rmse, ccc, or ccc+rmse')
    parser.add_argument('--ccc_weight', type=float, default=0.5,
                        help='Weight for CCC term when loss_type=ccc+rmse. Loss = ccc_weight*(1-CCC) + RMSE')

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

    if args.dropout > 0.:
        tags += [f"dropout={args.dropout}"]
    if hasattr(args, 'use_backbone_image') and args.use_backbone_image:
        tags += ["use_backbone_image"]
    return tags
