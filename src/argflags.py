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
    parser.add_argument('--wandb_project', type=str, default='XPASS', help='wandb project name')

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--no_save_model', action='store_true', default=False,
                        help='If set, keep best model in memory instead of saving to disk')

    parser.add_argument('--giaa_mode', action='store_true', default=False,
                        help='Use GIAA-only split files (train/val/test_images_GIAA.txt). '
                             'No PIAA files required. Evaluates on GIAA test set.')

    # Inference-only mode (skip training, load existing .pth and run inference)
    parser.add_argument('--inference_only', action='store_true', default=False,
                        help='Skip training and only run inference (save JSON). '
                             'Auto-discovers a .pth matching {domain_tag}_{method_tag}_NIMA_*.pth '
                             'under models_pth/{dataset_ver}/{domain_tag}/. '
                             'Use --inference_pattern to narrow down if multiple matches exist.')
    parser.add_argument('--inference_pattern', type=str, default=None,
                        help='Additional substring pattern used with --inference_only to filter '
                             'candidate .pth files (e.g. run name like "easy-plasma-7").')

    # Domain Adaptation
    parser.add_argument('--da_method', type=str, default=None,
                        help='Domain adaptation method and target domain. '
                             'Format: METHOD-target (e.g., DANN-fashion, DJDOT-scenery). '
                             'Omit to run source-only training.')
    parser.add_argument('--eval_target', type=str, default=None,
                        help='Target genre to monitor during source-only training (e.g., fashion). '
                             'Records target val EMD without doing domain adaptation.')
    # DA λ schedule hyperparameters (shared by DANN, DeepCORAL, ...)
    parser.add_argument('--da_schedule_epochs', type=int, default=50,
                        help='[DA] λ schedule: number of epochs over which λ reaches ~1.0. '
                             'Converted internally to total_steps = da_schedule_epochs × (data_size / batch_size).')
    parser.add_argument('--da_gamma', type=float, default=10.0,
                        help='[DA] λ schedule: sharpness of the sigmoid (Ganin et al.)')
    # DeepJDOT-specific hyperparameters
    parser.add_argument('--djdot_alpha', type=float, default=0.1,
                        help='[DJDOT] Weight for feature alignment term (L2 feature distance).')
    parser.add_argument('--djdot_lambda_t', type=float, default=0.1,
                        help='[DJDOT] Weight for label alignment term (EMD label cost).')
    # JUMBOT-specific hyperparameters (Unbalanced minibatch OT; aligns with DeepJDOT)
    #   eta1/eta2 are the feature/label weights -> matched to DeepJDOT alpha/lambda_t
    #   (GIAA: 0.1/0.1). eta3=1.0 keeps the effective loss (eta3*eta1, eta3*eta2)
    #   identical to DeepJDOT while staying a round value inside the design-doc range.
    parser.add_argument('--jumbot_eta1', type=float, default=0.1,
                        help='[JUMBOT] Weight of the feature-distance term (L2^2) in the OT cost matrix.')
    parser.add_argument('--jumbot_eta2', type=float, default=0.1,
                        help='[JUMBOT] Weight of the label-cost term (GIAA: EMD / PIAA: squared error) in the OT cost matrix.')
    parser.add_argument('--jumbot_eta3', type=float, default=1.0,
                        help='[JUMBOT] Scale of the transport loss <pi, C> added to the source task loss.')
    parser.add_argument('--jumbot_tau', type=float, default=0.5,
                        help='[JUMBOT] Marginal KL penalty (reg_m) of the Unbalanced OT. Smaller = looser marginals.')
    parser.add_argument('--jumbot_epsilon', type=float, default=0.1,
                        help='[JUMBOT] Entropic regularization (reg) of the Sinkhorn Unbalanced OT.')
    # MCD-specific hyperparameters
    parser.add_argument('--mcd_lambda', type=float, default=10.0,
                        help='[MCD] Weight for the discrepancy loss in Step B (lambda in L_s - lambda * L_adv).')
    parser.add_argument('--mcd_n_steps', type=int, default=4,
                        help='[MCD] Number of Step C (generator update) repetitions per batch.')
    # DAREGRAM-specific hyperparameters
    parser.add_argument('--daregram_alpha_cos', type=float, default=0.01,
                        help='[DAREGRAM] Weight for angle alignment loss L_cos.')
    parser.add_argument('--daregram_gamma_scale', type=float, default=0.01,
                        help='[DAREGRAM] Weight for scale alignment loss L_scale.')
    parser.add_argument('--daregram_T', type=float, default=0.95,
                        help='[DAREGRAM] Cumulative eigenvalue threshold for truncated pseudo-inverse.')
    parser.add_argument('--nima_da_method', type=str, default=None,
                        help='[DAREGRAM/UGAFEAT/DEEPCORAL/RSD] DA method whose pretrained NIMA to load for PIAA_pretrain. '
                             'E.g. "source_only" (load NIMA from models_pth/{ver}/{genre}/), '
                             '"DANN"/"MCD"/"DJDOT"/"DEEPCORAL"/"CDAN"/"ALDA" (load from models_pth/{ver}/{src2tgt}/, filtered by method). '
                             'Only meaningful for methods that have no GIAA-trained NIMA of their own.')
    # DeepCORAL-specific hyperparameters
    parser.add_argument('--coral_lambda', type=float, default=1.0,
                        help='[DEEPCORAL] Fixed weight for the CORAL alignment loss (no schedule, per design 4.5).')
    # CDAN-specific hyperparameters
    parser.add_argument('--cdan_sigma', type=float, default=1.0,
                        help='[CDAN] Width of the Gaussian Soft Ordinal Distribution used as conditioning '
                             'vector g in PIAA pretrain/finetune (per design 7.9). Fixed (not scheduled).')
    # UGAFEAT-specific hyperparameters
    parser.add_argument('--ugafeat_mmd_num', type=int, default=5,
                        help='[UGAFEAT] Number of bandwidths in the multi-bandwidth RBF MMD kernel.')
    parser.add_argument('--ugafeat_mmd_mul', type=float, default=2.0,
                        help='[UGAFEAT] Geometric multiplier between MMD bandwidths.')
    parser.add_argument('--ugafeat_lambda_evi', type=float, default=1.0,
                        help='[UGAFEAT] Weight of the DER regularization term (λ_EVI).')
    parser.add_argument('--ugafeat_lambda_align', type=float, default=1.0,
                        help='[UGAFEAT] Fixed weight for the MMD alignment loss (no schedule, per design 8.7).')
    # ALDA-specific hyperparameters
    parser.add_argument('--alda_sigma', type=float, default=1.0,
                        help='[ALDA] Width of the Gaussian Soft Ordinal Distribution used as p_t '
                             'for PIAA L_T (per design 6.11). Fixed (not scheduled). GIAA uses '
                             'softmax(logit_tgt) directly and ignores this flag.')
    parser.add_argument('--alda_threshold', type=float, default=0.2,
                        help='[ALDA] Confidence threshold δ for target pseudo-label filtering in L_T '
                             '(per design 6.5). Default 0.2 lets L_T contribute from early epochs '
                             '(GIAA max(p_t) starts ≈0.2 and converges ≈0.6, so 0.2 keeps the '
                             'self-reinforcing L_T → sharpen p_t loop alive). Higher values can '
                             'starve L_T early on (chicken-and-egg). Paper\'s δ=0.9 is unsuitable '
                             'for K=7 / EMD-trained NIMA / Gaussian soft labels.')
    # RSD-specific hyperparameters
    parser.add_argument('--rsd_beta', type=float, default=0.01,
                        help='[RSD] Weight β for the RSD loss (sum of sin of principal angles).')
    parser.add_argument('--rsd_gamma', type=float, default=1e-5,
                        help='[RSD] Weight γ for the BMP loss (basis mismatch penalization).')
    parser.add_argument('--rsd_eps', type=float, default=1e-8,
                        help='[RSD] Numerical stabilization ε for sqrt(1 - cos²θ).')

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
