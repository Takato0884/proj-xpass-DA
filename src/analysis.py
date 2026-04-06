import math
import json
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "exp"


def aggregate(args):
    """指定されたversionとgenreの各foldからJSONを集約し，全ユーザーの平均srocc/ndcgを出力する"""
    version = args.version
    genre = args.genre
    pattern = args.pattern
    method = args.method  # e.g., "ICI" (optional)
    reports_dir = Path(args.reports_dir)

    # version に該当する fold ディレクトリを検索
    fold_dirs = sorted(reports_dir.glob(f"{version}_fold*"))
    if not fold_dirs:
        print(
            f"Error: No fold directories found for version '{version}' in {reports_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 特定の fold のみに絞り込む
    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [
            d for d in fold_dirs
            if int(d.name.split("fold")[-1]) in fold_set
        ]
        if not fold_dirs:
            print(
                f"Error: No matching fold directories for folds {args.folds}",
                file=sys.stderr,
            )
            sys.exit(1)

    # genre が "art-scenery" のようなクロスドメインの場合、サブジャンルに分割
    sub_genres = genre.split("-")

    # サブジャンルごとに集約用辞書を用意
    all_user_srocc = {sg: {} for sg in sub_genres}
    all_user_ndcg = {sg: {} for sg in sub_genres}
    all_user_ccc = {sg: {} for sg in sub_genres}

    # クロスドメイン集約用: {target_genre: {user_id: {'srocc': [], 'ndcg': [], 'ccc': []}}}
    cd_user_srocc = {}
    cd_user_ndcg = {}
    cd_user_ccc = {}

    for fold_dir in fold_dirs:
        genre_dir = fold_dir / genre
        if not genre_dir.is_dir():
            print(f"Error: Genre directory not found: {genre_dir}", file=sys.stderr)
            sys.exit(1)

        # pattern に一致する JSON を検索（各 fold/genre に対して1つだけ存在する想定）
        if method and pattern:
            glob_pattern = f"*{method}*{pattern}*.json"
        elif method:
            glob_pattern = f"*{method}*.json"
        elif pattern:
            glob_pattern = f"*{pattern}*.json"
        else:
            glob_pattern = "*.json"
        matched_jsons = list(genre_dir.glob(glob_pattern))
        if len(matched_jsons) == 0:
            print(f"Error: No JSON matching '{glob_pattern}' found in {genre_dir}", file=sys.stderr)
            sys.exit(1)
        if len(matched_jsons) > 1:
            print(
                f"Error: Multiple JSONs matching '{glob_pattern}' found in {genre_dir}: {[f.name for f in matched_jsons]}",
                file=sys.stderr,
            )
            sys.exit(1)

        json_path = matched_jsons[0]
        with open(json_path) as f:
            data = json.load(f)

        per_user = data.get("per_user_metrics", {})
        for user_id, metrics in per_user.items():
            for sg in sub_genres:
                genre_metrics = metrics.get(sg, {})
                srocc = genre_metrics.get("srocc")
                ndcg = genre_metrics.get("ndcg@10")
                ccc = genre_metrics.get("ccc")
                if srocc is not None:
                    all_user_srocc[sg].setdefault(user_id, []).append(srocc)
                if ndcg is not None:
                    all_user_ndcg[sg].setdefault(user_id, []).append(ndcg)
                if ccc is not None:
                    all_user_ccc[sg].setdefault(user_id, []).append(ccc)

        # クロスドメイン結果の収集
        cross_domain = data.get("cross_domain_metrics", {})
        for target_genre, cd_data in cross_domain.items():
            if target_genre not in cd_user_srocc:
                cd_user_srocc[target_genre] = {}
                cd_user_ndcg[target_genre] = {}
                cd_user_ccc[target_genre] = {}
            per_user_cd = cd_data.get("per_user", {})
            for user_id, cd_metrics in per_user_cd.items():
                srocc = cd_metrics.get("srocc")
                ndcg = cd_metrics.get("ndcg@10")
                ccc = cd_metrics.get("ccc")
                if srocc is not None:
                    cd_user_srocc[target_genre].setdefault(user_id, []).append(srocc)
                if ndcg is not None:
                    cd_user_ndcg[target_genre].setdefault(user_id, []).append(ndcg)
                if ccc is not None:
                    cd_user_ccc[target_genre].setdefault(user_id, []).append(ccc)

        print(
            f"  Loaded: {json_path.relative_to(reports_dir)} ({len(per_user)} users)"
        )

    if not any(all_user_srocc[sg] for sg in sub_genres):
        print("Error: No user metrics found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Aggregated Results ({version}, {genre}, pattern='{pattern}') ===")
    print(f"  Folds:         {len(fold_dirs)}")

    for sg in sub_genres:
        if not all_user_srocc[sg]:
            continue

        # ユーザーごとの fold 平均を算出
        user_avg_srocc = [
            sum(vals) / len(vals) for vals in all_user_srocc[sg].values()
        ]
        user_avg_ndcg = [
            sum(vals) / len(vals) for vals in all_user_ndcg[sg].values()
        ]
        user_avg_ccc = [
            sum(vals) / len(vals) for vals in all_user_ccc[sg].values()
        ]

        avg_srocc = sum(user_avg_srocc) / len(user_avg_srocc)
        avg_ndcg = sum(user_avg_ndcg) / len(user_avg_ndcg)
        avg_ccc = sum(user_avg_ccc) / len(user_avg_ccc) if user_avg_ccc else None

        std_srocc = math.sqrt(
            sum((x - avg_srocc) ** 2 for x in user_avg_srocc) / len(user_avg_srocc)
        )
        std_ndcg = math.sqrt(
            sum((x - avg_ndcg) ** 2 for x in user_avg_ndcg) / len(user_avg_ndcg)
        )
        std_ccc = math.sqrt(
            sum((x - avg_ccc) ** 2 for x in user_avg_ccc) / len(user_avg_ccc)
        ) if user_avg_ccc else None

        print(f"  [{sg}]")
        print(f"    Total users:   {len(all_user_srocc[sg])}")
        print(f"    Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
        print(f"    Average NDCG@10: {avg_ndcg:.6f} (std: {std_ndcg:.6f})")
        if avg_ccc is not None:
            print(f"    Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")

    # クロスドメイン結果の出力
    if cd_user_srocc:
        print(f"\n  --- Cross-Domain (head average) ---")
        for target_genre in sorted(cd_user_srocc.keys()):
            if not cd_user_srocc[target_genre]:
                continue

            user_avg_srocc = [
                sum(vals) / len(vals) for vals in cd_user_srocc[target_genre].values()
            ]
            user_avg_ndcg = [
                sum(vals) / len(vals) for vals in cd_user_ndcg[target_genre].values()
            ]
            user_avg_ccc = [
                sum(vals) / len(vals) for vals in cd_user_ccc[target_genre].values()
            ]

            avg_srocc = sum(user_avg_srocc) / len(user_avg_srocc)
            avg_ndcg = sum(user_avg_ndcg) / len(user_avg_ndcg)
            avg_ccc = sum(user_avg_ccc) / len(user_avg_ccc) if user_avg_ccc else None

            std_srocc = math.sqrt(
                sum((x - avg_srocc) ** 2 for x in user_avg_srocc) / len(user_avg_srocc)
            )
            std_ndcg = math.sqrt(
                sum((x - avg_ndcg) ** 2 for x in user_avg_ndcg) / len(user_avg_ndcg)
            )
            std_ccc = math.sqrt(
                sum((x - avg_ccc) ** 2 for x in user_avg_ccc) / len(user_avg_ccc)
            ) if user_avg_ccc else None

            print(f"  [{genre} -> {target_genre}]")
            print(f"    Total users:   {len(cd_user_srocc[target_genre])}")
            print(f"    Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
            print(f"    Average NDCG@10: {avg_ndcg:.6f} (std: {std_ndcg:.6f})")
            if avg_ccc is not None:
                print(f"    Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analysis utilities for XPass project',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subcommand: aggregate
    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate results across folds",
    )
    agg_parser.add_argument(
        "--version", type=str, required=True, help="Dataset version (e.g., v3)"
    )
    agg_parser.add_argument(
        "--genre", type=str, required=True, help="Genre (e.g., art, scenery)"
    )
    agg_parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Glob pattern to match JSON files. e.g., pretrain, finetune",
    )
    agg_parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method name to filter JSON files (e.g., ICI). Used when multiple methods match the pattern.",
    )
    agg_parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold indices to aggregate (e.g., --folds 0 2 4). If omitted, all folds are used.",
    )
    agg_parser.add_argument(
        "--reports_dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Path to reports/exp directory",
    )

    args = parser.parse_args()

    if args.command == 'aggregate':
        aggregate(args)
    else:
        parser.print_help()
