import math
import json
import re
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "exp"


def aggregate(args):
    """指定されたversionとgenreの各foldからJSONを集約し，全ユーザーの平均srocc/ndcgを出力する"""
    version = args.version
    genre = args.genre
    pattern = args.pattern
    method = args.method  # e.g., "ICI" (optional)
    min_id = args.min_id
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
        if min_id is not None:
            def _extract_id(p):
                m = re.search(r'-(\d+)[_.]', p.name)
                return int(m.group(1)) if m else -1
            matched_jsons = [p for p in matched_jsons if _extract_id(p) >= min_id]
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


def plot_quality(args):
    """被験者ごとの品質管理指標（p_mode, MAE, r_fast）をプロットする"""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import sys as _sys
    import json as _json

    # preprocessing モジュールを import して rawデータ処理ロジックを再利用
    _src = Path(__file__).resolve().parent
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    from preprocessing import make_user_csv, make_ratings_csv

    score_col = args.score_col
    domains = args.domains
    mad_multiplier = args.mad_multiplier
    outlier_method = args.outlier_method
    fast_user_thresh = args.fast_user_thresh
    min_rt_art_fashion = args.min_rt_art_fashion
    min_rt_scenery = args.min_rt_scenery
    raw_dir = Path(args.raw_dir)

    # ── rawデータから完了者ユーザー情報と排除前ratingsを再構築 ────────────────
    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as _tmpdir:
        _tmp_users = _os.path.join(_tmpdir, "users.csv")
        make_user_csv(str(raw_dir), _tmp_users)
        ratings_df = make_ratings_csv(
            annotation_path=str(raw_dir / "user-annotation-data_rows.csv"),
            finished_users_path=_tmp_users,
            rel_tasks_users_path=str(raw_dir / "annotation-tasks_rows.csv"),
            user_path=str(raw_dir / "user-data_rows.csv"),
            url_filename_path=str(raw_dir / "url_filename_rows.csv"),
        )
        finished_uuids = set(pd.read_csv(_tmp_users)["uuid"].astype(str).unique())

    print(f"Loaded {len(finished_uuids)} finished UUIDs from raw data.")
    print(f"Reconstructed ratings: {len(ratings_df)} rows.")

    # ── r_fast 用: 生アノテーションデータから Time を取り出す ─────────────────
    _genre_map = {"アート作品": "art", "ファッション": "fashion", "映像": "scenery"}

    def _parse_data(val):
        if isinstance(val, dict):
            return val
        try:
            return _json.loads(val)
        except Exception:
            try:
                import ast as _ast
                return _ast.literal_eval(val)
            except Exception:
                return None

    raw_ann_df = pd.read_csv(str(raw_dir / "user-annotation-data_rows.csv"))
    raw_ann_df = raw_ann_df[raw_ann_df["uuid"].astype(str).isin(finished_uuids)]

    raw_time_rows = []
    for _, row in raw_ann_df.iterrows():
        d = _parse_data(row.get("data"))
        if not isinstance(d, dict):
            continue
        genre_jp = d.get("genre")
        genre_en = _genre_map.get(genre_jp)
        if genre_en is None:
            continue
        results = d.get("result", [])
        uuid = str(row.get("uuid", ""))
        for result in results:
            if isinstance(result, (list, tuple)) and len(result) > 10:
                raw_time_rows.append({"uuid": uuid, "genre": genre_en, "Time": result[10]})
    raw_time_df = pd.DataFrame(raw_time_rows)

    def _threshold_high(values: np.ndarray) -> float:
        if outlier_method == "mad":
            med = np.median(values)
            spread = np.median(np.abs(values - med))
            return float(med + mad_multiplier * spread)
        mu = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return mu + mad_multiplier * sd

    # ── 1. p_mode ──────────────────────────────────────────────────────────────
    p_mode_data: dict = {}
    for domain in domains:
        dom_df = ratings_df[ratings_df["genre"] == domain]
        if dom_df.empty or score_col not in dom_df.columns:
            continue
        pm: dict = {}
        for uid, udf in dom_df.groupby(dom_df["uuid"].astype(str)):
            scores = udf[score_col].dropna()
            if len(scores) > 0:
                pm[uid] = scores.value_counts().iloc[0] / len(scores)
        p_mode_data[domain] = pm

    # ── 2. MAE (retest) ────────────────────────────────────────────────────────
    mae_data: dict = {}
    for domain in domains:
        dom_df = ratings_df[ratings_df["genre"] == domain]
        if dom_df.empty or score_col not in dom_df.columns:
            continue
        mae: dict = {}
        for uid, udf in dom_df.groupby(dom_df["uuid"].astype(str)):
            dup_mask = udf.duplicated("sample_id", keep=False)
            dup_samples = udf.loc[dup_mask, "sample_id"].unique()
            r1_list, r2_list = [], []
            for sid in dup_samples:
                pair = udf[udf["sample_id"] == sid][score_col].dropna().values
                if len(pair) >= 2:
                    r1_list.append(pair[0])
                    r2_list.append(pair[1])
            if len(r1_list) >= 3:
                mae[uid] = float(np.mean(np.abs(np.array(r1_list, dtype=float) - np.array(r2_list, dtype=float))))
        mae_data[domain] = mae

    # ── 3. r_fast (rt_prop) ────────────────────────────────────────────────────
    # 生アノテーションデータ (raw_time_df) から計算
    r_fast_data: dict = {}
    if not raw_time_df.empty:
        raw_time_df["Time"] = pd.to_numeric(raw_time_df["Time"], errors="coerce")
        for domain in domains:
            thresh = min_rt_scenery if domain == "scenery" else min_rt_art_fashion
            dom_df = raw_time_df[raw_time_df["genre"] == domain]
            if dom_df.empty:
                continue
            rf: dict = {}
            for uid, udf in dom_df.groupby(dom_df["uuid"].astype(str)):
                valid = udf["Time"].dropna()
                if len(valid) > 0:
                    rf[uid] = float((valid < thresh).sum() / len(valid))
            r_fast_data[domain] = rf

    # ── Determine per-metric excluded UUIDs ───────────────────────────────────
    excluded_p_mode: set = set()
    for domain, pm in p_mode_data.items():
        if len(pm) < 2:
            continue
        uids = np.array(list(pm.keys()))
        vals = np.array(list(pm.values()), dtype=float)
        excluded_p_mode.update(uids[vals > _threshold_high(vals)].tolist())

    excluded_mae: set = set()
    for domain, mae in mae_data.items():
        if len(mae) < 2:
            continue
        uids = np.array(list(mae.keys()))
        vals = np.array(list(mae.values()), dtype=float)
        excluded_mae.update(uids[vals > _threshold_high(vals)].tolist())

    excluded_r_fast: set = set()
    for domain, rf in r_fast_data.items():
        for uid, val in rf.items():
            if val > fast_user_thresh:
                excluded_r_fast.add(uid)

    excluded_all = excluded_p_mode | excluded_mae | excluded_r_fast
    print(
        f"Excluded: p_mode={len(excluded_p_mode)}, MAE={len(excluded_mae)}, "
        f"r_fast={len(excluded_r_fast)}, total={len(excluded_all)}"
    )

    # ── per-domain excluded sets（そのドメイン・指標で閾値超えのみ赤）────────
    def _domain_excluded(data_dict, dynamic_thresh):
        """ドメインごとに閾値超えUUIDのsetを返す"""
        result = {}
        for domain, dd in data_dict.items():
            if len(dd) < 2:
                result[domain] = set()
                continue
            uids = np.array(list(dd.keys()))
            vals = np.array(list(dd.values()), dtype=float)
            thr = _threshold_high(vals) if dynamic_thresh else fast_user_thresh
            result[domain] = set(uids[vals > thr].tolist())
        return result

    domain_excl_p_mode = _domain_excluded(p_mode_data, True)
    domain_excl_mae    = _domain_excluded(mae_data,    True)
    domain_excl_r_fast = _domain_excluded(r_fast_data, False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(0)
    jitter_width = 0.12

    metrics = [
        ("p_mode", p_mode_data, "Mode proportion", True,  domain_excl_p_mode),
        ("mae",    mae_data,    "MAE",              True,  domain_excl_mae),
        ("r_fast", r_fast_data, "Fast response rate", False, domain_excl_r_fast),
    ]

    output_base = Path(args.output)
    stem = output_base.stem
    suffix = output_base.suffix
    output_base.parent.mkdir(parents=True, exist_ok=True)

    for metric_key, data_dict, metric_label, dynamic_thresh, domain_excl in metrics:
        active = [d for d in domains if data_dict.get(d)]
        if not active:
            continue

        fig, ax = plt.subplots(figsize=(1.3 * len(active) + 0.6, 4.0))
        ax.set_ylabel(metric_label, fontsize=13)
        ax.tick_params(axis="both", labelsize=12)

        x_positions = {d: i for i, d in enumerate(active)}

        for domain in active:
            domain_data = data_dict[domain]
            uids = np.array(list(domain_data.keys()))
            vals = np.array(list(domain_data.values()), dtype=float)
            excl_here = domain_excl.get(domain, set())

            x_base = x_positions[domain]
            x_jitter = x_base + rng.uniform(-jitter_width, jitter_width, size=len(vals))

            # 非排除（黒）→ 排除（赤）の順に描いて赤を前面に
            mask_excl = np.array([uid in excl_here for uid in uids])
            ax.scatter(x_jitter[~mask_excl], vals[~mask_excl],
                       c="black", s=18, alpha=0.5, linewidths=0, zorder=2)
            if mask_excl.any():
                ax.scatter(x_jitter[mask_excl], vals[mask_excl],
                           c="red", s=22, alpha=0.85, linewidths=0, zorder=3)

            # 箱ひげ図（外れ値非表示、黒）
            ax.boxplot(
                vals, positions=[x_base], widths=0.5,
                showfliers=False,
                patch_artist=False,
                boxprops=dict(color="black", linewidth=1.2),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color="black", linewidth=1.0),
                capprops=dict(color="black", linewidth=1.0),
                zorder=1,
            )

            # 排除水準（赤破線）
            thr = _threshold_high(vals) if dynamic_thresh else fast_user_thresh
            ax.plot(
                [x_base - 0.35, x_base + 0.35], [thr, thr],
                color="red", linestyle="--", linewidth=1.5, zorder=4,
            )

        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(list(x_positions.keys()), fontsize=12)
        ax.set_xlim(-0.6, len(active) - 0.4)

        all_vals = [v for d in active for v in data_dict[d].values()]
        all_thrs = [
            _threshold_high(np.array(list(data_dict[d].values()), dtype=float))
            if dynamic_thresh else fast_user_thresh
            for d in active if data_dict.get(d)
        ]
        y_max = max(max(all_vals), max(all_thrs)) if all_vals else fast_user_thresh
        ax.set_ylim(bottom=0, top=y_max * 1.15)

        # レジェンドはMAEのみ表示
        if metric_key == "mae":
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                       markersize=7, alpha=0.6, label="Annotator"),
                Line2D([0], [0], color="red", linestyle="--", linewidth=1.5,
                       label="Exclusion criterion"),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc="upper left",
                      framealpha=0.8)

        plt.tight_layout()
        out_path = output_base.parent / f"{stem}_{metric_key}{suffix}"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


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
        "--min-id",
        type=int,
        default=None,
        dest="min_id",
        help="Minimum run ID to include (e.g., 61 filters to files with ID >= 61, like 'name-61_pretrain.json')",
    )
    agg_parser.add_argument(
        "--reports_dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Path to reports/exp directory",
    )

    # Subcommand: plot_quality
    qc_parser = subparsers.add_parser(
        "plot_quality",
        help="Plot per-subject quality control metrics (p_mode, MAE, r_fast)",
    )
    qc_parser.add_argument(
        "ratings_csv",
        help="(unused for metric computation; kept for compatibility) Path to ratings CSV",
    )
    qc_parser.add_argument(
        "--raw-dir",
        default=str(Path.home() / "proj-xpass" / "data" / "raw"),
        dest="raw_dir",
        help=(
            "Path to raw data directory containing user-annotation-data_rows.csv etc. "
            "Default: ~/proj-xpass/data/raw"
        ),
    )
    qc_parser.add_argument(
        "--score-col",
        default="Aesthetic",
        help="Score column name used for p_mode and MAE (default: Aesthetic)",
    )
    qc_parser.add_argument(
        "--domains",
        nargs="+",
        default=["art", "fashion", "scenery"],
        help="Genres to evaluate (default: art fashion scenery)",
    )
    qc_parser.add_argument(
        "--min-rt-art-fashion",
        type=float,
        default=10.0,
        dest="min_rt_art_fashion",
        help="Fast-response threshold (s) for art/fashion (default: 10)",
    )
    qc_parser.add_argument(
        "--min-rt-scenery",
        type=float,
        default=30.0,
        dest="min_rt_scenery",
        help="Fast-response threshold (s) for scenery (default: 30)",
    )
    qc_parser.add_argument(
        "--fast-user-thresh",
        type=float,
        default=0.2,
        dest="fast_user_thresh",
        help="Max allowed proportion of fast responses per user (default: 0.2)",
    )
    qc_parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=2.5,
        dest="mad_multiplier",
        help="Multiplier k for outlier threshold (default: 2.5)",
    )
    qc_parser.add_argument(
        "--outlier-method",
        choices=["mad", "std"],
        default="std",
        dest="outlier_method",
        help="Outlier detection method: mad or std (default: std)",
    )
    qc_parser.add_argument(
        "-o", "--output",
        default="quality_check.png",
        help="Output figure path (default: quality_check.png)",
    )

    args = parser.parse_args()

    if args.command == 'aggregate':
        aggregate(args)
    elif args.command == 'plot_quality':
        plot_quality(args)
    else:
        parser.print_help()
