import math
import json
import re
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "exp"


def _spearman(x, y):
    """Spearman rank correlation (handles ties via average rank)."""
    n = len(x)

    def _rank(a):
        order = sorted(range(n), key=lambda i: a[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and a[order[j]] == a[order[i]]:
                j += 1
            avg = (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
            i = j
        return ranks

    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mx) ** 2 for i in range(n))
        * sum((ry[i] - my) ** 2 for i in range(n))
    )
    return num / (den + 1e-10)


def _ndcg_at_k(true_scores, pred_scores, k=10):
    """NDCG@k with exponential gain (2^rel - 1), matching sklearn default."""
    n = len(true_scores)
    k = min(k, n)
    order = sorted(range(n), key=lambda i: pred_scores[i], reverse=True)
    dcg = sum(
        (2.0 ** true_scores[order[i]] - 1.0) / math.log2(i + 2)
        for i in range(k)
    )
    ideal = sorted(range(n), key=lambda i: true_scores[i], reverse=True)
    idcg = sum(
        (2.0 ** true_scores[ideal[i]] - 1.0) / math.log2(i + 2)
        for i in range(k)
    )
    return dcg / idcg if idcg > 0.0 else 0.0


def _aggregate_model(args, model_name: str):
    """LLMモデル（Claude/Gemini/GPT）のPIAA評価。

    {genre}_piaa_results*.json の per-user 予測（ratings: [{user_id, pred_score}, ...]）を
    用いて，他のPIAA手法（NIMA/ICI/MIR 等）と同じプロトコルで指標を計算する:

      1. 各 fold の test_PIAA.txt で (user_id, sample_file) 対ごとに予測/GT を取得
      2. ユーザー単位で SROCC / NDCG@10 / MAE / CCC を計算
      3. ユーザーごとに fold 平均 → 全ユーザー平均と標準偏差を出力

    JSON に per-user の ratings が無く pred_dist のみある場合は，従来の
    zero-shot (期待値) フォールバックを使う。
    """
    import csv

    version = args.version
    genre = args.genre

    data_dir = Path(getattr(args, "data_dir", None) or
                    Path(__file__).resolve().parent.parent / "data")

    model_dir = Path(__file__).resolve().parent.parent / "reports" / "exp" / model_name
    # PIAA JSON (per-user ratings)が第一候補. 無ければ GIAA JSON (pred_distのみ) を
    # zero-shotフォールバックとして使う (後で警告を出す).
    matched = list(model_dir.glob(f"{genre}_piaa_results*.json"))
    if not matched:
        matched = list(model_dir.glob(f"{genre}_results*.json"))
    if not matched:
        matched = list(model_dir.glob(f"{genre}_giaa_results*.json"))
    if not matched:
        print(
            f"Error: {model_name} results not found in {model_dir} "
            f"(pattern: {genre}_piaa_results*.json / {genre}_giaa_results*.json)",
            file=sys.stderr,
        )
        sys.exit(1)
    model_json = matched[0]

    with open(model_json) as f:
        llm_data = json.load(f)

    # per-user 予測があれば PIAA モード，無ければ pred_dist 期待値で zero-shot
    per_user_pred = {}       # {(uid_str, sample_file): pred_score}
    per_user_pred_stem = {}  # {(uid_str, stem): pred_score}  cross-ext マッチ用
    fallback_pred = {}       # {sample_file: expected_score}
    fallback_pred_stem = {}  # {stem: expected_score}
    n_user_preds = 0
    for entry in llm_data["per_sample"]:
        sf = entry["sample_file"]
        stem = Path(sf).stem
        ratings = entry.get("ratings")
        if ratings:
            for r in ratings:
                uid = str(r["user_id"])
                p = float(r["pred_score"])
                per_user_pred[(uid, sf)] = p
                per_user_pred_stem[(uid, stem)] = p
                n_user_preds += 1
        dist = entry.get("pred_dist")
        if dist:
            e = sum(i * p for i, p in enumerate(dist))
            fallback_pred[sf] = e
            fallback_pred_stem[stem] = e

    piaa_mode = n_user_preds > 0
    if piaa_mode:
        print(
            f"Loaded {n_user_preds} per-user {model_name} predictions "
            f"over {len(llm_data['per_sample'])} samples  (genre='{genre}')"
        )
    else:
        if not fallback_pred:
            print(
                f"Error: {model_json.name} has neither per-user ratings nor pred_dist",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"[WARN] No per-user ratings in {model_json.name}; "
            f"falling back to zero-shot (pred_dist expected value). "
            f"Note: this is NOT a fair PIAA comparison — "
            f"generate {genre}_piaa_results.json for per-user predictions.",
            file=sys.stderr,
        )
        print(
            f"Loaded {len(fallback_pred)} {model_name} pred_dist predictions "
            f"(zero-shot fallback)  (genre='{genre}')"
        )

    def _lookup_pred(uid, sf):
        if piaa_mode:
            if (uid, sf) in per_user_pred:
                return per_user_pred[(uid, sf)]
            stem = Path(sf).stem
            return per_user_pred_stem.get((uid, stem))
        if sf in fallback_pred:
            return fallback_pred[sf]
        return fallback_pred_stem.get(Path(sf).stem)

    ratings_path = data_dir / "maked" / "ratings.csv"
    if not ratings_path.exists():
        print(f"Error: ratings.csv not found: {ratings_path}", file=sys.stderr)
        sys.exit(1)

    gt_scores = {}  # {(user_id_str, sample_file): aesthetic_score (0-6)}
    with open(ratings_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["genre"] != genre:
                continue
            try:
                gt_scores[(row["user_id"], row["sample_file"])] = float(row["Aesthetic"])
            except (ValueError, KeyError):
                pass

    print(f"Loaded {len(gt_scores)} ground-truth ratings  (genre='{genre}')")

    split_dir = data_dir / "split"
    fold_dirs = sorted(split_dir.glob(f"{version}_fold*"))
    if not fold_dirs:
        print(
            f"Error: No fold directories for version '{version}' in {split_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [
            d for d in fold_dirs
            if int(d.name.split("fold")[-1]) in fold_set
        ]
        if not fold_dirs:
            print(f"Error: No matching fold directories for folds {args.folds}", file=sys.stderr)
            sys.exit(1)

    all_user_mae   = {}  # {user_id: [mae per fold]}
    all_user_ndcg  = {}
    all_user_srocc = {}
    all_user_ccc   = {}

    skipped_missing_pred = 0

    for fold_dir in fold_dirs:
        test_file = fold_dir / genre / "test_PIAA.txt"
        if not test_file.exists():
            print(f"  Warning: {test_file} not found, skipping")
            continue

        user_test: dict = {}
        with open(test_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    user_test.setdefault(parts[0], []).append(parts[1])

        n_pairs = sum(len(v) for v in user_test.values())
        print(f"  [{fold_dir.name}] {len(user_test)} users, {n_pairs} test pairs")

        for uid, sample_files in user_test.items():
            preds, gts = [], []
            for sf in sample_files:
                p = _lookup_pred(uid, sf)
                if p is None:
                    skipped_missing_pred += 1
                    continue
                key = (uid, sf)
                if key not in gt_scores:
                    continue
                preds.append(p)
                gts.append(gt_scores[key])

            if len(preds) < 2:
                continue

            n = len(preds)
            srocc = _spearman(preds, gts)
            ndcg  = _ndcg_at_k(gts, preds, k=10)
            mae   = sum(abs(preds[i] / 6.0 - gts[i] / 6.0) for i in range(n)) / n

            mu_p  = sum(preds) / n
            mu_t  = sum(gts)   / n
            cov   = sum((preds[i] - mu_p) * (gts[i] - mu_t) for i in range(n)) / n
            var_p = sum((preds[i] - mu_p) ** 2 for i in range(n)) / n
            var_t = sum((gts[i]   - mu_t) ** 2 for i in range(n)) / n
            ccc   = float(2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8))

            all_user_mae.setdefault(uid,   []).append(mae)
            all_user_ndcg.setdefault(uid,  []).append(ndcg)
            all_user_srocc.setdefault(uid, []).append(srocc)
            all_user_ccc.setdefault(uid,   []).append(ccc)

    if not all_user_mae:
        print("Error: No user metrics computed.", file=sys.stderr)
        sys.exit(1)

    user_avg_mae   = [sum(v) / len(v) for v in all_user_mae.values()]
    user_avg_ndcg  = [sum(v) / len(v) for v in all_user_ndcg.values()]
    user_avg_srocc = [sum(v) / len(v) for v in all_user_srocc.values()]
    user_avg_ccc   = [sum(v) / len(v) for v in all_user_ccc.values()]

    n_users = len(user_avg_mae)

    avg_mae   = sum(user_avg_mae)   / n_users
    avg_ndcg  = sum(user_avg_ndcg)  / n_users
    avg_srocc = sum(user_avg_srocc) / n_users
    avg_ccc   = sum(user_avg_ccc)   / n_users

    std_mae   = math.sqrt(sum((x - avg_mae)   ** 2 for x in user_avg_mae)   / n_users)
    std_ndcg  = math.sqrt(sum((x - avg_ndcg)  ** 2 for x in user_avg_ndcg)  / n_users)
    std_srocc = math.sqrt(sum((x - avg_srocc) ** 2 for x in user_avg_srocc) / n_users)
    std_ccc   = math.sqrt(sum((x - avg_ccc)   ** 2 for x in user_avg_ccc)   / n_users)

    mode_tag = "PIAA" if piaa_mode else "Zero-Shot"
    print(f"\n=== {model_name.capitalize()} {mode_tag} Results ({version}, {genre}) ===")
    print(f"  Source:          {model_json.name}")
    print(f"  Folds:           {len(fold_dirs)}")
    print(f"  Total users:     {n_users}")
    if skipped_missing_pred:
        print(f"  Pairs skipped (no LLM prediction): {skipped_missing_pred}")
    print(f"  Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
    print(f"  Average NDCG@10: {avg_ndcg:.6f} (std: {std_ndcg:.6f})")
    print(f"  Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
    print(f"  Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")


def _aggregate_model_giaa(args, model_name: str):
    """LLMモデルのGIAA評価: test_images_GIAA.txt × 画像単位mean GT で指標を計算する。"""
    import csv

    version = args.version
    genre = args.genre
    data_dir = Path(getattr(args, "data_dir", None) or
                    Path(__file__).resolve().parent.parent / "data")

    model_dir = Path(__file__).resolve().parent.parent / "reports" / "exp" / model_name
    matched = list(model_dir.glob(f"{genre}_giaa_results*.json"))
    if not matched:
        matched = list(model_dir.glob(f"{genre}_results*.json"))
    if not matched:
        print(
            f"Error: {model_name} GIAA results not found in {model_dir} "
            f"(pattern: {genre}_giaa_results*.json)",
            file=sys.stderr,
        )
        sys.exit(1)
    model_json = matched[0]

    with open(model_json) as f:
        llm_data = json.load(f)

    pred_score = {}       # {sample_file: expected_score}
    pred_score_stem = {}  # {stem: expected_score}
    pred_dist = {}        # {sample_file: [p0..p6]}
    pred_dist_stem = {}   # {stem: [p0..p6]}
    for entry in llm_data["per_sample"]:
        dist = entry["pred_dist"]
        e = sum(i * p for i, p in enumerate(dist))
        sf = entry["sample_file"]
        pred_score[sf] = e
        pred_score_stem[Path(sf).stem] = e
        pred_dist[sf] = dist
        pred_dist_stem[Path(sf).stem] = dist

    print(f"Loaded {len(pred_score)} {model_name} predictions  (genre='{genre}')")

    # 画像単位の平均GTスコアとGTヒストグラムをratings.csvから構築
    NUM_BINS = 7
    ratings_path = data_dir / "maked" / "ratings.csv"
    img_sum: dict = {}
    img_cnt: dict = {}
    img_hist: dict = {}   # {sample_file: [count_bin0..count_bin6]}
    with open(ratings_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["genre"] != genre:
                continue
            try:
                sf = row["sample_file"]
                score = int(float(row["Aesthetic"]))
                img_sum[sf] = img_sum.get(sf, 0.0) + float(row["Aesthetic"])
                img_cnt[sf] = img_cnt.get(sf, 0) + 1
                hist = img_hist.setdefault(sf, [0] * NUM_BINS)
                if 0 <= score < NUM_BINS:
                    hist[score] += 1
            except (ValueError, KeyError):
                pass
    img_mean_gt = {sf: img_sum[sf] / img_cnt[sf] for sf in img_sum}
    # ヒストグラムを確率分布に正規化
    img_hist_norm = {sf: [c / img_cnt[sf] for c in hist] for sf, hist in img_hist.items()}
    print(f"Loaded mean GT for {len(img_mean_gt)} images  (genre='{genre}')")

    split_dir = data_dir / "split"
    fold_dirs = sorted(split_dir.glob(f"{version}_fold*"))
    if not fold_dirs:
        print(f"Error: No fold directories for version '{version}' in {split_dir}", file=sys.stderr)
        sys.exit(1)
    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [d for d in fold_dirs if int(d.name.split("fold")[-1]) in fold_set]

    def _emd(p, q):
        """L2 norm of CDF difference (same formula as EarthMoverDistance in train_common)."""
        cp, cq = 0.0, 0.0
        acc = 0.0
        for a, b in zip(p, q):
            cp += a;  cq += b
            acc += (cp - cq) ** 2
        return acc ** 0.5

    fold_srocc, fold_mae, fold_ccc, fold_emd = [], [], [], []

    for fold_dir in fold_dirs:
        test_file = fold_dir / genre / "test_images_GIAA.txt"
        if not test_file.exists():
            print(f"  Warning: {test_file} not found, skipping")
            continue

        with open(test_file) as f:
            test_images = [l.strip() for l in f if l.strip()]

        preds, gts, dists, gt_hists = [], [], [], []
        for sf in test_images:
            stem = Path(sf).stem
            p = pred_score.get(sf) or pred_score_stem.get(stem)
            d = pred_dist.get(sf) or pred_dist_stem.get(stem)
            if p is None:
                continue
            gt_key = sf if sf in img_mean_gt else next(
                (k for k in img_mean_gt if Path(k).stem == stem), None)
            if gt_key is None:
                continue
            preds.append(p)
            gts.append(img_mean_gt[gt_key])
            if d is not None and gt_key in img_hist_norm:
                dists.append(d)
                gt_hists.append(img_hist_norm[gt_key])

        if len(preds) < 2:
            print(f"  [{fold_dir.name}] Too few matched images ({len(preds)}), skipping")
            continue

        n = len(preds)
        srocc = _spearman(preds, gts)
        mae = sum(abs(preds[i] / 6.0 - gts[i] / 6.0) for i in range(n)) / n
        mu_p = sum(preds) / n;  mu_t = sum(gts) / n
        cov = sum((preds[i] - mu_p) * (gts[i] - mu_t) for i in range(n)) / n
        var_p = sum((preds[i] - mu_p) ** 2 for i in range(n)) / n
        var_t = sum((gts[i] - mu_t) ** 2 for i in range(n)) / n
        ccc = float(2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8))
        emd = sum(_emd(dists[i], gt_hists[i]) for i in range(len(dists))) / len(dists) if dists else float("nan")

        fold_srocc.append(srocc);  fold_mae.append(mae)
        fold_ccc.append(ccc);      fold_emd.append(emd)
        print(f"  [{fold_dir.name}] n={n}  EMD={emd:.4f}  SROCC={srocc:.4f}  MAE={mae:.6f}  CCC={ccc:.4f}")

    if not fold_srocc:
        print("Error: No fold metrics computed.", file=sys.stderr)
        sys.exit(1)

    def _stats(vals):
        avg = sum(vals) / len(vals)
        std = math.sqrt(sum((x - avg) ** 2 for x in vals) / len(vals))
        return avg, std

    avg_emd,   std_emd   = _stats(fold_emd)
    avg_srocc, std_srocc = _stats(fold_srocc)
    avg_mae,   std_mae   = _stats(fold_mae)
    avg_ccc,   std_ccc   = _stats(fold_ccc)

    print(f"\n=== {model_name.capitalize()} GIAA Results ({version}, {genre}) ===")
    print(f"  Folds:           {len(fold_srocc)}")
    print(f"  Average EMD:     {avg_emd:.6f} (std: {std_emd:.6f})")
    print(f"  Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
    print(f"  Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
    print(f"  Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")


def _aggregate_giaa(args):
    """GIAAモード: inference_giaa()が出力したJSONのaverage_metricsをfoldにわたって集約する。"""
    version = args.version
    genre = args.genre
    pattern = args.pattern
    method = args.method
    reports_dir = Path(args.reports_dir)

    fold_dirs = sorted(reports_dir.glob(f"{version}_fold*"))
    if not fold_dirs:
        print(f"Error: No fold directories for version '{version}' in {reports_dir}", file=sys.stderr)
        sys.exit(1)
    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [d for d in fold_dirs if int(d.name.split("fold")[-1]) in fold_set]

    # genre が "art2fashion" のような転移ドメインの場合、フォルダはそのまま使い
    # メトリクスのキーは source genre (genre1) を使う
    m2 = re.match(r'^(\w+)2(\w+)$', genre)
    if m2:
        metric_key = m2.group(1)
    else:
        metric_key = genre

    fold_emd, fold_srocc, fold_mae, fold_ccc = [], [], [], []
    cd_emd: dict = {}
    cd_srocc: dict = {}
    cd_mae: dict = {}
    cd_ccc: dict = {}
    cd_source_head: dict = {}

    for fold_dir in fold_dirs:
        genre_dir = fold_dir / genre
        if not genre_dir.is_dir():
            print(f"Error: Genre directory not found: {genre_dir}", file=sys.stderr)
            sys.exit(1)

        if method and pattern:
            glob_pattern = f"*{method}*{pattern}*.json"
        elif method:
            glob_pattern = f"*{method}*.json"
        elif pattern:
            glob_pattern = f"*{pattern}*.json"
        else:
            glob_pattern = "*.json"

        matched_jsons = [p for p in genre_dir.glob(glob_pattern)
                         if json.loads(p.read_text()).get("mode") == "GIAA"]
        if len(matched_jsons) == 0:
            print(f"Error: No GIAA JSON matching '{glob_pattern}' in {genre_dir}", file=sys.stderr)
            sys.exit(1)
        if len(matched_jsons) > 1:
            print(f"Error: Multiple GIAA JSONs found in {genre_dir}: {[f.name for f in matched_jsons]}", file=sys.stderr)
            sys.exit(1)

        data = json.loads(matched_jsons[0].read_text())
        m = data.get("average_metrics", {}).get(metric_key, {})
        if not m:
            print(f"  Warning: No average_metrics for genre '{metric_key}' in {matched_jsons[0].name}, skipping")
            continue

        fold_emd.append(m["emd"]);  fold_srocc.append(m["srocc"])
        fold_mae.append(m["mae"]);  fold_ccc.append(m["ccc"])
        print(f"  Loaded: {matched_jsons[0].relative_to(reports_dir)}  "
              f"EMD={m['emd']:.4f}  SROCC={m['srocc']:.4f}  CCC={m['ccc']:.4f}")

        # クロスドメイン結果の収集
        cross_domain = data.get("cross_domain_metrics", {})
        for target_genre, cd_data in cross_domain.items():
            avg = cd_data.get("average", {})
            if not avg:
                continue
            cd_emd.setdefault(target_genre, []).append(avg["emd"])
            cd_srocc.setdefault(target_genre, []).append(avg["srocc"])
            cd_mae.setdefault(target_genre, []).append(avg["mae"])
            cd_ccc.setdefault(target_genre, []).append(avg["ccc"])
            if "source_head" in cd_data:
                cd_source_head[target_genre] = cd_data["source_head"]

    if not fold_emd:
        print("Error: No fold metrics found.", file=sys.stderr)
        sys.exit(1)

    def _stats(vals):
        avg = sum(vals) / len(vals)
        std = math.sqrt(sum((x - avg) ** 2 for x in vals) / len(vals))
        return avg, std

    avg_emd,   std_emd   = _stats(fold_emd)
    avg_srocc, std_srocc = _stats(fold_srocc)
    avg_mae,   std_mae   = _stats(fold_mae)
    avg_ccc,   std_ccc   = _stats(fold_ccc)

    print(f"\n=== Aggregated GIAA Results ({version}, {genre}, pattern='{pattern}') ===")
    print(f"  Folds:           {len(fold_emd)}")
    print(f"  Average EMD:     {avg_emd:.6f} (std: {std_emd:.6f})")
    print(f"  Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
    print(f"  Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
    print(f"  Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")

    # クロスドメイン結果の出力
    if cd_emd:
        print(f"\n  --- Cross-Domain (GIAA) ---")
        for target_genre in sorted(cd_emd.keys()):
            if not cd_emd[target_genre]:
                continue
            cavg_emd,   cstd_emd   = _stats(cd_emd[target_genre])
            cavg_srocc, cstd_srocc = _stats(cd_srocc[target_genre])
            cavg_mae,   cstd_mae   = _stats(cd_mae[target_genre])
            cavg_ccc,   cstd_ccc   = _stats(cd_ccc[target_genre])
            src = cd_source_head.get(target_genre, metric_key)
            print(f"  [{src} -> {target_genre}]")
            print(f"    Folds:           {len(cd_emd[target_genre])}")
            print(f"    Average EMD:     {cavg_emd:.6f} (std: {cstd_emd:.6f})")
            print(f"    Average SROCC:   {cavg_srocc:.6f} (std: {cstd_srocc:.6f})")
            print(f"    Average MAE:     {cavg_mae:.6f} (std: {cstd_mae:.6f})")
            print(f"    Average CCC:     {cavg_ccc:.6f} (std: {cstd_ccc:.6f})")


def _aggregate_claude(args):
    _aggregate_model(args, "claude")


def aggregate(args):
    """指定されたversionとgenreの各foldからJSONを集約し，全ユーザーの平均srocc/ndcgを出力する"""
    version = args.version
    genre = args.genre
    pattern = args.pattern

    giaa_mode = getattr(args, "giaa_mode", False)

    if pattern in ("claude", "gemini", "gpt"):
        if giaa_mode:
            _aggregate_model_giaa(args, pattern)
        else:
            _aggregate_model(args, pattern)
        return

    if giaa_mode:
        _aggregate_giaa(args)
        return
    method = args.method  # e.g., "ICI" (optional)
    min_id = args.min_id
    max_id = args.max_id
    ids = set(args.ids) if args.ids is not None else None
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

    # genre が "art2fashion" のような転移ドメインの場合、フォルダはそのまま使い
    # メトリクスのキーは source genre (genre1) を使う
    m2 = re.match(r'^(\w+)2(\w+)$', genre)
    if m2:
        sub_genres = [m2.group(1)]
    else:
        # genre が "art-scenery" のようなクロスドメインの場合、サブジャンルに分割
        sub_genres = genre.split("-")

    # サブジャンルごとに集約用辞書を用意
    all_user_mae  = {sg: {} for sg in sub_genres}
    all_user_ndcg = {sg: {} for sg in sub_genres}
    all_user_srocc = {sg: {} for sg in sub_genres}
    all_user_ccc = {sg: {} for sg in sub_genres}

    # クロスドメイン集約用: {target_genre: {user_id: {'srocc': [], 'ndcg': [], 'ccc': []}}}
    cd_user_mae  = {}
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
        if min_id is not None or max_id is not None or ids is not None:
            def _extract_id(p):
                m = re.search(r'-(\d+)[_.]', p.name)
                return int(m.group(1)) if m else -1
            matched_jsons = [
                p for p in matched_jsons
                if (min_id is None or _extract_id(p) >= min_id)
                and (max_id is None or _extract_id(p) <= max_id)
                and (ids is None or _extract_id(p) in ids)
            ]
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
                mae  = genre_metrics.get("mae")
                ndcg = genre_metrics.get("ndcg@10")
                srocc = genre_metrics.get("srocc")
                ccc = genre_metrics.get("ccc")
                if mae is not None:
                    all_user_mae[sg].setdefault(user_id, []).append(mae)
                if ndcg is not None:
                    all_user_ndcg[sg].setdefault(user_id, []).append(ndcg)
                if srocc is not None:
                    all_user_srocc[sg].setdefault(user_id, []).append(srocc)
                if ccc is not None:
                    all_user_ccc[sg].setdefault(user_id, []).append(ccc)

        # クロスドメイン結果の収集
        cross_domain = data.get("cross_domain_metrics", {})
        for target_genre, cd_data in cross_domain.items():
            if target_genre not in cd_user_mae:
                cd_user_mae[target_genre]  = {}
                cd_user_srocc[target_genre] = {}
                cd_user_ndcg[target_genre] = {}
                cd_user_ccc[target_genre] = {}
            per_user_cd = cd_data.get("per_user", {})
            for user_id, cd_metrics in per_user_cd.items():
                mae  = cd_metrics.get("mae")
                ndcg = cd_metrics.get("ndcg@10")
                srocc = cd_metrics.get("srocc")
                ccc = cd_metrics.get("ccc")
                if mae is not None:
                    cd_user_mae[target_genre].setdefault(user_id, []).append(mae)
                if ndcg is not None:
                    cd_user_ndcg[target_genre].setdefault(user_id, []).append(ndcg)
                if srocc is not None:
                    cd_user_srocc[target_genre].setdefault(user_id, []).append(srocc)
                if ccc is not None:
                    cd_user_ccc[target_genre].setdefault(user_id, []).append(ccc)

        print(
            f"  Loaded: {json_path.relative_to(reports_dir)} ({len(per_user)} users)"
        )

    if not any(all_user_mae[sg] for sg in sub_genres):
        print("Error: No user metrics found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Aggregated Results ({version}, {genre}, pattern='{pattern}') ===")
    print(f"  Folds:         {len(fold_dirs)}")

    for sg in sub_genres:
        if not all_user_mae[sg]:
            continue

        # ユーザーごとの fold 平均を算出
        user_avg_mae  = [sum(vals) / len(vals) for vals in all_user_mae[sg].values()]
        user_avg_ndcg = [sum(vals) / len(vals) for vals in all_user_ndcg[sg].values()]
        user_avg_srocc = [sum(vals) / len(vals) for vals in all_user_srocc[sg].values()]
        user_avg_ccc  = [sum(vals) / len(vals) for vals in all_user_ccc[sg].values()]

        avg_mae   = sum(user_avg_mae)   / len(user_avg_mae)
        avg_ndcg  = sum(user_avg_ndcg)  / len(user_avg_ndcg)
        avg_srocc = sum(user_avg_srocc) / len(user_avg_srocc)
        avg_ccc   = sum(user_avg_ccc)   / len(user_avg_ccc) if user_avg_ccc else None

        std_mae  = math.sqrt(sum((x - avg_mae)   ** 2 for x in user_avg_mae)   / len(user_avg_mae))
        std_ndcg = math.sqrt(sum((x - avg_ndcg)  ** 2 for x in user_avg_ndcg)  / len(user_avg_ndcg))
        std_srocc = math.sqrt(sum((x - avg_srocc) ** 2 for x in user_avg_srocc) / len(user_avg_srocc))
        std_ccc  = math.sqrt(
            sum((x - avg_ccc) ** 2 for x in user_avg_ccc) / len(user_avg_ccc)
        ) if user_avg_ccc else None

        print(f"  [{sg}]")
        print(f"    Total users:     {len(all_user_mae[sg])}")
        print(f"    Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
        print(f"    Average NDCG@10: {avg_ndcg:.6f} (std: {std_ndcg:.6f})")
        print(f"    Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
        if avg_ccc is not None:
            print(f"    Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")

    # クロスドメイン結果の出力
    if cd_user_mae:
        print(f"\n  --- Cross-Domain (head average) ---")
        for target_genre in sorted(cd_user_mae.keys()):
            if not cd_user_mae[target_genre]:
                continue

            user_avg_mae  = [sum(vals) / len(vals) for vals in cd_user_mae[target_genre].values()]
            user_avg_ndcg = [sum(vals) / len(vals) for vals in cd_user_ndcg[target_genre].values()]
            user_avg_srocc = [sum(vals) / len(vals) for vals in cd_user_srocc[target_genre].values()]
            user_avg_ccc  = [sum(vals) / len(vals) for vals in cd_user_ccc[target_genre].values()]

            avg_mae   = sum(user_avg_mae)   / len(user_avg_mae)
            avg_ndcg  = sum(user_avg_ndcg)  / len(user_avg_ndcg)
            avg_srocc = sum(user_avg_srocc) / len(user_avg_srocc)
            avg_ccc   = sum(user_avg_ccc)   / len(user_avg_ccc) if user_avg_ccc else None

            std_mae  = math.sqrt(sum((x - avg_mae)   ** 2 for x in user_avg_mae)   / len(user_avg_mae))
            std_ndcg = math.sqrt(sum((x - avg_ndcg)  ** 2 for x in user_avg_ndcg)  / len(user_avg_ndcg))
            std_srocc = math.sqrt(sum((x - avg_srocc) ** 2 for x in user_avg_srocc) / len(user_avg_srocc))
            std_ccc  = math.sqrt(
                sum((x - avg_ccc) ** 2 for x in user_avg_ccc) / len(user_avg_ccc)
            ) if user_avg_ccc else None

            print(f"  [{genre} -> {target_genre}]")
            print(f"    Total users:     {len(cd_user_mae[target_genre])}")
            print(f"    Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
            print(f"    Average NDCG@10: {avg_ndcg:.6f} (std: {std_ndcg:.6f})")
            print(f"    Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
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


def visualize_features(args):
    """DAモデルと非DAモデルの特徴量を2D可視化して比較する。

    各foldの val_images_GIAA.txt から画像を収集し、NIMAfeatを抽出。
    ratings.csvの全ユーザー平均スコアで3クラス（low/mid/high）に分類し
    t-SNE/UMAP/PCAで2次元にプロット。非DAとDAを横並びサブプロットで比較。
    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image

    import sys as _sys
    _src = Path(__file__).resolve().parent
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    from train_common import NIMA, num_bins

    source_genre = args.source_genre
    target_genre = args.target_genre
    backbone = args.backbone
    root_dir = Path(args.root_dir)
    models_pth_dir = Path(args.models_pth_dir)
    method = args.method
    percentile = args.percentile
    dataset_ver = args.dataset_ver
    uda_methods = args.uda_methods  # e.g. ["DANN"], ["DJDOT"], ["DANN", "DJDOT"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Task: {source_genre} → {target_genre}  |  UDA: {', '.join(uda_methods)}")

    # ── 1. 画像ごとの平均スコアを計算・閾値を自動算出 ────────────────────────
    ratings = pd.read_csv(root_dir / "maked" / "ratings.csv")
    target_ratings = ratings[ratings["genre"] == target_genre]
    img_mean_score = target_ratings.groupby("sample_file")["Aesthetic"].mean()

    low_thresh = float(np.percentile(img_mean_score.values, percentile))
    high_thresh = float(np.percentile(img_mean_score.values, 100 - percentile))
    print(f"Percentile: {percentile}% / {100 - percentile}%  →  low<{low_thresh:.2f}, high≥{high_thresh:.2f}")

    # ── 2. 画像変換（CLIP-ViT-B/16の標準前処理） ─────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # ── 3. 全foldを回して特徴量を収集 ─────────────────────────────────────────
    split_dir = root_dir / "split"
    fold_dirs = sorted(split_dir.glob(f"{dataset_ver}_fold*"))
    if not fold_dirs:
        print(f"Error: No fold dirs for version '{dataset_ver}' in {split_dir}", file=_sys.stderr)
        _sys.exit(1)
    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [d for d in fold_dirs if int(d.name.split("fold")[-1]) in fold_set]
        if not fold_dirs:
            print(f"Error: No fold dirs matched --folds {args.folds}", file=_sys.stderr)
            _sys.exit(1)

    samples_dir = Path.home() / "proj-xpass" / "data" / "samples"

    def find_nima_pth(fold_name, subdir, uda_method=None):
        d = models_pth_dir / fold_name / subdir
        if uda_method:
            ptns = list(d.glob(f"{subdir}_{uda_method}_NIMA_*.pth"))
        else:
            ptns = list(d.glob(f"{subdir}_NIMA_*.pth"))
        return ptns[0] if ptns else None

    def load_model(pth_path):
        model = NIMA(num_bins, backbone=backbone)
        state = torch.load(pth_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return model

    def extract_features(model, val_images):
        feats, labels = [], []
        with torch.no_grad():
            for img_file in val_images:
                if img_file not in img_mean_score.index:
                    continue
                score = img_mean_score[img_file]
                label = 0 if score < low_thresh else (1 if score < high_thresh else 2)

                img_path = samples_dir / target_genre / img_file
                try:
                    img = Image.open(img_path).convert("RGB")
                    t = transform(img).unsqueeze(0).to(device)
                    _, feat, _ = model(t, return_feat=True)
                    feats.append(feat.cpu().float().numpy()[0])
                    labels.append(label)
                except Exception as e:
                    print(f"  Warning: skip {img_file}: {e}")
        return feats, labels

    keys = ["nonda"] + uda_methods
    all_feats = {k: [] for k in keys}
    all_labels = {k: [] for k in keys}
    all_fold_ids = {k: [] for k in keys}  # foldごとのID
    fold_sil_scores = {k: [] for k in keys}  # foldごとのSilhouette Score

    from sklearn.metrics import silhouette_score

    for fold_idx, fold_dir in enumerate(fold_dirs):
        fold_name = fold_dir.name
        val_img_file = fold_dir / target_genre / "train_images_GIAA.txt"
        if not val_img_file.exists():
            print(f"Warning: {val_img_file} not found, skipping")
            continue

        with open(val_img_file) as f:
            val_images = [line.strip() for line in f if line.strip()]

        nonda_pth = find_nima_pth(fold_name, source_genre)
        if nonda_pth is None:
            print(f"Warning: Non-DA model not found for {fold_name}/{source_genre}, skipping")
            continue

        pth_pairs = [("nonda", nonda_pth)]
        skip_fold = False
        for um in uda_methods:
            da_pth = find_nima_pth(fold_name, f"{source_genre}2{target_genre}", uda_method=um)
            if da_pth is None:
                print(f"Warning: {um} model not found for {fold_name}/{source_genre}2{target_genre}, skipping")
                skip_fold = True
                break
            pth_pairs.append((um, da_pth))
        if skip_fold:
            continue

        print(f"\n[{fold_name}]")
        print(f"  Non-DA: {nonda_pth.name}")
        for um, pth in pth_pairs[1:]:
            print(f"  {um}:     {pth.name}")
        print(f"  Images: {len(val_images)}")

        for key, pth_path in pth_pairs:
            model = load_model(pth_path)
            feats, labels = extract_features(model, val_images)
            all_feats[key].extend(feats)
            all_labels[key].extend(labels)
            all_fold_ids[key].extend([fold_idx] * len(feats))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # foldごとのSilhouette Score
            f_arr = np.array(feats)
            l_arr = np.array(labels)
            mask = l_arr != 1
            f_lh, l_lh = f_arr[mask], l_arr[mask]
            if len(np.unique(l_lh)) >= 2 and len(f_lh) >= 2:
                s = silhouette_score(f_lh, l_lh, metric="euclidean")
                fold_sil_scores[key].append(s)
                print(f"  Silhouette ({key}): {s:.4f}  (low={( l_lh==0).sum()}, high={(l_lh==2).sum()})")
            else:
                print(f"  Silhouette ({key}): N/A (insufficient samples)")

    for key in keys:
        n = len(all_feats[key])
        if n == 0:
            print(f"Error: No features extracted for {key} model.", file=_sys.stderr)
            _sys.exit(1)
        print(f"\nTotal samples ({key}): {n}")

    # ── 4. Silhouette Score 集計（平均±std） ──────────────────────────────────
    print("\n=== Silhouette Score (256-dim domain_feat, low vs high) ===")
    key_label_pairs = [("nonda", "Non-DA")] + [(um, um) for um in uda_methods]
    for key, label in key_label_pairs:
        scores = fold_sil_scores[key]
        if not scores:
            print(f"  {label}: N/A")
            continue
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        print(f"  {label}: {mean_s:.4f} ± {std_s:.4f}  (n_folds={len(scores)})")

    # ── 5. 次元削減＋プロット ─────────────────────────────────────────────────
    class_names = [f"Low (<{low_thresh:.2f})", f"Mid ({low_thresh:.2f}-{high_thresh:.2f})", f"High (≥{high_thresh:.2f})"]
    colors = ["#e74c3c", "#f39c12", "#2980b9"]
    markers = ["o", "s", "^"]
    hide_mid = args.hide_mid
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _reduce(feats_arr, m):
        if m == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats_arr) - 1)).fit_transform(feats_arr)
        elif m == "umap":
            import umap as umap_lib
            return umap_lib.UMAP(n_components=2, random_state=42).fit_transform(feats_arr)
        elif m == "pca":
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=42).fit_transform(feats_arr)

    def _plot_and_save(m):
        # 先に全モデルの埋め込みを計算してからスケールを揃える
        plot_keys = keys  # ["nonda"] + uda_methods
        subtitles = [f"Non-DA  ({source_genre} only)"] + [
            f"{um}  ({source_genre}→{target_genre})" for um in uda_methods
        ]
        embeds = {}
        for key in plot_keys:
            feats_arr = np.array(all_feats[key])
            print(f"\nRunning {m.upper()} on {key} ({len(feats_arr)} samples)...")
            embeds[key] = _reduce(feats_arr, m)

        # 全埋め込みのグローバル軸範囲を算出
        all_x = np.concatenate([embeds[k][:, 0] for k in plot_keys])
        all_y = np.concatenate([embeds[k][:, 1] for k in plot_keys])
        margin_x = (all_x.max() - all_x.min()) * 0.05 or 1.0
        margin_y = (all_y.max() - all_y.min()) * 0.05 or 1.0
        xlim = (all_x.min() - margin_x, all_x.max() + margin_x)
        ylim = (all_y.min() - margin_y, all_y.max() + margin_y)

        n_plots = len(plot_keys)
        fig, axes = plt.subplots(1, n_plots, figsize=(6.5 * n_plots, 5.5))
        if n_plots == 1:
            axes = [axes]
        for ax, key, subtitle in zip(axes, plot_keys, subtitles):
            labels_arr = np.array(all_labels[key])
            fold_ids_arr = np.array(all_fold_ids[key])
            embed = embeds[key]
            unique_folds = np.unique(fold_ids_arr)
            for cls_idx, (cname, color, marker) in enumerate(zip(class_names, colors, markers)):
                if hide_mid and cls_idx == 1:
                    continue
                mask = labels_arr == cls_idx
                ax.scatter(
                    embed[mask, 0], embed[mask, 1],
                    c=color, label=f"{cname} (n={mask.sum()})",
                    s=18, alpha=0.7, linewidths=0, marker=marker,
                )
                if mask.sum() > 0:
                    cx, cy = embed[mask, 0].mean(), embed[mask, 1].mean()
                    ax.scatter(cx, cy, marker="*", c=color, s=250,
                               edgecolors="black", linewidths=0.5, zorder=5,
                               label="_nolegend_")
                # foldごとのクラスター重心をプロット
                for fold_id in unique_folds:
                    fold_cls_mask = mask & (fold_ids_arr == fold_id)
                    if fold_cls_mask.sum() == 0:
                        continue
                    cx = embed[fold_cls_mask, 0].mean()
                    cy = embed[fold_cls_mask, 1].mean()
                    ax.scatter(
                        cx, cy,
                        c=color, marker="*", s=220, edgecolors="k",
                        linewidths=0.8, zorder=5,
                    )
                    ax.annotate(
                        f"f{fold_id}",
                        xy=(cx, cy), xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7, color=color, zorder=6,
                    )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(subtitle, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9, loc="best")
            ax.set_xlabel(f"{m.upper()} 1", fontsize=10)
            ax.set_ylabel(f"{m.upper()} 2", fontsize=10)
            ax.tick_params(labelsize=9)
        methods_str = "_".join(uda_methods)
        fig.suptitle(
            f"Feature space: {source_genre} → {target_genre}  [{m.upper()}]",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        out = output_dir / f"{source_genre}2{target_genre}_{methods_str}_{m}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    # ── 5. 指定手法（またはall）で出力 ──────────────────────────────────────────
    if not args.score_only:
        methods_to_run = ["tsne", "umap", "pca"] if method == "all" else [method]
        for m in methods_to_run:
            _plot_and_save(m)


def visualize_domain_gap(args):
    """DAモデルと非DAモデルでソース・ターゲットドメイン間の特徴量ギャップを比較可視化する。

    各foldのソース画像とターゲット画像の特徴量を両モデルで抽出し、t-SNE/UMAP/PCAで2次元にプロット。
    非DA（左）とDA（右）を横並びサブプロットで比較し、DAによるドメインギャップ縮小を可視化。
    ドメイン分離度をSilhouette Scoreで定量評価（低いほどドメインギャップが小さい）。
    """
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image

    import sys as _sys
    _src = Path(__file__).resolve().parent
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    from train_common import NIMA, num_bins

    source_genre = args.source_genre
    target_genre = args.target_genre
    backbone = args.backbone
    root_dir = Path(args.root_dir)
    models_pth_dir = Path(args.models_pth_dir)
    method = args.method
    dataset_ver = args.dataset_ver
    split_file = args.split_file
    n_source = args.n_source
    n_target = args.n_target
    uda_methods = args.uda_methods  # e.g. ["DANN"], ["DJDOT"], ["DANN", "DJDOT"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Task: {source_genre} → {target_genre}  |  UDA: {', '.join(uda_methods)}")

    # ── 1. 画像変換（CLIP-ViT-B/16の標準前処理） ─────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # ── 2. foldディレクトリの取得 ─────────────────────────────────────────────
    split_dir = root_dir / "split"
    fold_dirs = sorted(split_dir.glob(f"{dataset_ver}_fold*"))
    if not fold_dirs:
        print(f"Error: No fold dirs for version '{dataset_ver}' in {split_dir}", file=_sys.stderr)
        _sys.exit(1)
    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [d for d in fold_dirs if int(d.name.split("fold")[-1]) in fold_set]
        if not fold_dirs:
            print(f"Error: No fold dirs matched --folds {args.folds}", file=_sys.stderr)
            _sys.exit(1)

    samples_dir = Path.home() / "proj-xpass" / "data" / "samples"

    # split ファイルのファイル名と実際のサンプルディレクトリが異なるジャンルのマッピング
    # "genre_name": ("samples_subdir", "image_extension")
    GENRE_SAMPLES_MAP = {
        "scenery": ("scenery_image", ".jpg"),
    }

    def find_nima_pth(fold_name, subdir, uda_method=None):
        d = models_pth_dir / fold_name / subdir
        if uda_method:
            ptns = list(d.glob(f"{subdir}_{uda_method}_NIMA_*.pth"))
        else:
            ptns = list(d.glob(f"{subdir}_NIMA_*.pth"))
        return ptns[0] if ptns else None

    def load_model(pth_path):
        model = NIMA(num_bins, backbone=backbone)
        state = torch.load(pth_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return model

    def extract_domain_features(model, img_files, genre, domain_label, max_n):
        """指定ドメインの画像から特徴量を抽出する。domain_label: 0=source, 1=target"""
        feats, labels = [], []
        targets = img_files[:max_n] if max_n is not None else img_files
        samples_subdir, img_ext = GENRE_SAMPLES_MAP.get(genre, (genre, None))
        with torch.no_grad():
            for img_file in targets:
                # split ファイルの拡張子を画像拡張子に置き換える（必要な場合）
                fname = (Path(img_file).stem + img_ext) if img_ext else img_file
                img_path = samples_dir / samples_subdir / fname
                try:
                    img = Image.open(img_path).convert("RGB")
                    t = transform(img).unsqueeze(0).to(device)
                    _, feat, _ = model(t, return_feat=True)
                    feats.append(feat.cpu().float().numpy()[0])
                    labels.append(domain_label)
                except Exception as e:
                    print(f"  Warning: skip {img_file}: {e}")
        return feats, labels

    # domain_label: 0=source, 1=target
    keys = ["nonda"] + uda_methods
    all_feats = {k: [] for k in keys}
    all_labels = {k: [] for k in keys}
    all_fold_ids = {k: [] for k in keys}
    fold_sil_scores = {k: [] for k in keys}

    from sklearn.metrics import silhouette_score

    for fold_idx, fold_dir in enumerate(fold_dirs):
        fold_name = fold_dir.name

        src_img_file = fold_dir / source_genre / split_file
        tgt_img_file = fold_dir / target_genre / split_file

        if not src_img_file.exists():
            print(f"Warning: {src_img_file} not found, skipping")
            continue
        if not tgt_img_file.exists():
            print(f"Warning: {tgt_img_file} not found, skipping")
            continue

        with open(src_img_file) as f:
            src_images = [line.strip() for line in f if line.strip()]
        with open(tgt_img_file) as f:
            tgt_images = [line.strip() for line in f if line.strip()]

        nonda_pth = find_nima_pth(fold_name, source_genre)
        if nonda_pth is None:
            print(f"Warning: Non-DA model not found for {fold_name}/{source_genre}, skipping")
            continue

        pth_pairs = [("nonda", nonda_pth)]
        skip_fold = False
        for um in uda_methods:
            da_pth = find_nima_pth(fold_name, f"{source_genre}2{target_genre}", uda_method=um)
            if da_pth is None:
                print(f"Warning: {um} model not found for {fold_name}/{source_genre}2{target_genre}, skipping")
                skip_fold = True
                break
            pth_pairs.append((um, da_pth))
        if skip_fold:
            continue

        print(f"\n[{fold_name}]")
        print(f"  Non-DA: {nonda_pth.name}")
        for um, pth in pth_pairs[1:]:
            print(f"  {um}:     {pth.name}")
        print(f"  Source images: {len(src_images)}, Target images: {len(tgt_images)}")

        for key, pth_path in pth_pairs:
            model = load_model(pth_path)
            src_feats, src_labs = extract_domain_features(model, src_images, source_genre, 0, n_source)
            tgt_feats, tgt_labs = extract_domain_features(model, tgt_images, target_genre, 1, n_target)
            feats = src_feats + tgt_feats
            labels = src_labs + tgt_labs
            all_feats[key].extend(feats)
            all_labels[key].extend(labels)
            all_fold_ids[key].extend([fold_idx] * len(feats))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # foldごとのSilhouette Score（低いほどドメインギャップが小さい）
            f_arr = np.array(feats)
            l_arr = np.array(labels)
            if len(np.unique(l_arr)) >= 2 and len(f_arr) >= 2:
                s = silhouette_score(f_arr, l_arr, metric="euclidean")
                fold_sil_scores[key].append(s)
                print(f"  Silhouette ({key}): {s:.4f}  "
                      f"(source={( l_arr==0).sum()}, target={(l_arr==1).sum()})")
            else:
                print(f"  Silhouette ({key}): N/A (insufficient samples)")

    for key in keys:
        n = len(all_feats[key])
        if n == 0:
            print(f"Error: No features extracted for {key} model.", file=_sys.stderr)
            _sys.exit(1)
        print(f"\nTotal samples ({key}): {n}")

    # ── 4. Silhouette Score 集計（平均±std） ──────────────────────────────────
    print("\n=== Domain Separation Score (256-dim domain_feat, source vs target) ===")
    print("  (lower Silhouette = better domain alignment)")
    key_label_pairs = [("nonda", "Non-DA")] + [(um, um) for um in uda_methods]
    for key, label in key_label_pairs:
        scores = fold_sil_scores[key]
        if not scores:
            print(f"  {label}: N/A")
            continue
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        print(f"  {label}: {mean_s:.4f} ± {std_s:.4f}  (n_folds={len(scores)})")

    if args.score_only:
        return

    # ── 5. 次元削減＋プロット ─────────────────────────────────────────────────
    domain_names = [source_genre, target_genre]
    domain_colors = ["#e74c3c", "#2980b9"]
    domain_markers = ["o", "s"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _reduce(feats_arr, m):
        if m == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats_arr) - 1)).fit_transform(feats_arr)
        elif m == "umap":
            import umap as umap_lib
            return umap_lib.UMAP(n_components=2, random_state=42).fit_transform(feats_arr)
        elif m == "pca":
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=42).fit_transform(feats_arr)

    sil_means = {}
    for key in keys:
        s = fold_sil_scores[key]
        sil_means[key] = float(np.mean(s)) if s else float("nan")

    def _plot_and_save(m):
        plot_keys = keys  # ["nonda"] + uda_methods
        subtitles = [f"Non-DA  ({source_genre} only)"] + [
            f"{um}  ({source_genre}→{target_genre})" for um in uda_methods
        ]
        embeds = {}
        for key in plot_keys:
            feats_arr = np.array(all_feats[key])
            print(f"\nRunning {m.upper()} on {key} ({len(feats_arr)} samples)...")
            embeds[key] = _reduce(feats_arr, m)

        all_x = np.concatenate([embeds[k][:, 0] for k in plot_keys])
        all_y = np.concatenate([embeds[k][:, 1] for k in plot_keys])
        margin_x = (all_x.max() - all_x.min()) * 0.05 or 1.0
        margin_y = (all_y.max() - all_y.min()) * 0.05 or 1.0
        xlim = (all_x.min() - margin_x, all_x.max() + margin_x)
        ylim = (all_y.min() - margin_y, all_y.max() + margin_y)

        n_plots = len(plot_keys)
        fig, axes = plt.subplots(1, n_plots, figsize=(6.5 * n_plots, 5.5))
        if n_plots == 1:
            axes = [axes]
        for ax, key, subtitle in zip(axes, plot_keys, subtitles):
            labels_arr = np.array(all_labels[key])
            embed = embeds[key]
            for dom_idx, (dname, color, marker) in enumerate(
                zip(domain_names, domain_colors, domain_markers)
            ):
                mask = labels_arr == dom_idx
                ax.scatter(
                    embed[mask, 0], embed[mask, 1],
                    c=color, label=f"{dname} (n={mask.sum()})",
                    s=18, alpha=0.7, linewidths=0, marker=marker,
                )
                if mask.sum() > 0:
                    cx, cy = embed[mask, 0].mean(), embed[mask, 1].mean()
                    ax.scatter(cx, cy, marker="*", c=color, s=250,
                               edgecolors="black", linewidths=0.5, zorder=5,
                               label="_nolegend_")
            sil_val = sil_means[key]
            sil_str = f"{sil_val:.4f}" if not np.isnan(sil_val) else "N/A"
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(f"{subtitle}\nSilhouette={sil_str}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9, loc="best")
            ax.set_xlabel(f"{m.upper()} 1", fontsize=10)
            ax.set_ylabel(f"{m.upper()} 2", fontsize=10)
            ax.tick_params(labelsize=9)
        methods_str = "_".join(uda_methods)
        fig.suptitle(
            f"Domain gap: {source_genre} vs {target_genre}  [{m.upper()}]",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        out = output_dir / f"{source_genre}2{target_genre}_{methods_str}_domain_gap_{m}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    methods_to_run = ["tsne", "umap", "pca"] if method == "all" else [method]
    for m in methods_to_run:
        _plot_and_save(m)


# ---------------------------------------------------------------------------
# analyze_da_factors: linear-regression analysis of DA-success factors
# ---------------------------------------------------------------------------

def _canonical_rename_map(src: str, tgt: str, metric: str) -> dict:
    """Map pair-specific feature names → canonical src/tgt names.

    Features with no src/tgt analog (Big5, age, demographics) are not in the
    returned map; callers should drop them. scenery's learn/interest map to
    users.csv `photoVideo_*` columns.
    """
    learn_prefix = {"art": "art", "fashion": "fashion", "scenery": "photoVideo"}
    src_lp = learn_prefix.get(src, src)
    tgt_lp = learn_prefix.get(tgt, tgt)

    rename = {
        f"generality_{src}": "generality_src",
        f"generality_{tgt}": "generality_tgt",
        f"retest_mae_{src}": "retest_mae_src",
        f"retest_mae_{tgt}": "retest_mae_tgt",
        f"shift_mean_{src}_to_{tgt}": "shift_mean_src_to_tgt",
        f"shift_std_{src}_to_{tgt}": "shift_std_src_to_tgt",
        f"shift_skew_{src}_to_{tgt}": "shift_skew_src_to_tgt",
        f"shift_kurt_{src}_to_{tgt}": "shift_kurt_src_to_tgt",
        f"shift_retest_mae_{src}_to_{tgt}": "shift_retest_mae_src_to_tgt",
        f"shift_generality_{src}_to_{tgt}": "shift_generality_src_to_tgt",
        f"shift_interest_{src}_to_{tgt}": "shift_interest_src_to_tgt",
        f"shift_learn_{src}_to_{tgt}": "shift_learn_src_to_tgt",
        f"{src_lp}_learn": "learn_src",
        f"{tgt_lp}_learn": "learn_tgt",
        f"{src_lp}_interest": "interest_src",
        f"{tgt_lp}_interest": "interest_tgt",
        f"baseline_{metric}_target": "baseline_tgt",
        f"baseline_{metric}_source": "baseline_src",
    }
    if src_lp == tgt_lp:
        rename.pop(f"{tgt_lp}_learn", None)
        rename.pop(f"{tgt_lp}_interest", None)
    for stat in ("mean", "std", "skew", "kurt"):
        rename[f"src_{src}_{stat}"] = f"style_src_{stat}"
        rename[f"tgt_{tgt}_{stat}"] = f"style_tgt_{stat}"
    # When scenery is off-domain (i.e., art↔fashion pairs), keep photoVideo_*
    # columns as scenery-tagged off-domain features instead of dropping them.
    if "scenery" not in (src, tgt):
        rename["photoVideo_learn"] = "scenery_learn"
        rename["photoVideo_interest"] = "scenery_interest"
    return rename


def _is_global_canonical_feature(f: str) -> bool:
    """Features whose name is identical across all pairs (no src/tgt suffix).
    Big5 / age / one-hot demographics — pass through without renaming.
    """
    if f == "age":
        return True
    return f.startswith(("big5_", "gender_", "edu_", "nationality_"))


def _aggregate_canonical_ranking(args, pairs):
    """Average standardized OLS β across (src→tgt) pairs.

    For each pair's regression_results.csv:
      - Paired features (e.g. `generality_art`, `tgt_fashion_mean`) are
        renamed to canonical src/tgt names (`generality_src`, `style_tgt_mean`).
      - Globally canonical features (Big5, age, gender/edu/nationality
        one-hots) pass through unchanged — they are the same column in
        every pair and are the cleanest "domain-general DA-success factors".
      - Off-domain learn/interest that wasn't renamed for this pair (e.g.
        `photoVideo_learn` in art↔fashion) is dropped, because the same
        users.csv column would otherwise appear under two different aggregated
        rows depending on pair direction.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    base_dir = Path(args.output_dir)
    out_dir = base_dir / f"_aggregated_{args.model_type}_{args.da_method}_{args.metric}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_pair = {}  # pair → {canonical_feat: {beta, se, t, p, p_fdr, vif}}
    used_pairs = []
    for src, tgt in pairs:
        pair = f"{src}2{tgt}"
        csv_path = base_dir / f"{pair}_{args.model_type}_{args.da_method}" / "regression_results.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found", file=sys.stderr)
            continue
        reg = pd.read_csv(csv_path)
        rename = _canonical_rename_map(src, tgt, args.metric)
        d = {}
        n_paired = n_global = n_skipped = 0
        for _, row in reg.iterrows():
            feat = row["feature"]
            if feat in rename:
                canonical = rename[feat]
                n_paired += 1
            elif _is_global_canonical_feature(feat):
                canonical = feat
                n_global += 1
            else:
                # off-domain learn/interest, or anything else without a clear
                # cross-pair identity — skip
                n_skipped += 1
                continue
            d[canonical] = {
                "beta": float(row["ols_coef_std"]),
                "se": float(row["ols_se"]),
                "t": float(row["ols_t"]),
                "p": float(row["ols_p"]),
                "p_fdr": float(row["ols_p_fdr"]),
                "vif": float(row["vif"]),
            }
        per_pair[pair] = d
        used_pairs.append(pair)
        print(f"  [load] {pair}: {len(d)} features "
              f"(paired={n_paired}, global={n_global}, skipped={n_skipped})")

    if not used_pairs:
        print("[warn] no pair regression_results.csv found for aggregation",
              file=sys.stderr)
        return

    # Union of all canonical feature names seen across pairs.
    all_feats = sorted({f for p in used_pairs for f in per_pair[p]})

    rows = []
    for feat in all_feats:
        betas = [per_pair[p][feat]["beta"] for p in used_pairs if feat in per_pair[p]]
        if not betas:
            continue
        ses = [per_pair[p][feat]["se"] for p in used_pairs if feat in per_pair[p]]
        n_sig = sum(1 for p in used_pairs if feat in per_pair[p]
                    and per_pair[p][feat]["p_fdr"] < 0.05)
        row = {"feature": feat}
        for p in used_pairs:
            row[f"beta_{p}"] = per_pair[p].get(feat, {}).get("beta", np.nan)
        row["beta_mean"] = float(np.mean(betas))
        row["beta_std"] = float(np.std(betas, ddof=1)) if len(betas) > 1 else 0.0
        row["abs_beta_mean"] = float(np.mean(np.abs(betas)))
        row["se_mean"] = float(np.mean(ses))
        row["n_pairs"] = len(betas)
        row["n_sig_fdr"] = int(n_sig)
        rows.append(row)

    wide = pd.DataFrame(rows)
    wide = (wide.sort_values("abs_beta_mean", ascending=False)
                .drop(columns="abs_beta_mean")
                .reset_index(drop=True))
    wide_path = out_dir / "aggregated_ols_betas.csv"
    wide.to_csv(wide_path, index=False)
    print(f"[save] {wide_path}  "
          f"({len(wide)} canonical features × {len(used_pairs)} pairs)")

    long_rows = []
    for p in used_pairs:
        for feat in all_feats:
            if feat in per_pair[p]:
                d = per_pair[p][feat]
                long_rows.append({
                    "pair": p, "feature": feat,
                    "ols_coef_std": d["beta"], "ols_se": d["se"],
                    "ols_t": d["t"], "ols_p": d["p"],
                    "ols_p_fdr": d["p_fdr"], "vif": d["vif"],
                })
    long_path = out_dir / "per_pair_ols_betas.csv"
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    print(f"[save] {long_path}")

    print(f"\n[mean OLS β across {len(used_pairs)} pairs, |β| descending]")
    header = (f"  {'feature':25s}  {'mean β':>8s}  {'± std':>7s}  "
              f"{'n_sig':>5s}  ") + "  ".join(f"{p:>14s}" for p in used_pairs)
    print(header)
    for _, r in wide.iterrows():
        line = (f"  {r['feature']:25s}  {r['beta_mean']:+8.3f}  "
                f"{r['beta_std']:7.3f}  {int(r['n_sig_fdr']):5d}  ")
        line += "  ".join(f"{r[f'beta_{p}']:+14.3f}" for p in used_pairs)
        print(line)

    if not getattr(args, "no_plots", False):
        top_k = getattr(args, "top_k", 10)
        top = wide.head(min(top_k, len(wide))).iloc[::-1]  # bottom→top for barh
        sig_thresh = max(1, len(used_pairs) // 2)
        colors = ["tab:red" if int(n) >= sig_thresh else "tab:gray"
                  for n in top["n_sig_fdr"]]
        # ±95% CI on the mean β across pairs (SEM = std / sqrt(n_pairs))
        sem = top["beta_std"].values / np.sqrt(np.maximum(top["n_pairs"].values, 1))
        ci95 = 1.96 * sem

        fig, ax = plt.subplots(figsize=(7, max(3.0, 0.35 * len(top) + 1.5)))
        ax.barh(range(len(top)), top["beta_mean"].values, xerr=ci95,
                color=colors, alpha=0.8, ecolor="black", capsize=2)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=8)
        ax.set_xlabel("mean OLS β across pairs (standardized, ±95% CI)")
        ax.set_title(
            f"aggregated across {len(used_pairs)} pairs  "
            f"(red = FDR-sig in ≥{sig_thresh}/{len(used_pairs)} pairs)",
            fontsize=10,
        )
        fig.suptitle(
            f"{args.model_type} | {args.da_method} | metric={args.metric}",
            fontsize=11,
        )
        fig.tight_layout()
        fig_path = out_dir / "aggregated_feature_importance.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {fig_path}")


def _find_finetune_jsons(reports_dir: Path, version: str, subdir: str,
                         model_type: str, da_method: str | None,
                         folds: list[int] | None) -> dict[int, Path]:
    """Find one finetune JSON per fold under {reports_dir}/{version}_fold*/{subdir}/.

    Filters by:
      - filename contains "finetune"
      - filename contains model_type (ICI / MIR)
      - if da_method is None: filename contains "Only"  (no-DA)
        else: filename contains da_method
    Returns: {fold_num: path}.
    """
    fold_dirs = sorted(reports_dir.glob(f"{version}_fold*"))
    out = {}
    for fold_dir in fold_dirs:
        m = re.search(r"fold(\d+)$", fold_dir.name)
        if not m:
            continue
        fold_num = int(m.group(1))
        if folds is not None and fold_num not in folds:
            continue
        target_dir = fold_dir / subdir
        if not target_dir.is_dir():
            continue
        cands = []
        for p in sorted(target_dir.glob("*finetune*.json")):
            name = p.name
            if model_type not in name:
                continue
            if da_method is None:
                if "Only" not in name:
                    continue
            else:
                if da_method not in name:
                    continue
            cands.append(p)
        if not cands:
            continue
        if len(cands) > 1:
            print(f"  [warn] {target_dir.name}/: {len(cands)} matches, using {cands[0].name}",
                  file=sys.stderr)
        out[fold_num] = cands[0]
    return out


def _load_per_user_target(json_path: Path, target_genre: str, metric: str):
    """Return dict {uid_str: metric_value} for cross-domain target evaluation."""
    with open(json_path) as f:
        d = json.load(f)
    cdm = d.get("cross_domain_metrics", {}).get(target_genre, {})
    pu = cdm.get("per_user", {})
    return {uid: float(v[metric]) for uid, v in pu.items() if metric in v}


def _load_per_user_source(json_path: Path, source_genre: str, metric: str):
    """Return dict {uid_str: metric_value} for source-domain evaluation."""
    with open(json_path) as f:
        d = json.load(f)
    pum = d.get("per_user_metrics", {})
    out = {}
    for uid, by_genre in pum.items():
        if source_genre in by_genre and metric in by_genre[source_genre]:
            out[uid] = float(by_genre[source_genre][metric])
    return out


def _build_user_features(users_csv: Path, ratings_csv: Path,
                          source_genre: str, target_genre: str,
                          score_col: str = "Aesthetic"):
    """Build per-user feature matrix.

    Returns DataFrame indexed by user_id (int), columns = features.
    """
    import pandas as pd
    import numpy as np

    users = pd.read_csv(users_csv)
    ratings = pd.read_csv(ratings_csv)

    # --- attribute features (from users.csv) ---
    # TIPI Big5: Q1..Q10 stored 0-based (0..6); shift to 1..7 then average each
    # factor's forward item with its reverse-scored partner (8 - reverse).
    users_idx = users.set_index("user_id")
    q = {i: users_idx[f"Q{i}"].astype(float) + 1.0 for i in range(1, 11)}
    big5 = pd.DataFrame({
        "big5_E":  (q[1]  + (8 - q[6]))  / 2.0,
        "big5_A":  ((8 - q[2])  + q[7])  / 2.0,
        "big5_C":  (q[3]  + (8 - q[8]))  / 2.0,
        "big5_ES": ((8 - q[4])  + q[9])  / 2.0,
        "big5_O":  (q[5]  + (8 - q[10])) / 2.0,
    })

    attr_cols = ["age",
                 "art_interest", "fashion_interest", "photoVideo_interest",
                 "art_learn", "fashion_learn", "photoVideo_learn"]
    cat_cols = ["gender", "nationality"]
    feat = users_idx[attr_cols].copy()
    feat = feat.join(big5)
    # edu → ordinal scale, ordered by years of formal education
    edu_order = {
        "high_school": 1,
        "vocational": 2,
        "junior_college": 3,
        "technical_college": 4,
        "university": 5,
        "graduate": 6,
        "博士": 7,
    }
    feat["edu_level"] = users_idx["edu"].map(edu_order)
    # drop_first=True avoids the dummy-variable trap (perfect collinearity)
    # so the resulting feature matrix is full-rank for linear regression.
    cats = pd.get_dummies(users_idx[cat_cols],
                          prefix=cat_cols, dummy_na=False,
                          drop_first=True, dtype=int)
    feat = feat.join(cats)

    # --- per-domain rating-style features ---
    def _style(df_g, prefix):
        agg = df_g.groupby("user_id")[score_col].agg(
            ["mean", "std",
             lambda x: float(pd.Series(x).skew()),
             lambda x: float(pd.Series(x).kurt())]
        )
        agg.columns = [f"{prefix}_mean", f"{prefix}_std",
                       f"{prefix}_skew", f"{prefix}_kurt"]
        return agg

    src_df = ratings[ratings["genre"] == source_genre]
    tgt_df = ratings[ratings["genre"] == target_genre]
    feat = feat.join(_style(src_df, f"src_{source_genre}"), how="left")
    feat = feat.join(_style(tgt_df, f"tgt_{target_genre}"), how="left")

    # --- test-retest reliability per user (within domain) ---
    def _retest_mae(df_g):
        # average over (user, sample_file) pairs that appear >=2 times
        out = {}
        grp = df_g.groupby(["user_id", "sample_file"])[score_col]
        for (uid, _), s in grp:
            if len(s) >= 2:
                out.setdefault(uid, []).append(abs(s.iloc[0] - s.iloc[1]))
        return pd.Series({uid: float(np.mean(v)) for uid, v in out.items()})

    feat[f"retest_mae_{source_genre}"] = _retest_mae(src_df)
    feat[f"retest_mae_{target_genre}"] = _retest_mae(tgt_df)

    # --- generality: corr(user's ratings, mean of others) on shared images ---
    def _generality(df_g):
        # image-level mean of all OTHER users (leave-one-out via global mean trick)
        img_sum = df_g.groupby("sample_file")[score_col].sum()
        img_cnt = df_g.groupby("sample_file")[score_col].count()
        out = {}
        for uid, sub in df_g.groupby("user_id"):
            if len(sub) < 5:
                continue
            sums = img_sum.loc[sub["sample_file"].values].values
            cnts = img_cnt.loc[sub["sample_file"].values].values
            others = (sums - sub[score_col].values) / np.maximum(cnts - 1, 1)
            x = sub[score_col].values.astype(float)
            if np.std(x) == 0 or np.std(others) == 0:
                continue
            r = float(np.corrcoef(x, others)[0, 1])
            out[uid] = r
        return pd.Series(out)

    feat[f"generality_{source_genre}"] = _generality(src_df)
    feat[f"generality_{target_genre}"] = _generality(tgt_df)

    # --- shift features |target - source| across every src/tgt-paired column ---
    # Absolute difference (magnitude only). Captures "how dissimilar across domains",
    # which is the per-user analog of domain gap / similarity.
    # scenery has no native learn/interest column; it maps to photoVideo_*.
    learn_prefix = {"art": "art", "fashion": "fashion", "scenery": "photoVideo"}
    src_lp = learn_prefix.get(source_genre, source_genre)
    tgt_lp = learn_prefix.get(target_genre, target_genre)

    for stat in ("mean", "std", "skew", "kurt"):
        feat[f"shift_{stat}_{source_genre}_to_{target_genre}"] = (
            feat[f"tgt_{target_genre}_{stat}"] - feat[f"src_{source_genre}_{stat}"]
        ).abs()
    feat[f"shift_retest_mae_{source_genre}_to_{target_genre}"] = (
        feat[f"retest_mae_{target_genre}"] - feat[f"retest_mae_{source_genre}"]
    ).abs()
    feat[f"shift_generality_{source_genre}_to_{target_genre}"] = (
        feat[f"generality_{target_genre}"] - feat[f"generality_{source_genre}"]
    ).abs()
    if src_lp != tgt_lp:
        feat[f"shift_interest_{source_genre}_to_{target_genre}"] = (
            feat[f"{tgt_lp}_interest"] - feat[f"{src_lp}_interest"]
        ).abs()
        feat[f"shift_learn_{source_genre}_to_{target_genre}"] = (
            feat[f"{tgt_lp}_learn"] - feat[f"{src_lp}_learn"]
        ).abs()

    return feat


def _benjamini_hochberg(pvals):
    """Return BH-FDR adjusted p-values (same length as input)."""
    import numpy as np
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (right-to-left cumulative min)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def _ols_fit(X, y):
    """Fit y ~ const + X. Return coefs (incl. intercept) and per-coef SE/t/p.

    X: (n, p) ndarray (NOT including a constant column).
    Returns dict with keys: coef, se, t, p (length p+1, intercept first),
    r2, adj_r2, n, p.
    """
    import numpy as np
    from scipy import stats
    n, p = X.shape
    Xc = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.pinv(Xc.T @ Xc)
    beta = XtX_inv @ Xc.T @ y
    yhat = Xc @ beta
    resid = y - yhat
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    dof = max(n - (p + 1), 1)
    sigma2 = rss / dof
    var_beta = np.diag(sigma2 * XtX_inv)
    se = np.sqrt(np.maximum(var_beta, 0.0))
    t = np.where(se > 0, beta / np.where(se > 0, se, 1.0), 0.0)
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(t), df=dof))
    r2 = 1.0 - rss / max(tss, 1e-12)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof
    return {
        "coef": beta, "se": se, "t": t, "p": pvals,
        "r2": float(r2), "adj_r2": float(adj_r2),
        "n_obs": int(n), "n_feat": int(p),
    }


def _vif(X):
    """Variance inflation factor for each column of X."""
    import numpy as np
    n, p = X.shape
    vifs = np.full(p, np.nan)
    for j in range(p):
        other = np.delete(X, j, axis=1)
        Xc = np.column_stack([np.ones(n), other])
        try:
            beta = np.linalg.pinv(Xc.T @ Xc) @ Xc.T @ X[:, j]
        except np.linalg.LinAlgError:
            continue
        yhat = Xc @ beta
        ss_res = float(((X[:, j] - yhat) ** 2).sum())
        ss_tot = float(((X[:, j] - X[:, j].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        vifs[j] = 1.0 / max(1.0 - r2, 1e-12)
    return vifs


def analyze_da_factors(args):
    """Linear-regression analysis of DA-success factors.

    For each PIAA test user (across folds), compute:
      Δ = metric(DA finetune, target) - metric(no-DA finetune, target)
    Then fit OLS on standardized features and report β / SE / t / p (BH-FDR
    adjusted) with VIF for multicollinearity diagnostics. R² and adj-R² are
    reported for the full-data fit (no cross-validation).
    `baseline_{metric}_target` and `baseline_{metric}_source` are included
    as user-level controls.

    Always runs all 6 ordered pairs of (art, fashion, scenery) and writes
    one output directory per pair.
    """
    GENRES = ("art", "fashion", "scenery")
    if getattr(args, "source_genre", None) is None or getattr(args, "target_genre", None) is None:
        import copy
        all_pairs = [(s, t) for s in GENRES for t in GENRES if s != t]
        for src, tgt in all_pairs:
            print(f"\n{'=' * 30} {src} → {tgt} {'=' * 30}")
            args_pair = copy.copy(args)
            args_pair.source_genre = src
            args_pair.target_genre = tgt
            analyze_da_factors(args_pair)
        print(f"\n{'=' * 30} aggregated canonical ranking {'=' * 30}")
        _aggregate_canonical_ranking(args, all_pairs)
        return

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    reports_dir = Path(args.reports_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / (
        f"{args.source_genre}2{args.target_genre}_{args.model_type}_{args.da_method}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric
    higher_is_better = metric in ("ccc", "srocc", "ndcg@10")

    # 1. discover JSONs
    nodA_jsons = _find_finetune_jsons(
        reports_dir, args.version, args.source_genre,
        args.model_type, da_method=None, folds=args.folds,
    )
    da_jsons = _find_finetune_jsons(
        reports_dir, args.version, f"{args.source_genre}2{args.target_genre}",
        args.model_type, da_method=args.da_method, folds=args.folds,
    )
    common_folds = sorted(set(nodA_jsons) & set(da_jsons))
    if not common_folds:
        print(f"[warn] {args.source_genre}→{args.target_genre} ({args.da_method}): "
              f"no folds with both no-DA and DA finetune JSONs found; skipping.\n"
              f"  no-DA folds: {sorted(nodA_jsons)}\n"
              f"  DA folds:    {sorted(da_jsons)}", file=sys.stderr)
        return
    print(f"[discover] folds with both no-DA and DA: {common_folds}")
    for f in common_folds:
        print(f"  fold{f}: noDA={nodA_jsons[f].name} | DA={da_jsons[f].name}")

    # 2. per-user metrics on TARGET (primary) and SOURCE (secondary)
    rows = []
    for f in common_folds:
        no_tgt = _load_per_user_target(nodA_jsons[f], args.target_genre, metric)
        da_tgt = _load_per_user_target(da_jsons[f], args.target_genre, metric)
        no_src = _load_per_user_source(nodA_jsons[f], args.source_genre, metric)
        da_src = _load_per_user_source(da_jsons[f], args.source_genre, metric)
        for uid in sorted(set(no_tgt) & set(da_tgt)):
            sign = 1.0 if higher_is_better else -1.0
            d_tgt = sign * (da_tgt[uid] - no_tgt[uid])
            d_src = (sign * (da_src.get(uid, np.nan) - no_src.get(uid, np.nan))
                     if uid in no_src and uid in da_src else np.nan)
            rows.append({
                "user_id": int(uid), "fold": f,
                f"baseline_{metric}_target": no_tgt[uid],
                f"baseline_{metric}_source": no_src.get(uid, np.nan),
                f"da_{metric}_target": da_tgt[uid],
                f"da_{metric}_source": da_src.get(uid, np.nan),
                "delta_target": d_tgt,
                "delta_source": d_src,
            })
    delta_df = pd.DataFrame(rows)
    print(f"[delta] N users (target Δ): {len(delta_df)} "
          f"across {delta_df['fold'].nunique()} folds")

    # 3. user features
    users_csv = data_dir / "maked" / "users.csv"
    ratings_csv = data_dir / "maked" / "ratings.csv"
    feat_df = _build_user_features(
        users_csv, ratings_csv,
        source_genre=args.source_genre, target_genre=args.target_genre,
        score_col=args.score_col,
    )
    merged = delta_df.merge(feat_df, left_on="user_id", right_index=True, how="left")
    merged.to_csv(out_dir / "per_user_features.csv", index=False)
    print(f"[save] per_user_features.csv ({len(merged)} rows × {merged.shape[1]} cols)")

    # 4. descriptive stats on Δ
    def _summary(s):
        s = s.dropna()
        if len(s) == 0:
            return {}
        return {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "p_positive": float((s > 0).mean()),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    desc = {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "source": args.source_genre,
        "target": args.target_genre,
        "model_type": args.model_type,
        "da_method": args.da_method,
        "delta_target": _summary(merged["delta_target"]),
        "delta_source": _summary(merged["delta_source"]),
    }
    with open(out_dir / "delta_summary.json", "w") as f:
        json.dump(desc, f, indent=2)
    print(f"[summary] Δ target: mean={desc['delta_target'].get('mean', float('nan')):.4f}, "
          f"median={desc['delta_target'].get('median', float('nan')):.4f}, "
          f"P(Δ>0)={desc['delta_target'].get('p_positive', float('nan')):.3f}, "
          f"n={desc['delta_target'].get('n', 0)}")

    # 5. build feature matrix for regression
    skip_cols = {"user_id", "fold",
                 f"baseline_{metric}_target", f"baseline_{metric}_source",
                 f"da_{metric}_target", f"da_{metric}_source",
                 "delta_target", "delta_source"}
    feat_cols = [c for c in merged.columns if c not in skip_cols]
    # Drop learn_* features: learn_src/learn_tgt/shift_learn_* are linearly
    # dependent (shift = |tgt - src|), producing VIF ≈ 1e12 and rank-deficient X.
    dropped_learn = [c for c in feat_cols
                     if c.endswith("_learn") or c.startswith("shift_learn_")]
    if dropped_learn:
        print(f"[filter] drop learn-* (collinear with shift_learn_*): {dropped_learn}")
    feat_cols = [c for c in feat_cols if c not in dropped_learn]
    # include source/target baseline as additional features:
    # - baseline_target: regression to the mean control
    # - baseline_source: how well the user is modeled on source (no-DA)
    feat_cols = [f"baseline_{metric}_target",
                 f"baseline_{metric}_source"] + feat_cols

    # coerce to numeric, drop constant / near-empty columns
    Xy = merged[feat_cols + ["delta_target"]].apply(pd.to_numeric, errors="coerce")
    keep = []
    for c in feat_cols:
        s = Xy[c]
        if s.notna().sum() < 10:
            continue
        if s.dropna().nunique() < 2:
            continue
        keep.append(c)
    feat_cols = keep
    Xy = Xy[feat_cols + ["delta_target"]].dropna()
    y = Xy["delta_target"].values.astype(float)
    X_raw = Xy[feat_cols].values.astype(float)
    n_obs, p = X_raw.shape
    if n_obs < p + 5 or p == 0:
        print(f"[warn] not enough samples for regression "
              f"(n={n_obs}, p={p}); skipping linear analysis", file=sys.stderr)
        print(f"[done] outputs in: {out_dir}")
        return

    # standardize so β are directly comparable (β = effect of +1 SD on Δ)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_raw)
    print(f"[regression] n={n_obs} users, p={p} features (post-filter)")

    # 5a. OLS with SE / t / p (BH-FDR) and VIF
    ols = _ols_fit(Xz, y)
    vifs = _vif(Xz)
    p_fdr = _benjamini_hochberg(ols["p"][1:])  # exclude intercept

    # 5b. feature-importance table
    reg_df = pd.DataFrame({
        "feature": feat_cols,
        "ols_coef_std": ols["coef"][1:],
        "ols_se": ols["se"][1:],
        "ols_t": ols["t"][1:],
        "ols_p": ols["p"][1:],
        "ols_p_fdr": p_fdr,
        "vif": vifs,
    })
    reg_df = (reg_df.assign(_abs=reg_df["ols_coef_std"].abs())
                    .sort_values("_abs", ascending=False)
                    .drop(columns="_abs")
                    .reset_index(drop=True))
    reg_df.to_csv(out_dir / "regression_results.csv", index=False)
    print(f"[save] regression_results.csv ({len(reg_df)} features)")

    summary = {
        "metric": metric,
        "n_users": int(n_obs),
        "n_features": int(p),
        "ols_R2": ols["r2"],
        "ols_R2_adj": ols["adj_r2"],
    }
    with open(out_dir / "regression_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OLS] R²={ols['r2']:.3f}, adj-R²={ols['adj_r2']:.3f}")
    print("[top-10 |OLS β| (standardized features)]")
    for _, r in reg_df.head(10).iterrows():
        sig = "*" if r["ols_p_fdr"] < 0.05 else " "
        print(f"  {sig} {r['feature']:42s} "
              f"β={r['ols_coef_std']:+.3f} (SE={r['ols_se']:.3f}, "
              f"p_fdr={r['ols_p_fdr']:.4f}, VIF={r['vif']:.1f})")

    # 6. plots
    if not args.no_plots:
        title_tag = (f"{args.source_genre}→{args.target_genre} | "
                     f"{args.model_type} | {args.da_method} | metric={metric}")
        # 6a. histogram of Δ
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, col, lbl in zip(
            axes, ["delta_target", "delta_source"], ["target", "source"]
        ):
            s = merged[col].dropna()
            if len(s) == 0:
                ax.set_title(f"Δ {lbl} (no data)")
                continue
            ax.hist(s, bins=30, edgecolor="black", alpha=0.75)
            ax.axvline(0, color="red", lw=1, label="Δ=0")
            ax.axvline(s.mean(), color="green", lw=1, ls="--",
                       label=f"mean={s.mean():.3f}")
            ax.set_xlabel(f"Δ {metric} ({lbl})")
            ax.set_ylabel("# users")
            ax.set_title(f"Δ {lbl}  (n={len(s)}, P(Δ>0)={(s>0).mean():.2f})")
            ax.legend(fontsize=8)
        fig.suptitle(title_tag, fontsize=11)
        fig.tight_layout()
        out_path = out_dir / "delta_histogram.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {out_path}")

        # 6b. feature-importance bar plot (top-K OLS β with 95% CI)
        top = reg_df.head(min(args.top_k, len(reg_df))).iloc[::-1]  # bottom→top
        fig, ax = plt.subplots(
            figsize=(7, max(3.0, 0.35 * len(top) + 1.5)),
        )
        ols_colors = ["tab:red" if pf < 0.05 else "tab:gray"
                      for pf in top["ols_p_fdr"]]
        ax.barh(range(len(top)), top["ols_coef_std"].values,
                xerr=top["ols_se"].values * 1.96,
                color=ols_colors, alpha=0.8, ecolor="black", capsize=2)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=8)
        ax.set_xlabel("OLS β (standardized, ±95% CI)")
        ax.set_title(f"OLS  adj-R²={ols['adj_r2']:.3f}  "
                     f"(red = p_fdr<0.05)", fontsize=10)
        fig.suptitle(title_tag, fontsize=11)
        fig.tight_layout()
        out_path = out_dir / "feature_importance.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {out_path}")

    print(f"[done] outputs in: {out_dir}")


def basic_stats(args):
    """ratings.csv と users.csv から基礎的な統計量を算出して表示する.

    出力内容:
      - データ全体のサイズ (users / ratings / 一意な sample 数)
      - ジャンル(ドメイン)別: rating 数, 一意 sample 数, 参加ユーザー数
      - set / fold 別: rating 数, ユーザー数
      - ユーザー属性 (gender / edu / nationality / age) の分布
      - ジャンル別の主要評定列 (Aesthetic / Like / Beautiful) と応答時間 Time の要約
    """
    import pandas as pd

    data_dir = Path(args.data_dir)
    ratings_csv = data_dir / "ratings.csv"
    users_csv = data_dir / "users.csv"

    if not ratings_csv.exists():
        print(f"Error: not found: {ratings_csv}", file=sys.stderr)
        sys.exit(1)
    if not users_csv.exists():
        print(f"Error: not found: {users_csv}", file=sys.stderr)
        sys.exit(1)

    ratings = pd.read_csv(ratings_csv)
    users = pd.read_csv(users_csv)

    def _hr(title):
        print()
        print(f"=== {title} ===")

    _hr("Files")
    print(f"  ratings_csv: {ratings_csv}  ({len(ratings):,} rows)")
    print(f"  users_csv  : {users_csv}  ({len(users):,} rows)")

    _hr("Overall")
    print(f"  # users (users.csv)        : {len(users):,}")
    print(f"  # users in ratings         : {ratings['user_id'].nunique():,}")
    print(f"  # ratings (rows)           : {len(ratings):,}")
    print(f"  # unique samples           : {ratings['sample_file'].nunique():,}")
    print(f"  # genres (domains)         : {ratings['genre'].nunique()}  "
          f"({sorted(ratings['genre'].dropna().unique().tolist())})")

    _hr("By genre (domain)")
    by_g = ratings.groupby("genre", dropna=False).agg(
        n_ratings=("user_id", "size"),
        n_users=("user_id", "nunique"),
        n_samples=("sample_file", "nunique"),
    ).sort_values("n_ratings", ascending=False)
    print(f"  {'genre':<12}{'n_ratings':>12}{'n_users':>10}{'n_samples':>12}"
          f"{'ratings/user':>14}{'ratings/sample':>16}")
    for g, row in by_g.iterrows():
        rpu = row["n_ratings"] / row["n_users"] if row["n_users"] else 0
        rps = row["n_ratings"] / row["n_samples"] if row["n_samples"] else 0
        print(f"  {str(g):<12}{int(row['n_ratings']):>12,}"
              f"{int(row['n_users']):>10,}{int(row['n_samples']):>12,}"
              f"{rpu:>14.2f}{rps:>16.2f}")

    if "set" in ratings.columns:
        _hr("By set")
        by_s = ratings.groupby("set", dropna=False).agg(
            n_ratings=("user_id", "size"),
            n_users=("user_id", "nunique"),
            n_samples=("sample_file", "nunique"),
        ).sort_index()
        print(f"  {'set':<6}{'n_ratings':>12}{'n_users':>10}{'n_samples':>12}")
        for s, row in by_s.iterrows():
            print(f"  {str(s):<6}{int(row['n_ratings']):>12,}"
                  f"{int(row['n_users']):>10,}{int(row['n_samples']):>12,}")

    if "fold" in ratings.columns:
        _hr("By fold")
        by_f = ratings.groupby("fold", dropna=False).agg(
            n_ratings=("user_id", "size"),
            n_users=("user_id", "nunique"),
        ).sort_index()
        print(f"  {'fold':<6}{'n_ratings':>12}{'n_users':>10}")
        for f_, row in by_f.iterrows():
            print(f"  {str(f_):<6}{int(row['n_ratings']):>12,}"
                  f"{int(row['n_users']):>10,}")

    _hr("Genre × fold (n_users)")
    if {"genre", "fold"}.issubset(ratings.columns):
        gf = ratings.groupby(["genre", "fold"])["user_id"].nunique().unstack(fill_value=0)
        print(gf.to_string())

    _hr("User demographics (users.csv)")
    if "age" in users.columns:
        age = pd.to_numeric(users["age"], errors="coerce").dropna()
        print(f"  age: n={len(age)}  mean={age.mean():.2f}  std={age.std():.2f}  "
              f"median={age.median():.1f}  min={int(age.min())}  max={int(age.max())}")
    for col in ("gender", "edu", "nationality"):
        if col in users.columns:
            vc = users[col].value_counts(dropna=False)
            total = vc.sum()
            print(f"  {col}:")
            for v, c in vc.items():
                pct = 100.0 * c / total if total else 0
                print(f"    {str(v):<20}{int(c):>6,}  ({pct:5.1f}%)")

    _hr("Domain experience (binary)")
    # ドメイン経験を 2 値化:
    #   learn_bin    = (*_learn   == 1)   ← もとから 0/1
    #   interest_bin = (*_interest >  0)   ← 0-6 Likert を >0 で 2 値化
    #   any_exp      = learn_bin OR interest_bin
    # scenery は photoVideo_* 列にマップ (analyze_da_factors と同じ規約).
    learn_prefix = {"art": "art", "fashion": "fashion", "scenery": "photoVideo"}
    n_users_total = len(users)
    print(f"  {'domain':<10}{'learn=1':>14}{'interest>0':>16}"
          f"{'any_exp':>14}   (n_users={n_users_total})")
    for dom in ("art", "fashion", "scenery"):
        lp = learn_prefix[dom]
        lc = f"{lp}_learn"
        ic = f"{lp}_interest"
        if lc not in users.columns or ic not in users.columns:
            continue
        learn_bin = (pd.to_numeric(users[lc], errors="coerce").fillna(0) > 0)
        intr_bin = (pd.to_numeric(users[ic], errors="coerce").fillna(0) > 0)
        any_bin = learn_bin | intr_bin
        def _fmt(s):
            n = int(s.sum())
            pct = 100.0 * n / n_users_total if n_users_total else 0
            return f"{n:>4} ({pct:5.1f}%)"
        print(f"  {dom:<10}{_fmt(learn_bin):>14}{_fmt(intr_bin):>16}"
              f"{_fmt(any_bin):>14}")

    # 評定参加者のうちドメイン経験ありの割合 (genre × any_exp).
    if "user_id" in users.columns and "user_id" in ratings.columns:
        print()
        print(f"  Among raters of each genre (any_exp = learn=1 or interest>0):")
        print(f"    {'genre':<10}{'raters':>10}{'with_exp':>12}{'pct':>10}")
        for dom in ("art", "fashion", "scenery"):
            lp = learn_prefix[dom]
            lc, ic = f"{lp}_learn", f"{lp}_interest"
            if lc not in users.columns or ic not in users.columns:
                continue
            any_bin = (
                (pd.to_numeric(users[lc], errors="coerce").fillna(0) > 0)
                | (pd.to_numeric(users[ic], errors="coerce").fillna(0) > 0)
            )
            exp_uids = set(users.loc[any_bin, "user_id"].astype(int).tolist())
            raters = ratings.loc[ratings["genre"] == dom, "user_id"].unique()
            n_r = len(raters)
            n_exp = sum(1 for u in raters if int(u) in exp_uids)
            pct = 100.0 * n_exp / n_r if n_r else 0
            print(f"    {dom:<10}{n_r:>10,}{n_exp:>12,}{pct:>9.1f}%")

    _hr("Score distribution by genre")
    score_cols = [c for c in ("Aesthetic", "Like", "Beautiful") if c in ratings.columns]
    for col in score_cols:
        print(f"  [{col}]")
        sub = ratings.groupby("genre")[col].agg(["count", "mean", "std", "min", "max"])
        for g, row in sub.iterrows():
            print(f"    {str(g):<12}n={int(row['count']):>7,}  "
                  f"mean={row['mean']:.3f}  std={row['std']:.3f}  "
                  f"min={row['min']:.0f}  max={row['max']:.0f}")

    if "Time" in ratings.columns:
        _hr("Response time (Time, seconds) by genre")
        t = ratings.copy()
        t["Time"] = pd.to_numeric(t["Time"], errors="coerce")
        sub = t.groupby("genre")["Time"].agg(["count", "mean", "median", "std",
                                              "min", "max"])
        for g, row in sub.iterrows():
            print(f"    {str(g):<12}n={int(row['count']):>7,}  "
                  f"mean={row['mean']:.2f}  median={row['median']:.2f}  "
                  f"std={row['std']:.2f}  min={row['min']:.2f}  max={row['max']:.2f}")

    print()


def plot_user_traits(args):
    """users.csv から Domain Interest と Personality Trait (TIPI Big5) の
    分布を 2 枚並びの箱ひげ図で可視化する.

    - Domain Interest:  art_interest / fashion_interest / photoVideo_interest
                        (0-6 Likert)
    - Personality Trait: TIPI Big5 (Q1..Q10 → 1-7 スケールで forward + reverse 平均)
                         Extraversion / Agreeableness / Conscientiousness /
                         Emotional Stability / Openness
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    users_csv = Path(args.users_csv)
    if not users_csv.exists():
        print(f"Error: not found: {users_csv}", file=sys.stderr)
        sys.exit(1)

    users = pd.read_csv(users_csv)

    # --- Domain Interest ---
    # 元データは 0-6 Likert. Personality Trait と同じ 1-7 スケールに揃えるため +1.
    interest_map = {
        "Art": "art_interest",
        "Fashion": "fashion_interest",
        "Scenery": "photoVideo_interest",
    }
    interest_data = {}
    for label, col in interest_map.items():
        if col not in users.columns:
            print(f"Error: column '{col}' not found in {users_csv}", file=sys.stderr)
            sys.exit(1)
        interest_data[label] = (
            pd.to_numeric(users[col], errors="coerce").dropna().values + 1.0
        )

    # --- Personality Trait: TIPI Big5 ---
    # Q1..Q10 は 0..6 で保存されているので +1 して 1..7 に. 逆転項目は (8 - x).
    for i in range(1, 11):
        if f"Q{i}" not in users.columns:
            print(f"Error: column 'Q{i}' not found in {users_csv}", file=sys.stderr)
            sys.exit(1)
    q = {i: pd.to_numeric(users[f"Q{i}"], errors="coerce") + 1.0 for i in range(1, 11)}
    big5 = pd.DataFrame({
        "Ext.": (q[1]  + (8 - q[6]))  / 2.0,
        "Agr.": ((8 - q[2])  + q[7])  / 2.0,
        "Con.": (q[3]  + (8 - q[8]))  / 2.0,
        "E.S.": ((8 - q[4])  + q[9])  / 2.0,
        "Opn.": (q[5]  + (8 - q[10])) / 2.0,
    })

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             gridspec_kw={"width_ratios": [3, 5]})

    fs = float(args.font_size)
    mean_props = {"marker": "D", "markerfacecolor": "red",
                  "markeredgecolor": "red", "markersize": max(4, fs * 0.45)}
    median_props = {"color": "black", "linewidth": 1.5}

    def _color_boxes(bp, colors):
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor("black")
            patch.set_alpha(0.7)

    def _style_axis(ax):
        ax.set_ylim(0.5, 7.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", labelsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.yaxis.label.set_size(fs)
        ax.title.set_size(fs * 1.1)

    ax = axes[0]
    labels = list(interest_data.keys())
    data = [interest_data[k] for k in labels]
    interest_colors = ["#4C72B0", "#DD8452", "#55A467"]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True,
                    patch_artist=True,
                    meanprops=mean_props, medianprops=median_props)
    _color_boxes(bp, interest_colors)
    ax.set_title(f"Domain Interest  (n={len(users):,})")
    ax.set_ylabel("Interest score (1-7)")
    _style_axis(ax)

    ax = axes[1]
    labels = list(big5.columns)
    data = [big5[c].dropna().values for c in labels]
    big5_colors = ["#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True,
                    patch_artist=True,
                    meanprops=mean_props, medianprops=median_props)
    _color_boxes(bp, big5_colors)
    ax.set_title(f"Personality Trait (TIPI Big5)  (n={len(users):,})")
    ax.set_ylabel("Score (1-7)")
    _style_axis(ax)

    fig.tight_layout()

    out_path = Path(args.output)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_rating_histograms(args):
    """ratings.csv から美的評価＋美的感情の正規化ヒストグラムを 2×5 で描画する.

    各サブプロットは 1-7 Likert 上の確率分布 (count / n_per_genre) で,
    ジャンル (art / fashion / scenery) ごとに横並び棒グラフを描画.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratings_csv = Path(args.ratings_csv)
    if not ratings_csv.exists():
        print(f"Error: not found: {ratings_csv}", file=sys.stderr)
        sys.exit(1)

    ratings = pd.read_csv(ratings_csv)

    cols = list(args.columns)
    if len(cols) != 10:
        print(f"Error: --columns must have exactly 10 names "
              f"(got {len(cols)}: {cols})", file=sys.stderr)
        sys.exit(1)
    missing = [c for c in cols if c not in ratings.columns]
    if missing:
        print(f"Error: columns not in ratings.csv: {missing}", file=sys.stderr)
        sys.exit(1)
    genres = list(args.genres)
    missing_g = [g for g in genres if g not in set(ratings["genre"].unique())]
    if missing_g:
        print(f"Error: genres not found in ratings.csv: {missing_g}",
              file=sys.stderr)
        sys.exit(1)

    fs = float(args.font_size)

    # 添付画像に倣ったジャンル別スタイル
    genre_styles = {
        "art":     {"color": "#F2A93B", "hatch": ""},
        "fashion": {"color": "#A6CE5A", "hatch": "//"},
        "scenery": {"color": "#5B8AB8", "hatch": "xx"},
    }

    bins_centers = np.arange(1, 8)            # 1..7
    bin_edges = np.arange(0.5, 8.5, 1.0)
    n_g = len(genres)
    bar_width = 0.8 / n_g
    offsets = np.linspace(-(n_g - 1) / 2.0, (n_g - 1) / 2.0, n_g) * bar_width

    fig, axes = plt.subplots(2, 5,
                             figsize=tuple(args.figsize),
                             sharex=True, sharey=args.share_y)
    axes_flat = axes.flatten()

    handles, labels_ = [], []
    for i, col in enumerate(cols):
        ax = axes_flat[i]
        for g_idx, g in enumerate(genres):
            sub = ratings.loc[ratings["genre"] == g, col]
            vals = pd.to_numeric(sub, errors="coerce").dropna() + 1.0  # 0-6 → 1-7
            counts, _ = np.histogram(vals, bins=bin_edges)
            total = counts.sum()
            prop = counts / total if total else counts
            x = bins_centers + offsets[g_idx]
            style = genre_styles.get(g, {"color": f"C{g_idx}", "hatch": ""})
            bars = ax.bar(x, prop, width=bar_width,
                          color=style["color"], edgecolor="black",
                          linewidth=0.6, hatch=style["hatch"], label=g)
            if i == 0:
                handles.append(bars[0])
                labels_.append(g)

        ax.set_title(col, fontsize=fs * 1.05)
        ax.set_xticks(bins_centers)
        ax.tick_params(axis="x", labelsize=fs * 0.9)
        ax.tick_params(axis="y", labelsize=fs * 0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if i % 5 == 0:
            ax.set_ylabel("Proportion", fontsize=fs)

    fig.legend(handles, labels_, loc="upper center",
               ncol=n_g, fontsize=fs, frameon=True,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path(args.output)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    if not args.no_stats:
        print()
        print("=== Per-item rating statistics "
              "(raw + 1, plot display scale; per-item range visible in min/max) ===")
        header = (f"  {'genre':<10}{'n':>7}{'mean':>8}{'std':>8}"
                  f"{'median':>9}{'Q1':>7}{'Q3':>7}{'min':>5}{'max':>5}{'mode':>6}")
        for col in cols:
            print()
            print(f"[{col}]")
            print(header)
            # per-genre rows + aggregate
            rows = []
            for g in genres:
                v = pd.to_numeric(
                    ratings.loc[ratings["genre"] == g, col],
                    errors="coerce",
                ).dropna() + 1.0
                rows.append((g, v))
            v_all = pd.to_numeric(
                ratings.loc[ratings["genre"].isin(genres), col],
                errors="coerce",
            ).dropna() + 1.0
            rows.append(("all", v_all))
            for name, v in rows:
                if len(v) == 0:
                    print(f"  {name:<10}{0:>7}  (no data)")
                    continue
                mode_val = int(v.round().mode().iloc[0])
                print(
                    f"  {name:<10}{len(v):>7,}"
                    f"{v.mean():>8.2f}{v.std():>8.2f}"
                    f"{v.median():>9.2f}"
                    f"{v.quantile(0.25):>7.2f}{v.quantile(0.75):>7.2f}"
                    f"{int(v.min()):>5}{int(v.max()):>5}"
                    f"{mode_val:>6}"
                )
        print()


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
        "--ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific run IDs to include (e.g., --ids 61 65 70). Only files whose ID matches one of these values are aggregated.",
    )
    agg_parser.add_argument(
        "--min-id",
        type=int,
        default=None,
        dest="min_id",
        help="Minimum run ID to include (e.g., 61 filters to files with ID >= 61, like 'name-61_pretrain.json')",
    )
    agg_parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        dest="max_id",
        help="Maximum run ID to include (e.g., 80 filters to files with ID <= 80, like 'name-80_pretrain.json')",
    )
    agg_parser.add_argument(
        "--reports_dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Path to reports/exp directory",
    )
    agg_parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        dest="data_dir",
        help="Path to data directory containing split/ and maked/ (used with --pattern claude). "
             "Default: <project_root>/data",
    )
    agg_parser.add_argument(
        "--giaa_mode",
        action="store_true",
        default=False,
        dest="giaa_mode",
        help="Aggregate GIAA results (EMD/SROCC/MSE/MAE/CCC). "
             "For NN: reads average_metrics from GIAA JSONs. "
             "For LLM (--pattern claude/gemini/gpt): evaluates image-level predictions against test_images_GIAA.txt.",
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

    # Subcommand: visualize_features
    vf_parser = subparsers.add_parser(
        "visualize_features",
        help="Visualize DA vs non-DA model features on target domain (2D projection)",
    )
    vf_parser.add_argument(
        "--source-genre", type=str, default="art", dest="source_genre",
        help="Source domain genre used to train the model (default: art)",
    )
    vf_parser.add_argument(
        "--target-genre", type=str, default="fashion", dest="target_genre",
        help="Target domain genre to visualize features on (default: fashion)",
    )
    vf_parser.add_argument(
        "--dataset-ver", type=str, default="v1", dest="dataset_ver",
        help="Dataset version prefix for fold discovery (default: v1)",
    )
    vf_parser.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="Specific fold numbers to use (e.g. --folds 1 3). If omitted, all folds are used.",
    )
    vf_parser.add_argument(
        "--backbone", type=str, default="clip_vit_b16",
        choices=["resnet50", "vit_b_16", "clip_rn50", "clip_vit_b16"],
        help="Backbone architecture (must match saved model, default: clip_vit_b16)",
    )
    vf_parser.add_argument(
        "--method", type=str, default="tsne",
        choices=["tsne", "umap", "pca", "all"],
        help="Dimensionality reduction method; 'all' runs tsne/umap/pca and saves each (default: tsne)",
    )
    vf_parser.add_argument(
        "--percentile", type=float, default=25.0,
        help="Bottom/top percentile for low/high class split (default: 25 → bottom 25%% = low, top 25%% = high)",
    )
    vf_parser.add_argument(
        "--hide-mid", action="store_true", dest="hide_mid",
        help="Hide the mid class from the plot (show only low and high)",
    )
    vf_parser.add_argument(
        "--score-only", action="store_true", dest="score_only",
        help="Only compute Silhouette Score; skip dimensionality reduction and plotting",
    )
    vf_parser.add_argument(
        "--root-dir", type=str,
        default="/home/hayashi0884/proj-xpass-DA/data",
        dest="root_dir",
        help="Root data directory containing maked/ and split/ (default: proj-xpass-DA/data)",
    )
    vf_parser.add_argument(
        "--models-pth-dir", type=str,
        default="/home/hayashi0884/proj-xpass-DA/models_pth",
        dest="models_pth_dir",
        help="Root directory of saved .pth models (default: proj-xpass-DA/models_pth)",
    )
    vf_parser.add_argument(
        "--uda-methods", type=str, nargs="+", default=["DANN"],
        dest="uda_methods",
        help="UDA method name(s) to compare against Non-DA (e.g. DANN, DJDOT). "
             "Multiple values produce one subplot per method. (default: DANN)",
    )
    vf_parser.add_argument(
        "-o", "--output-dir", default="reports/feature_viz",
        dest="output_dir",
        help="Output directory for figures; filenames are auto-generated as "
             "{source}2{target}_{methods}_{dim_method}.png (default: reports/feature_viz)",
    )

    # Subcommand: visualize_domain_gap
    vdg_parser = subparsers.add_parser(
        "visualize_domain_gap",
        help="Visualize domain gap reduction between source and target (non-DA vs DA)",
    )
    vdg_parser.add_argument(
        "--source-genre", type=str, default="art", dest="source_genre",
        help="Source domain genre (default: art)",
    )
    vdg_parser.add_argument(
        "--target-genre", type=str, default="fashion", dest="target_genre",
        help="Target domain genre (default: fashion)",
    )
    vdg_parser.add_argument(
        "--dataset-ver", type=str, default="v1", dest="dataset_ver",
        help="Dataset version prefix for fold discovery (default: v1)",
    )
    vdg_parser.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="Specific fold numbers to use (e.g. --folds 1 3). If omitted, all folds are used.",
    )
    vdg_parser.add_argument(
        "--split-file", type=str, default="train_images_GIAA.txt", dest="split_file",
        help="Image list filename inside each fold/<genre>/ directory (default: train_images_GIAA.txt)",
    )
    vdg_parser.add_argument(
        "--n-source", type=int, default=None, dest="n_source",
        help="Max number of source images per fold (default: all)",
    )
    vdg_parser.add_argument(
        "--n-target", type=int, default=None, dest="n_target",
        help="Max number of target images per fold (default: all)",
    )
    vdg_parser.add_argument(
        "--backbone", type=str, default="clip_vit_b16",
        choices=["resnet50", "vit_b_16", "clip_rn50", "clip_vit_b16"],
        help="Backbone architecture (must match saved model, default: clip_vit_b16)",
    )
    vdg_parser.add_argument(
        "--method", type=str, default="tsne",
        choices=["tsne", "umap", "pca", "all"],
        help="Dimensionality reduction method; 'all' runs tsne/umap/pca (default: tsne)",
    )
    vdg_parser.add_argument(
        "--score-only", action="store_true", dest="score_only",
        help="Only compute Silhouette Score; skip dimensionality reduction and plotting",
    )
    vdg_parser.add_argument(
        "--root-dir", type=str,
        default="/home/hayashi0884/proj-xpass-DA/data",
        dest="root_dir",
        help="Root data directory containing maked/ and split/ (default: proj-xpass-DA/data)",
    )
    vdg_parser.add_argument(
        "--models-pth-dir", type=str,
        default="/home/hayashi0884/proj-xpass-DA/models_pth",
        dest="models_pth_dir",
        help="Root directory of saved .pth models (default: proj-xpass-DA/models_pth)",
    )
    vdg_parser.add_argument(
        "--uda-methods", type=str, nargs="+", default=["DANN"],
        dest="uda_methods",
        help="UDA method name(s) to compare against Non-DA (e.g. DANN, DJDOT). "
             "Multiple values produce one subplot per method. (default: DANN)",
    )
    vdg_parser.add_argument(
        "-o", "--output-dir", default="reports/feature_viz",
        dest="output_dir",
        help="Output directory for figures; filenames are auto-generated as "
             "{source}2{target}_{methods}_domain_gap_{dim_method}.png (default: reports/feature_viz)",
    )

    # Subcommand: analyze_da_factors
    adf_parser = subparsers.add_parser(
        "analyze_da_factors",
        help="Linear-regression analysis of DA-success factors "
             "(Δ = DA finetune − no-DA finetune, per user, on target domain). "
             "Fits OLS on standardized features and reports β / SE / FDR-adjusted p / "
             "VIF and R² / adj-R² (single fit, no cross-validation).",
    )
    adf_parser.add_argument("--version", type=str, required=True,
                            help="Dataset version (e.g., v3) → searches v3_fold*/")
    adf_parser.add_argument("--model-type", type=str, default="ICI",
                            dest="model_type", choices=["ICI", "MIR"],
                            help="PIAA model type (default: ICI)")
    adf_parser.add_argument("--da-method", type=str, required=True,
                            dest="da_method",
                            help="DA method tag in DA filename (e.g., DJDOT, DANN, "
                                 "DAREGRAM, MCD)")
    adf_parser.add_argument("--metric", type=str, default="ccc",
                            choices=["ccc", "srocc", "ndcg@10", "mae"],
                            help="Per-user metric to use for Δ (default: ccc). "
                                 "For mae, sign is flipped so Δ>0 always = improvement.")
    adf_parser.add_argument("--folds", type=int, nargs="+", default=None,
                            help="Restrict to specific fold numbers (default: all common)")
    adf_parser.add_argument("--score-col", type=str, default="Aesthetic",
                            dest="score_col",
                            help="Rating column for style/consistency (default: Aesthetic)")
    adf_parser.add_argument("--reports-dir", type=str,
                            default=str(REPORTS_DIR),
                            dest="reports_dir",
                            help="Reports directory (default: reports/exp)")
    adf_parser.add_argument("--data-dir", type=str,
                            default=str(Path(__file__).resolve().parent.parent / "data"),
                            dest="data_dir",
                            help="Data directory containing maked/users.csv, ratings.csv")
    adf_parser.add_argument("-o", "--output-dir", type=str,
                            default="reports/da_factors",
                            dest="output_dir",
                            help="Output directory (default: reports/da_factors)")
    adf_parser.add_argument("--top-k", type=int, default=10, dest="top_k",
                            help="# of top features (by |OLS β|) in the feature-importance "
                                 "bar plot (default: 15)")
    adf_parser.add_argument("--no-plots", action="store_true", dest="no_plots",
                            help="Skip plot generation")

    # Subcommand: basic_stats
    bs_parser = subparsers.add_parser(
        "basic_stats",
        help="ratings.csv と users.csv から基礎的な統計量を算出 "
             "(ユーザー数, ドメイン別サンプル数, 評定/応答時間の要約 など)",
    )
    bs_parser.add_argument(
        "--data-dir", type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "maked"),
        dest="data_dir",
        help="ratings.csv と users.csv を含むディレクトリ "
             "(default: <project_root>/data/maked)",
    )

    # Subcommand: plot_user_traits
    put_parser = subparsers.add_parser(
        "plot_user_traits",
        help="users.csv から Domain Interest と Personality Trait (TIPI Big5) の "
             "分布を箱ひげ図で可視化",
    )
    put_parser.add_argument(
        "--users-csv", type=str,
        default=str(Path(__file__).resolve().parent.parent
                    / "data" / "maked" / "users.csv"),
        dest="users_csv",
        help="users.csv のパス (default: <project_root>/data/maked/users.csv)",
    )
    put_parser.add_argument(
        "-o", "--output", type=str,
        default="reports/user_traits.png",
        help="出力する図のパス (default: reports/user_traits.png)",
    )
    put_parser.add_argument(
        "--font-size", type=float, default=12.0,
        dest="font_size",
        help="軸ラベル・目盛・タイトルの基本フォントサイズ (default: 12)",
    )

    # Subcommand: plot_rating_histograms
    prh_parser = subparsers.add_parser(
        "plot_rating_histograms",
        help="ratings.csv から美的評価＋美的感情の正規化ヒストグラムを "
             "2×5 で描画 (ジャンル別)",
    )
    prh_parser.add_argument(
        "--ratings-csv", type=str,
        default=str(Path(__file__).resolve().parent.parent
                    / "data" / "maked" / "ratings.csv"),
        dest="ratings_csv",
        help="ratings.csv のパス (default: <project_root>/data/maked/ratings.csv)",
    )
    prh_parser.add_argument(
        "--columns", type=str, nargs="+",
        default=["Aesthetic", "Like", "Beautiful",
                 "Impressed", "Intellectually",
                 "Motivated", "Amused", "Nostalgic", "Sad", "Distasteful"],
        help="プロットする 10 カラム (row-major で 2×5 に並ぶ. "
             "default: 美的評価 3 + 美的感情 7)",
    )
    prh_parser.add_argument(
        "--genres", type=str, nargs="+",
        default=["art", "fashion", "scenery"],
        help="比較するジャンル (default: art fashion scenery)",
    )
    prh_parser.add_argument(
        "--share-y", action=argparse.BooleanOptionalAction,
        default=True, dest="share_y",
        help="サブプロット間で Y 軸スケールを共有 (default: True. "
             "個別スケールにしたいときは --no-share-y)",
    )
    prh_parser.add_argument(
        "--figsize", type=float, nargs=2, default=[18.0, 7.0],
        help="figure サイズ (W H) (default: 18 7)",
    )
    prh_parser.add_argument(
        "--font-size", type=float, default=12.0, dest="font_size",
        help="フォントサイズ (default: 12)",
    )
    prh_parser.add_argument(
        "-o", "--output", type=str,
        default="reports/rating_histograms.png",
        help="出力する図のパス (default: reports/rating_histograms.png)",
    )
    prh_parser.add_argument(
        "--no-stats", action="store_true", dest="no_stats",
        help="項目ごとの基本統計量 (n, mean, std, median, Q1, Q3, min, max, mode) "
             "の標準出力をスキップ",
    )

    args = parser.parse_args()

    if args.command == 'aggregate':
        aggregate(args)
    elif args.command == 'plot_quality':
        plot_quality(args)
    elif args.command == 'visualize_features':
        visualize_features(args)
    elif args.command == 'visualize_domain_gap':
        visualize_domain_gap(args)
    elif args.command == 'analyze_da_factors':
        analyze_da_factors(args)
    elif args.command == 'basic_stats':
        basic_stats(args)
    elif args.command == 'plot_user_traits':
        plot_user_traits(args)
    elif args.command == 'plot_rating_histograms':
        plot_rating_histograms(args)
    else:
        parser.print_help()
