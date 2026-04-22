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

    args = parser.parse_args()

    if args.command == 'aggregate':
        aggregate(args)
    elif args.command == 'plot_quality':
        plot_quality(args)
    elif args.command == 'visualize_features':
        visualize_features(args)
    elif args.command == 'visualize_domain_gap':
        visualize_domain_gap(args)
    else:
        parser.print_help()
