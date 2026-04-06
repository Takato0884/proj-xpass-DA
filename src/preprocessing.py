import pandas as pd
import json
import ast
from pathlib import Path
import random
from typing import List, Optional

def make_user_csv(raw_dir: str, output_path: Optional[str] = None):
    """Build a filtered user CSV from raw inputs.

    Steps:
      - Read raw annotation and user CSVs from raw_dir
      - Filter annotations where order == 3.6 and step == 37
      - Collect UUIDs from filtered annotations and filter the user CSV to those UUIDs
      - Compute user_id based on appearance order in the filtered user rows
      - Parse titpj into Q1..Q10 (0-based), and fill art/fashion/photoVideo learn (binary from year != -1) and interest (0-based)
      - Save selected columns to the provided output_path

    Returns:
      pd.DataFrame: The filtered annotation DataFrame (for optional downstream use)
    """
    raw_dir = Path(raw_dir)
    ann_path = raw_dir / "user-annotation-data_rows.csv"
    user_path = raw_dir / "user-data_rows.csv"

    # Read raw CSVs
    ann_df = pd.read_csv(ann_path)
    users_df = pd.read_csv(user_path)

    # Filter annotations where order == 3.6 (robust to numeric/str)
    filtered_ann_df = ann_df.copy()
    if "order" in filtered_ann_df.columns:
        order_num = pd.to_numeric(filtered_ann_df["order"], errors="coerce")
        mask_num = order_num == 3.6
        mask_str = filtered_ann_df["order"].astype(str) == "3.6"
        mask = mask_num | mask_str
        filtered_ann_df = filtered_ann_df[mask]

    # Further filter: step == 37 if available
    step_filtered_df = filtered_ann_df
    if "step" in step_filtered_df.columns:
        step_num = pd.to_numeric(step_filtered_df["step"], errors="coerce")
        mask_num = step_num == 37
        mask_str = step_filtered_df["step"].astype(str) == "37"
        step_mask = mask_num | mask_str
        step_filtered_df = step_filtered_df[step_mask]

    # Collect UUIDs from filtered annotations
    uuids = step_filtered_df["uuid"].astype(str).dropna().unique().tolist() if "uuid" in step_filtered_df.columns else []

    # Filter user rows to selected UUIDs
    filtered_users_df = users_df.copy()
    if uuids and "uuid" in filtered_users_df.columns:
        filtered_users_df = filtered_users_df[filtered_users_df["uuid"].astype(str).isin(uuids)].copy()

    # Build user_id from appearance order
    if "uuid" in filtered_users_df.columns:
        ordered_unique_uuids = pd.unique(filtered_users_df["uuid"].astype(str))
        uuid_to_id = {u: i for i, u in enumerate(ordered_unique_uuids)}
        filtered_users_df.insert(0, "user_id", filtered_users_df["uuid"].astype(str).map(uuid_to_id).astype("Int64"))

    # Parse titpj into Q1..Q10 (0-based)
    q_cols = [f"Q{i}" for i in range(1, 11)]
    for qc in q_cols:
        filtered_users_df[qc] = pd.NA
    if "titpj" in filtered_users_df.columns:
        def _safe_parse_titpj(val):
            if isinstance(val, (dict, list)):
                return val
            if pd.isna(val):
                return None
            s = str(val)
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return None
        parsed_titpj = filtered_users_df["titpj"].apply(_safe_parse_titpj)
        for i in range(1, 11):
            key = str(i)
            col = f"Q{i}"
            filtered_users_df[col] = (
                pd.to_numeric(
                    parsed_titpj.apply(lambda d: d.get(key) if isinstance(d, dict) else pd.NA),
                    errors="coerce"
                ).astype("Int64") - 1
            )

    # Initialize learn/interest columns
    placeholder_cols = [
        "art_learn","art_interest","fashion_learn","fashion_interest","photoVideo_learn","photoVideo_interest"
    ]
    for pc in placeholder_cols:
        filtered_users_df[pc] = pd.NA

    # Parse experience: learn -> binary from year, interest -> 0-based
    if "experience" in filtered_users_df.columns:
        def _safe_parse_exp(val):
            if isinstance(val, dict):
                return val
            if pd.isna(val):
                return None
            s = str(val)
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return None

        parsed_exp = filtered_users_df["experience"].apply(_safe_parse_exp)

        def _domain(d, domain):
            if not isinstance(d, dict):
                return None
            v = d.get(domain)
            return v if isinstance(v, dict) else None

        for domain in ["art", "fashion", "photoVideo"]:
            learn_col = f"{domain}_learn"
            interest_col = f"{domain}_interest"

            def _learn_val(d):
                sub = _domain(d, domain)
                learn = sub.get("learn") if isinstance(sub, dict) else None
                year = learn.get("year", -1) if isinstance(learn, dict) else -1
                try:
                    yr = int(year)
                except Exception:
                    yr = -1
                return 0 if yr == -1 else 1

            def _interest_val(d):
                sub = _domain(d, domain)
                val = sub.get("interest") if isinstance(sub, dict) else pd.NA
                return pd.to_numeric(val, errors="coerce") - 1

            filtered_users_df[learn_col] = parsed_exp.apply(_learn_val).astype("Int64")
            filtered_users_df[interest_col] = parsed_exp.apply(_interest_val).astype("Int64")

    # Select and order requested columns
    requested_cols = [
        "user_id","uuid","age","gender","edu","nationality",
        "Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10",
        "art_learn","art_interest","fashion_learn","fashion_interest","photoVideo_learn","photoVideo_interest"
    ]
    present_cols = [c for c in requested_cols if c in filtered_users_df.columns]
    out_df = filtered_users_df[present_cols].copy()

    # Save to output_path if provided; otherwise just return DataFrame and print count
    if output_path is not None:
        try:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_path, index=False)
            print(f"Saved users CSV with {len(out_df)} rows to {out_path}")
        except Exception as e:
            print(f"Failed to save filtered users to {output_path}: {e}")
    else:
        print(f"Built users DataFrame with {len(out_df)} rows (not saved)")

    return step_filtered_df

def make_ratings_csv(annotation_path, finished_users_path, rel_tasks_users_path, user_path=None, url_filename_path=None, output_path=None):
    """Create ratings CSV enriched with IDs and filename (same logic as create_annotation_df).

    入力:
      - annotation_path: user-annotation-data_rows.csv のパス
      - finished_users_path: 抽出済みユーザーCSV (uuid 列を含む)
      - rel_tasks_users_path: annotation-tasks_rows.csv のうち task_id と identifier を含むCSVのパス
      - user_path: ユーザーIDの安定な割り当てのための raw の user-data_rows.csv のパス（省略可）
      - url_filename_path: url_filename_rows.csv のパス（省略可）
      - output_path: 保存先CSVのパス（省略可。指定がなければ返却のみ）

    出力:
      - pandas.DataFrame を返し、output_path が指定されていれば CSV 保存も行う
    """

    annotation_df = pd.read_csv(annotation_path)
    finished_users_df = pd.read_csv(finished_users_path)
    rel_tasks_users_df = pd.read_csv(rel_tasks_users_path)
    # rel_tasks_users_dfはtask_idとidentifierの列のみ
    rel_tasks_users_df = rel_tasks_users_df[['task_id', 'identifier']]

    # master_task_id と task_id をキーに結合
    annotation_df = annotation_df.merge(rel_tasks_users_df, left_on='master_task_id', right_on='task_id', how='left')
    finished_users_uuids = finished_users_df['uuid'].tolist()
    filtered_annotation_df = annotation_df[annotation_df['uuid'].isin(finished_users_uuids)]
    filtered_annotation_df.loc[:, 'data'] = filtered_annotation_df['data'].apply(eval)

    # ジャンルの日本語→英語マップ
    genre_map = {
        'アート作品': 'art',
        'ファッション': 'fashion',
        '映像': 'scenery',
    }

    sample_rows = []
    for _, row in filtered_annotation_df.iterrows():
        data = row['data']
        urls = data['urls']
        results = data['result']
        genre_val = data.get('genre')
        genre_en = genre_map.get(genre_val, genre_val)
        # result の要素はすべて -1
        for url, result in zip(urls, results):
            sample_rows.append({
                'uuid': row['uuid'],
                'set': row['identifier'],
                'fold': int(row['identifier']) // 2 if pd.notna(row['identifier']) else pd.NA,
                'genre': genre_en,
                'sample': url.split('/')[-1],
                'Like': result[0] - 1,
                'Beautiful': result[1] - 1,
                'Distasteful': result[2] - 1,
                'Impressed': result[3] - 1,
                'Intellectually': result[4] - 1,
                'Motivated': result[5] - 1,
                'Nostalgic': result[6] - 1,
                'Sad': result[7] - 1,
                'Amused': result[8] - 1,
                'Aesthetic': result[9] - 1,
                'Time': round(result[10], 2)
            })

    out = pd.DataFrame(sample_rows)

    # user_id の割当（安定性のため raw ユーザーファイルの出現順に基づく）
    if user_path is None:
        user_path = str(Path(annotation_path).parent / 'user-data_rows.csv')
    try:
        raw_users_df = pd.read_csv(user_path)
        uuids_in_user_file = raw_users_df['uuid'].astype(str)
        ordered_unique_uuids = pd.unique(uuids_in_user_file)
        uuid_to_id = {u: i for i, u in enumerate(ordered_unique_uuids)}
    except Exception:
        ordered_unique_uuids = pd.unique(out['uuid'].astype(str))
        uuid_to_id = {u: i for i, u in enumerate(ordered_unique_uuids)}

    out['user_id'] = out['uuid'].astype(str).map(uuid_to_id).astype('Int64')

    # sample_id の割当（出現順）
    ordered_unique_samples = pd.unique(out['sample'].astype(str))
    sample_to_id = {s: i for i, s in enumerate(ordered_unique_samples)}
    out['sample_id'] = out['sample'].astype(str).map(sample_to_id).astype('Int64')

    # url_filename_rows.csv からファイル名を結合
    if url_filename_path is None:
        url_filename_path = str(Path(annotation_path).parent / 'url_filename_rows.csv')
    try:
        url_file_df = pd.read_csv(url_filename_path, usecols=['url', 'filename'])
        url_file_df['sample'] = url_file_df['url'].astype(str).str.split('/').str[-1]
        url_file_df = url_file_df[['sample', 'filename']].drop_duplicates('sample')
        url_file_df = url_file_df.rename(columns={'filename': 'sample_file'})
        out = out.merge(url_file_df, on='sample', how='left')
    except Exception:
        out['sample_file'] = pd.NA

    # 列の並び替え
    front_cols = ['user_id', 'uuid', 'set', 'fold', 'genre', 'sample_id', 'sample', 'sample_file']
    rating_cols = [
        'Like', 'Beautiful', 'Distasteful', 'Impressed', 'Intellectually',
        'Motivated', 'Nostalgic', 'Sad', 'Amused', 'Aesthetic', 'Time'
    ]
    cols = [c for c in front_cols + rating_cols if c in out.columns] + [
        c for c in out.columns if c not in front_cols + rating_cols
    ]
    out = out[cols]

    # 保存（指定がない場合も行数を表示）
    if output_path is not None:
        try:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)
            print(f"Saved ratings CSV with {len(out)} rows to {out_path}")
        except Exception as e:
            print(f"Failed to save ratings CSV to {output_path}: {e}")
    else:
        print(f"Built ratings DataFrame with {len(out)} rows (not saved)")

    return out

def get_quality_outlier_uuids(
    ratings_df: pd.DataFrame,
    domains: tuple = ('art', 'fashion', 'scenery'),
    score_col: str = "Aesthetic",
    min_rt_art_fashion: float = 0.0,
    min_rt_scenery: float = 0.0,
    fast_user_thresh: float = 0.2,
    retest_method: str = "spearman",
    mad_multiplier: float = 2.5,
    outlier_method: str = "mad",
) -> set:
    """Detect annotator UUIDs to exclude based on rating quality criteria.

    Criteria (all use ANY-domain flagging — flagged in even one domain → excluded):
      - p_mode  : proportion of modal score is an outlier (> centre + k * spread)
      - retest  : test-retest reliability is an outlier in ANY domain
                  (MAE: high = bad; ICC / Spearman: low = bad)
      - rt_prop : proportion of fast responses (Time < threshold) > fast_user_thresh

    Args:
        ratings_df     : ratings DataFrame with columns uuid, genre, sample_id,
                         score_col, and optionally Time.
        domains        : genres to evaluate (default: art, fashion, scenery).
        score_col      : column name used for p_mode and retest metrics.
        min_rt_art_fashion : fast-response threshold (ms) for art/fashion.
        min_rt_scenery : fast-response threshold (ms) for scenery.
        fast_user_thresh   : max allowed proportion of fast responses per user.
        retest_method  : "spearman" | "icc" | "mae".
        mad_multiplier : multiplier k for outlier detection spread.
        outlier_method : "mad" (median ± k*MAD) or "mean" (mean ± k*SD).

    Returns:
        Set of UUID strings to exclude.
    """
    import numpy as np
    from scipy.stats import spearmanr

    suspect_uuids: set = set()

    # ── outlier-detection helpers ─────────────────────────────────────────────
    def _flag_high(values: np.ndarray) -> np.ndarray:
        """Boolean mask: value > centre + k*spread  (high = bad)."""
        if outlier_method == "mad":
            med = np.median(values)
            spread = np.median(np.abs(values - med))
            return values > med + mad_multiplier * spread
        mu = values.mean()
        sd = values.std(ddof=1) if len(values) > 1 else 0.0
        return values > mu + mad_multiplier * sd

    def _flag_low(values: np.ndarray) -> np.ndarray:
        """Boolean mask: value < centre - k*spread  (low = bad)."""
        if outlier_method == "mad":
            med = np.median(values)
            spread = np.median(np.abs(values - med))
            return values < med - mad_multiplier * spread
        mu = values.mean()
        sd = values.std(ddof=1) if len(values) > 1 else 0.0
        return values < mu - mad_multiplier * sd

    def _icc_two_ratings(x1: np.ndarray, x2: np.ndarray) -> float:
        """ICC(1) for two repeated measures per item."""
        n = len(x1)
        if n < 2:
            return float('nan')
        item_means = (x1 + x2) / 2.0
        grand_mean = item_means.mean()
        ms_between = np.sum((item_means - grand_mean) ** 2) * 2 / (n - 1)
        ms_within = np.sum((x1 - x2) ** 2) / (2 * n)
        denom = ms_between + ms_within
        return 1.0 if denom == 0 else (ms_between - ms_within) / denom

    # ── 1. p_mode ─────────────────────────────────────────────────────────────
    for domain in domains:
        dom_df = ratings_df[ratings_df['genre'] == domain]
        if dom_df.empty or score_col not in dom_df.columns:
            continue

        p_modes: dict = {}
        for uid, udf in dom_df.groupby(dom_df['uuid'].astype(str)):
            scores = udf[score_col].dropna()
            if len(scores) == 0:
                continue
            p_modes[uid] = scores.value_counts().iloc[0] / len(scores)

        if len(p_modes) < 2:
            continue

        uid_arr = np.array(list(p_modes.keys()))
        val_arr = np.array(list(p_modes.values()), dtype=float)
        flagged = uid_arr[_flag_high(val_arr)]
        if len(flagged):
            print(f"[quality] p_mode domain={domain}: {len(flagged)} flagged")
        suspect_uuids.update(flagged.tolist())

    # ── 2. retest ─────────────────────────────────────────────────────────────
    for domain in domains:
        dom_df = ratings_df[ratings_df['genre'] == domain]
        if dom_df.empty or score_col not in dom_df.columns:
            continue

        retest_scores: dict = {}
        for uid, udf in dom_df.groupby(dom_df['uuid'].astype(str)):
            # Find sample_ids rated more than once by this user
            dup_mask = udf.duplicated('sample_id', keep=False)
            dup_samples = udf.loc[dup_mask, 'sample_id'].unique()
            r1_list, r2_list = [], []
            for sid in dup_samples:
                pair = udf[udf['sample_id'] == sid][score_col].dropna().values
                if len(pair) >= 2:
                    r1_list.append(pair[0])
                    r2_list.append(pair[1])
            if len(r1_list) < 3:  # need ≥3 pairs for a meaningful metric
                continue
            r1 = np.array(r1_list, dtype=float)
            r2 = np.array(r2_list, dtype=float)
            if retest_method == "mae":
                retest_scores[uid] = float(np.mean(np.abs(r1 - r2)))
            elif retest_method == "icc":
                retest_scores[uid] = _icc_two_ratings(r1, r2)
            else:  # spearman
                rho, _ = spearmanr(r1, r2)
                retest_scores[uid] = float(rho)

        # Drop NaN entries (e.g., constant arrays → undefined correlation)
        retest_scores = {k: v for k, v in retest_scores.items() if not (v != v)}
        if len(retest_scores) < 2:
            continue

        uid_arr = np.array(list(retest_scores.keys()))
        val_arr = np.array(list(retest_scores.values()), dtype=float)
        flagged = uid_arr[_flag_high(val_arr) if retest_method == "mae" else _flag_low(val_arr)]
        if len(flagged):
            print(f"[quality] retest({retest_method}) domain={domain}: {len(flagged)} flagged")
        suspect_uuids.update(flagged.tolist())

    # ── 3. rt_prop ────────────────────────────────────────────────────────────
    if 'Time' in ratings_df.columns:
        rt_thresh_map = {d: (min_rt_scenery if d == 'scenery' else min_rt_art_fashion) for d in domains}
        for domain in domains:
            thresh = rt_thresh_map[domain]
            if thresh <= 0.0:
                continue
            dom_df = ratings_df[ratings_df['genre'] == domain].copy()
            if dom_df.empty:
                continue
            dom_df['Time'] = pd.to_numeric(dom_df['Time'], errors='coerce')
            for uid, udf in dom_df.groupby(dom_df['uuid'].astype(str)):
                valid = udf['Time'].dropna()
                if len(valid) == 0:
                    continue
                fast_prop = (valid < thresh).sum() / len(valid)
                if fast_prop > fast_user_thresh:
                    print(f"[quality] rt_prop domain={domain} uuid={uid}: {fast_prop:.2f} > {fast_user_thresh}")
                    suspect_uuids.add(uid)

    return suspect_uuids


def make_users_and_ratings_pipeline(
    raw_dir: str,
    output_dir: str,
    raw_annotation_rows_path: Optional[str] = None,
    raw_tasks_rows_path: Optional[str] = None,
    raw_user_rows_path: Optional[str] = None,
    raw_url_filename_rows_path: Optional[str] = None,
    finished_users_output_csv: Optional[str] = None,
    ratings_output_csv: Optional[str] = None,
    score_col: str = "Aesthetic",
    exclude_video_files: Optional[List[str]] = None,
    exclude_fashion_files: Optional[List[str]] = None,
    min_rs_art_fashion: Optional[float] = None,
    min_rs_video: Optional[float] = None,
    fast_user_thresh: float = 0.2,
    retest_method: str = "spearman",
    mad_multiplier: float = 2.5,
    outlier_method: str = "mad",
    quality_domains: Optional[tuple] = None,
    exclude_uuids_file: Optional[str] = None,
):
    """Build users.csv and ratings.csv, then remove outlier UUIDs using annotator-quality criteria.

    raw_dir and output_dir are used to resolve default file paths.
    Individual path arguments override the defaults derived from those folders.

    User exclusion logic (Combined exclusion, same as annotator-quality):
      - p_mode  : flagged in ANY domain  (> mean/median + k * SD/MAD)
      - retest  : flagged in ANY domain  (MAE↑ / ICC↓ / Spearman↓)
      - rt_prop : flagged in ANY domain  (proportion of fast responses > fast_user_thresh)

    When exclude_uuids_file is given, its UUID list is used instead of running detection.

    Steps:
      1) Create users.csv via make_user_csv
      2) Create ratings.csv via make_ratings_csv
      3) Exclude specified sample files (video / fashion)
      4) Align ratings.user_id via users.csv uuid mapping
      5) Detect outlier UUIDs (quality-based or from file)
      6) Remove per-sample rows below RT threshold
      7) Filter outlier UUIDs from both users.csv and ratings.csv and save
    """
    # Resolve paths from folders when not explicitly provided
    _raw = Path(raw_dir)
    _out = Path(output_dir)
    if raw_annotation_rows_path is None:
        raw_annotation_rows_path = str(_raw / "user-annotation-data_rows.csv")
    if raw_tasks_rows_path is None:
        raw_tasks_rows_path = str(_raw / "annotation-tasks_rows.csv")
    if raw_user_rows_path is None:
        raw_user_rows_path = str(_raw / "user-data_rows.csv")
    if raw_url_filename_rows_path is None:
        raw_url_filename_rows_path = str(_raw / "url_filename_rows.csv")
    if finished_users_output_csv is None:
        finished_users_output_csv = str(_out / "users.csv")
    if ratings_output_csv is None:
        ratings_output_csv = str(_out / "ratings.csv")

    # 1) Build users.csv
    _ = make_user_csv(raw_dir, finished_users_output_csv)

    # 2) Build ratings.csv using newly created users.csv
    ratings_df = make_ratings_csv(
        annotation_path=raw_annotation_rows_path,
        finished_users_path=finished_users_output_csv,
        rel_tasks_users_path=raw_tasks_rows_path,
        user_path=raw_user_rows_path,
        url_filename_path=raw_url_filename_rows_path,
        output_path=ratings_output_csv,
    )

    # Optionally exclude specified scenery files from ratings (only affects genre='scenery')
    if exclude_video_files:
        try:
            # Normalize to strings and unique set
            excluded = {str(x).strip() for x in exclude_video_files if str(x).strip()}
            before = len(ratings_df)
            ratings_df = ratings_df[~((ratings_df.get('genre') == 'scenery') & (ratings_df.get('sample_file').astype(str).isin(excluded)))]
            after = len(ratings_df)
            print(f"[make_users_and_ratings_pipeline] excluded {before - after} rows for {len(excluded)} scenery files.")
            Path(ratings_output_csv).parent.mkdir(parents=True, exist_ok=True)
            ratings_df.to_csv(ratings_output_csv, index=False)
        except Exception as e:
            print(f"[make_users_and_ratings_pipeline] Warning: failed to exclude videos: {e}")

    # Optionally exclude specified fashion files from ratings (only affects genre='fashion')
    if exclude_fashion_files:
        try:
            excluded = {str(x).strip() for x in exclude_fashion_files if str(x).strip()}
            before = len(ratings_df)
            ratings_df = ratings_df[~((ratings_df.get('genre') == 'fashion') & (ratings_df.get('sample_file').astype(str).isin(excluded)))]
            after = len(ratings_df)
            print(f"[make_users_and_ratings_pipeline] excluded {before - after} rows for {len(excluded)} fashion files.")
            Path(ratings_output_csv).parent.mkdir(parents=True, exist_ok=True)
            ratings_df.to_csv(ratings_output_csv, index=False)
        except Exception as e:
            print(f"[make_users_and_ratings_pipeline] Warning: failed to exclude fashion files: {e}")

    # Align ratings.user_id to users.csv via uuid mapping (users.csv is authoritative)
    try:
        users_map_df = pd.read_csv(finished_users_output_csv, usecols=["uuid", "user_id"])\
            .drop_duplicates(subset=["uuid"], keep="first")
        users_map_df["uuid"] = users_map_df["uuid"].astype(str)
        users_map_df["user_id"] = users_map_df["user_id"].astype("Int64")
        ratings_df["uuid"] = ratings_df["uuid"].astype(str)
        ratings_df = ratings_df.merge(users_map_df, on="uuid", how="left", suffixes=("", "_users"))
        # prefer users.csv user_id when available
        ratings_df["user_id"] = ratings_df["user_id_users"].combine_first(ratings_df["user_id"]).astype("Int64")
        ratings_df.drop(columns=["user_id_users"], inplace=True)
        Path(ratings_output_csv).parent.mkdir(parents=True, exist_ok=True)
        ratings_df.to_csv(ratings_output_csv, index=False)
        print("[make_users_and_ratings_pipeline] ratings.user_id aligned to users.csv using uuid mapping.")
    except Exception as e:
        print(f"[make_users_and_ratings_pipeline] Warning: could not align user_id via users.csv: {e}")

    # 5) Detect outlier UUIDs using annotator-quality criteria (or from pre-computed file)
    if exclude_uuids_file is not None:
        try:
            with open(exclude_uuids_file, 'r', encoding='utf-8') as _f:
                suspect_uuids = {ln.strip() for ln in _f if ln.strip()}
            print(f"[make_users_and_ratings_pipeline] Loaded {len(suspect_uuids)} UUIDs from {exclude_uuids_file}")
        except Exception as e:
            print(f"[make_users_and_ratings_pipeline] Warning: failed to read exclude-uuids-file: {e}")
            suspect_uuids = set()
    else:
        domains = quality_domains or ('art', 'fashion', 'scenery')
        suspect_uuids = get_quality_outlier_uuids(
            ratings_df=ratings_df,
            domains=domains,
            score_col=score_col,
            min_rt_art_fashion=min_rs_art_fashion if min_rs_art_fashion is not None else 0.0,
            min_rt_scenery=min_rs_video if min_rs_video is not None else 0.0,
            fast_user_thresh=fast_user_thresh,
            retest_method=retest_method,
            mad_multiplier=mad_multiplier,
            outlier_method=outlier_method,
        )
        print(f"[make_users_and_ratings_pipeline] {len(suspect_uuids)} UUIDs flagged by annotator-quality criteria")

    # 6) Remove individual samples with response time below threshold
    if (min_rs_art_fashion is not None or min_rs_video is not None) and "Time" in ratings_df.columns:
        ratings_df["Time"] = pd.to_numeric(ratings_df["Time"], errors="coerce")
        n_before = len(ratings_df)
        mask = pd.Series(True, index=ratings_df.index)
        if min_rs_art_fashion is not None:
            af_mask = ratings_df["genre"].isin(["art", "fashion"]) & ratings_df["Time"].notna() & (ratings_df["Time"] < min_rs_art_fashion)
            mask &= ~af_mask
        if min_rs_video is not None:
            v_mask = (ratings_df["genre"] == "scenery") & ratings_df["Time"].notna() & (ratings_df["Time"] < min_rs_video)
            mask &= ~v_mask
        ratings_df = ratings_df[mask]
        n_removed = n_before - len(ratings_df)
        print(f"[make_users_and_ratings_pipeline] Removed {n_removed} samples with response time below threshold")

    # 7) Filter out suspect uuids from users and ratings
    users_out_df = pd.read_csv(finished_users_output_csv)
    users_out_df = users_out_df[~users_out_df["uuid"].astype(str).isin(set(map(str, suspect_uuids)))]
    ratings_out_df = ratings_df[~ratings_df["uuid"].astype(str).isin(set(map(str, suspect_uuids)))]

    # 8) Save both
    Path(finished_users_output_csv).parent.mkdir(parents=True, exist_ok=True)
    users_out_df.to_csv(finished_users_output_csv, index=False)
    Path(ratings_output_csv).parent.mkdir(parents=True, exist_ok=True)
    ratings_out_df.to_csv(ratings_output_csv, index=False)
    print(
        f"Saved users.csv ({len(users_out_df)}) and ratings.csv ({len(ratings_out_df)}) "
        f"after removing {len(suspect_uuids)} UUIDs (annotator-quality criteria)"
    )

def make_data_split_giaa(
    annotation_df,
    genre=None,
    val_frac=0.15,
    test_frac=0.15,
    seed=42,
    out_dir='data/split',
    version='v1',
):
    """
    Create GIAA-only train/val/test split at image level.

    Splits all unique images into three disjoint sets (train/val/test).
    Independent of cross-validation and PIAA splits.

    Args:
      annotation_df (pd.DataFrame): Must contain columns ['genre','sample_file'].
      genre (str | None): If provided, filter rows to this genre.
      val_frac (float): Fraction of images for validation (default: 0.15).
      test_frac (float): Fraction of images for test (default: 0.15).
      seed (int): Random seed.
      out_dir (str | Path): Base output directory (defaults to ./data/split).
      version (str): Version folder name under out_dir.

    Writes:
      - {out_dir}/{version}/{genre}/train_images_GIAA.txt
      - {out_dir}/{version}/{genre}/val_images_GIAA.txt
      - {out_dir}/{version}/{genre}/test_images_GIAA.txt
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally filter by genre
    if genre is not None:
        annotation_df = annotation_df[annotation_df['genre'] == genre]

    # Get all unique images
    all_images = annotation_df['sample_file'].astype(str).drop_duplicates().tolist()
    random.seed(seed)
    shuffled = all_images[:]
    random.shuffle(shuffled)

    # Split: test -> val -> train
    n_test = int(len(shuffled) * float(test_frac))
    n_val = int(len(shuffled) * float(val_frac))
    test_images = shuffled[:n_test]
    val_images = shuffled[n_test:n_test + n_val]
    train_images = shuffled[n_test + n_val:]

    # Ensure output directory
    out_base = out_dir / version / genre if genre is not None else out_dir / version
    out_base.mkdir(parents=True, exist_ok=True)

    # Write split files
    for fname, images in [
        ('train_images_GIAA.txt', train_images),
        ('val_images_GIAA.txt', val_images),
        ('test_images_GIAA.txt', test_images),
    ]:
        with (out_base / fname).open('w', encoding='utf-8') as f:
            for im in images:
                f.write(im + '\n')

    print(
        f"[make_data_split_giaa] genre={genre}: "
        f"train={len(train_images)}, val={len(val_images)}, test={len(test_images)} "
        f"(total unique images={len(all_images)}). Output: {out_base}"
    )


def make_data_split_cv(
    annotation_df,
    n_folds=5,
    genre=None,
    val_frac_images_giaa=0.2,
    val_frac_users_giaa=0.2,
    n_train_PIAA=100,
    n_test_PIAA=60,
    seed=42,
    out_dir='data/split',
    version='v3',
):
    """
    Create cross-validation data splits with user-level folding.

    Behavior:
      - Split all users into n_folds groups evenly
      - For each fold i, use group i as test users (PIAA)
      - Remaining users form GIAA pool:
        * Split their images into train/val by val_frac_images_giaa
        * Split the users themselves into train/val by val_frac_users_giaa
      - For each test user, split their images into train/test/val for PIAA

    Args:
      annotation_df (pd.DataFrame): Must contain columns ['user_id','genre','sample_file']
      n_folds (int): Number of cross-validation folds (default: 5)
      genre (str | None): If provided, filter rows to this genre
      val_frac_images_giaa (float): Fraction of GIAA images for validation (default: 0.2)
      val_frac_users_giaa (float): Fraction of GIAA users for validation (default: 0.2)
      n_train_PIAA (int): Per-user train images count (PIAA) sampled from test pool
      n_test_PIAA (int): Per-user test images count (PIAA) sampled from test pool
      seed (int): Random seed
      out_dir (str | Path): Base output directory (defaults to ./data/split)
      version (str): Version folder name prefix (e.g., 'v3' -> 'v3_fold1', 'v3_fold2', ...)

    Writes for each fold:
      - {out_dir}/{version}_fold{i}/{genre}/train_images_GIAA.txt
      - {out_dir}/{version}_fold{i}/{genre}/val_images_GIAA.txt
      - {out_dir}/{version}_fold{i}/{genre}/train_users_GIAA.txt
      - {out_dir}/{version}_fold{i}/{genre}/val_users_GIAA.txt
      - {out_dir}/{version}_fold{i}/{genre}/train_PIAA.txt (lines: user_id\tfilename)
      - {out_dir}/{version}_fold{i}/{genre}/val_PIAA.txt (lines: user_id\tfilename)
      - {out_dir}/{version}_fold{i}/{genre}/test_PIAA.txt (lines: user_id\tfilename)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally filter by genre
    if genre is not None:
        annotation_df = annotation_df[annotation_df['genre'] == genre]

    # Get all unique users and split them into n_folds groups
    all_users = annotation_df['user_id'].dropna().astype(str).drop_duplicates().tolist()
    random.seed(seed)
    shuffled_users = all_users[:]
    random.shuffle(shuffled_users)

    # Split users into n_folds groups as evenly as possible
    fold_size = len(shuffled_users) // n_folds
    user_folds = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else len(shuffled_users)
        user_folds.append(shuffled_users[start_idx:end_idx])

    # Process each fold
    for fold_idx in range(n_folds):
        fold_num = fold_idx + 1
        test_users = user_folds[fold_idx]
        train_users = [u for i, fold in enumerate(user_folds) if i != fold_idx for u in fold]

        # Split rows into test and train pools
        test_rows = annotation_df[annotation_df['user_id'].astype(str).isin(test_users)]
        train_rows = annotation_df[annotation_df['user_id'].astype(str).isin(train_users)]

        # GIAA: split images from train pool
        train_images = train_rows['sample_file'].astype(str).drop_duplicates().tolist()
        random.seed(seed + fold_idx)  # Different seed per fold for variance
        train_images_shuffled = train_images[:]
        random.shuffle(train_images_shuffled)
        n_val_images_giaa = int(len(train_images_shuffled) * float(val_frac_images_giaa))
        val_images_GIAA = train_images_shuffled[:n_val_images_giaa]
        train_images_GIAA = train_images_shuffled[n_val_images_giaa:]

        # GIAA: split users from train pool
        random.seed(seed + fold_idx)
        train_users_shuffled = train_users[:]
        random.shuffle(train_users_shuffled)
        n_val_users_giaa = int(len(train_users_shuffled) * float(val_frac_users_giaa))
        val_users_GIAA = train_users_shuffled[:n_val_users_giaa]
        train_users_GIAA = train_users_shuffled[n_val_users_giaa:]

        # PIAA: per-user split from test pool
        train_images_PIAA, test_images_PIAA, val_images_PIAA = [], [], []
        for uid in test_users:
            user_rows = test_rows[test_rows['user_id'].astype(str) == uid]
            user_images = user_rows['sample_file'].astype(str).drop_duplicates().tolist()
            user_shuffled = user_images[:]
            random.seed(seed + fold_idx + hash(uid) % 10000)  # Reproducible per-user shuffle
            random.shuffle(user_shuffled)
            t = user_shuffled[:n_test_PIAA]
            tr = user_shuffled[n_test_PIAA:n_test_PIAA + n_train_PIAA]
            va = user_shuffled[n_test_PIAA + n_train_PIAA:]
            test_images_PIAA.extend([(uid, im) for im in t])
            train_images_PIAA.extend([(uid, im) for im in tr])
            val_images_PIAA.extend([(uid, im) for im in va])

        # Create output directory for this fold
        fold_version = f"{version}_fold{fold_num}"
        out_base = out_dir / fold_version / genre if genre is not None else out_dir / fold_version
        out_base.mkdir(parents=True, exist_ok=True)

        # Write GIAA image files
        train_path_GIAA = out_base / 'train_images_GIAA.txt'
        val_path_GIAA = out_base / 'val_images_GIAA.txt'
        with train_path_GIAA.open('w', encoding='utf-8') as f:
            for im in train_images_GIAA:
                f.write(im + '\n')
        with val_path_GIAA.open('w', encoding='utf-8') as f:
            for im in val_images_GIAA:
                f.write(im + '\n')

        # Write GIAA user files
        train_users_path_GIAA = out_base / 'train_users_GIAA.txt'
        val_users_path_GIAA = out_base / 'val_users_GIAA.txt'
        with train_users_path_GIAA.open('w', encoding='utf-8') as f:
            for uid in train_users_GIAA:
                f.write(uid + '\n')
        with val_users_path_GIAA.open('w', encoding='utf-8') as f:
            for uid in val_users_GIAA:
                f.write(uid + '\n')

        # Write PIAA user-image pair files
        train_path_PIAA = out_base / 'train_PIAA.txt'
        val_path_PIAA = out_base / 'val_PIAA.txt'
        test_path_PIAA = out_base / 'test_PIAA.txt'
        with train_path_PIAA.open('w', encoding='utf-8') as f:
            for uid, im in train_images_PIAA:
                f.write(f"{uid}\t{im}\n")
        with val_path_PIAA.open('w', encoding='utf-8') as f:
            for uid, im in val_images_PIAA:
                f.write(f"{uid}\t{im}\n")
        with test_path_PIAA.open('w', encoding='utf-8') as f:
            for uid, im in test_images_PIAA:
                f.write(f"{uid}\t{im}\n")

        # Summary for this fold
        piaa_train_users = len({uid for uid, _ in train_images_PIAA})
        piaa_val_users = len({uid for uid, _ in val_images_PIAA})
        piaa_test_users = len({uid for uid, _ in test_images_PIAA})

        print(
            f"[make_data_split_cv] Fold {fold_num}/{n_folds}: "
            f"GIAA train_images={len(train_images_GIAA)}, val_images={len(val_images_GIAA)}, "
            f"train_users={len(train_users_GIAA)}, val_users={len(val_users_GIAA)}. "
            f"PIAA train_samples={len(train_images_PIAA)} (unique_users={piaa_train_users}), "
            f"val_samples={len(val_images_PIAA)} (unique_users={piaa_val_users}), "
            f"test_samples={len(test_images_PIAA)} (unique_users={piaa_test_users}). "
            f"Output: {out_base}"
        )

    print(
        f"[make_data_split_cv] Completed {n_folds}-fold cross-validation split. "
        f"Total users: {len(all_users)}, users per fold: ~{len(all_users)//n_folds}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Utilities for preprocessing XPASS data")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: make_users_and_ratings (unified)
    both_parser = subparsers.add_parser(
        "make_users_and_ratings",
        help="Create users.csv and ratings.csv together, then remove outlier UUIDs using annotator-quality criteria.")
    both_parser.add_argument(
        "--raw-dir",
        default="/home/hayashi0884/proj-xpass/data/raw",
        help="Folder containing the 4 raw CSVs "
             "(user-annotation-data_rows.csv, annotation-tasks_rows.csv, "
             "user-data_rows.csv, url_filename_rows.csv). "
             "Default: /home/hayashi0884/proj-xpass/data/raw")
    both_parser.add_argument(
        "--output-dir",
        default="data/maked",
        help="Output folder; users.csv and ratings.csv are written here. Default: data/maked")
    both_parser.add_argument("--score-col", default="Aesthetic", help="Score column for p_mode and retest reliability")
    both_parser.add_argument(
        "--exclude-videos",
        nargs='*',
        default=["E5zeYBrVd5o_0034350_0036150.mp4"],
        help="List of video filenames (sample_file) to exclude from ratings for genre=scenery "
             "(default: E5zeYBrVd5o_0034350_0036150.mp4)")
    both_parser.add_argument(
        "--exclude-fashion",
        nargs='*',
        default=["0940.jpg", "1034.jpg"],
        help="List of fashion filenames (sample_file) to exclude from ratings for genre=fashion "
             "(default: 0940.jpg 1034.jpg)")
    both_parser.add_argument(
        "--min-rs-art-fashion",
        type=float,
        default=10,
        help="Min response time (seconds) for art/fashion — used for both sample removal and rt_prop user flagging (default: 10)")
    both_parser.add_argument(
        "--min-rs-video",
        type=float,
        default=30,
        help="Min response time (seconds) for scenery — used for both sample removal and rt_prop user flagging (default: 30)")
    both_parser.add_argument(
        "--fast-user-thresh",
        type=float,
        default=0.2,
        help="Fraction threshold for rt_prop: users with more than this proportion of fast responses are excluded (default: 0.2)")
    both_parser.add_argument(
        "--retest-method",
        type=str,
        default="mae",
        choices=["spearman", "icc", "mae"],
        help="Test-retest reliability metric for outlier detection: spearman, icc, or mae (default: mae)")
    both_parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=2.5,
        help="Multiplier k for outlier threshold (default: 2.5)")
    both_parser.add_argument(
        "--outlier-method",
        type=str,
        default="std",
        choices=["mad", "std"],
        help="Outlier detection method: mad (median ± k*MAD) or std (mean ± k*SD) (default: std)")

    # Subcommand: make_data_split_giaa (GIAA-only train/val/test image split)
    split_giaa_parser = subparsers.add_parser(
        "make_data_split_giaa",
        help="Create GIAA-only train/val/test split at image level. Independent of CV and PIAA splits.")
    split_giaa_parser.add_argument(
        "ratings_csv",
        help="Path to ratings CSV (e.g., data/maked/ratings.csv) containing genre, sample_file, etc.")
    split_giaa_parser.add_argument(
        "--genre",
        choices=["art", "fashion", "scenery", "all"],
        default=None,
        help="Genre to split. Use 'all' to generate splits for art, fashion, and scenery in one run.")
    split_giaa_parser.add_argument(
        "--version",
        required=True,
        help="Version name to create under output directory (e.g., v_giaa).")
    split_giaa_parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of images for validation (0-1, default: 0.15).")
    split_giaa_parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Fraction of images for test (0-1, default: 0.15).")
    split_giaa_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).")
    split_giaa_parser.add_argument(
        "--out-dir",
        default=str(Path.cwd() / "data" / "split"),
        help="Output directory (default: ./data/split relative to current working directory).")

    # Subcommand: make_data_split_cv (cross-validation)
    split_cv_parser = subparsers.add_parser(
        "make_data_split_cv",
        help="Create cross-validation data splits with user-level folding. Each user appears in exactly one test fold.")
    split_cv_parser.add_argument(
        "ratings_csv",
        nargs="?",
        default="data/maked/ratings.csv",
        help="Path to ratings CSV (default: data/maked/ratings.csv) containing user_id, genre, sample_file, etc.")
    split_cv_parser.add_argument(
        "--genre",
        choices=["art", "fashion", "scenery", "all"],
        default="all",
        help="Genre to split. Use 'all' to generate splits for art, fashion, and scenery in one run (default: all).")
    split_cv_parser.add_argument(
        "--version",
        required=True,
        help="Version name prefix (e.g., 'v3' creates 'v3_fold1', 'v3_fold2', ...).")
    split_cv_parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).")
    split_cv_parser.add_argument(
        "--val-frac-images-giaa",
        type=float,
        default=0.1,
        help="Validation fraction for GIAA images (0-1, default: 0.1).")
    split_cv_parser.add_argument(
        "--val-frac-users-giaa",
        type=float,
        default=0.1,
        help="Validation fraction for GIAA users (0-1, default: 0.1).")
    split_cv_parser.add_argument(
        "--n-train-piaa",
        type=int,
        default=100,
        help="Number of per-user train images for PIAA (default: 100).")
    split_cv_parser.add_argument(
        "--n-test-piaa",
        type=int,
        default=50,
        help="Number of per-user test images for PIAA (default: 50).")
    split_cv_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).")
    split_cv_parser.add_argument(
        "--out-dir",
        default=str(Path.cwd() / "data" / "split"),
        help="Output directory (default: ./data/split relative to current working directory).")

    args = parser.parse_args()

    if args.command == "make_users_and_ratings":
        make_users_and_ratings_pipeline(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            score_col=args.score_col,
            exclude_video_files=args.exclude_videos if args.exclude_videos else None,
            exclude_fashion_files=args.exclude_fashion if args.exclude_fashion else None,
            min_rs_art_fashion=args.min_rs_art_fashion,
            min_rs_video=args.min_rs_video,
            fast_user_thresh=args.fast_user_thresh,
            retest_method=args.retest_method,
            mad_multiplier=args.mad_multiplier,
            outlier_method=args.outlier_method,
        )
    elif args.command == "make_data_split_giaa":
        # Load ratings CSV
        df = pd.read_csv(args.ratings_csv)
        # Determine target genres
        target_genres = (
            ["art", "fashion", "scenery"] if args.genre == "all" else [args.genre]
            if args.genre is not None else [None]
        )
        # Generate GIAA-only train/val/test splits per requested genre
        for g in target_genres:
            make_data_split_giaa(
                annotation_df=df,
                genre=g,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=args.seed,
                out_dir=args.out_dir,
                version=args.version,
            )
    elif args.command == "make_data_split_cv":
        # Load ratings CSV
        df = pd.read_csv(args.ratings_csv)
        # Determine target genres
        target_genres = (
            ["art", "fashion", "scenery"] if args.genre == "all" else [args.genre]
            if args.genre is not None else [None]
        )
        # Generate cross-validation splits per requested genre
        for g in target_genres:
            make_data_split_cv(
                annotation_df=df,
                n_folds=args.n_folds,
                genre=g,
                val_frac_images_giaa=args.val_frac_images_giaa,
                val_frac_users_giaa=args.val_frac_users_giaa,
                n_train_PIAA=args.n_train_piaa,
                n_test_PIAA=args.n_test_piaa,
                seed=args.seed,
                out_dir=args.out_dir,
                version=args.version,
            )
