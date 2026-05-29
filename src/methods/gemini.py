"""
Zero-shot evaluation using Gemini via Google GenAI API.
Runs on ALL images for the given genre (no train/test split required).

Usage:
    python -m src.methods.gemini --mode giaa --genre art
    python -m src.methods.gemini --mode giaa --genre fashion --trial 10
"""
import os
import json
import time
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Settings
# ──────────────────────────────────────────────────────────────────────────────
_MAKED_DIR = '/home/hayashi0884/proj-xpass-DA/data/maked'
_SAVE_DIR = '/home/hayashi0884/proj-xpass-DA/reports/exp/gemini'
_EXCLUDE_DIR = '/home/hayashi0884/proj-xpass-DA/documents/exclude_samole'
_SAMPLES_DIR_MAP = {
    'art':     '/home/hayashi0884/proj-xpass/data/samples/art',
    'fashion': '/home/hayashi0884/proj-xpass/data/samples/fashion',
    'scenery': '/home/hayashi0884/proj-xpass/data/samples/scenery_image',
}

_GENRE_IMG_EXT = {'scenery': '.jpg'}

_MODEL = "gemini-3-flash-preview"
_MAX_TOKENS = 5000
_SYSTEM_PROMPT = (
    "You are a researcher specializing in empirical aesthetics, skilled at predicting "
    "how general audiences perceive and rate visual content."
)

_GENRE_LABEL_EN = {
    'art':     'art image',
    'fashion': 'fashion image',
    'scenery': 'landscape image',
}

def _make_user_prompt(genre: str) -> str:
    label_en = _GENRE_LABEL_EN.get(genre, genre)
    return (
        f"Imagine approximately 13 ordinary people with no special training in art or photography "
        f"are shown the {label_en} below and asked to rate its aesthetic quality.\n\n"
        f"In the study, participants were asked the following question:\n"
        f"\"Overall, how aesthetic do you find this {label_en}?\"\n\n"
        f"Each person rates the {label_en} using the following 7-point scale:\n"
        f"- 1 = Highly unaesthetic\n"
        f"- 2 = Unaesthetic\n"
        f"- 3 = Slightly unaesthetic\n"
        f"- 4 = Neutral\n"
        f"- 5 = Slightly aesthetic\n"
        f"- 6 = Aesthetic\n"
        f"- 7 = Highly aesthetic\n\n"
        f"Predict the distribution of their ratings as a probability distribution over scores "
        f"1 through 7. The 7 probabilities must sum to 1.0.\n\n"
        f"Respond only with a valid JSON array of exactly 7 floats representing the predicted "
        f"proportion of raters for each score from 1 to 7, in order, and nothing else.\n"
        f"Round each value to 3 decimal places.\n"
        f"Example: [0.020, 0.050, 0.100, 0.200, 0.350, 0.200, 0.080]"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_distribution(text: str) -> list:
    """Parse a 7-element probability distribution from model output text."""
    try:
        # JSON部分のみを抽出（念のため）
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != -1:
            dist = json.loads(text[start:end])
            if len(dist) == 7:
                total = sum(dist)
                return [round(float(x) / total, 3) for x in dist] if total > 0 else [0.143]*7
    except:
        pass
    return [0.143] * 7 # Fallback to uniform distribution

# ──────────────────────────────────────────────────────────────────────────────
# GIAA
# ──────────────────────────────────────────────────────────────────────────────

def run_giaa(genre: str, n: int = 0, resume: bool = False):
    """逐次処理モード。n=0 のときは全件処理する。resume=True で途中から再開。"""
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY in .env")

    client = genai.Client(api_key=api_key)
    samples_dir = _SAMPLES_DIR_MAP[genre]

    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    all_files = sorted([f for f in os.listdir(samples_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    if n > 0:
        all_files = all_files[:n]
        print(f"[trial] {len(all_files)} images (sequential)")
    else:
        print(f"Total images to process: {len(all_files)} (sequential)")

    _MIME_MAP = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
    user_prompt = _make_user_prompt(genre)
    per_sample_results = {}

    os.makedirs(_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, f'{genre}_results_sequential.json')
    _CHECKPOINT_INTERVAL = 100

    # resume: 既存の結果を読み込み、処理済みファイルをスキップ
    completed = []
    if resume and os.path.exists(save_path):
        with open(save_path) as fp:
            existing = json.load(fp)
        for entry in existing.get('per_sample', []):
            per_sample_results[entry['sample_file']] = entry['pred_dist']
            completed.append(entry['sample_file'])
        print(f"[resume] Loaded {len(completed)} already-processed results from {save_path}")

    done_set = set(completed)

    def _save(completed_files):
        output = {
            "genre": genre,
            "model": _MODEL,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "n_total_images": len(all_files),
            "per_sample": [
                {"sample_file": f, "pred_dist": per_sample_results.get(f, [0.143]*7)}
                for f in completed_files
            ]
        }
        with open(save_path, 'w') as fp:
            json.dump(output, fp, indent=2)

    for idx, fname in enumerate(all_files):
        if fname in done_set:
            continue
        img_path = os.path.join(samples_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        mime_type = _MIME_MAP.get(ext, 'image/jpeg')
        with open(img_path, 'rb') as f:
            img_bytes = f.read()

        response = client.models.generate_content(
            model=_MODEL,
            contents=[
                types.Content(parts=[
                    types.Part(inline_data=types.Blob(mime_type=mime_type, data=img_bytes)),
                    types.Part(text=user_prompt),
                ]),
            ],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=_MAX_TOKENS,
                temperature=0.0,
            ),
        )
        dist = _parse_distribution(response.text)
        per_sample_results[fname] = dist
        completed.append(fname)
        print(f"  [{idx + 1}/{len(all_files)}] {fname} → {dist}")

        if (idx + 1) % _CHECKPOINT_INTERVAL == 0:
            _save(completed)
            print(f"  [checkpoint] {idx + 1} images saved → {save_path}")

    _save(completed)
    print(f"\nSequential completed! Results saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# PIAA
# ──────────────────────────────────────────────────────────────────────────────

_PIAA_SYSTEM_PROMPT = (
    "You are a researcher specializing in empirical aesthetics, skilled at predicting "
    "how a specific individual perceives and rates visual content based on their "
    "psychological profile and demographic background."
)

_NATIONALITY_MAP = {'JPN': 'Japan', 'KOR': 'Korea', 'CHN': 'China'}

_PIAA_MAX_TOKENS = 5000


def _make_piaa_user_prompt(user: dict) -> str:
    nat = _NATIONALITY_MAP.get(user['nationality'], user['nationality'])
    return (
        f"A specific individual with the following profile is shown the image above\n"
        f"and asked to rate its aesthetic quality.\n\n"
        f"In the study, the participant was asked the following question:\n"
        f"\"Overall, how aesthetic do you find this image?\"\n\n"
        f"=== Individual Profile ===\n"
        f"Age            : {user['age']}\n"
        f"Gender         : {user['gender']}\n"
        f"Education      : {user['edu']}\n"
        f"Nationality    : {nat}\n\n"
        f"Domain training (0 = no formal training, 1 = formally trained):\n"
        f"  Art: {user['art_learn']},  Fashion: {user['fashion_learn']},  Photo/Video: {user['photoVideo_learn']}\n\n"
        f"Domain interest (1\u20137 scale, 1 = not interested at all, 7 = strongly interested):\n"
        f"  Art: {int(user['art_interest']) + 1},  Fashion: {int(user['fashion_interest']) + 1},  Photo/Video: {int(user['photoVideo_interest']) + 1}\n\n"
        f"Psychological questionnaire (1\u20137 scale, 1 = Disagree strongly, 7 = Agree strongly):\n"
        f"  Q1  (Extraverted, enthusiastic):          {int(user['Q1']) + 1}\n"
        f"  Q2  (Critical, quarrelsome):              {int(user['Q2']) + 1}\n"
        f"  Q3  (Dependable, self-disciplined):       {int(user['Q3']) + 1}\n"
        f"  Q4  (Anxious, easily upset):              {int(user['Q4']) + 1}\n"
        f"  Q5  (Open to new experiences, complex):   {int(user['Q5']) + 1}\n"
        f"  Q6  (Reserved, quiet):                    {int(user['Q6']) + 1}\n"
        f"  Q7  (Sympathetic, warm):                  {int(user['Q7']) + 1}\n"
        f"  Q8  (Disorganized, careless):             {int(user['Q8']) + 1}\n"
        f"  Q9  (Calm, emotionally stable):           {int(user['Q9']) + 1}\n"
        f"  Q10 (Conventional, uncreative):           {int(user['Q10']) + 1}\n\n"
        f"The individual rates the image using the following 7-point scale:\n"
        f"- 1 = Highly unaesthetic\n"
        f"- 2 = Unaesthetic\n"
        f"- 3 = Slightly unaesthetic\n"
        f"- 4 = Neutral\n"
        f"- 5 = Slightly aesthetic\n"
        f"- 6 = Aesthetic\n"
        f"- 7 = Highly aesthetic\n\n"
        f"Respond only with a single integer from 1 to 7."
    )


def _parse_piaa_score(text: str) -> int:
    """Parse a single integer 1-7 from model output."""
    text = text.strip()
    try:
        val = int(text)
        if 1 <= val <= 7:
            return val
    except ValueError:
        pass
    for char in text:
        if char.isdigit():
            val = int(char)
            if 1 <= val <= 7:
                return val
    return 4


def run_piaa(genre: str, n: int = 0, resume: bool = False):
    """逐次処理モード（PIAA）。n=0 のときは全件処理する。resume=True で途中から再開。"""
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY in .env")

    import csv
    from collections import defaultdict

    client = genai.Client(api_key=api_key)

    users = {}
    with open(os.path.join(_MAKED_DIR, 'users.csv')) as f:
        for row in csv.DictReader(f):
            users[int(row['user_id'])] = row

    ratings_by_image = defaultdict(list)
    with open(os.path.join(_MAKED_DIR, 'ratings.csv')) as f:
        for row in csv.DictReader(f):
            if row['genre'] == genre:
                ratings_by_image[row['sample_file']].append(int(row['user_id']))

    samples_dir = _SAMPLES_DIR_MAP[genre]
    all_images = sorted(ratings_by_image.keys())
    if n > 0:
        all_images = all_images[:n]
        print(f"[trial] {len(all_images)} images")
    n_pairs = sum(len(ratings_by_image[f]) for f in all_images)
    print(f"Total images: {len(all_images)}, total (image, user) pairs: {n_pairs}")

    _MIME_MAP = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
    os.makedirs(_SAVE_DIR, exist_ok=True)
    os.makedirs(_EXCLUDE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, f'{genre}_piaa_results.json')
    exclude_path = os.path.join(_EXCLUDE_DIR, f'{genre}_piaa_excluded.txt')
    _CHECKPOINT_INTERVAL = 100

    per_sample_results = defaultdict(dict)
    done_pairs_set = set()

    if resume and os.path.exists(save_path):
        with open(save_path) as fp:
            existing = json.load(fp)
        for entry in existing.get('per_sample', []):
            fname = entry['sample_file']
            for r in entry.get('ratings', []):
                per_sample_results[fname][r['user_id']] = r['pred_score']
                done_pairs_set.add((fname, r['user_id']))
        print(f"[resume] Loaded {len(done_pairs_set)} already-processed pairs")

    def _save():
        output = {
            "genre": genre,
            "model": _MODEL,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "n_total_pairs": n_pairs,
            "per_sample": [
                {
                    "sample_file": f,
                    "ratings": [
                        {"user_id": uid, "pred_score": per_sample_results[f][uid]}
                        for uid in ratings_by_image[f]
                        if uid in per_sample_results[f]
                    ]
                }
                for f in all_images if f in per_sample_results
            ]
        }
        with open(save_path, 'w') as fp:
            json.dump(output, fp, indent=2)

    img_ext = _GENRE_IMG_EXT.get(genre)
    pair_idx = len(done_pairs_set)
    for fname in all_images:
        img_fname = os.path.splitext(fname)[0] + img_ext if img_ext else fname
        img_path = os.path.join(samples_dir, img_fname)
        ext = os.path.splitext(img_fname)[1].lower()
        mime_type = _MIME_MAP.get(ext, 'image/jpeg')
        with open(img_path, 'rb') as f:
            img_bytes = f.read()

        for user_id in ratings_by_image[fname]:
            if (fname, user_id) in done_pairs_set:
                continue
            user = users[user_id]
            user_prompt = _make_piaa_user_prompt(user)

            response = client.models.generate_content(
                model=_MODEL,
                contents=[
                    types.Content(parts=[
                        types.Part(inline_data=types.Blob(mime_type=mime_type, data=img_bytes)),
                        types.Part(text=user_prompt),
                    ]),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=_PIAA_SYSTEM_PROMPT,
                    max_output_tokens=_PIAA_MAX_TOKENS,
                    temperature=0.0,
                ),
            )
            if response.text is None:
                cand = response.candidates[0] if response.candidates else None
                finish_reason = getattr(cand, 'finish_reason', None)
                pf = getattr(response, 'prompt_feedback', None)
                block_reason = getattr(pf, 'block_reason', None)
                print(f"[WARN] empty response for {fname} / user {user_id}: "
                      f"finish_reason={finish_reason} block_reason={block_reason} → fallback score=4")
                score = 4
                with open(exclude_path, 'a') as ef:
                    ef.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{fname}\tuser_{user_id}\t"
                             f"finish_reason={finish_reason}\tblock_reason={block_reason}\n")
            else:
                score = _parse_piaa_score(response.text)
            per_sample_results[fname][user_id] = score
            done_pairs_set.add((fname, user_id))
            pair_idx += 1
            print(f"  [{pair_idx}/{n_pairs}] {fname} / user {user_id} → {score}")

            if pair_idx % _CHECKPOINT_INTERVAL == 0:
                _save()
                print(f"  [checkpoint] {pair_idx} pairs saved → {save_path}")

    _save()
    print(f"\nResults saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Batch Mode (async queue) — shared core
#
# 同期 generate_content の代わりに client.batches.create でまとめて投入する。
# Google 側のキューで空きが出たら処理されるため 503(高需要)で落ちない。
# 画像は Files API にアップロードして file_uri 参照で渡す(リクエストを小さく保つ)。
# 投入済みジョブ名を state ファイルに保存するので、ポーリング中に中断しても
# --batch --resume で再アタッチでき、再投入しない。
# ──────────────────────────────────────────────────────────────────────────────

_BATCH_CHUNK_SIZE = 500       # 1 ジョブあたりの最大リクエスト数(順次投入)
_BATCH_POLL_INTERVAL = 30     # ジョブ状態のポーリング間隔(秒)
_KEY_SEP = '\t'               # metadata key の区切り(ファイル名に出現しない文字)

_BATCH_TERMINAL_STATES = {
    'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED', 'JOB_STATE_PARTIALLY_SUCCEEDED',
}


def _state_name(state) -> str:
    """JobState(enum/str) を 'JOB_STATE_*' 文字列に正規化する。"""
    return getattr(state, 'name', None) or str(state).split('.')[-1]


def _upload_image_part(client, img_path: str, mime_type: str, cache: dict):
    """画像を Files API にアップロードし file_data Part を返す。同一パスはキャッシュ。"""
    if img_path in cache:
        return cache[img_path]
    f = client.files.upload(file=img_path, config=types.UploadFileConfig(mime_type=mime_type))
    part = types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=mime_type))
    cache[img_path] = part
    return part


def _poll_until_done(client, job_name: str, poll_interval: int):
    """ジョブが終端状態になるまでポーリングして BatchJob を返す。"""
    while True:
        job = client.batches.get(name=job_name)
        state = _state_name(job.state)
        print(f"  [batch] {job_name} state={state} ({datetime.now().strftime('%H:%M:%S')})")
        if state in _BATCH_TERMINAL_STATES:
            return job
        time.sleep(poll_interval)


def _collect_responses(job, keys: list) -> dict:
    """完了ジョブから {key: GenerateContentResponse | None} を取り出す。"""
    out = {}
    dest = getattr(job, 'dest', None)
    responses = getattr(dest, 'inlined_responses', None) if dest else None
    if not responses:
        print(f"  [batch][WARN] no inlined_responses (state={_state_name(job.state)}, error={job.error})")
        return out
    for i, ir in enumerate(responses):
        key = None
        if ir.metadata and 'key' in ir.metadata:
            key = ir.metadata['key']
        elif keys and i < len(keys):
            key = keys[i]            # metadata が無い場合は投入順で対応付け
        if key is None:
            continue
        if ir.error:
            print(f"  [batch][WARN] item error for {key}: {ir.error}")
            out[key] = None
        else:
            out[key] = ir.response
    return out


def _run_batch_jobs(client, model: str, requests_by_key: dict, state_path: str,
                    on_chunk, display_prefix: str,
                    chunk_size: int = _BATCH_CHUNK_SIZE,
                    poll_interval: int = _BATCH_POLL_INTERVAL):
    """requests_by_key を chunk ごとにバッチ投入し、各 chunk 完了時に on_chunk(responses) を呼ぶ。

    on_chunk: dict {key: GenerateContentResponse | None} を受け取り、パース・保存する callback。
    state ファイルに投入中ジョブ名を保存し、完了・保存後に削除する。
    """
    # 1) 投入済みで未回収のジョブがあれば、まず再アタッチして回収する
    if os.path.exists(state_path):
        with open(state_path) as fp:
            st = json.load(fp)
        print(f"  [batch] re-attaching to in-flight job {st['job_name']}")
        job = _poll_until_done(client, st['job_name'], poll_interval)
        on_chunk(_collect_responses(job, st.get('keys')))
        os.remove(state_path)
        for k in st.get('keys', []):
            requests_by_key.pop(k, None)   # 回収済みは残りから除外

    # 2) 残りを chunk 単位で順次投入(同時アクティブジョブは常に 1 つ)
    keys = list(requests_by_key.keys())
    if not keys:
        print("  [batch] nothing to submit (all pairs already done)")
        return
    total_chunks = (len(keys) + chunk_size - 1) // chunk_size
    for ci in range(0, len(keys), chunk_size):
        chunk_keys = keys[ci:ci + chunk_size]
        reqs = [requests_by_key[k] for k in chunk_keys]
        display_name = f"{display_prefix}-c{ci // chunk_size + 1}-{datetime.now().strftime('%H%M%S')}"
        job = client.batches.create(
            model=model,
            src=reqs,
            config=types.CreateBatchJobConfig(display_name=display_name),
        )
        print(f"  [batch] submitted chunk {ci // chunk_size + 1}/{total_chunks} "
              f"({len(chunk_keys)} requests) → {job.name}")
        with open(state_path, 'w') as fp:
            json.dump({'job_name': job.name, 'keys': chunk_keys}, fp)
        job = _poll_until_done(client, job.name, poll_interval)
        on_chunk(_collect_responses(job, chunk_keys))
        os.remove(state_path)


# ──────────────────────────────────────────────────────────────────────────────
# Batch Mode — GIAA
# ──────────────────────────────────────────────────────────────────────────────

def run_giaa_batch(genre: str, n: int = 0, resume: bool = False):
    """GIAA をバッチモードで実行。終了済み(resume)はやり直さない。"""
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY in .env")

    client = genai.Client(api_key=api_key)
    samples_dir = _SAMPLES_DIR_MAP[genre]

    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    all_files = sorted([f for f in os.listdir(samples_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    if n > 0:
        all_files = all_files[:n]
    print(f"Total images: {len(all_files)} (batch)")

    _MIME_MAP = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
    user_prompt = _make_user_prompt(genre)

    os.makedirs(_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, f'{genre}_results_sequential.json')
    state_path = os.path.join(_SAVE_DIR, f'{genre}_giaa_batch_state.json')

    per_sample_results = {}
    completed = []
    if resume and os.path.exists(save_path):
        with open(save_path) as fp:
            existing = json.load(fp)
        for entry in existing.get('per_sample', []):
            per_sample_results[entry['sample_file']] = entry['pred_dist']
            completed.append(entry['sample_file'])
        print(f"[resume] Loaded {len(completed)} already-processed results")
    done_set = set(completed)

    def _save():
        output = {
            "genre": genre, "model": _MODEL,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "n_total_images": len(all_files),
            "per_sample": [
                {"sample_file": f, "pred_dist": per_sample_results.get(f, [0.143] * 7)}
                for f in completed
            ],
        }
        with open(save_path, 'w') as fp:
            json.dump(output, fp, indent=2)

    # 未処理画像のリクエストを構築(画像は Files API にアップロード)
    img_cache = {}
    requests_by_key = {}
    pending = [f for f in all_files if f not in done_set]
    print(f"  building {len(pending)} requests (uploading images)…")
    for fname in pending:
        img_path = os.path.join(samples_dir, fname)
        mime_type = _MIME_MAP.get(os.path.splitext(fname)[1].lower(), 'image/jpeg')
        img_part = _upload_image_part(client, img_path, mime_type, img_cache)
        requests_by_key[fname] = types.InlinedRequest(
            contents=[types.Content(parts=[img_part, types.Part(text=user_prompt)])],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT, max_output_tokens=_MAX_TOKENS, temperature=0.0,
            ),
            metadata={'key': fname},
        )

    def _on_chunk(responses: dict):
        for key, resp in responses.items():
            dist = _parse_distribution(resp.text) if (resp and resp.text) else [0.143] * 7
            per_sample_results[key] = dist
            if key not in done_set:
                completed.append(key)
                done_set.add(key)
        _save()
        print(f"  [batch] saved {len(completed)}/{len(all_files)} → {save_path}")

    _run_batch_jobs(client, _MODEL, requests_by_key, state_path, _on_chunk,
                    display_prefix=f"{genre}-giaa")
    _save()
    print(f"\nBatch GIAA completed! Results saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Batch Mode — PIAA
# ──────────────────────────────────────────────────────────────────────────────

def run_piaa_batch(genre: str, n: int = 0, resume: bool = False):
    """PIAA をバッチモードで実行。終了済み(resume)ペアはやり直さない。"""
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY in .env")

    import csv
    from collections import defaultdict

    client = genai.Client(api_key=api_key)

    users = {}
    with open(os.path.join(_MAKED_DIR, 'users.csv')) as f:
        for row in csv.DictReader(f):
            users[int(row['user_id'])] = row

    ratings_by_image = defaultdict(list)
    with open(os.path.join(_MAKED_DIR, 'ratings.csv')) as f:
        for row in csv.DictReader(f):
            if row['genre'] == genre:
                ratings_by_image[row['sample_file']].append(int(row['user_id']))

    samples_dir = _SAMPLES_DIR_MAP[genre]
    all_images = sorted(ratings_by_image.keys())
    if n > 0:
        all_images = all_images[:n]
    n_pairs = sum(len(ratings_by_image[f]) for f in all_images)
    print(f"Total images: {len(all_images)}, total (image, user) pairs: {n_pairs} (batch)")

    _MIME_MAP = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
    os.makedirs(_SAVE_DIR, exist_ok=True)
    os.makedirs(_EXCLUDE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, f'{genre}_piaa_results.json')
    exclude_path = os.path.join(_EXCLUDE_DIR, f'{genre}_piaa_excluded.txt')
    state_path = os.path.join(_SAVE_DIR, f'{genre}_piaa_batch_state.json')

    per_sample_results = defaultdict(dict)
    done_pairs_set = set()
    if resume and os.path.exists(save_path):
        with open(save_path) as fp:
            existing = json.load(fp)
        for entry in existing.get('per_sample', []):
            fname = entry['sample_file']
            for r in entry.get('ratings', []):
                per_sample_results[fname][r['user_id']] = r['pred_score']
                done_pairs_set.add((fname, r['user_id']))
        print(f"[resume] Loaded {len(done_pairs_set)} already-processed pairs")

    def _save():
        output = {
            "genre": genre, "model": _MODEL,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "n_total_pairs": n_pairs,
            "per_sample": [
                {
                    "sample_file": f,
                    "ratings": [
                        {"user_id": uid, "pred_score": per_sample_results[f][uid]}
                        for uid in ratings_by_image[f] if uid in per_sample_results[f]
                    ],
                }
                for f in all_images if f in per_sample_results
            ],
        }
        with open(save_path, 'w') as fp:
            json.dump(output, fp, indent=2)

    # 未処理ペアのリクエストを構築(画像は 1 枚につき 1 度だけアップロード)
    img_ext = _GENRE_IMG_EXT.get(genre)
    img_cache = {}
    requests_by_key = {}
    n_pending = 0
    for fname in all_images:
        pending_uids = [u for u in ratings_by_image[fname] if (fname, u) not in done_pairs_set]
        if not pending_uids:
            continue
        img_fname = os.path.splitext(fname)[0] + img_ext if img_ext else fname
        img_path = os.path.join(samples_dir, img_fname)
        mime_type = _MIME_MAP.get(os.path.splitext(img_fname)[1].lower(), 'image/jpeg')
        img_part = _upload_image_part(client, img_path, mime_type, img_cache)
        for user_id in pending_uids:
            key = f"{fname}{_KEY_SEP}{user_id}"
            requests_by_key[key] = types.InlinedRequest(
                contents=[types.Content(parts=[img_part, types.Part(text=_make_piaa_user_prompt(users[user_id]))])],
                config=types.GenerateContentConfig(
                    system_instruction=_PIAA_SYSTEM_PROMPT, max_output_tokens=_PIAA_MAX_TOKENS, temperature=0.0,
                ),
                metadata={'key': key},
            )
            n_pending += 1
    print(f"  building {n_pending} requests across {len(img_cache)} images (uploading images)…")

    def _on_chunk(responses: dict):
        for key, resp in responses.items():
            fname, uid_str = key.split(_KEY_SEP)
            user_id = int(uid_str)
            if resp is None or resp.text is None:
                cand = resp.candidates[0] if (resp and resp.candidates) else None
                finish_reason = getattr(cand, 'finish_reason', None)
                pf = getattr(resp, 'prompt_feedback', None) if resp else None
                block_reason = getattr(pf, 'block_reason', None)
                print(f"  [batch][WARN] empty response {fname}/user {user_id}: "
                      f"finish_reason={finish_reason} block_reason={block_reason} → fallback 4")
                score = 4
                with open(exclude_path, 'a') as ef:
                    ef.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{fname}\tuser_{user_id}\t"
                             f"finish_reason={finish_reason}\tblock_reason={block_reason}\n")
            else:
                score = _parse_piaa_score(resp.text)
            per_sample_results[fname][user_id] = score
            done_pairs_set.add((fname, user_id))
        _save()
        print(f"  [batch] saved {len(done_pairs_set)}/{n_pairs} pairs → {save_path}")

    _run_batch_jobs(client, _MODEL, requests_by_key, state_path, _on_chunk,
                    display_prefix=f"{genre}-piaa")
    _save()
    print(f"\nBatch PIAA completed! Results saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Gemini zero-shot evaluation (all images)')
    parser.add_argument('--mode',  required=True, choices=['giaa', 'piaa'], help='Evaluation mode')
    parser.add_argument('--genre', required=True, choices=['art', 'fashion', 'scenery'])
    parser.add_argument('--trial', type=int, default=0, help='Limit to N images (0 = all)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results JSON')
    parser.add_argument('--batch', action='store_true',
                        help='Batch Mode: キュー投入して空き次第処理(503回避・約半額)。中断時は --batch --resume で再アタッチ')
    args = parser.parse_args()

    if args.batch:
        if args.mode == 'giaa':
            run_giaa_batch(genre=args.genre, n=args.trial, resume=args.resume)
        elif args.mode == 'piaa':
            run_piaa_batch(genre=args.genre, n=args.trial, resume=args.resume)
    elif args.mode == 'giaa':
        run_giaa(genre=args.genre, n=args.trial, resume=args.resume)
    elif args.mode == 'piaa':
        run_piaa(genre=args.genre, n=args.trial, resume=args.resume)
