"""
Zero-shot GIAA evaluation using GPT (gpt-4o) via OpenAI API.
Runs on ALL images for the given genre (no train/test split required).

Usage:
    python -m src.methods.gpt_giaa --genre art --mode sequential
    python -m src.methods.gpt_giaa --genre fashion --mode batch
    python -m src.methods.gpt_giaa --genre art --mode batch --trial 10
    python -m src.methods.gpt_giaa --genre art --mode batch --batch_ids batch_xxx,batch_yyy
"""
import os
import json
import base64
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Settings
# ──────────────────────────────────────────────────────────────────────────────

_SAVE_DIR = '/home/hayashi0884/proj-xpass-DA/reports/exp/gpt'

_SAMPLES_DIR_MAP = {
    'art':     '/home/hayashi0884/proj-xpass/data/samples/art',
    'fashion': '/home/hayashi0884/proj-xpass/data/samples/fashion',
    'scenery': '/home/hayashi0884/proj-xpass/data/samples/scenery_image',
}

_MODEL         = "gpt-4o"
_MAX_TOKENS    = 64
_POLL_INTERVAL = 60   # seconds
_BATCH_SIZE    = 500  # images per batch

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

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
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def _media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        '.jpg':  'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png':  'image/png',
        '.gif':  'image/gif',
        '.webp': 'image/webp',
    }.get(ext, 'image/jpeg')


def _encode_image(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')


# ──────────────────────────────────────────────────────────────────────────────
# Distribution helpers
# ──────────────────────────────────────────────────────────────────────────────

_UNIFORM_DIST = np.ones(7, dtype=np.float32) / 7


def _parse_distribution(text: str) -> np.ndarray:
    """Parse a 7-element probability distribution from model output."""
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != 0:
            dist = np.array(json.loads(text[start:end]), dtype=np.float32)
            if dist.shape == (7,) and dist.min() >= 0:
                total = dist.sum()
                if total > 0:
                    return dist / total
    except Exception:
        pass
    return _UNIFORM_DIST.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Sequential mode
# ──────────────────────────────────────────────────────────────────────────────

def run_sequential(genre: str, trial: int = 0):
    """逐次処理モード。trial=0 のときは全件処理する。"""
    from openai import OpenAI

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in .env")

    client = OpenAI(api_key=api_key)
    samples_dir = _SAMPLES_DIR_MAP[genre]
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    all_files = sorted([f for f in os.listdir(samples_dir)
                        if os.path.splitext(f)[1].lower() in valid_exts])
    if trial > 0:
        all_files = all_files[:trial]
        print(f"[trial] mode ON — using {len(all_files)} images (sequential)")
    else:
        print(f"Total images to process: {len(all_files)} (sequential)")

    user_prompt = _make_user_prompt(genre)
    per_sample_results = {}

    os.makedirs(_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, f'{genre}_results_sequential.json')
    _CHECKPOINT_INTERVAL = 100

    def _save(completed_files):
        output = {
            "genre": genre,
            "model": _MODEL,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "n_total_images": len(all_files),
            "per_sample": [
                {"sample_file": f, "pred_dist": per_sample_results.get(f, [round(float(x), 3) for x in _UNIFORM_DIST])}
                for f in completed_files
            ]
        }
        with open(save_path, 'w') as fp:
            json.dump(output, fp, indent=2)

    completed = []
    for idx, fname in enumerate(all_files):
        img_path = os.path.join(samples_dir, fname)
        mime = _media_type(img_path)
        b64 = _encode_image(img_path)

        response = client.chat.completions.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": user_prompt},
                ]},
            ],
        )
        text = response.choices[0].message.content or ""
        dist = _parse_distribution(text)
        per_sample_results[fname] = [round(float(x), 3) for x in dist]
        completed.append(fname)
        print(f"  [{idx + 1}/{len(all_files)}] {fname} → {per_sample_results[fname]}")

        if (idx + 1) % _CHECKPOINT_INTERVAL == 0:
            _save(completed)
            print(f"  [checkpoint] {idx + 1} images saved → {save_path}")

    _save(completed)
    print(f"\nSequential completed! Results saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Batch mode
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(genre: str, trial: int = 0, batch_ids: list = None):
    """OpenAI Batch API を使った処理。

    Parameters
    ----------
    genre     : 'art' | 'fashion' | 'scenery'
    trial     : if > 0, limit to the first N images (for quick tests)
    batch_ids : if given, skip submission and re-fetch results from these batches
    """
    from openai import OpenAI

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in .env")

    client = OpenAI(api_key=api_key)
    samples_dir = _SAMPLES_DIR_MAP[genre]
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    # ── 全画像ファイルの列挙 ─────────────────────────────────────────────────
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    all_files = sorted([f for f in os.listdir(samples_dir)
                        if os.path.splitext(f)[1].lower() in valid_exts])
    if trial > 0:
        all_files = all_files[:trial]
        print(f"[trial] mode ON — using {len(all_files)} images")
    print(f"Total images: {len(all_files)}")

    user_prompt = _make_user_prompt(genre)

    if batch_ids:
        # ── 既存バッチの結果を再取得 ─────────────────────────────────────────
        print(f"Re-fetching results from {len(batch_ids)} existing batch(es): {batch_ids}")
    else:
        # ── Batch API リクエスト構築（チャンク分割） ──────────────────────────
        chunks = [all_files[i:i + _BATCH_SIZE] for i in range(0, len(all_files), _BATCH_SIZE)]
        batch_ids = []

        for chunk_no, chunk_files in enumerate(chunks):
            offset = chunk_no * _BATCH_SIZE
            print(f"\n=== Chunk {chunk_no + 1}/{len(chunks)} ({len(chunk_files)} images) ===")

            # JSONL リクエストの構築
            jsonl_lines = []
            for i, fname in enumerate(chunk_files):
                global_idx = offset + i
                img_path = os.path.join(samples_dir, fname)
                mime = _media_type(img_path)
                b64 = _encode_image(img_path)

                req = {
                    "custom_id": f"sample_{global_idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": _MODEL,
                        "max_tokens": _MAX_TOKENS,
                        "temperature": 0.0,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                                {"type": "text", "text": user_prompt},
                            ]},
                        ],
                    },
                }
                jsonl_lines.append(json.dumps(req))
                if (i + 1) % 50 == 0:
                    print(f"  Prepared: {i + 1}/{len(chunk_files)}")

            # JSONL ファイルの作成・アップロード
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            input_jsonl = f"batch_input_{genre}_chunk{chunk_no}_{ts}.jsonl"
            with open(input_jsonl, 'w') as f:
                f.write('\n'.join(jsonl_lines))

            print(f"Uploading request file: {input_jsonl}")
            with open(input_jsonl, 'rb') as f:
                batch_file = client.files.create(file=f, purpose="batch")
            os.remove(input_jsonl)

            # バッチジョブの作成
            print(f"Starting batch job with model: {_MODEL}...")
            batch = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batch_ids.append(batch.id)
            print(f"  Batch ID: {batch.id}")

        # ── 全バッチ完了待ち (ポーリング) ────────────────────────────────────
        pending = set(batch_ids)
        while pending:
            time.sleep(_POLL_INTERVAL)
            still_pending = set()
            for bid in pending:
                b = client.batches.retrieve(bid)
                c = b.request_counts
                print(
                    f"  [{bid}] {b.status} | "
                    f"completed={c.completed}  failed={c.failed}  total={c.total}"
                )
                if b.status not in ('completed', 'failed', 'expired', 'cancelled'):
                    still_pending.add(bid)
                elif b.status != 'completed':
                    raise RuntimeError(f"Batch {bid} ended with status: {b.status}")
            pending = still_pending

    # ── 結果パース（全バッチ） ────────────────────────────────────────────────
    pred_dists: dict = {}
    for bid in batch_ids:
        b = client.batches.retrieve(bid)
        if not b.output_file_id:
            print(f"  Warning: no output file for batch {bid}")
            continue
        content = client.files.content(b.output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            res_data = json.loads(line)
            custom_id = res_data.get("custom_id", "")
            idx = int(custom_id.split('_')[1])
            try:
                text = res_data["response"]["body"]["choices"][0]["message"]["content"]
                pred_dists[idx] = _parse_distribution(text)
            except Exception:
                pred_dists[idx] = _UNIFORM_DIST.copy()

    # ── per-sample 結果の構築 ─────────────────────────────────────────────────
    per_sample = []
    for idx, fname in enumerate(all_files):
        pred_hist = pred_dists.get(idx, _UNIFORM_DIST.copy())
        per_sample.append({
            'sample_file': fname,
            'pred_dist':   [round(float(x), 3) for x in pred_hist],
        })

    # ── JSON 保存 ─────────────────────────────────────────────────────────────
    os.makedirs(_SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    meta = {
        'genre':          genre,
        'model':          _MODEL,
        'timestamp':      timestamp,
        'batch_ids':      batch_ids,
        'n_total_images': len(all_files),
    }

    save_path = os.path.join(_SAVE_DIR, f'{genre}_results.json')
    with open(save_path, 'w') as f:
        f.write('{\n')
        for key, val in meta.items():
            f.write(f'  {json.dumps(key)}: {json.dumps(val)},\n')
        f.write('  "per_sample": [\n')
        for i, s in enumerate(per_sample):
            dist_str = '[' + ', '.join(f'{x:.3f}' for x in s['pred_dist']) + ']'
            comma = ',' if i < len(per_sample) - 1 else ''
            f.write(
                f'    {{\n'
                f'      "sample_file": {json.dumps(s["sample_file"])},\n'
                f'      "pred_dist": {dist_str}\n'
                f'    }}{comma}\n'
            )
        f.write('  ]\n')
        f.write('}\n')
    print(f"\nResults saved → {save_path}")

    return {**meta, 'per_sample': per_sample}


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GPT zero-shot GIAA evaluation (all images)')
    parser.add_argument('--genre',     required=True, choices=['art', 'fashion', 'scenery'])
    parser.add_argument('--mode',      required=True, choices=['sequential', 'batch'])
    parser.add_argument('--trial',     type=int, default=0,
                        help='Limit to first N images for quick testing (0 = all)')
    parser.add_argument('--batch_ids', type=str, default=None,
                        help='Re-fetch results from existing batches (comma-separated, skip submission)')
    cli = parser.parse_args()

    batch_ids = [b.strip() for b in cli.batch_ids.split(',')] if cli.batch_ids else None

    if cli.mode == 'sequential':
        run_sequential(genre=cli.genre, trial=cli.trial)
    else:
        run_batch(genre=cli.genre, trial=cli.trial, batch_ids=batch_ids)
