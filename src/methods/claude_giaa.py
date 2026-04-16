"""
Zero-shot GIAA evaluation using Claude (claude-opus-4-6) via Anthropic Batch API.
Runs on ALL images for the given genre (no train/test split required).

Usage:
    python -m src.methods.claude_giaa --genre art
    python -m src.methods.claude_giaa --genre fashion --trial 10
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
# Paths
# ──────────────────────────────────────────────────────────────────────────────

_MAKED_DIR = '/home/hayashi0884/proj-xpass-DA/data/maked'
_SAVE_DIR   = '/home/hayashi0884/proj-xpass-DA/reports/exp/claude'

_SAMPLES_DIR_MAP = {
    'art':     '/home/hayashi0884/proj-xpass/data/samples/art',
    'fashion': '/home/hayashi0884/proj-xpass/data/samples/fashion',
    'scenery': '/home/hayashi0884/proj-xpass/data/samples/scenery_image',
}

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

_MODEL         = "claude-opus-4-6"
_MAX_TOKENS    = 64
_POLL_INTERVAL = 60   # seconds
_BATCH_SIZE    = 500  # images per batch (to stay under 32 MB payload limit)


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
        dist = np.array(json.loads(text.strip()), dtype=np.float32)
        if dist.shape == (7,) and dist.min() >= 0:
            total = dist.sum()
            if total > 0:
                return dist / total
    except Exception:
        pass
    return _UNIFORM_DIST.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(genre: str, trial: int = 0, batch_ids: list = None) -> dict:
    """Run Claude zero-shot GIAA on all images for *genre*.

    Parameters
    ----------
    genre     : 'art' | 'fashion' | 'scenery'
    trial     : if > 0, limit to the first N images (for quick tests)
    batch_ids : if given, skip submission and re-fetch results from these batches

    Saves
    -----
    {_SAVE_DIR}/{genre}_results.json
    """
    import anthropic

    samples_dir = _SAMPLES_DIR_MAP[genre]
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    # ── 全画像ファイルの列挙 ─────────────────────────────────────────────────
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    all_files = sorted([
        f for f in os.listdir(samples_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    if trial > 0:
        all_files = all_files[:trial]
        print(f"[trial] mode ON — using {len(all_files)} images")
    print(f"Total images: {len(all_files)}")

    client = anthropic.Anthropic()

    if batch_ids:
        # ── 既存バッチの結果を再取得 ─────────────────────────────────────────
        print(f"Re-fetching results from {len(batch_ids)} existing batch(es): {batch_ids}")
    else:
        # ── Batch API リクエスト構築（チャンク分割） ──────────────────────────
        user_prompt = _make_user_prompt(genre)

        def _make_request(idx, fname):
            img_path = os.path.join(samples_dir, fname)
            return {
                'custom_id': f'sample_{idx}',
                'params': {
                    'model': _MODEL,
                    'max_tokens': _MAX_TOKENS,
                    'system': _SYSTEM_PROMPT,
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image',
                                'source': {
                                    'type': 'base64',
                                    'media_type': _media_type(img_path),
                                    'data': _encode_image(img_path),
                                },
                            },
                            {'type': 'text', 'text': user_prompt},
                        ],
                    }],
                },
            }

        # _BATCH_SIZE 枚ずつ分割して送信
        chunks = [
            all_files[i:i + _BATCH_SIZE]
            for i in range(0, len(all_files), _BATCH_SIZE)
        ]
        batch_ids = []
        for chunk_no, chunk in enumerate(chunks):
            offset = chunk_no * _BATCH_SIZE
            requests = [_make_request(offset + i, fname) for i, fname in enumerate(chunk)]
            print(f"Submitting batch {chunk_no + 1}/{len(chunks)} ({len(requests)} images) ...")
            batch = client.messages.batches.create(requests=requests)
            batch_ids.append(batch.id)
            print(f"  Batch ID: {batch.id}")

        # ── 全バッチ完了待ち (ポーリング) ────────────────────────────────────
        pending = set(batch_ids)
        while pending:
            time.sleep(_POLL_INTERVAL)
            still_pending = set()
            for bid in pending:
                b = client.messages.batches.retrieve(bid)
                c = b.request_counts
                print(
                    f"  [{bid}] {b.processing_status} | "
                    f"succeeded={c.succeeded}  errored={c.errored}  processing={c.processing}"
                )
                if b.processing_status != 'ended':
                    still_pending.add(bid)
            pending = still_pending

    # ── 結果パース（全バッチ） ────────────────────────────────────────────────
    pred_dists: dict = {}
    for bid in batch_ids:
        for result in client.messages.batches.results(bid):
            idx = int(result.custom_id.split('_')[1])
            if result.result.type == 'succeeded':
                text = result.result.message.content[0].text
                pred_dists[idx] = _parse_distribution(text)
            else:
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

    parser = argparse.ArgumentParser(description='Claude zero-shot GIAA evaluation (all images)')
    parser.add_argument('--genre',    required=True, choices=['art', 'fashion', 'scenery'])
    parser.add_argument('--trial',    type=int, default=0,
                        help='Limit to first N images for quick testing (0 = all)')
    parser.add_argument('--batch_ids', type=str, default=None,
                        help='Re-fetch results from existing batches (comma-separated, skip submission)')
    cli = parser.parse_args()

    batch_ids = [b.strip() for b in cli.batch_ids.split(',')] if cli.batch_ids else None
    run(genre=cli.genre, trial=cli.trial, batch_ids=batch_ids)
