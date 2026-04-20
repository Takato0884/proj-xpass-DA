"""
Zero-shot evaluation using Claude (claude-opus-4-6) via Anthropic API.
Runs on ALL images for the given genre (no train/test split required).

Usage:
    python -m src.methods.claude --mode giaa --genre art
    python -m src.methods.claude --mode giaa --genre fashion --trial 10
"""
import os
import json
import base64
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

_MODEL      = "claude-opus-4-6"
_MAX_TOKENS = 5000


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
# GIAA
# ──────────────────────────────────────────────────────────────────────────────

def run_giaa(genre: str, trial: int = 0) -> dict:
    """Run Claude zero-shot GIAA on all images for *genre*.

    Parameters
    ----------
    genre  : 'art' | 'fashion' | 'scenery'
    trial  : if > 0, limit to the first N images (for quick tests)

    Saves
    -----
    {_SAVE_DIR}/{genre}_results.json
    """
    import anthropic

    samples_dir = _SAMPLES_DIR_MAP[genre]
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

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
    user_prompt = _make_user_prompt(genre)
    per_sample_results = {}
    completed = []

    os.makedirs(_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, f'{genre}_results.json')
    _CHECKPOINT_INTERVAL = 100

    def _save():
        output = {
            "genre": genre,
            "model": _MODEL,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "n_total_images": len(all_files),
            "per_sample": [
                {"sample_file": f, "pred_dist": [round(float(x), 3) for x in per_sample_results.get(f, _UNIFORM_DIST)]}
                for f in completed
            ]
        }
        with open(save_path, 'w') as fp:
            json.dump(output, fp, indent=2)

    for idx, fname in enumerate(all_files):
        img_path = os.path.join(samples_dir, fname)
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{
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
        )
        text = response.content[0].text
        dist = _parse_distribution(text)
        per_sample_results[fname] = dist
        completed.append(fname)
        print(f"  [{idx + 1}/{len(all_files)}] {fname} → {[round(float(x), 3) for x in dist]}")

        if (idx + 1) % _CHECKPOINT_INTERVAL == 0:
            _save()
            print(f"  [checkpoint] {idx + 1} images saved → {save_path}")

    _save()
    print(f"\nResults saved → {save_path}")

    return {
        'genre': genre,
        'model': _MODEL,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_total_images': len(all_files),
        'per_sample': [
            {'sample_file': f, 'pred_dist': [round(float(x), 3) for x in per_sample_results.get(f, _UNIFORM_DIST)]}
            for f in completed
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# PIAA
# ──────────────────────────────────────────────────────────────────────────────

_PIAA_SYSTEM_PROMPT = (
    "You are a researcher specializing in empirical aesthetics, skilled at predicting "
    "how a specific individual perceives and rates visual content based on their "
    "psychological profile and demographic background."
)

_NATIONALITY_MAP = {'JPN': 'Japan', 'KOR': 'Korea', 'CHN': 'China'}

_PIAA_MAX_TOKENS = 8


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


def run_piaa(genre: str, trial: int = 0, resume: bool = False) -> dict:
    """Run Claude zero-shot PIAA on all (image, user) pairs for *genre*.

    Parameters
    ----------
    genre  : 'art' | 'fashion' | 'scenery'
    trial  : if > 0, limit to the first N images (for quick tests)
    resume : if True, skip already-processed pairs from existing JSON

    Saves
    -----
    {_SAVE_DIR}/{genre}_piaa_results.json
    """
    import anthropic
    import csv
    from collections import defaultdict

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
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    all_images = sorted(ratings_by_image.keys())
    if trial > 0:
        all_images = all_images[:trial]
        print(f"[trial] mode ON — using {len(all_images)} images")
    n_pairs = sum(len(ratings_by_image[f]) for f in all_images)
    print(f"Total images: {len(all_images)}, total (image, user) pairs: {n_pairs}")

    client = anthropic.Anthropic()
    os.makedirs(_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, f'{genre}_piaa_results.json')
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

    pair_idx = len(done_pairs_set)
    for fname in all_images:
        img_path = os.path.join(samples_dir, fname)
        img_b64 = _encode_image(img_path)
        mime = _media_type(img_path)

        for user_id in ratings_by_image[fname]:
            if (fname, user_id) in done_pairs_set:
                continue
            user = users[user_id]
            user_prompt = _make_piaa_user_prompt(user)

            response = client.messages.create(
                model=_MODEL,
                max_tokens=_PIAA_MAX_TOKENS,
                system=_PIAA_SYSTEM_PROMPT,
                messages=[{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': mime,
                                'data': img_b64,
                            },
                            'cache_control': {'type': 'ephemeral'},
                        },
                        {'type': 'text', 'text': user_prompt},
                    ],
                }],
            )
            score = _parse_piaa_score(response.content[0].text)
            per_sample_results[fname][user_id] = score
            done_pairs_set.add((fname, user_id))
            pair_idx += 1
            print(f"  [{pair_idx}/{n_pairs}] {fname} / user {user_id} → {score}")

            if pair_idx % _CHECKPOINT_INTERVAL == 0:
                _save()
                print(f"  [checkpoint] {pair_idx} pairs saved → {save_path}")

    _save()
    print(f"\nResults saved → {save_path}")

    return {
        'genre': genre,
        'model': _MODEL,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_total_pairs': n_pairs,
        'per_sample': [
            {
                'sample_file': f,
                'ratings': [
                    {'user_id': uid, 'pred_score': per_sample_results[f][uid]}
                    for uid in ratings_by_image[f]
                ]
            }
            for f in all_images
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Claude zero-shot evaluation (all images)')
    parser.add_argument('--mode',  required=True, choices=['giaa', 'piaa'], help='Evaluation mode')
    parser.add_argument('--genre', required=True, choices=['art', 'fashion', 'scenery'])
    parser.add_argument('--trial', type=int, default=0,
                        help='Limit to first N images for quick testing (0 = all)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results JSON')
    cli = parser.parse_args()

    if cli.mode == 'giaa':
        run_giaa(genre=cli.genre, trial=cli.trial)
    elif cli.mode == 'piaa':
        run_piaa(genre=cli.genre, trial=cli.trial, resume=cli.resume)
