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
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Gemini zero-shot evaluation (all images)')
    parser.add_argument('--mode',  required=True, choices=['giaa', 'piaa'], help='Evaluation mode')
    parser.add_argument('--genre', required=True, choices=['art', 'fashion', 'scenery'])
    parser.add_argument('--trial', type=int, default=0, help='Limit to N images (0 = all)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results JSON')
    args = parser.parse_args()

    if args.mode == 'giaa':
        run_giaa(genre=args.genre, n=args.trial, resume=args.resume)
    elif args.mode == 'piaa':
        run_piaa(genre=args.genre, n=args.trial, resume=args.resume)
