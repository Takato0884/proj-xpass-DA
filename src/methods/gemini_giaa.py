import os
import json
import time
import base64
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Settings
# ──────────────────────────────────────────────────────────────────────────────
_SAVE_DIR = '/home/hayashi0884/proj-xpass-DA/reports/exp/gemini'
_SAMPLES_DIR_MAP = {
    'art':     '/home/hayashi0884/proj-xpass/data/samples/art',
    'fashion': '/home/hayashi0884/proj-xpass/data/samples/fashion',
    'scenery': '/home/hayashi0884/proj-xpass/data/samples/scenery_image',
}

# Batch APIを利用する場合、最新の3.1 Pro プレビュー版が推奨されます
_MODEL = "gemini-3-flash-preview"
_CHUNK_SIZE = 500
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
# Main Batch Logic
# ──────────────────────────────────────────────────────────────────────────────

def run_sequential(genre: str, n: int = 0):
    """逐次処理モード。n=0 のときは全件処理する。"""
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

    for idx, fname in enumerate(all_files):
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
        print(f"  [{idx + 1}/{len(all_files)}] {fname} → {dist}")

    os.makedirs(_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, f'{genre}_results_sequential.json')
    final_output = {
        "genre": genre,
        "model": _MODEL,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_total_images": len(all_files),
        "per_sample": [
            {"sample_file": f, "pred_dist": per_sample_results.get(f, [0.143]*7)}
            for f in all_files
        ]
    }
    with open(save_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSequential completed! Results saved → {save_path}")


def run_batch(genre: str, n: int = 0):
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY in .env")

    client = genai.Client(api_key=api_key)
    samples_dir = _SAMPLES_DIR_MAP[genre]

    # 1. 画像ファイルの列挙
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    all_files = sorted([f for f in os.listdir(samples_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    if n > 0:
        all_files = all_files[:n]
        print(f"[trial] {len(all_files)} images (batch)")
    else:
        print(f"Total images to process: {len(all_files)} (batch)")
    user_prompt = _make_user_prompt(genre)

    _MIME_MAP = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}

    # チャンク分割
    chunks = [all_files[i:i + _CHUNK_SIZE] for i in range(0, len(all_files), _CHUNK_SIZE)]
    n_chunks = len(chunks)
    print(f"Split into {n_chunks} chunk(s) of up to {_CHUNK_SIZE} images each.")

    per_sample_results = {}
    file_uri_map = {}  # custom_id とファイル名の紐付け用（全チャンク共通）

    for chunk_idx, chunk_files in enumerate(chunks):
        print(f"\n=== Chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_files)} images) ===")

        # 2. 画像をbase64エンコードしてリクエスト作成（OpenAI互換形式）
        requests = []
        print("Preparing batch requests...")
        for idx, fname in enumerate(chunk_files):
            global_idx = chunk_idx * _CHUNK_SIZE + idx
            img_path = os.path.join(samples_dir, fname)
            ext = os.path.splitext(fname)[1].lower()
            mime_type = _MIME_MAP.get(ext, 'image/jpeg')
            with open(img_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')

            custom_id = f"req_{global_idx}_{fname.replace('.', '_')}"
            file_uri_map[custom_id] = fname

            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": _MODEL,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                            {"type": "text", "text": user_prompt}
                        ]}
                    ],
                    "max_tokens": _MAX_TOKENS,
                    "temperature": 0.0
                }
            }
            requests.append(req)
            if (idx + 1) % 50 == 0:
                print(f"  Prepared: {idx + 1}/{len(chunk_files)}")

        # 3. JSONLファイルの作成とアップロード
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_jsonl = f"batch_input_{genre}_chunk{chunk_idx}_{ts}.jsonl"
        with open(input_jsonl, "w") as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")

        print(f"Uploading request file: {input_jsonl}")
        batch_input_file = client.files.upload(file=input_jsonl, config={'mime_type': 'application/jsonl'})

        # 4. バッチジョブの作成
        print(f"Starting batch job with model: {_MODEL}...")
        job = client.batches.create(
            model=f"models/{_MODEL}",
            src=types.BatchJobSource(file_name=batch_input_file.name),
        )

        job_id = job.name
        print(f"Batch Job Created! ID: {job_id}")

        # 5. ステータス監視
        print("Waiting for job completion (polling every 60s)...")
        while True:
            job_status = client.batches.get(name=job_id)
            state = job_status.state
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Current state: {state}")

            if state == "SUCCEEDED":
                print("Job finished successfully!")
                output_uri = job_status.output_file
                break
            elif state in ["FAILED", "CANCELLED"]:
                raise RuntimeError(f"Batch job {state} (chunk {chunk_idx}). Check the Google AI Studio console.")

            time.sleep(60)

        # 6. 結果のダウンロードとパース
        print("Downloading results...")
        output_content = client.files.download(name=output_uri)
        results_lines = output_content.decode('utf-8').splitlines()

        for line in results_lines:
            res_data = json.loads(line)
            custom_id = res_data.get("custom_id")
            fname = file_uri_map.get(custom_id)

            # 応答テキストの抽出（OpenAI互換形式）
            try:
                response_text = res_data["response"]["body"]["choices"][0]["message"]["content"]
                dist = _parse_distribution(response_text)
            except Exception:
                dist = [0.143] * 7

            per_sample_results[fname] = dist

        print(f"Chunk {chunk_idx + 1} done. Collected {len(per_sample_results)} results so far.")

    # 7. 最終的なJSON保存
    os.makedirs(_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(_SAVE_DIR, f'{genre}_results_batch.json')
    
    final_output = {
        "genre": genre,
        "model": _MODEL,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_total_images": len(all_files),
        "per_sample": [
            {"sample_file": f, "pred_dist": per_sample_results.get(f, [0.143]*7)}
            for f in all_files
        ]
    }
    
    with open(save_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\nAll process completed! Results saved → {save_path}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--genre', required=True, choices=['art', 'fashion', 'scenery'])
    parser.add_argument('--mode', required=True, choices=['sequential', 'batch'])
    parser.add_argument('--trial', type=int, default=0, help='Limit to N images (0 = all)')
    args = parser.parse_args()

    if args.mode == 'sequential':
        run_sequential(genre=args.genre, n=args.trial)
    else:
        run_batch(genre=args.genre, n=args.trial)