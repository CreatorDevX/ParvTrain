"""
Standalone script: stream Phase 2 (openbmb/Ultra-FineWeb) → tokenize directly to shards → upload to HF.
Zero temporary files on disk, zero memory spikes.
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

from dataset import (
    load_hastings,
    DTYPE,
)


def stream_and_tokenize_phase2(
    tokenizer,
    out_dir: Path,
    hf_dataset: str = "openbmb/Ultra-FineWeb",
    split: str = "train",
    n_samples: int = 50000,
    shard_size: int = 500_000_000,
    text_field: str = "text",
):
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("phase2_*.bin"))
    if existing:
        total = sum(os.path.getsize(p) // DTYPE().itemsize for p in existing)
        print(f"  Cache found: {len(existing)} shards, {total:,} tokens")
        return

    print(f"  Streaming and tokenizing {hf_dataset} (n_samples={n_samples})...")
    ds = load_dataset(hf_dataset, split=split, streaming=True)

    shard_idx = 0
    current_tokens = 0
    current_file = None
    eos = tokenizer.eos_token_id

    pbar = tqdm(total=n_samples, desc="Processing samples", unit="sample")

    for i, row in enumerate(ds):
        if n_samples is not None and i >= n_samples:
            break

        text = row[text_field]
        ids = tokenizer.enc.encode(text, allowed_special="all") + [eos]
        arr = np.array(ids, dtype=DTYPE)

        if current_file is None:
            shard_path = out_dir / f"phase2_{shard_idx:04d}.bin"
            current_file = open(shard_path, "wb")
            shard_idx += 1

        current_file.write(arr.tobytes())
        current_tokens += len(ids)

        if current_tokens >= shard_size:
            current_file.close()
            current_file = None
            current_tokens = 0

        pbar.update(1)

    pbar.close()
    if current_file is not None:
        current_file.close()

    total_tokens = sum(os.path.getsize(p) // DTYPE().itemsize for p in out_dir.glob("phase2_*.bin"))
    print(f"  Done! Tokenized {total_tokens:,} tokens into {shard_idx} shard(s).")


def upload_to_hf(local_dir, repo_id, hf_token, tokenizer, revision=None):
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=hf_token)
    create_repo(repo_id, repo_type="dataset", private=True, token=hf_token, exist_ok=True)

    kwargs = dict(repo_id=repo_id, token=hf_token, repo_type="dataset")
    if revision:
        kwargs["revision"] = revision

    files = [f for f in sorted(os.listdir(local_dir)) if (Path(local_dir) / f).is_file()]
    for fname in tqdm(files, desc="Uploading", unit="file"):
        local = Path(local_dir) / fname
        api.upload_file(
            path_or_fileobj=str(local), path_in_repo=fname, **kwargs,
        )

    # tokenizer alongside
    tok_dir = Path(local_dir) / "_tokenizer"
    tok_dir.mkdir(exist_ok=True)
    with open(tok_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer.enc, f)
    with open(tok_dir / "info.json", "w") as f:
        json.dump({"vocab_size": tokenizer.vocab_size}, f)
    for fname in os.listdir(str(tok_dir)):
        api.upload_file(
            path_or_fileobj=str(tok_dir / fname),
            path_in_repo=f"_tokenizer/{fname}", **kwargs,
        )

    tqdm.write(f"  Done — {repo_id} (rev={revision})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hastings", default="Hastings.pkl", dest="hastings_path")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--upload-repo", required=True)
    parser.add_argument("--cache-dir", default="data/tokenized_npy")
    parser.add_argument("--phase2-samples", type=int, default=50_000)
    parser.add_argument("--shard-tokens", type=int, default=500_000_000)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    tokenizer = load_hastings(args.hastings_path)
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # ── Phase 2: Ultra-FineWeb ──
    print("\n=== Phase 2: Ultra-FineWeb ===")
    stream_and_tokenize_phase2(
        tokenizer,
        cache / "phase2",
        n_samples=args.phase2_samples,
        shard_size=args.shard_tokens,
    )

    # ── Upload ──
    if not args.skip_upload:
        print("\n=== Uploading Phase 2 ===")
        upload_to_hf(cache / "phase2", args.upload_repo, args.hf_token, tokenizer, revision="phase2")
