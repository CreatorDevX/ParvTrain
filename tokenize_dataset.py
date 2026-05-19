"""
Standalone script: tokenize the entire Themelios-11 corpus + Ultra-FineWeb subset
and upload as .npy files to Hugging Face Datasets.

Usage:
  python tokenize_dataset.py \
    --hastings Hastings.pkl \
    --hf-token hf_... \
    --upload-repo your-user/parv-tokenized \
    --phase2-samples 50000
"""

import os
import sys
import math
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset import (
    load_hastings,
    resolve_sources,
    download_text_files,
    load_hf_dataset_subset,
    DTYPE,
)

THEMELIOS_URLS = [
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/Currentaffairs.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/TimeMagazine.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/arxiv.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/arxiv_abstracts.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/books.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/code1.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/dialogs.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/maths.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/stackoverflow.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/stanfordphilosophy.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/webscale.txt",
    "https://huggingface.co/datasets/CreatorDevX/Themelios-11/resolve/main/wikidata5m_text.txt",
]

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB text chunks
NPY_MAX_BYTES = 2 * 1024**3    # 2 GB per .npy shard


def tokenize_to_npy_shards(
    text_paths,
    tokenizer,
    out_dir,
    name="phase1",
    skip_if_exists=False,
):
    """Tokenize files and write to 2 GB .npy shards."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if skip_if_exists and list(out_dir.glob(f"{name}_*.npy")):
        npy_paths = sorted(out_dir.glob(f"{name}_*.npy"))
        total = sum(np.load(p, mmap_mode="r").size for p in npy_paths)
        print(f"  Cache found: {len(npy_paths)} shards, {total:,} tokens")
        return

    print(f"Tokenizing {len(text_paths)} files...")
    shard_idx = 0
    shard_buffer = []
    shard_bytes = 0
    total_tokens = 0

    for p in text_paths:
        fname = Path(p).name
        print(f"  {fname} ...")
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                ids = tokenizer.enc.encode(chunk, allowed_special="all")
                ids.append(tokenizer.eos_token_id)
                shard_buffer.extend(ids)
                shard_bytes += len(ids) * 2  # uint16 bytes
                total_tokens += len(ids)

                if shard_bytes >= NPY_MAX_BYTES:
                    arr = np.array(shard_buffer, dtype=DTYPE)
                    npy_path = out_dir / f"{name}_{shard_idx:04d}.npy"
                    np.save(npy_path, arr)
                    print(f"    -> {npy_path.name}  ({len(arr):,} tokens)")
                    shard_buffer = []
                    shard_bytes = 0
                    shard_idx += 1

    # flush remainder
    if shard_buffer:
        arr = np.array(shard_buffer, dtype=DTYPE)
        npy_path = out_dir / f"{name}_{shard_idx:04d}.npy"
        np.save(npy_path, arr)
        print(f"    -> {npy_path.name}  ({len(arr):,} tokens)")

    print(f"  Total: {total_tokens:,} tokens across {shard_idx + 1} shards")


def upload_to_hf(local_dir, repo_id, hf_token, revision=None):
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=hf_token)
    create_repo(repo_id, repo_type="dataset", private=True, token=hf_token, exist_ok=True)

    kwargs = dict(repo_id=repo_id, token=hf_token, repo_type="dataset")
    if revision:
        kwargs["revision"] = revision

    for fname in sorted(os.listdir(local_dir)):
        local = Path(local_dir) / fname
        if local.is_file():
            print(f"  Uploading {fname} ...")
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=fname,
                **kwargs,
            )

    # also upload tokenizer
    tok_dir = Path(local_dir) / "_tokenizer"
    tok_dir.mkdir(exist_ok=True)
    with open(tok_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer.enc, f)
    with open(tok_dir / "info.json", "w") as f:
        json.dump({"vocab_size": tokenizer.vocab_size}, f)
    for fname in os.listdir(str(tok_dir)):
        api.upload_file(
            path_or_fileobj=str(tok_dir / fname),
            path_in_repo=f"_tokenizer/{fname}",
            **kwargs,
        )

    print(f"Done — uploaded to {repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hastings", default="Hastings.pkl")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--upload-repo", required=True)
    parser.add_argument("--cache-dir", default="data/tokenized_npy")
    parser.add_argument("--phase2-samples", type=int, default=50_000)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    tokenizer = load_hastings(args.hastings_path)
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Themelios-11 ──
    print("\n=== Phase 1: Themelios-11 ===")
    local_files = download_text_files(THEMELIOS_URLS, cache_dir=str(cache / "_raw"))
    tokenize_to_npy_shards(local_files, tokenizer, cache / "phase1", name="phase1")

    # ── Phase 2: Ultra-FineWeb subset ──
    print("\n=== Phase 2: Ultra-FineWeb ===")
    text = load_hf_dataset_subset("openbmb/Ultra-FineWeb", n_samples=args.phase2_samples)
    tmp_path = cache / "_tmp_ultra.txt"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    tokenize_to_npy_shards([str(tmp_path)], tokenizer, cache / "phase2", name="phase2")
    os.remove(tmp_path)

    # ── Upload ──
    if not args.skip_upload:
        print(f"\n=== Uploading to {args.upload_repo} ===")
        upload_to_hf(cache / "phase1", args.upload_repo, args.hf_token, revision="phase1")
        upload_to_hf(cache / "phase2", args.upload_repo, args.hf_token, revision="phase2")
