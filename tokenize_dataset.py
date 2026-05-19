"""
Standalone script: tokenize corpus → .npy shards → upload to HF Datasets.
Zero OOM — tokens stream to disk, never accumulate in RAM.

Usage:
  python tokenize_dataset.py \
    --hastings Hastings.pkl \
    --hf-token hf_... \
    --upload-repo your-user/parv-tokenized \
    --phase2-samples 50000
"""

import os
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from dataset import (
    load_hastings,
    download_text_files,
    load_hf_dataset_to_file,
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

CHUNK_SIZE = 64 * 1024 * 1024      # 64 MB text chunks
SHARD_TOKENS = 500_000_000         # 500M tokens per .npy shard (~1 GB as uint16)


def tokenize_stream_to_bin(text_paths, tokenizer, dst):
    """Tokenize files chunk-by-chunk, append tokens to a .bin file on disk.
    Peak RAM: one 64 MB text chunk + its ~14M encoded tokens (~28 MB)."""
    total = 0
    with open(dst, "wb") as out:
        for p in text_paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    text = f.read(CHUNK_SIZE)
                    if not text:
                        break
                    ids = tokenizer.enc.encode(text, allowed_special="all")
                    ids.append(tokenizer.eos_token_id)
                    total += len(ids)
                    arr = np.array(ids, dtype=DTYPE)
                    out.write(arr.tobytes())
    return total


def split_bin(bin_path, out_dir, name="phase1"):
    """Split a .bin file into SHARD_TOKENS-sized .bin shards via memmap.
    Peak RAM: negligible."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = os.path.getsize(bin_path) // DTYPE().itemsize
    data = np.memmap(bin_path, dtype=DTYPE, mode="r")

    shard_count = (total + SHARD_TOKENS - 1) // SHARD_TOKENS
    for i in range(shard_count):
        start = i * SHARD_TOKENS
        end = min(start + SHARD_TOKENS, total)
        shard = data[start:end]
        shard_path = out_dir / f"{name}_{i:04d}.bin"
        shard.tofile(str(shard_path))
        print(f"  -> {shard_path.name}  ({len(shard):,} tokens)")

    del data
    print(f"  Total: {total:,} tokens across {shard_count} shard(s)")


def tokenize_to_shards(text_paths, tokenizer, out_dir, name="phase1"):
    out_dir = Path(out_dir)
    existing = sorted(out_dir.glob(f"{name}_*.bin"))
    if existing:
        total = sum(os.path.getsize(p) // DTYPE().itemsize for p in existing)
        print(f"  Cache found: {len(existing)} shards, {total:,} tokens")
        return

    tmp_bin = out_dir / f"_{name}_tmp.bin"
    print(f"  Streaming tokens to temp .bin...")
    n = tokenize_stream_to_bin(text_paths, tokenizer, str(tmp_bin))
    print(f"    {n:,} tokens written")
    split_bin(str(tmp_bin), out_dir, name=name)
    tmp_bin.unlink()


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

    print(f"  Done — {repo_id} (rev={revision})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hastings", default="Hastings.pkl", dest="hastings_path")
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
    tokenize_to_shards(local_files, tokenizer, cache / "phase1", name="phase1")

    # ── Phase 2: Ultra-FineWeb subset (streamed to file, never in RAM) ──
    print("\n=== Phase 2: Ultra-FineWeb ===")
    tmp_txt = cache / "_ultra_temp.txt"
    load_hf_dataset_to_file(
        "openbmb/Ultra-FineWeb", dst=str(tmp_txt),
        n_samples=args.phase2_samples,
    )
    tokenize_to_shards([str(tmp_txt)], tokenizer, cache / "phase2", name="phase2")
    tmp_txt.unlink()

    # ── Upload ──
    if not args.skip_upload:
        print("\n=== Uploading ===")
        upload_to_hf(cache / "phase1", args.upload_repo, args.hf_token, revision="phase1")
        upload_to_hf(cache / "phase2", args.upload_repo, args.hf_token, revision="phase2")
