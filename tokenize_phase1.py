"""
Standalone script: tokenize Phase 1 corpus → .bin shards → upload to HF Datasets.
Zero OOM — tokens stream to disk, never accumulate in RAM.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

from dataset import (
    load_hastings,
    download_text_files,
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
SHARD_TOKENS = 500_000_000         # 500M tokens per .bin shard (~1 GB as uint16)


def tokenize_stream_to_bin(text_paths, tokenizer, dst):
    """Tokenize files chunk-by-chunk, append tokens to a .bin file on disk."""
    total = 0
    file_pbar = tqdm(text_paths, desc="Files", unit="file", position=0)
    with open(dst, "wb") as out:
        for p in file_pbar:
            fname = Path(p).name
            file_pbar.set_postfix_str(fname)
            file_tok = 0
            fsize = os.path.getsize(p)
            chunk_pbar = tqdm(
                total=fsize, desc=f"  {fname}", unit="B", unit_scale=True,
                leave=False, position=1,
            )
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    text = f.read(CHUNK_SIZE)
                    if not text:
                        break
                    ids = tokenizer.enc.encode(text, allowed_special="all")
                    total += len(ids)
                    file_tok += len(ids)
                    arr = np.array(ids, dtype=DTYPE)
                    out.write(arr.tobytes())
                    chunk_pbar.update(len(text.encode("utf-8")))
            chunk_pbar.close()
            # Append EOS token once per file after whole-file stream completes
            eos_arr = np.array([tokenizer.eos_token_id], dtype=DTYPE)
            out.write(eos_arr.tobytes())
            total += 1
            file_tok += 1
            tqdm.write(f"    {fname}: {file_tok:>10,} tokens")
    tqdm.write(f"  Total: {total:,} tokens")
    return total


def split_bin(bin_path, out_dir, name="phase1"):
    """Split a .bin file into SHARD_TOKENS-sized .bin shards via memmap."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = os.path.getsize(bin_path) // DTYPE().itemsize
    data = np.memmap(bin_path, dtype=DTYPE, mode="r")

    shard_count = (total + SHARD_TOKENS - 1) // SHARD_TOKENS
    pbar = tqdm(range(shard_count), desc="Writing shards", unit="shard")
    for i in pbar:
        start = i * SHARD_TOKENS
        end = min(start + SHARD_TOKENS, total)
        shard = data[start:end]
        shard_path = out_dir / f"{name}_{i:04d}.bin"
        shard.tofile(str(shard_path))
        pbar.set_postfix_str(f"{shard_path.name} ({len(shard):,} tok)")

    del shard
    if hasattr(data, "_mmap") and data._mmap is not None:
        data._mmap.close()
    del data
    tqdm.write(f"  Total: {total:,} tokens across {shard_count} shard(s)")


def tokenize_to_shards(text_paths, tokenizer, out_dir, name="phase1"):
    out_dir = Path(out_dir)
    existing = sorted(out_dir.glob(f"{name}_*.bin"))
    if existing:
        total = sum(os.path.getsize(p) // DTYPE().itemsize for p in existing)
        print(f"  Cache found: {len(existing)} shards, {total:,} tokens")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_bin = out_dir / f"_{name}_tmp.bin"
    print(f"  Streaming tokens to temp .bin...")
    n = tokenize_stream_to_bin(text_paths, tokenizer, str(tmp_bin))
    print(f"  Total: {n:,} tokens across {len(text_paths)} files")
    split_bin(str(tmp_bin), out_dir, name=name)
    tmp_bin.unlink()


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
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    tokenizer = load_hastings(args.hastings_path)
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Themelios-11 ──
    print("\n=== Phase 1: Themelios-11 ===")
    print("Downloading source files...")
    local_files = download_text_files(THEMELIOS_URLS, cache_dir=str(cache / "_raw"))
    print(f"  {len(local_files)} files cached")
    tokenize_to_shards(local_files, tokenizer, cache / "phase1", name="phase1")

    # ── Upload ──
    if not args.skip_upload:
        print("\n=== Uploading Phase 1 ===")
        upload_to_hf(cache / "phase1", args.upload_repo, args.hf_token, tokenizer, revision="phase1")
