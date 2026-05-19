import io
import os
import math
import json
import pickle
import random
import shutil
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Iterator, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import PreTrainedTokenizerFast


DTYPE = np.uint16


class TiktokenTokenizer:
    def __init__(self, enc, bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"):
        self.enc = enc
        self._bos = bos_token
        self._eos = eos_token
        self._pad = pad_token
        self.vocab_size = enc.n_vocab
        self.bos_token_id = enc.encode_single_token(bos_token) if bos_token in enc.special_tokens_set else 0
        self.eos_token_id = enc.encode_single_token(eos_token) if eos_token in enc.special_tokens_set else 1
        self.pad_token_id = enc.encode_single_token(pad_token) if pad_token in enc.special_tokens_set else 2
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.special_tokens_set = enc.special_tokens_set

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special="all")

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    def save_pretrained(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tokenizer.pkl", "wb") as f:
            pickle.dump(self.enc, f)
        info = {
            "bos_token": self._bos,
            "eos_token": self._eos,
            "pad_token": self._pad,
            "vocab_size": self.vocab_size,
        }
        with open(path / "tokenizer_info.json", "w") as f:
            json.dump(info, f)

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)
        with open(path / "tokenizer.pkl", "rb") as f:
            enc = pickle.load(f)
        info_path = path / "tokenizer_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            return TiktokenTokenizer(enc, info["bos_token"], info["eos_token"], info["pad_token"])
        return TiktokenTokenizer(enc)


def load_hastings(pkl_path: str = "Hastings.pkl") -> TiktokenTokenizer:
    import tiktoken
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    name = data.pop("name")
    enc = tiktoken.core.Encoding(name=name, **data)
    return TiktokenTokenizer(enc)


# ---------------------------------------------------------------------------
# Tokenization pipeline  (saves token IDs as raw binary arrays for zero-copy)
# ---------------------------------------------------------------------------

def tokenize_to_bin(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    dst: str,
    bos: bool = False,
    eos: bool = True,
) -> int:
    ids = tokenizer.encode(text)
    if bos:
        ids = [tokenizer.bos_token_id] + ids
    if eos:
        ids = ids + [tokenizer.eos_token_id]
    arr = np.array(ids, dtype=DTYPE)
    with open(dst, "wb") as f:
        f.write(arr.tobytes())
    return len(arr)


def load_bin(path: str) -> np.ndarray:
    return np.memmap(path, dtype=DTYPE, mode="r")


def bin_n_tokens(path: str) -> int:
    return os.path.getsize(path) // DTYPE().itemsize


def concatenate_bins(src_paths: List[str], dst: str):
    total = sum(bin_n_tokens(p) for p in src_paths)
    out = np.memmap(dst, dtype=DTYPE, mode="w+", shape=(total,))
    offset = 0
    for p in src_paths:
        chunk = np.memmap(p, dtype=DTYPE, mode="r")
        out[offset : offset + len(chunk)] = chunk[:]
        offset += len(chunk)
        del chunk
    out.flush()
    return total


# ---------------------------------------------------------------------------
# TXT ingestion
# ---------------------------------------------------------------------------

def download_text_files(urls: List[str], cache_dir: str = "data/raw") -> List[str]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in urls:
        fname = url.split("/")[-1] or f"file_{len(paths)}.txt"
        dest = cache / fname
        if not dest.exists():
            print(f"Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)
        paths.append(str(dest))
    return paths


def resolve_sources(sources: List[str], cache_dir: str = "data/raw") -> List[str]:
    local = []
    to_dl = []
    for s in sources:
        if s.startswith(("http://", "https://")):
            to_dl.append(s)
        else:
            local.append(s)
    if to_dl:
        local.extend(download_text_files(to_dl, cache_dir))
    return local


def load_text_files(paths: List[str]) -> str:
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "\n\n".join(texts)


def download_and_load_text(sources: List[str], cache_dir: str = "data/raw") -> str:
    paths = resolve_sources(sources, cache_dir)
    return load_text_files(paths)


# ---------------------------------------------------------------------------
# HF dataset ingestion  (Ultra-FineWeb)
# ---------------------------------------------------------------------------

def load_hf_dataset_subset(
    name: str,
    split: str = "train",
    n_samples: Optional[int] = None,
    text_field: str = "text",
) -> str:
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=False)
    if n_samples is not None and n_samples < len(ds):
        indices = random.sample(range(len(ds)), n_samples)
        ds = ds.select(indices)
    texts = [row[text_field] for row in ds]
    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def train_tokenizer(
    corpus_iterator,
    vocab_size: int = 32000,
    save_path: str = "data/tokenizer.json",
):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(corpus_iterator, trainer)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)


def _stream_lines(sources: List[str], cache_dir: str = "data/raw"):
    """Yield lines from all sources one by one (never loads everything)."""
    paths = resolve_sources(sources, cache_dir)
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line


def load_or_train_tokenizer(
    text_paths: List[str],
    vocab_size: int = 32000,
    tokenizer_path: str = "data/tokenizer.json",
) -> PreTrainedTokenizerFast:
    if os.path.exists(tokenizer_path):
        return PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
    print("=" * 60)
    print("  BPE tokenizer training (first run only)")
    print(f"  Corpus: {len(text_paths)} files, ~12 GB")
    print("  This takes ~1 hour — it only happens once.")
    print("=" * 60)
    resolve_sources(text_paths, cache_dir="data/raw")
    train_tokenizer(_stream_lines(text_paths), vocab_size=vocab_size, save_path=tokenizer_path)
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )


# ---------------------------------------------------------------------------
# Prepare tokenized binary  (txt files → one big .bin)
# ---------------------------------------------------------------------------

CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB text chunks


def prepare_phase1_data(
    data_paths: List[str],
    tokenizer: "TiktokenTokenizer",
    cache_dir: str = "data",
) -> str:
    bin_path = Path(cache_dir) / "phase1.bin"
    if bin_path.exists():
        n = bin_n_tokens(str(bin_path))
        print(f"Phase 1 cache found: {bin_path} ({n:,} tokens)")
        return str(bin_path)

    print("Downloading phase 1 data...")
    resolved = resolve_sources(data_paths, cache_dir=os.path.join(cache_dir, "raw"))

    tmp_dir = Path(cache_dir) / "tmp_p1"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def _tokenize_one(idx_and_path):
        """Read & tokenize a file in 32 MB chunks — peak memory ~ one chunk."""
        i, path = idx_and_path
        out = str(tmp_dir / f"shard_{i}.bin")
        total = 0
        eos = tokenizer.eos_token_id
        with open(out, "wb") as bin_f:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    ids = tokenizer.enc.encode(chunk, allowed_special="all")
                    total += len(ids)
                    arr = np.array(ids, dtype=np.uint16)
                    bin_f.write(arr.tobytes())
            # append EOS once per file
            bin_f.write(np.array([eos], dtype=np.uint16).tobytes())
            total += 1
        return out, total, Path(path).name

    n_workers = min(len(resolved), 2)  # 2 max → at most 2 chunks in RAM = 64 MB + overhead
    print(f"  Tokenizing {len(resolved)} files (2 workers, 32 MB chunks)...")
    shard_bins = [None] * len(resolved)
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_map = {pool.submit(_tokenize_one, (i, p)): i for i, p in enumerate(resolved)}
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            shard, n, name = fut.result()
            shard_bins[i] = shard
            total_tokens += n
            print(f"    [{i+1}/{len(resolved)}] {name}: {n:,} tokens")

    print(f"  Total across all files: {total_tokens:,} tokens")
    concatenate_bins([s for s in shard_bins if s], str(bin_path))
    print(f"  Saved to {bin_path}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return str(bin_path)


def prepare_phase2_data(
    hf_dataset: str,
    n_samples: int,
    tokenizer: PreTrainedTokenizerFast,
    cache_dir: str = "data",
    text_field: str = "text",
) -> str:
    bin_path = Path(cache_dir) / "phase2.bin"
    if bin_path.exists():
        n = bin_n_tokens(str(bin_path))
        print(f"Phase 2 cache found: {bin_path} ({n:,} tokens)")
        return str(bin_path)

    print(f"Loading {hf_dataset} ({n_samples} samples)...")
    text = load_hf_dataset_subset(hf_dataset, n_samples=n_samples, text_field=text_field)
    print(f"  Raw text: {len(text):,} chars")

    tmp_dir = Path(cache_dir) / "tmp_p2"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    n_tok = tokenize_to_bin(text, tokenizer, str(tmp_dir / "all.bin"), eos=True)
    total = concatenate_bins([str(tmp_dir / "all.bin")], str(bin_path))
    print(f"  Tokenized: {total:,} tokens → {bin_path}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return str(bin_path)


# ---------------------------------------------------------------------------
# Load pre-tokenized .npy shards from Hugging Face  (downloaded → merged .bin)
# ---------------------------------------------------------------------------

def download_npy_shards(
    repo_id: str,
    revision: str = "phase1",
    cache_dir: str = "data",
    hf_token: Optional[str] = None,
) -> str:
    """Download .npy shards from a HF dataset repo and merge into a single .bin."""
    from huggingface_hub import HfApi, hf_hub_download

    bin_path = Path(cache_dir) / f"{revision}.bin"
    done_flag = Path(cache_dir) / f".{revision}_done"

    if done_flag.exists() and bin_path.exists():
        n = bin_n_tokens(str(bin_path))
        print(f"  {revision} cache found: {bin_path} ({n:,} tokens)")
        return str(bin_path)

    api = HfApi(token=hf_token)
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    npy_files = sorted(f for f in files if f.endswith(".npy"))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {repo_id}@{revision}")

    print(f"  Downloading {len(npy_files)} .npy shards from {repo_id}@{revision} ...")
    tmp_dir = Path(cache_dir) / f"tmp_{revision}_npy"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shard_bins = []

    for i, fname in enumerate(npy_files):
        local = hf_hub_download(
            repo_id, fname, repo_type="dataset",
            revision=revision, token=hf_token,
            cache_dir=str(tmp_dir / "hf_cache"),
        )
        arr = np.load(local)
        shard = tmp_dir / f"shard_{i}.bin"
        arr.tofile(str(shard))
        shard_bins.append(str(shard))
        print(f"    [{i+1}/{len(npy_files)}] {Path(fname).name}  ({len(arr):,} tokens)")

    total = concatenate_bins(shard_bins, str(bin_path))
    print(f"  Merged → {bin_path} ({total:,} tokens)")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    done_flag.touch()
    return str(bin_path)


# ---------------------------------------------------------------------------
# Fast memory-mapped dataset  (zero-copy, multi-worker safe)
# ---------------------------------------------------------------------------

class MemmapDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int, stride: int = 512):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.stride = stride
        item_size = DTYPE().itemsize
        n_total = os.path.getsize(bin_path) // item_size
        self._len = max(0, (n_total - seq_len) // stride) + 1

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.memmap(self.bin_path, dtype=DTYPE, mode="r")
        start = idx * self.stride
        end = start + self.seq_len
        if end > len(data):
            chunk = np.zeros(self.seq_len, dtype=np.int64)
            avail = len(data) - start
            chunk[:avail] = data[start:start+avail].astype(np.int64)
        else:
            chunk = data[start:end].astype(np.int64)
        ids = torch.from_numpy(chunk)
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones(self.seq_len, dtype=torch.long)}


def build_dataloader(
    bin_path: str,
    seq_len: int = 2048,
    batch_size: int = 8,
    stride: int = 512,
    num_workers: int = 4,
) -> DataLoader:
    dataset = MemmapDataset(bin_path, seq_len=seq_len, stride=stride)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Phase 2 curriculum  (mixed context lengths)
# ---------------------------------------------------------------------------

class CurriculumSpec:
    def __init__(self, seq_len: int, token_budget: float):
        self.seq_len = seq_len
        self.token_budget = token_budget


PHASE2_CURRICULUM = [
    CurriculumSpec(4096, 0.50),
    CurriculumSpec(8192, 0.3125),
    CurriculumSpec(16384, 0.125),
    CurriculumSpec(32768, 0.0625),
]


class CurriculumDataloader(IterableDataset):
    def __init__(
        self,
        bin_path: str,
        curriculum: List[CurriculumSpec],
        batch_size: int,
        stride: int = 512,
        num_workers: int = 4,
    ):
        self.bin_path = bin_path
        self.curriculum = curriculum
        self.batch_size = batch_size
        self.stride = stride
        self.num_workers = num_workers

        total_weight = sum(spec.token_budget for spec in curriculum)
        self.probs = [spec.token_budget / total_weight for spec in curriculum]

        self._loaders = None
        self._iters = None

    def _build_loaders(self):
        self._loaders = []
        for spec in self.curriculum:
            loader = build_dataloader(
                self.bin_path, seq_len=spec.seq_len,
                batch_size=self.batch_size, stride=self.stride,
                num_workers=self.num_workers,
            )
            self._loaders.append(loader)
        self._iters = [iter(ld) for ld in self._loaders]

    def __iter__(self):
        self._build_loaders()
        return self

    def __next__(self):
        if self._iters is None:
            self._build_loaders()
        idx = random.choices(range(len(self.curriculum)), weights=self.probs, k=1)[0]
        try:
            batch = next(self._iters[idx])
        except StopIteration:
            self._iters[idx] = iter(self._loaders[idx])
            batch = next(self._iters[idx])
        return batch, self.curriculum[idx].seq_len
