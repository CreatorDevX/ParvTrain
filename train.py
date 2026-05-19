import os
import math
import random
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

import torch
from torch.optim import AdamW
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import GradientAccumulationPlugin
from huggingface_hub import HfApi, create_repo

from config import ModelConfig, ParvHFConfig
from hf_model import ParvForCausalLM
from lora import apply_lora, merge_lora, reset_lora, extract_lora_state_dict
from dataset import (
    prepare_phase1_data,
    prepare_phase2_data,
    build_dataloader,
    load_or_train_tokenizer,
    PHASE2_CURRICULUM,
)

DEFAULT_DATA = [
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


def train_cli(args):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    notebook_launcher(
        _train_impl,
        (args,),
        num_processes=args.num_gpus,
    )


def _train_impl(args):
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key

    ga_plugin = GradientAccumulationPlugin(num_steps=args.grad_accum)
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_plugin=ga_plugin,
    )
    device = accelerator.device
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    eff_bs = args.batch_size * world_size * args.grad_accum

    def log(msg):
        if is_main:
            accelerator.print(msg)

    log("=== Parv Training ===")
    log(f"  GPUs: {world_size}  |  batch/GPU: {args.batch_size}  (total/step: {args.batch_size * world_size})")
    log(f"  grad_accum: {args.grad_accum}  |  effective batch: {eff_bs}")

    # ── wandb init ──
    run_name = args.wandb_name or f"parv-{os.path.splitext(os.path.basename(args.tokenizer_path))[0]}"
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config={
            "lora_r": args.lora_r,
            "batch_per_gpu": args.batch_size,
            "effective_batch": eff_bs,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "total_tokens_p1": args.total_tokens_p1,
            "total_tokens_p2": args.total_tokens_p2,
            "merge_interval": args.merge_interval,
            "upload_model_interval": args.upload_model_interval,
            "num_gpus": world_size,
            "num_workers": args.num_workers,
            "seq_len_p1": 2048,
            "p2_curriculum": [(s.seq_len, s.token_budget) for s in PHASE2_CURRICULUM],
            "model_config": {
                "d_model": ModelConfig().d_model,
                "n_layers": ModelConfig().n_layers,
                "n_heads": ModelConfig().n_heads,
                "n_kv_heads": ModelConfig().n_kv_heads,
                "d_ff": ModelConfig().d_ff,
                "n_routed_experts": ModelConfig().moe.n_routed_experts,
                "top_k": ModelConfig().moe.top_k,
                "n_global_tokens": ModelConfig().n_global_tokens,
            },
        },
        init_kwargs={"wandb": {"name": run_name, "dir": args.checkpoint_dir}} if is_main else None,
    )

    # ── tokenizer ──
    log(f"Tokenizer: {args.tokenizer_path}")
    tokenizer = load_or_train_tokenizer(
        args.data,
        vocab_size=ModelConfig().vocab_size,
        tokenizer_path=args.tokenizer_path,
    )

    # ── pre-tokenize phase 1 → memmap ──
    if is_main:
        prepare_phase1_data(args.data, tokenizer, cache_dir="data")
    if world_size > 1:
        torch.distributed.barrier()
    phase1_bin = prepare_phase1_data(args.data, tokenizer, cache_dir="data")

    # ── model ──
    log("Creating model...")
    mc = ModelConfig()
    hf_config = ParvHFConfig(
        vocab_size=mc.vocab_size, d_model=mc.d_model, n_layers=mc.n_layers,
        n_moe_layers=mc.n_moe_layers, n_dense_layers=mc.n_dense_layers,
        n_heads=mc.n_heads, n_kv_heads=mc.n_kv_heads, d_head=mc.d_head,
        d_ff=mc.d_ff, activation=mc.activation, rope_base=mc.rope_base,
        max_seq_len=mc.max_seq_len, sliding_window=mc.sliding_window,
        n_global_tokens=mc.n_global_tokens, use_flash_attn=mc.use_flash_attn,
        tie_word_embeddings=mc.tie_word_embeddings, use_kv_8bit=mc.use_kv_8bit,
        rope_scaling=mc.rope_scaling.__dict__ if mc.rope_scaling else None,
        moe=mc.moe.__dict__ if mc.moe else None,
    )
    model = ParvForCausalLM(hf_config)
    log(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # ── LoRA ──
    log(f"LoRA r={args.lora_r} on all linears + embedding...")
    apply_lora(model, r=args.lora_r)
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    n_lora = sum(p.numel() for p in lora_params)
    log(f"LoRA params: {n_lora:,}")

    optimizer = AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    # ── dataloader (phase 1) ──
    dataloader = build_dataloader(
        phase1_bin, seq_len=2048, batch_size=args.batch_size, stride=512,
        num_workers=args.num_workers,
    )

    tokens_per_step = args.batch_size * world_size * 2048
    total_steps_p1 = math.ceil(args.total_tokens_p1 / (tokens_per_step * args.grad_accum))
    total_steps_p2 = math.ceil(args.total_tokens_p2 / (tokens_per_step * args.grad_accum))
    total_steps = total_steps_p1 + total_steps_p2
    log(f"Effective tok/step: {tokens_per_step * args.grad_accum}")
    log(f"Steps — P1: ~{total_steps_p1}  P2: ~{total_steps_p2}  total: ~{total_steps}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # ── HF repos ──
    hf_api = HfApi(token=args.hf_token) if args.hf_token else None
    if is_main and hf_api:
        for repo in filter(None, [args.lora_repo, args.model_repo]):
            try:
                create_repo(repo, private=True, token=args.hf_token, exist_ok=True)
                log(f"HF repo ready: {repo}")
            except Exception as e:
                log(f"HF repo warning: {e}")

    # ── resume ──
    step = 0
    tokens_seen = 0
    ckpt_dir = Path(args.checkpoint_dir) / "latest"
    if not args.no_resume and ckpt_dir.exists():
        log(f"Resuming from {ckpt_dir}")
        accelerator.load_state(str(ckpt_dir))
        state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu")
        step = state["step"]
        tokens_seen = state["tokens_seen"]
        log(f"  step={step}  tokens={tokens_seen:,}")

    def save_ckpt():
        if not is_main:
            return
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(ckpt_dir))
        torch.save({"step": step, "tokens_seen": tokens_seen}, ckpt_dir / "trainer_state.pt")

    def push_lora():
        if hf_api is None or not args.lora_repo or not is_main:
            return
        state = extract_lora_state_dict(model)
        p = Path(args.checkpoint_dir) / f"lora_{step}.pt"
        torch.save(state, p)
        hf_api.upload_file(path_or_fileobj=str(p), path_in_repo=f"lora_{step}.pt",
                           repo_id=args.lora_repo, token=args.hf_token)
        log(f"  LoRA pushed (step={step})")
        p.unlink()
        if step == args.merge_interval:
            td = Path(args.checkpoint_dir) / "tok_push"
            td.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(td))
            for fname in os.listdir(str(td)):
                hf_api.upload_file(path_or_fileobj=str(td / fname),
                                   path_in_repo=f"tokenizer/{fname}",
                                   repo_id=args.model_repo, token=args.hf_token)
            log("  Tokenizer pushed")

    def push_model():
        if hf_api is None or not args.model_repo or not is_main:
            return
        unwrapped = accelerator.unwrap_model(model)
        d = Path(args.checkpoint_dir) / f"model_{step}"
        d.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(str(d))
        tokenizer.save_pretrained(str(d))
        hf_api.upload_folder(folder_path=str(d), repo_id=args.model_repo,
                             revision=f"step-{step}", token=args.hf_token)
        log(f"  Model pushed (step={step})")
        shutil.rmtree(d)

    def log_metrics(phase, seq_len, loss_val, extra=None):
        if not is_main:
            return
        metrics = {
            "loss": loss_val,
            "phase": phase,
            "seq_len": seq_len,
            "tokens": tokens_seen,
            "tok/step": tokens_per_step if phase == 1 else (args.batch_size * world_size * seq_len),
            "tok/step_eff": (tokens_per_step if phase == 1 else (args.batch_size * world_size * seq_len)) * args.grad_accum,
            "step": step,
            "lr": scheduler.get_last_lr()[0],
        }
        if extra:
            metrics.update(extra)
        accelerator.log(metrics, step=step)

    # ==================================================================
    # Phase 1
    # ==================================================================
    log("=== Phase 1: Short-Context Pretraining ===")
    model.train()
    data_iter = iter(dataloader)
    best_loss = float("inf")

    while tokens_seen < args.total_tokens_p1:
        try:
            batch = next(data_iter)
        except StopIteration:
            dataloader = build_dataloader(
                phase1_bin, seq_len=2048, batch_size=args.batch_size, stride=512,
                num_workers=args.num_workers,
            )
            dataloader = accelerator.prepare(dataloader)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            out = model(input_ids=batch["input_ids"], labels=batch["labels"])
            accelerator.backward(out.loss)
            accelerator.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                tokens_seen += batch["input_ids"].numel() * world_size
                best_loss = min(best_loss, out.loss.item())

                if step % 20 == 0:
                    log(f"  P1 step={step:>6}  tok={tokens_seen:>10,}  loss={out.loss.item():.4f}")
                    log_metrics(1, 2048, out.loss.item())

                if step % args.merge_interval == 0:
                    log(f"Merging LoRA (step {step})...")
                    merge_lora(model)
                    push_lora()
                    log_metrics(1, 2048, out.loss.item(), {"event": "lora_merge"})
                    reset_lora(model)
                    save_ckpt()

                if step % args.upload_model_interval == 0:
                    push_model()
                    log_metrics(1, 2048, out.loss.item(), {"event": "model_push"})

    # ==================================================================
    # Phase 2  (Ultra-FineWeb curriculum)
    # ==================================================================
    log("=== Phase 2: Long-Context Curriculum ===")

    if is_main:
        prepare_phase2_data(args.hf_dataset_p2, args.n_samples_p2, tokenizer, cache_dir="data")
    if world_size > 1:
        torch.distributed.barrier()
    phase2_bin = prepare_phase2_data(args.hf_dataset_p2, args.n_samples_p2, tokenizer, cache_dir="data")

    p2_curriculum = PHASE2_CURRICULUM
    p2_probs = [s.token_budget / sum(s.token_budget for s in p2_curriculum) for s in p2_curriculum]

    p2_loaders, p2_iters = [], []
    for spec in p2_curriculum:
        loader = build_dataloader(phase2_bin, seq_len=spec.seq_len,
                                  batch_size=args.batch_size, stride=512,
                                  num_workers=args.num_workers)
        loader = accelerator.prepare(loader)
        p2_loaders.append(loader)
        p2_iters.append(iter(loader))

    model.train()
    p2_tokens = 0
    while p2_tokens < args.total_tokens_p2:
        idx = random.choices(range(len(p2_curriculum)), weights=p2_probs, k=1)[0]
        spec = p2_curriculum[idx]
        try:
            batch = next(p2_iters[idx])
        except StopIteration:
            p2_iters[idx] = iter(p2_loaders[idx])
            batch = next(p2_iters[idx])

        with accelerator.accumulate(model):
            out = model(input_ids=batch["input_ids"], labels=batch["labels"])
            accelerator.backward(out.loss)
            accelerator.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                tok = batch["input_ids"].numel() * world_size
                tokens_seen += tok
                p2_tokens += tok
                best_loss = min(best_loss, out.loss.item())

                if step % 20 == 0:
                    log(f"  P2 step={step:>6}  tok={tokens_seen:>10,}  seq={spec.seq_len}  loss={out.loss.item():.4f}")
                    log_metrics(2, spec.seq_len, out.loss.item())

                if step % args.merge_interval == 0:
                    log(f"Merging LoRA (step {step})...")
                    merge_lora(model)
                    push_lora()
                    log_metrics(2, spec.seq_len, out.loss.item(), {"event": "lora_merge"})
                    reset_lora(model)
                    save_ckpt()

                if step % args.upload_model_interval == 0:
                    push_model()
                    log_metrics(2, spec.seq_len, out.loss.item(), {"event": "model_push"})

    log("=== Training Complete ===")
    push_model()
    log_metrics(2, 0, out.loss.item(), {"event": "training_complete", "best_loss": best_loss})
    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    parser.add_argument("--hf-dataset-p2", default="openbmb/Ultra-FineWeb")
    parser.add_argument("--n-samples-p2", type=int, default=50_000)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32, help="per GPU")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--total-tokens-p1", type=int, default=3_000_000_000)
    parser.add_argument("--total-tokens-p2", type=int, default=250_000_000)
    parser.add_argument("--merge-interval", type=int, default=100)
    parser.add_argument("--upload-model-interval", type=int, default=2000)
    parser.add_argument("--lora-repo", type=str, default=None)
    parser.add_argument("--model-repo", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--tokenizer-path", type=str, default="data/tokenizer.json")
    parser.add_argument("--wandb-project", type=str, default="parv")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    train_cli(args)
