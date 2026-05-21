import os
import math
import random
import shutil
from pathlib import Path
from typing import List, Optional

import torch
import torch.multiprocessing as tmp
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, DistributedDataParallelKwargs
from huggingface_hub import HfApi, create_repo

from config import ModelConfig, ParvHFConfig
from hf_model import ParvForCausalLM
from lora import apply_lora, merge_lora, reset_lora, extract_lora_state_dict
from dataset import (
    prepare_phase1_data,
    prepare_phase2_data,
    build_dataloader,
    load_hastings,
    download_bin_shards,
    PHASE2_CURRICULUM,
    StreamingHFDataset,
)

P1_SEQ_LEN = 2048

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
    # ── All I/O in parent process (no NCCL, no GPU) ──
    print("=== Parv: Preparing data (parent process) ===")
    tokenizer = load_hastings(args.hastings_path)
    print(f"  Tokenizer: vocab_size={tokenizer.vocab_size}")

    if args.phase1data_ufw_override:
        print("Phase 1 data override enabled. Will stream openbmb/Ultra-FineWeb asynchronously during training.")
    elif args.bin_repo_p1:
        print(f"Downloading pre-tokenized Phase 1 .bin shards from {args.bin_repo_p1} ...")
        download_bin_shards(args.bin_repo_p1, revision="main", hf_token=args.hf_token, out_name="phase1")
    else:
        # Tokenize from scratch
        prepare_phase1_data(args.data, tokenizer, cache_dir="data")

    if args.bin_repo_p2:
        print(f"Downloading pre-tokenized Phase 2 .bin shards from {args.bin_repo_p2} ...")
        download_bin_shards(args.bin_repo_p2, revision="main", hf_token=args.hf_token, out_name="phase2")
    else:
        ds_exists = os.path.exists("data/phase2.bin")
        if not ds_exists:
            prepare_phase2_data(args.hf_dataset_p2, args.n_samples_p2, tokenizer, cache_dir="data")
        else:
            print(f"  Phase 2 cache found: data/phase2.bin")

    if args.tpu:
        import torch_xla.distributed.xla_multiprocessing as xmp
        n = args.num_tpus or 8
        print(f"Launching TPU processes via XLA (cores={n})...")
        os.environ["WORLD_SIZE"] = str(n)
        xmp.spawn(_spawn_wrapper, args=(args,), nprocs=n, start_method="fork")
    else:
        print("Data ready. Launching GPU processes...")
        print()
        tmp.set_start_method("spawn", force=True)
        n = args.num_gpus or torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = str(n)
        tmp.spawn(_spawn_wrapper, args=(args,), nprocs=n, join=True)


def _spawn_wrapper(local_rank, args):
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(os.environ["WORLD_SIZE"])
    _train_impl(args)


def _train_impl(args):
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key

    ga_plugin = GradientAccumulationPlugin(num_steps=args.grad_accum_p1)
    kwargs_handlers = []
    if not args.tpu:
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_plugin=ga_plugin,
        kwargs_handlers=kwargs_handlers,
    )
    device = accelerator.device
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    p1_bs = args.batch_size_p1
    p1_ga = args.grad_accum_p1
    p2_bs = args.batch_size_p2
    p2_ga = args.grad_accum_p2
    eff_p1 = p1_bs * world_size * p1_ga
    eff_p2 = p2_bs * world_size * p2_ga

    def log(msg):
        if is_main:
            accelerator.print(msg)

    log("=== Parv Training ===")
    log(f"  GPUs: {world_size}")
    log(f"  Phase 1:  batch/GPU={p1_bs}  grad_accum={p1_ga}  effective={eff_p1}")
    log(f"  Phase 2:  batch/GPU={p2_bs}  grad_accum={p2_ga}  effective={eff_p2}")

    # ── tokenizer (cached from parent — instant) ──
    log(f"Loading tokenizer from {args.hastings_path}")
    tokenizer = load_hastings(args.hastings_path)
    log(f"  vocab_size={tokenizer.vocab_size}")

    # ── wandb init ──
    run_name = args.wandb_name or "parv-hastings"
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config={
            "tokenizer": "Hastings",
            "tokenizer_vocab": tokenizer.vocab_size,
            "lora_r": args.lora_r,
            "p1_batch_per_gpu": p1_bs,
            "p1_grad_accum": p1_ga,
            "p1_effective": eff_p1,
            "p2_batch_per_gpu": p2_bs,
            "p2_grad_accum": p2_ga,
            "p2_effective": eff_p2,
            "lr": args.lr,
            "total_tokens_p1": args.total_tokens_p1,
            "total_tokens_p2": args.total_tokens_p2,
            "merge_interval": args.merge_interval,
            "upload_model_interval": args.upload_model_interval,
            "num_gpus": world_size,
            "num_workers": args.num_workers,
            "seq_len_p1": P1_SEQ_LEN,
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

    # ── data (cached from parent — instant) ──
    if not args.phase1data_ufw_override:
        phase1_bin = prepare_phase1_data(args.data, tokenizer, cache_dir="data")
    else:
        phase1_bin = None

    # ── model ──
    log("Creating model...")
    mc = ModelConfig()
    # override vocab from tokenizer
    actual_vocab = tokenizer.vocab_size
    log(f"  vocab_size: {actual_vocab} (from Hastings)")
    hf_config = ParvHFConfig(
        vocab_size=actual_vocab, d_model=mc.d_model, n_layers=mc.n_layers,
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

    # Enable gradient checkpointing only when the user explicitly asks for it.
    if args.gradient_checkpointing:
        log("Enabling gradient checkpointing to save GPU memory...")
        model.gradient_checkpointing_enable()

    # Apply optional compute‑efficiency flags before training starts.
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log("TF32 matmul enabled for faster but slightly less precise training.")
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        log("cudnn.benchmark enabled for kernel auto‑tuning.")
    # Set high‑precision matmul policy for float32 (helps on newer GPUs).
    torch.set_float32_matmul_precision('high')

    # ── Parameters & Optimizer ──
    if args.no_lora:
        log("LoRA disabled. Training full parameters.")
        train_params = [p for p in model.parameters() if p.requires_grad]
    else:
        log(f"LoRA r={args.lora_r} on all linears + embedding...")
        apply_lora(model, r=args.lora_r)
        train_params = [p for n, p in model.named_parameters() if "lora_" in n]
        n_lora = sum(p.numel() for p in train_params)
        log(f"LoRA params: {n_lora:,}")

    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(train_params, lr=args.lr, weight_decay=0.1)
        log("Using 8-bit AdamW optimizer (bitsandbytes).")
    except ImportError:
        optimizer = AdamW(train_params, lr=args.lr, weight_decay=0.1)
        log("bitsandbytes not installed, falling back to 32-bit AdamW optimizer.")

    # ── dataloader (phase 1) ──
    if args.phase1data_ufw_override:
        log("Overriding Phase 1 data with streaming openbmb/Ultra-FineWeb...")
        from torch.utils.data import DataLoader
        train_ds = StreamingHFDataset(
            dataset_name="openbmb/Ultra-FineWeb",
            tokenizer=tokenizer,
            seq_len=P1_SEQ_LEN,
            split="train",
            text_field="text",
            is_val=False,
            val_size=1000
        )
        val_ds = StreamingHFDataset(
            dataset_name="openbmb/Ultra-FineWeb",
            tokenizer=tokenizer,
            seq_len=P1_SEQ_LEN,
            split="train",
            text_field="text",
            is_val=True,
            val_size=1000
        )
        dataloader = DataLoader(train_ds, batch_size=p1_bs, pin_memory=True, num_workers=0)
        val_dataloader_p1 = DataLoader(val_ds, batch_size=p1_bs, pin_memory=True, num_workers=0)
    else:
        dataloader = build_dataloader(
            phase1_bin, seq_len=P1_SEQ_LEN, batch_size=p1_bs, stride=256,
            num_workers=args.num_workers, limit_range=(0.0, 0.99)
        )
        val_dataloader_p1 = build_dataloader(
            phase1_bin, seq_len=P1_SEQ_LEN, batch_size=p1_bs, stride=256,
            num_workers=args.num_workers, limit_range=(0.99, 1.0)
        )

    p1_tok_step = p1_bs * world_size * P1_SEQ_LEN
    total_steps_p1 = math.ceil(args.total_tokens_p1 / (p1_tok_step * p1_ga))
    # avg seq_len across curriculum: 4096×0.5 + 8192×0.3125 + 16384×0.125 + 32768×0.0625 = 8704
    avg_seq_p2 = 8704
    p2_tok_step = p2_bs * world_size * avg_seq_p2
    total_steps_p2 = math.ceil(args.total_tokens_p2 / (p2_tok_step * p2_ga))
    total_steps = total_steps_p1 + total_steps_p2
    log(f"Effective tok/step (P1): {p1_tok_step * p1_ga}")
    log(f"Steps — P1: ~{total_steps_p1}  P2: ~{total_steps_p2}  total: ~{total_steps}")

    # Theorem: polynomially decaying learning rate \eta_t = 1/t^\gamma
    gamma = 0.5
    def poly_decay_lr(current_step: int):
        warmup = args.warmup_steps
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        return (float(current_step) / float(max(1, warmup))) ** (-gamma)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_decay_lr)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    val_dataloader_p1 = accelerator.prepare(val_dataloader_p1)

    if args.compile:
        log("Compiling model via torch.compile...")
        model = torch.compile(model)

    # ── HF repos ──
    hf_api = HfApi(token=args.hf_token) if args.hf_token else None
    username = None
    bucket_name = None
    if hf_api:
        if is_main:
            for repo in filter(None, [args.lora_repo, args.model_repo]):
                try:
                    create_repo(repo, private=True, token=args.hf_token, exist_ok=True)
                    log(f"HF repo ready: {repo}")
                except Exception as e:
                    log(f"HF repo warning: {e}")
        try:
            username = hf_api.whoami()["name"]
        except Exception:
            pass



    # ── resume ──
    step = 0
    tokens_seen = 0
    ckpt_dir = Path(args.checkpoint_dir) / "latest"
    
    repo_id = args.lora_repo if (args.lora_repo and not args.no_lora) else args.model_repo
    if repo_id and not args.no_resume:
        try:
            if is_main:
                log(f"Checking HF Repo for existing checkpoint: {repo_id}")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns="latest/*",
                    local_dir=str(ckpt_dir.parent),
                    token=args.hf_token,
                    repo_type="model"
                )
                if (ckpt_dir / "trainer_state.pt").exists():
                    log("Checkpoint downloaded successfully from HF Repo.")
                else:
                    log("No checkpoint found in HF Repo under 'latest/'. Starting from scratch.")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception as e:
            log(f"HF Repo download failed or not found: {e}")

    if not args.no_resume and ckpt_dir.exists():
        log(f"Resuming from {ckpt_dir}")
        accelerator.load_state(str(ckpt_dir))
        state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu")
        step = state["step"]
        tokens_seen = state["tokens_seen"]
        log(f"  step={step}  tokens={tokens_seen:,}")

    pbar = None
    if is_main:
        from tqdm.auto import tqdm
        pbar = tqdm(total=total_steps, initial=step, desc="Training Progress")

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
                                   repo_id=args.lora_repo, token=args.hf_token)
            log("  Tokenizer pushed")

    def push_ckpt():
        """Upload local checkpoint to lora repo under 'latest/' for resume."""
        if hf_api is None or not is_main:
            return
        if not repo_id:
            return
        log(f"  Uploading checkpoint to {repo_id}/latest ...")
        try:
            hf_api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=repo_id,
                path_in_repo="latest",
                repo_type="model",
                token=args.hf_token,
                delete_patterns="*",
            )
            log("  Checkpoint uploaded.")
        except Exception as e:
            log(f"  Checkpoint upload failed: {e}")

    def push_model():
        if hf_api is None or not is_main:
            return
        target = args.model_repo or repo_id
        if not target:
            return
        unwrapped = accelerator.unwrap_model(model)
        d = Path(args.checkpoint_dir) / f"model_{step}"
        d.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(str(d))
        tokenizer.save_pretrained(str(d))
        log(f"  Uploading model to {target}/step-{step} ...")
        hf_api.upload_folder(folder_path=str(d), repo_id=target,
                             path_in_repo=f"step-{step}", token=args.hf_token)
        log(f"  Model pushed (step={step})")
        shutil.rmtree(d)

    def log_metrics(phase, seq_len, loss_val, extra=None):
        if not is_main:
            return
        bs = p1_bs if phase == 1 else p2_bs
        ga = p1_ga if phase == 1 else p2_ga
        raw = bs * world_size * seq_len
        metrics = {
            "loss": loss_val,
            "phase": phase,
            "seq_len": seq_len,
            "tokens": tokens_seen,
            "tok/step": raw,
            "tok/step_eff": raw * ga,
            "step": step,
            "lr": scheduler.get_last_lr()[0],
        }
        if extra:
            metrics.update(extra)
        accelerator.log(metrics, step=step)

    def run_validation(phase):
        val_loader = val_dataloader_p1 if phase == 1 else val_dataloader_p2
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 20:
                    break
                out = model(input_ids=batch["input_ids"], labels=batch["labels"])
                val_loss += out.loss.item()
                val_steps += 1
        avg_val_loss = val_loss / max(1, val_steps)
        if torch.distributed.is_initialized():
            avg_val_loss_t = torch.tensor(avg_val_loss, device=device)
            torch.distributed.all_reduce(avg_val_loss_t, op=torch.distributed.ReduceOp.SUM)
            avg_val_loss = avg_val_loss_t.item() / torch.distributed.get_world_size()
        model.train()
        return avg_val_loss

    def run_inference_example():
        model.eval()
        prompt = "Once upon a time,"
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        unwrapped = accelerator.unwrap_model(model)
        with torch.no_grad():
            gen_ids = unwrapped.model.generate(input_ids, max_new_tokens=48, temperature=0.7, top_k=50)
        gen_tokens = gen_ids[0].tolist()
        gen_text = tokenizer.decode(gen_tokens)
        if is_main:
            log(f"--- Inference Example (step {step}) ---")
            log(gen_text)
            log("-----------------------------------------")
        model.train()
        return gen_text

    def log_expert_utilization():
        unwrapped = accelerator.unwrap_model(model)
        moe_layers = []
        for i, layer in enumerate(unwrapped.model.layers):
            if layer.is_moe:
                moe_layers.append((i, layer.ffn))
        extra_metrics = {}
        for i, moe_layer in moe_layers:
            counts = moe_layer.expert_counts.clone()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            if is_main:
                total_tokens = counts.sum().item()
                if total_tokens > 0:
                    percentages = (counts.float() / total_tokens * 100.0).cpu().tolist()
                    for exp_idx, pct in enumerate(percentages):
                        extra_metrics[f"expert_L{i}/exp_{exp_idx}_pct"] = pct
            moe_layer.expert_counts.zero_()
        return extra_metrics

    def check_lr_override():
        nonlocal scheduler
        new_lr_tensor = torch.tensor([-1.0], device=device)
        if is_main:
            override_file = Path("lr_override.txt")
            if override_file.exists():
                try:
                    content = override_file.read_text().strip()
                    if content:
                        new_lr_tensor[0] = float(content)
                except Exception as e:
                    log(f"Error reading lr_override.txt: {e}")
                finally:
                    try:
                        override_file.unlink()
                    except Exception:
                        pass
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(new_lr_tensor, src=0)
        new_lr = new_lr_tensor[0].item()
        if new_lr > 0.0:
            current_lr = scheduler.get_last_lr()[0]
            if current_lr > 0:
                factor = new_lr / current_lr
                scheduler.base_lrs = [base_lr * factor for base_lr in scheduler.base_lrs]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                log(f"Dynamic LR override applied: {current_lr:.6e} -> {new_lr:.6e} (scaled scheduler by {factor:.4f})")

    # ==================================================================
    # Phase 1
    # ==================================================================
    log("=== Phase 1: Short-Context Pretraining ===")
    model.train()
    data_iter = iter(dataloader)
    best_loss = float("inf")
    tokens_at_start = tokens_seen
    max_tokens_this_run = 150_000_000 if world_size == 1 else 400_000_000
    log(f"Run token limit: {max_tokens_this_run:,} tokens. Resume point: {tokens_at_start:,} tokens.")
    import time
    last_time = time.time()
    last_tokens = tokens_seen

    while tokens_seen < args.total_tokens_p1 and (tokens_seen - tokens_at_start) < max_tokens_this_run:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        check_lr_override()

        with accelerator.accumulate(model):
            out = model(input_ids=batch["input_ids"], labels=batch["labels"],
                        global_step=step, warmup_steps=args.warmup_steps)
            loss_val = out.loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                log(f"WARNING: NaN/Inf loss ({loss_val}) detected at step {step}. Skipping backward and zeroing grads.")
                optimizer.zero_grad()
            else:
                accelerator.backward(out.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                    )
                    has_nan_or_inf = False
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                has_nan_or_inf = True
                                break
                    if has_nan_or_inf:
                        log(f"WARNING: NaN/Inf detected in gradients at step {step}. Skipping optimizer step.")
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                scheduler.step()
                tokens_seen += batch["input_ids"].numel() * world_size * p1_ga
                if pbar is not None:
                    pbar.update(1)

                if step % 20 == 0:
                    loss_val = out.loss.item()
                    best_loss = min(best_loss, loss_val)
                    
                    current_time = time.time()
                    elapsed_seconds = current_time - last_time
                    tokens_delta = tokens_seen - last_tokens
                    tokens_per_sec = tokens_delta / max(1e-6, elapsed_seconds)
                    sec_per_20 = elapsed_seconds
                    
                    log(f"  P1 step={step:>6}  tok={tokens_seen:>10,}  loss={loss_val:.4f}  tok/s={tokens_per_sec:.1f}  sec/20_steps={sec_per_20:.2f}s")
                    log_metrics(1, P1_SEQ_LEN, loss_val)

                    if pbar is not None:
                        pbar.set_postfix({
                            "loss": f"{loss_val:.4f}",
                            "tok/s": f"{tokens_per_sec:.1f}",
                            "sec/20": f"{sec_per_20:.1f}s"
                        })

                    last_time = current_time
                    last_tokens = tokens_seen

                if step % 100 == 0:
                    val_loss = run_validation(1)
                    log(f"  P1 val_loss={val_loss:.4f} (step {step})")
                    extra = log_expert_utilization()
                    extra["val/loss"] = val_loss
                    log_metrics(1, P1_SEQ_LEN, out.loss.item(), extra)

                if step % 250 == 0:
                    gen_text = run_inference_example()
                    if is_main:
                        log_metrics(1, P1_SEQ_LEN, out.loss.item(), {"val/generation": gen_text})

                if step % args.merge_interval == 0:
                    if not args.no_lora:
                        log(f"Merging LoRA (step {step})...")
                        merge_lora(model)
                        push_lora()
                        loss_val = out.loss.item()
                        log_metrics(1, P1_SEQ_LEN, loss_val, {"event": "lora_merge"})
                        reset_lora(model, r=args.lora_r)
                        optimizer.state.clear()
                    save_ckpt()

                if step % args.upload_model_interval == 0:
                    push_ckpt()
                    push_model()
                    loss_val = out.loss.item()
                    log_metrics(1, P1_SEQ_LEN, loss_val, {"event": "model_push"})

    if (tokens_seen - tokens_at_start) >= max_tokens_this_run:
        log(f"Reached run token limit ({max_tokens_this_run:,} tokens). Saving checkpoint and exiting.")
        if pbar is not None:
            pbar.close()
        save_ckpt()
        push_ckpt()
        push_model()
        accelerator.end_training()
        return

    if not args.enable_phase2:
        log("Phase 1 complete/resumed. Phase 2 not enabled (--enable-phase2 not set). Saving checkpoint and exiting.")
        if pbar is not None:
            pbar.close()
        save_ckpt()
        push_ckpt()
        push_model()
        accelerator.end_training()
        return

    # ==================================================================
    # Phase 2  (Ultra-FineWeb curriculum)
    # ==================================================================
    log("=== Phase 2: Long-Context Curriculum ===")
    accelerator.gradient_accumulation_plugin.num_steps = p2_ga
    phase2_bin = prepare_phase2_data(args.hf_dataset_p2, args.n_samples_p2, tokenizer, cache_dir="data")

    p2_curriculum = PHASE2_CURRICULUM
    p2_probs = [s.token_budget / sum(s.token_budget for s in p2_curriculum) for s in p2_curriculum]

    p2_loaders, p2_iters = [], []
    for spec in p2_curriculum:
        loader = build_dataloader(phase2_bin, seq_len=spec.seq_len,
                                  batch_size=p2_bs, stride=512,
                                  num_workers=args.num_workers,
                                  limit_range=(0.0, 0.99))
        loader = accelerator.prepare(loader)
        p2_loaders.append(loader)
        p2_iters.append(iter(loader))

    val_dataloader_p2 = build_dataloader(
        phase2_bin, seq_len=4096, batch_size=p2_bs, stride=512,
        num_workers=args.num_workers, limit_range=(0.99, 1.0)
    )
    val_dataloader_p2 = accelerator.prepare(val_dataloader_p2)

    model.train()
    p2_tokens = 0
    p2_weight_t = torch.tensor(p2_probs)
    while p2_tokens < args.total_tokens_p2 and (tokens_seen - tokens_at_start) < max_tokens_this_run:
        # deterministic across DDP ranks (seeded by optimizer step)
        p2_rng = torch.Generator(device="cpu").manual_seed(42 + step * 1000003)
        idx = torch.multinomial(p2_weight_t, 1, generator=p2_rng).item()
        spec = p2_curriculum[idx]
        try:
            batch = next(p2_iters[idx])
        except StopIteration:
            p2_iters[idx] = iter(p2_loaders[idx])
            batch = next(p2_iters[idx])

        check_lr_override()

        with accelerator.accumulate(model):
            out = model(input_ids=batch["input_ids"], labels=batch["labels"],
                        global_step=step, warmup_steps=args.warmup_steps)
            loss_val = out.loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                log(f"WARNING: NaN/Inf loss ({loss_val}) detected at step {step}. Skipping backward and zeroing grads.")
                optimizer.zero_grad()
            else:
                accelerator.backward(out.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                    )
                    has_nan_or_inf = False
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                has_nan_or_inf = True
                                break
                    if has_nan_or_inf:
                        log(f"WARNING: NaN/Inf detected in gradients at step {step}. Skipping optimizer step.")
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                scheduler.step()
                tok = batch["input_ids"].numel() * world_size * p2_ga
                tokens_seen += tok
                p2_tokens += tok
                if pbar is not None:
                    pbar.update(1)

                if step % 20 == 0:
                    loss_val = out.loss.item()
                    best_loss = min(best_loss, loss_val)
                    
                    current_time = time.time()
                    elapsed_seconds = current_time - last_time
                    tokens_delta = tokens_seen - last_tokens
                    tokens_per_sec = tokens_delta / max(1e-6, elapsed_seconds)
                    sec_per_20 = elapsed_seconds
                    
                    log(f"  P2 step={step:>6}  tok={tokens_seen:>10,}  seq={spec.seq_len}  loss={loss_val:.4f}  tok/s={tokens_per_sec:.1f}  sec/20_steps={sec_per_20:.2f}s")
                    log_metrics(2, spec.seq_len, loss_val)
                    
                    if pbar is not None:
                        pbar.set_postfix({
                            "loss": f"{loss_val:.4f}",
                            "tok/s": f"{tokens_per_sec:.1f}",
                            "sec/20": f"{sec_per_20:.1f}s"
                        })
                    
                    last_time = current_time
                    last_tokens = tokens_seen

                if step % 100 == 0:
                    val_loss = run_validation(2)
                    log(f"  P2 val_loss={val_loss:.4f} (step {step})")
                    extra = log_expert_utilization()
                    extra["val/loss"] = val_loss
                    log_metrics(2, spec.seq_len, out.loss.item(), extra)

                if step % 250 == 0:
                    gen_text = run_inference_example()
                    if is_main:
                        log_metrics(2, spec.seq_len, out.loss.item(), {"val/generation": gen_text})

                if step % args.merge_interval == 0:
                    if not args.no_lora:
                        log(f"Merging LoRA (step {step})...")
                        merge_lora(model)
                        push_lora()
                        loss_val = out.loss.item()
                        log_metrics(2, spec.seq_len, loss_val, {"event": "lora_merge"})
                        reset_lora(model, r=args.lora_r)
                        optimizer.state.clear()
                    save_ckpt()

                if step % args.upload_model_interval == 0:
                    push_ckpt()
                    push_model()
                    loss_val = out.loss.item()
                    log_metrics(2, spec.seq_len, loss_val, {"event": "model_push"})

    if pbar is not None:
        pbar.close()

    if (tokens_seen - tokens_at_start) >= max_tokens_this_run:
        log(f"=== Stopped: Reached run token limit ({max_tokens_this_run:,} tokens) ===")
    else:
        log("=== Training Complete ===")
    save_ckpt()
    push_ckpt()
    push_model()
    final_loss = out.loss.item() if 'out' in dir() else best_loss
    log_metrics(2, 0, final_loss, {"event": "training_complete", "best_loss": best_loss})
    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    parser.add_argument("--hf-dataset-p2", default="openbmb/Ultra-FineWeb")
    parser.add_argument("--n-samples-p2", type=int, default=50_000)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--batch-size-p1", type=int, default=12, help="per GPU, phase 1")
    parser.add_argument("--grad-accum-p1", type=int, default=4)
    parser.add_argument("--batch-size-p2", type=int, default=2, help="per GPU, phase 2")
    parser.add_argument("--grad-accum-p2", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-tokens-p1", type=int, default=3_500_000_000)
    parser.add_argument("--total-tokens-p2", type=int, default=500_000_000)
    parser.add_argument("--merge-interval", type=int, default=100)
    parser.add_argument("--upload-model-interval", type=int, default=2000)
    parser.add_argument("--lora-repo", type=str, default=None)
    parser.add_argument("--model-repo", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--hastings-path", type=str, default="Hastings.pkl")
    parser.add_argument("--wandb-project", type=str, default="ParvLM")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--bin-repo-p1", type=str, default=None,
                        help="HF dataset repo with pre-tokenized Phase 1 .bin shards")
    parser.add_argument("--bin-repo-p2", type=str, default=None,
                        help="HF dataset repo with pre-tokenized Phase 2 .bin shards")
    parser.add_argument("--tpu", action="store_true", help="Train on TPU using torch_xla")
    parser.add_argument("--num-tpus", type=int, default=8, help="Number of TPU cores to spawn")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA and train all parameters")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile to optimize the model graph")
    parser.add_argument("--hf-bucket", type=str, default=None, help="Hugging Face Bucket name to sync checkpoints")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save GPU memory")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--enable-phase2", action="store_true", help="Enable transition to Phase 2 training")
    parser.add_argument("--phase1data-ufw-override", action="store_true", help="Replace Phase 1 data with streaming openbmb/Ultra-FineWeb (~20k tok/s)")
    # extra performance knobs
    parser.add_argument("--use-tf32", action="store_true", help="Allow TF32 matmul on Ampere GPUs for speed (may reduce precision)")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Enable torch.backends.cudnn.benchmark for faster kernels")
    args = parser.parse_args()

    train_cli(args)
