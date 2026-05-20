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
    # ── All I/O in parent process (no NCCL, no GPU) ──
    print("=== Parv: Preparing data (parent process) ===")
    tokenizer = load_hastings(args.hastings_path)
    print(f"  Tokenizer: vocab_size={tokenizer.vocab_size}")

    if args.bin_repo_p1:
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
            "seq_len_p1": 1024,
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
    phase1_bin = prepare_phase1_data(args.data, tokenizer, cache_dir="data")

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
        from Sophia import SophiaG
        optimizer = SophiaG(train_params, lr=args.lr, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1)
        log("Using SophiaG optimizer.")
    except ImportError:
        optimizer = AdamW(train_params, lr=args.lr, weight_decay=0.01)
        log("Sophia not installed, using 32-bit AdamW optimizer.")

    # ── dataloader (phase 1) ──
    dataloader = build_dataloader(
        phase1_bin, seq_len=1024, batch_size=p1_bs, stride=256,
        num_workers=args.num_workers, limit_range=(0.0, 0.99)
    )
    val_dataloader_p1 = build_dataloader(
        phase1_bin, seq_len=1024, batch_size=p1_bs, stride=256,
        num_workers=args.num_workers, limit_range=(0.99, 1.0)
    )

    p1_tok_step = p1_bs * world_size * 1024
    total_steps_p1 = math.ceil(args.total_tokens_p1 / (p1_tok_step * p1_ga))
    # avg seq_len across curriculum: 4096×0.5 + 8192×0.3125 + 16384×0.125 + 32768×0.0625 = 8704
    avg_seq_p2 = 8704
    p2_tok_step = p2_bs * world_size * avg_seq_p2
    total_steps_p2 = math.ceil(args.total_tokens_p2 / (p2_tok_step * p2_ga))
    total_steps = total_steps_p1 + total_steps_p2
    log(f"Effective tok/step (P1): {p1_tok_step * p1_ga}")
    log(f"Steps — P1: ~{total_steps_p1}  P2: ~{total_steps_p2}  total: ~{total_steps}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
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

    if args.hf_bucket:
        bucket_name = args.hf_bucket
        if "/" not in bucket_name and username:
            bucket_name = f"{username}/{bucket_name}"
        
        if is_main and hf_api:
            short_bucket_name = bucket_name.split("/")[-1]
            try:
                from huggingface_hub import create_bucket
                create_bucket(short_bucket_name, exist_ok=True, token=args.hf_token)
                log(f"HF Bucket ready: {bucket_name}")
            except Exception as e:
                log(f"HF Bucket warning during creation: {e}")

    # ── resume ──
    step = 0
    tokens_seen = 0
    ckpt_dir = Path(args.checkpoint_dir) / "latest"
    
    if args.hf_bucket and not args.no_resume:
        try:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem(token=args.hf_token)
            remote_latest = f"hf://buckets/{bucket_name}/latest"
            if is_main:
                if fs.exists(remote_latest):
                    log(f"Downloading checkpoint from HF Bucket: {remote_latest}")
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    files = fs.ls(remote_latest, detail=False)
                    for f in files:
                        filename = Path(f).name
                        local_path = ckpt_dir / filename
                        log(f"  Downloading {filename}...")
                        fs.get(f"hf://{f}", str(local_path))
                    log("Checkpoint downloaded successfully.")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception as e:
            log(f"HF Bucket download failed or not found: {e}")

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

        if args.hf_bucket:
            try:
                from huggingface_hub import batch_bucket_files
                files_to_upload = []
                for p in ckpt_dir.iterdir():
                    if p.is_file():
                        files_to_upload.append((str(p), f"latest/{p.name}"))
                if files_to_upload:
                    log(f"Uploading {len(files_to_upload)} files to HF Bucket {bucket_name}...")
                    batch_bucket_files(
                        bucket_name,
                        add=files_to_upload,
                        token=args.hf_token
                    )
                    log("HF Bucket upload complete.")
            except Exception as e:
                log(f"HF Bucket upload failed: {e}")

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
    tokens_seen = 0

    while tokens_seen < args.total_tokens_p1:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        check_lr_override()

        with accelerator.accumulate(model):
            out = model(input_ids=batch["input_ids"], labels=batch["labels"],
                        global_step=step, warmup_steps=args.warmup_steps)
            accelerator.backward(out.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                )
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                scheduler.step()
                tokens_seen += batch["input_ids"].numel() * world_size * p1_ga

                if step % 20 == 0:
                    loss_val = out.loss.item()
                    best_loss = min(best_loss, loss_val)
                    log(f"  P1 step={step:>6}  tok={tokens_seen:>10,}  loss={loss_val:.4f}")
                    log_metrics(1, 1024, loss_val)

                if step % 100 == 0:
                    val_loss = run_validation(1)
                    log(f"  P1 val_loss={val_loss:.4f} (step {step})")
                    extra = log_expert_utilization()
                    extra["val/loss"] = val_loss
                    log_metrics(1, 1024, out.loss.item(), extra)

                if step % 250 == 0:
                    gen_text = run_inference_example()
                    if is_main:
                        log_metrics(1, 1024, out.loss.item(), {"val/generation": gen_text})

                if step % args.merge_interval == 0:
                    if not args.no_lora:
                        log(f"Merging LoRA (step {step})...")
                        merge_lora(model)
                        push_lora()
                        loss_val = out.loss.item()
                        log_metrics(1, 1024, loss_val, {"event": "lora_merge"})
                        reset_lora(model, r=args.lora_r)
                        optimizer.state.clear()
                    save_ckpt()

                if step % args.upload_model_interval == 0:
                    push_model()
                    loss_val = out.loss.item()
                    log_metrics(1, 1024, loss_val, {"event": "model_push"})

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
    while p2_tokens < args.total_tokens_p2:
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
            accelerator.backward(out.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                )
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                scheduler.step()
                tok = batch["input_ids"].numel() * world_size * p2_ga
                tokens_seen += tok
                p2_tokens += tok

                if step % 20 == 0:
                    loss_val = out.loss.item()
                    best_loss = min(best_loss, loss_val)
                    log(f"  P2 step={step:>6}  tok={tokens_seen:>10,}  seq={spec.seq_len}  loss={loss_val:.4f}")
                    log_metrics(2, spec.seq_len, loss_val)

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
                    push_model()
                    loss_val = out.loss.item()
                    log_metrics(2, spec.seq_len, loss_val, {"event": "model_push"})

    log("=== Training Complete ===")
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
    parser.add_argument("--wandb-project", type=str, default="parv")
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
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    train_cli(args)
