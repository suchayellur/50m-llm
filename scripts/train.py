import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from data.datasets import choose_text_stream, dataset_metadata, token_batches
from scripts.experiment_logging import git_commit, software_versions, utc_now_iso, write_json, write_markdown
from src import HackathonLMConfig, HackathonLMForCausalLM

PARAMETER_LIMIT = 50_000_000


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32, "fp32"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_names():
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def save_checkpoint(path, model, optimizer, scheduler, step, meta):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    shutil.copyfile("src/modeling_hackathon_lm.py", path / "modeling_hackathon_lm.py")
    torch.save(model.state_dict(), path / "model_state.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "meta": meta,
        },
        path / "trainer_state.pt",
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    state_dict = torch.load(path / "model_state.pt", map_location=device)
    model.load_state_dict(state_dict)
    state = torch.load(path / "trainer_state.pt", map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"]), state.get("meta", {})


def main():
    parser = argparse.ArgumentParser(description="Train the hackathon LM from random weights.")
    parser.add_argument("--config", default="configs/debug.json")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--dataset", choices=["tinystories", "fineweb", "local"], default="tinystories")
    parser.add_argument("--dataset-revision", default="", help="Dataset git revision/version. Defaults to main.")
    parser.add_argument("--dataset-mixture", default="", help="Human-readable mixture, for example tinystories:0.8,crypto:0.2")
    parser.add_argument("--local-files", nargs="*", default=[])
    parser.add_argument("--output-dir", default="runs/debug")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype, dtype_name = pick_dtype()
    config = HackathonLMConfig.from_json_file(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tokenizer.vocab_size == config.vocab_size, "Tokenizer vocab size must match model config."

    model = HackathonLMForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= PARAMETER_LIMIT, (
        f"Trainable parameters {total_params:,} exceed the {PARAMETER_LIMIT:,} limit."
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    start_step = 0
    meta = {}
    output_dir = Path(args.output_dir)
    if args.resume_from:
        start_step, meta = load_checkpoint(Path(args.resume_from), model, optimizer, scheduler, device)
        print(f"Resumed from step {start_step}")

    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_name_list = gpu_names()
    gpu_name = gpu_name_list[0] if gpu_name_list else "CPU"
    config.save_pretrained(output_dir)
    print(json.dumps(config.to_dict(), indent=2))
    print(f"Trainable parameters: {total_params:,}")
    print(f"Device: {gpu_name}")
    print(f"Precision: {dtype_name}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ds_meta = dataset_metadata(
        args.dataset,
        revision=args.dataset_revision or None,
        mixture=args.dataset_mixture or None,
        local_files=args.local_files,
    )
    run_log = {
        "status": "running",
        "git_commit": git_commit(),
        "model": {
            "config_path": args.config,
            "config": config.to_dict(),
            "trainable_parameters": total_params,
            "parameter_limit": PARAMETER_LIMIT,
            "random_initialization": True,
            "pretrained_initialization_used": False,
            "weight_tying": bool(config.tie_word_embeddings),
        },
        "data": {
            "dataset": args.dataset,
            "dataset_path": ds_meta["path"],
            "dataset_name": ds_meta["name"],
            "dataset_version": ds_meta["revision"],
            "dataset_mixture": ds_meta["mixture"],
            "local_files": ds_meta["local_files"],
        },
        "training": {
            "seed": args.seed,
            "start_time_utc": utc_now_iso(),
            "end_time_utc": None,
            "duration_seconds": None,
            "total_training_tokens": 0,
            "optimizer": "AdamW",
            "optimizer_settings": {"betas": [0.9, 0.95], "weight_decay": 0.1},
            "learning_rate": args.lr,
            "warmup_steps": args.warmup_steps,
            "steps": args.steps,
            "start_step": start_step,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "sequence_length": config.max_position_embeddings,
            "checkpoints": [],
            "final_loss": None,
            "estimated_flops": None,
        },
        "hardware": {
            "precision": dtype_name,
            "gpu_names": gpu_name_list,
            "gpu_count": len(gpu_name_list),
            "peak_gpu_memory_bytes": None,
        },
        "software": software_versions(),
    }
    write_json(output_dir / "run_log.json", run_log)
    write_markdown(output_dir / "run_report.md", run_log)

    text_iter = choose_text_stream(args.dataset, args.local_files, revision=args.dataset_revision or None)
    batches = token_batches(text_iter, tokenizer, config.max_position_embeddings, args.batch_size)

    model.train()
    t0 = time.time()
    tokens_seen = 0
    final_loss = None
    try:
        for step in trange(start_step, args.steps, initial=start_step, total=args.steps):
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for _ in range(args.grad_accum):
                batch = torch.tensor(next(batches), dtype=torch.long, device=device)
                labels = batch.clone()
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                    loss = model(batch, labels=labels).loss / args.grad_accum
                scaler.scale(loss).backward()
                running_loss += loss.item()
                tokens_seen += batch.numel()
            final_loss = running_loss
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (step + 1) % 10 == 0:
                elapsed = max(time.time() - t0, 1e-9)
                tps = tokens_seen / elapsed
                print(f"step={step + 1} loss={running_loss:.4f} tokens/sec={tps:.1f}")

            if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
                elapsed_hours = (time.time() - t0) / 3600
                # Dense transformer estimate: about 6 parameter ops per token.
                estimated_flops = 6 * total_params * tokens_seen
                meta = {
                    "step": step + 1,
                    "gpu": gpu_name,
                    "gpu_count": len(gpu_name_list),
                    "precision": dtype_name,
                    "tokens_seen": tokens_seen,
                    "elapsed_hours": elapsed_hours,
                    "estimated_flops": estimated_flops,
                }
                checkpoint_path = output_dir / f"checkpoint-{step + 1}"
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, step + 1, meta)
                run_log["training"]["checkpoints"].append(str(checkpoint_path))
                run_log["training"]["total_training_tokens"] = tokens_seen
                run_log["training"]["final_loss"] = final_loss
                run_log["training"]["estimated_flops"] = estimated_flops
                if torch.cuda.is_available():
                    run_log["hardware"]["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
                with (output_dir / "training_meta.json").open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                write_json(output_dir / "run_log.json", run_log)
                write_markdown(output_dir / "run_report.md", run_log)
    except Exception as exc:
        run_log["status"] = "failed"
        run_log["training"]["end_time_utc"] = utc_now_iso()
        run_log["training"]["duration_seconds"] = time.time() - t0
        run_log["training"]["total_training_tokens"] = tokens_seen
        run_log["training"]["final_loss"] = final_loss
        run_log["error"] = repr(exc)
        if torch.cuda.is_available():
            run_log["hardware"]["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        write_json(output_dir / "run_log.json", run_log)
        write_markdown(output_dir / "run_report.md", run_log)
        raise

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    shutil.copyfile("src/modeling_hackathon_lm.py", final_dir / "modeling_hackathon_lm.py")
    tokenizer.save_pretrained(output_dir / "final")
    end_time = time.time()
    run_log["status"] = "completed"
    run_log["training"]["end_time_utc"] = utc_now_iso()
    run_log["training"]["duration_seconds"] = end_time - t0
    run_log["training"]["total_training_tokens"] = tokens_seen
    run_log["training"]["final_loss"] = final_loss
    run_log["training"]["final_checkpoint"] = str(final_dir)
    run_log["training"]["estimated_flops"] = 6 * total_params * tokens_seen
    if torch.cuda.is_available():
        run_log["hardware"]["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    write_json(output_dir / "run_log.json", run_log)
    write_markdown(output_dir / "run_report.md", run_log)
    print(f"Saved final checkpoint to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
