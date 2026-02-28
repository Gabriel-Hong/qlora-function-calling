"""
QLoRA fine-tuning script for Qwen2.5-1.5B-Instruct on GEN NX API tool calling.

This is the core training script that:
  - Loads YAML configs (model, LoRA, training)
  - Loads preprocessed train/eval JSONL datasets
  - Sets up the model with 4-bit quantization (standard HF or Unsloth)
  - Trains with SFTTrainer + EarlyStopping
  - Supports checkpoint resumption
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ── Environment setup (must come before any torch / transformers imports) ───
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset

from src.data_utils import load_yaml_config, load_jsonl, get_tokenizer


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen2.5-1.5B-Instruct function calling",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing YAML config files (default: config)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing train.jsonl and eval.jsonl (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Output directory for checkpoints (default: models/checkpoints)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in output-dir",
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        help="Use Unsloth for faster training (requires unsloth package)",
    )
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────
def find_latest_checkpoint(output_dir: Path) -> str | None:
    """Return the path of the most recent checkpoint-* directory, or None."""
    checkpoints = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if checkpoints:
        return str(checkpoints[-1])
    return None


def print_trainable_parameters(model) -> None:
    """Print the number of trainable vs total parameters."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable %          : {pct:.4f}%")


def get_vram_peak_mb() -> float:
    """Return peak VRAM usage in MB (0.0 if CUDA is unavailable)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


# ── Model setup ────────────────────────────────────────────────────────────
def setup_model_unsloth(model_config: dict, lora_config: dict):
    """Load model + tokenizer via Unsloth (faster, fused kernels)."""
    from unsloth import FastLanguageModel

    lora_cfg = lora_config["lora"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["model"]["name"],
        max_seq_length=model_config["model"]["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )

    target_modules = lora_cfg["target_modules"]
    if isinstance(target_modules, str) and target_modules != "all-linear":
        target_modules = [m.strip() for m in target_modules.split(",")]

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=target_modules,
    )

    return model, tokenizer


def setup_model_standard(model_config: dict, lora_config: dict):
    """Load model + tokenizer via HuggingFace + BitsAndBytes + PEFT."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lora_cfg = lora_config["lora"]

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    target_modules = lora_cfg["target_modules"]
    if isinstance(target_modules, str) and target_modules != "all-linear":
        target_modules = [m.strip() for m in target_modules.split(",")]

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=target_modules,
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, peft_config)
    return model


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    config_dir = Path(args.config_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load configs ────────────────────────────────────────────────
    print("=" * 60)
    print("QLoRA Fine-Tuning — Qwen2.5-1.5B-Instruct")
    print("=" * 60)

    model_config = load_yaml_config(str(config_dir / "model_config.yaml"))
    lora_config = load_yaml_config(str(config_dir / "lora_config.yaml"))
    training_config = load_yaml_config(str(config_dir / "training_config.yaml"))

    train_cfg = training_config["training"]

    print(f"\n[Config]  model      : {model_config['model']['name']}")
    print(f"[Config]  LoRA r     : {lora_config['lora']['r']}")
    print(f"[Config]  epochs     : {train_cfg['num_train_epochs']}")
    print(f"[Config]  batch size : {train_cfg['per_device_train_batch_size']}")
    print(f"[Config]  grad accum : {train_cfg['gradient_accumulation_steps']}")
    print(f"[Config]  lr         : {train_cfg['learning_rate']}")
    print(f"[Config]  output dir : {output_dir}")

    # ── 2. Load data ───────────────────────────────────────────────────
    print("\n[Data] Loading datasets...")
    train_records = load_jsonl(str(data_dir / "train.jsonl"))
    eval_records = load_jsonl(str(data_dir / "eval.jsonl"))

    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records)

    print(f"[Data]  Train samples : {len(train_dataset)}")
    print(f"[Data]  Eval samples  : {len(eval_dataset)}")

    # ── 3. Setup model ─────────────────────────────────────────────────
    print("\n[Model] Loading model...")
    if args.use_unsloth:
        print("[Model] Using Unsloth backend")
        model, tokenizer = setup_model_unsloth(model_config, lora_config)
    else:
        print("[Model] Using standard HuggingFace + BitsAndBytes backend")
        model = setup_model_standard(model_config, lora_config)
        tokenizer = None  # will be set up below

    # ── 4. Setup tokenizer ─────────────────────────────────────────────
    if tokenizer is None:
        tokenizer = get_tokenizer(model_config["model"]["name"])
    else:
        # Ensure Unsloth tokenizer has the correct special tokens
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.padding_side = "left"

    print(f"[Tokenizer] eos_token : {tokenizer.eos_token!r}")
    print(f"[Tokenizer] pad_token : {tokenizer.pad_token!r}")

    # ── Print parameter summary ────────────────────────────────────────
    print("\n[Parameters]")
    print_trainable_parameters(model)

    # ── 5. Setup trainer ───────────────────────────────────────────────
    from trl import SFTTrainer
    from transformers import TrainingArguments, EarlyStoppingCallback

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        num_train_epochs=train_cfg["num_train_epochs"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        eval_strategy=train_cfg["eval_strategy"],
        logging_steps=train_cfg["logging_steps"],
        logging_dir=train_cfg["logging_dir"],
        report_to=train_cfg["report_to"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        dataloader_pin_memory=train_cfg["dataloader_pin_memory"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=model_config["model"]["max_seq_length"],
        packing=False,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg["early_stopping_patience"],
                early_stopping_threshold=train_cfg["early_stopping_threshold"],
            ),
        ],
    )

    # ── 6. Resume handling ─────────────────────────────────────────────
    resume_from_checkpoint = None
    if args.resume:
        resume_from_checkpoint = find_latest_checkpoint(output_dir)
        if resume_from_checkpoint:
            print(f"\n[Resume] Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            print("\n[Resume] --resume was set but no checkpoint found. Starting fresh.")

    # ── 7. Train ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_time = time.time()

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    elapsed = time.time() - start_time

    # Save final adapter
    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    print(f"\n[Save] Final adapter saved to: {final_adapter_dir}")

    # ── 8. Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"  Training time     : {int(hours)}h {int(minutes)}m {seconds:.1f}s")

    metrics = train_result.metrics
    final_loss = metrics.get("train_loss", None)
    if final_loss is not None:
        print(f"  Final train loss  : {final_loss:.4f}")

    vram_peak = get_vram_peak_mb()
    if vram_peak > 0:
        print(f"  Peak VRAM usage   : {vram_peak:.1f} MB ({vram_peak / 1024:.2f} GB)")

    print(f"  Output directory  : {output_dir}")
    print(f"  Final adapter     : {final_adapter_dir}")
    print()


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("\n" + "!" * 60)
        print("CUDA OUT OF MEMORY")
        print("!" * 60)
        print()
        print("Your GPU ran out of VRAM during training.")
        print()
        print("Suggestions:")
        print("  1. Use the attention-only LoRA fallback config.")
        print("     In config/lora_config.yaml, replace the 'lora' section with")
        print("     the 'lora_fallback' values (target_modules: [q_proj, v_proj]).")
        print()
        print("  2. Reduce max_seq_length in config/model_config.yaml")
        print("     (e.g., 2048 -> 1024).")
        print()
        print("  3. Reduce per_device_train_batch_size to 1 (if not already).")
        print()
        print("  4. Increase gradient_accumulation_steps to compensate for")
        print("     smaller batch sizes.")
        print()
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"  Peak VRAM before OOM: {vram_peak:.1f} MB ({vram_peak / 1024:.2f} GB)")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Training failed: {exc}")
        raise
