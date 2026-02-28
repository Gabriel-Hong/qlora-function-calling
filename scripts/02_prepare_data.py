"""
02_prepare_data.py

Prepares training data from JSONL files for QLoRA fine-tuning.
Validates, formats, shuffles, and splits data into train/eval/test sets.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random

from src.data_utils import (
    load_jsonl,
    save_jsonl,
    validate_data_format,
    format_tool_calls_for_qwen,
    get_tokenizer,
    compute_token_stats,
)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from JSONL files for QLoRA fine-tuning."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/samples/gennx_tool_calling_samples.jsonl",
        help="Path to input JSONL file (default: data/samples/gennx_tool_calling_samples.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed train/eval/test splits (default: data/processed)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for evaluation (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path for tokenizer (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1: Load raw data
    # ------------------------------------------------------------------ #
    print(f"\n[Step 1/6] Loading data from '{args.input}' ...")
    raw_data = load_jsonl(args.input)
    print(f"[Step 1/6] Loaded {len(raw_data)} samples.")

    # ------------------------------------------------------------------ #
    # Step 2: Validate each sample
    # ------------------------------------------------------------------ #
    print(f"\n[Step 2/6] Validating data format ...")
    valid_samples = []
    invalid_count = 0
    for i, sample in enumerate(raw_data):
        is_valid, errors = validate_data_format(sample)
        if is_valid:
            valid_samples.append(sample)
        else:
            invalid_count += 1
            print(f"  [INVALID] Sample {i}: {errors}")

    print(
        f"[Step 2/6] Validation complete: {len(valid_samples)} valid, "
        f"{invalid_count} invalid (skipped)."
    )

    if not valid_samples:
        print("[ERROR] No valid samples remaining. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 3: Format tool calls for Qwen2.5
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3/6] Formatting tool calls for Qwen2.5 chat template ...")
    formatted_samples = [format_tool_calls_for_qwen(s) for s in valid_samples]
    print(f"[Step 3/6] Formatting complete for {len(formatted_samples)} samples.")

    # ------------------------------------------------------------------ #
    # Step 4: Shuffle and split
    # ------------------------------------------------------------------ #
    print(f"\n[Step 4/6] Shuffling (seed=42) and splitting data ...")
    random.seed(42)
    random.shuffle(formatted_samples)

    total = len(formatted_samples)
    train_end = int(total * args.train_ratio)
    eval_end = train_end + int(total * args.eval_ratio)

    train_data = formatted_samples[:train_end]
    eval_data = formatted_samples[train_end:eval_end]
    test_data = formatted_samples[eval_end:]

    print(
        f"[Step 4/6] Split sizes: "
        f"train={len(train_data)}, eval={len(eval_data)}, test={len(test_data)}"
    )

    # ------------------------------------------------------------------ #
    # Step 5: Save splits
    # ------------------------------------------------------------------ #
    print(f"\n[Step 5/6] Saving splits to '{args.output_dir}' ...")
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.jsonl")
    eval_path = os.path.join(args.output_dir, "eval.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    save_jsonl(train_data, train_path)
    print(f"  Saved {len(train_data)} training samples   -> {train_path}")

    save_jsonl(eval_data, eval_path)
    print(f"  Saved {len(eval_data)} evaluation samples -> {eval_path}")

    save_jsonl(test_data, test_path)
    print(f"  Saved {len(test_data)} test samples       -> {test_path}")

    print(f"[Step 5/6] All splits saved.")

    # ------------------------------------------------------------------ #
    # Step 6: Token statistics
    # ------------------------------------------------------------------ #
    print(f"\n[Step 6/6] Computing token statistics ...")
    tokenizer = get_tokenizer(args.model_name)
    stats = compute_token_stats(formatted_samples, tokenizer)

    print(f"\n{'='*60}")
    print("Token Statistics (full dataset)")
    print(f"{'='*60}")
    print(f"  Sample count: {stats['count']}")
    print(f"  Min tokens:   {stats['min']}")
    print(f"  Max tokens:   {stats['max']}")
    print(f"  Mean tokens:  {stats['mean']}")
    print(f"  Median tokens:{stats['median']}")
    print(f"  P95 tokens:   {stats['p95']}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Data Preparation Summary")
    print(f"{'='*60}")
    print(f"  Input file:      {args.input}")
    print(f"  Total raw:       {len(raw_data)}")
    print(f"  Valid samples:   {len(valid_samples)}")
    print(f"  Invalid (skip):  {invalid_count}")
    print(f"  Train samples:   {len(train_data)} ({args.train_ratio*100:.0f}%)")
    print(f"  Eval samples:    {len(eval_data)} ({args.eval_ratio*100:.0f}%)")
    print(f"  Test samples:    {len(test_data)} ({args.test_ratio*100:.0f}%)")
    print(f"  Output dir:      {args.output_dir}")
    print(f"{'='*60}")

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()
