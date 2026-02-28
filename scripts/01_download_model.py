"""
01_download_model.py

Downloads the Qwen2.5-1.5B-Instruct model from Hugging Face and verifies
it can be loaded in 4-bit quantization (QLoRA-ready).
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def print_vram_usage():
    """Print current VRAM usage for all visible NVIDIA GPUs using pynvml."""
    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlDeviceGetCount,
            nvmlShutdown,
        )

        nvmlInit()
        device_count = nvmlDeviceGetCount()
        print(f"\n{'='*60}")
        print(f"VRAM Usage ({device_count} GPU(s) detected)")
        print(f"{'='*60}")
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            used_mb = mem_info.used / (1024 ** 2)
            total_mb = mem_info.total / (1024 ** 2)
            free_mb = mem_info.free / (1024 ** 2)
            print(
                f"  GPU {i} ({name}): "
                f"{used_mb:.0f} MB used / {total_mb:.0f} MB total "
                f"({free_mb:.0f} MB free)"
            )
        print(f"{'='*60}")
        nvmlShutdown()
    except Exception as e:
        print(f"\n[WARNING] Could not read VRAM usage via pynvml: {e}")


def print_parameter_counts(model):
    """Print total and trainable parameter counts for the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print("Model Parameter Summary")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total_params:>14,}")
    print(f"  Trainable parameters: {trainable_params:>14,}")
    print(
        f"  Trainable %:          {100 * trainable_params / total_params:>13.4f}%"
        if total_params > 0
        else "  Trainable %:                    N/A"
    )
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify the Qwen2.5-1.5B-Instruct model."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model identifier to download (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/base/Qwen2.5-1.5B-Instruct",
        help="Local directory to save the model (default: models/base/Qwen2.5-1.5B-Instruct)",
    )
    args = parser.parse_args()

    model_name = args.model_name
    output_dir = args.output_dir

    # ------------------------------------------------------------------ #
    # Step 1: Download model snapshot from Hugging Face
    # ------------------------------------------------------------------ #
    print(f"\n[Step 1/3] Downloading model '{model_name}' to '{output_dir}' ...")
    snapshot_download(
        repo_id=model_name,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[Step 1/3] Download complete. Files saved to: {output_dir}")

    # ------------------------------------------------------------------ #
    # Step 2: Verify model loads in 4-bit quantization
    # ------------------------------------------------------------------ #
    print(f"\n[Step 2/3] Verifying model loads in 4-bit quantization (QLoRA config) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    print("[Step 2/3] Model and tokenizer loaded successfully.")

    # ------------------------------------------------------------------ #
    # Step 3: Print diagnostics
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3/3] Diagnostics")

    print_vram_usage()
    print_parameter_counts(model)

    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model dtype:          {model.dtype}")
    print(f"Device map:           {model.hf_device_map}")

    print("\nModel download and verification completed successfully.")


if __name__ == "__main__":
    main()
