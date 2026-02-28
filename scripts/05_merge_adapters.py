"""
05_merge_adapters.py

Merges a LoRA adapter with the base model and saves the result.
Optionally provides instructions for GGUF export and Ollama deployment.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_dir_size_mb(path: str) -> float:
    """Compute the total size of all files in a directory in megabytes.

    Args:
        path: Filesystem path to the directory.

    Returns:
        Total size in MB.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total += os.path.getsize(filepath)
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter with the base model."
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name or path (default: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/final",
        help="Output directory for the merged model (default: models/final).",
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Print instructions for exporting to GGUF format for Ollama.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1: Load base model in float16 (full precision for merge)
    # ------------------------------------------------------------------ #
    print(f"\n[Step 1/4] Loading base model '{args.base_model}' in float16 ...")
    print("  NOTE: Loading on CPU with full precision for clean merge.")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    print("[Step 1/4] Base model loaded.")

    # ------------------------------------------------------------------ #
    # Step 2: Load and apply the LoRA adapter
    # ------------------------------------------------------------------ #
    print(f"\n[Step 2/4] Loading LoRA adapter from '{args.adapter_path}' ...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    print("[Step 2/4] Adapter loaded and applied.")

    # ------------------------------------------------------------------ #
    # Step 3: Merge adapter weights and unload
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3/4] Merging adapter weights into base model ...")
    model = model.merge_and_unload()
    print("[Step 3/4] Merge complete.")

    # ------------------------------------------------------------------ #
    # Step 4: Save merged model and tokenizer
    # ------------------------------------------------------------------ #
    print(f"\n[Step 4/4] Saving merged model to '{args.output_dir}' ...")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(args.output_dir)
    print(f"  Model saved to '{args.output_dir}'.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Tokenizer saved to '{args.output_dir}'.")

    # Print model size on disk
    size_mb = get_dir_size_mb(args.output_dir)
    print(f"\n{'='*60}")
    print("Merged Model Summary")
    print(f"{'='*60}")
    print(f"  Base model:    {args.base_model}")
    print(f"  Adapter:       {args.adapter_path}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Size on disk:  {size_mb:.1f} MB")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # Optional: GGUF export instructions
    # ------------------------------------------------------------------ #
    if args.export_gguf:
        print(f"\n{'='*60}")
        print("GGUF Export Instructions")
        print(f"{'='*60}")
        print()
        print("To convert the merged model to GGUF format for use with Ollama,")
        print("you need llama.cpp's convert script. Follow these steps:")
        print()
        print("1. Clone llama.cpp (if not already done):")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp")
        print()
        print("2. Install Python dependencies:")
        print("   pip install -r requirements/requirements-convert_hf_to_gguf.txt")
        print()
        print("3. Convert the merged model to GGUF:")
        print(f"   python convert_hf_to_gguf.py {args.output_dir} \\")
        print(f"       --outfile {args.output_dir}/model.gguf \\")
        print("       --outtype q4_k_m")
        print()
        print("4. Create an Ollama Modelfile:")
        modelfile_path = os.path.join(args.output_dir, "Modelfile")
        print(f"   Save the following to '{modelfile_path}':")
        print()

        modelfile_content = (
            f'FROM ./model.gguf\n'
            f'TEMPLATE "{{{{ .System }}}}\\n{{{{ .Prompt }}}}"\n'
            f'PARAMETER stop "<|im_end|>"\n'
        )

        print("   ------- Modelfile -------")
        for line in modelfile_content.strip().split("\n"):
            print(f"   {line}")
        print("   -------------------------")
        print()
        print("5. Create and run the Ollama model:")
        print(f"   ollama create gennx-tool-calling -f {modelfile_path}")
        print("   ollama run gennx-tool-calling")
        print()
        print("NOTE: Actual GGUF conversion requires llama.cpp which is out of scope")
        print("for this script. The steps above are provided as a reference.")
        print(f"{'='*60}")

    print("\nMerge completed successfully.")


if __name__ == "__main__":
    main()
