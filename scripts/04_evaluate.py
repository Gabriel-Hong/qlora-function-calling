"""
04_evaluate.py

Evaluates a fine-tuned QLoRA model on the test set.
Computes tool-calling accuracy metrics, latency, and VRAM usage.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.data_utils import get_tokenizer, load_jsonl
from src.eval_metrics import compute_full_evaluation, measure_latency, measure_vram_usage


def load_model(args):
    """Load the model either as a merged model or as base + adapter.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Loaded model instance on GPU with 4-bit quantization.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    if args.is_merged:
        print(f"  Loading merged model from '{args.model_path}' ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print(f"  Loading base model '{args.base_model}' ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print(f"  Loading adapter from '{args.model_path}' ...")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.model_path)

    model.eval()
    return model


def extract_prompt_messages(messages):
    """Extract messages up to and including the first user turn, excluding the
    assistant response and everything after it.

    Args:
        messages: Full list of conversation messages.

    Returns:
        List of messages forming the prompt (system + user turns only).
    """
    prompt_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            break
        prompt_messages.append(msg)
    return prompt_messages


def extract_reference_tool_calls(messages):
    """Extract the reference tool_calls from the first assistant message.

    Args:
        messages: Full list of conversation messages.

    Returns:
        List of tool_call dicts from the assistant turn, or an empty list.
    """
    for msg in messages:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            return msg["tool_calls"]
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on the test set."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned adapter or merged model.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to test data JSONL file (default: data/processed/test.jsonl).",
    )
    parser.add_argument(
        "--tools-schema",
        type=str,
        default="data/samples/gennx_tool_schemas_tier1.json",
        help="Path to tools schema JSON file (default: data/samples/gennx_tool_schemas_tier1.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/eval_results.json",
        help="Output path for evaluation results JSON (default: data/eval/eval_results.json).",
    )
    parser.add_argument(
        "--is-merged",
        action="store_true",
        help="Whether the model is already merged (not an adapter).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1: Load model
    # ------------------------------------------------------------------ #
    print(f"\n[Step 1/6] Loading model ...")
    model = load_model(args)
    print("[Step 1/6] Model loaded successfully.")

    # ------------------------------------------------------------------ #
    # Step 2: Load tokenizer
    # ------------------------------------------------------------------ #
    print(f"\n[Step 2/6] Loading tokenizer ...")
    tokenizer_path = args.model_path if args.is_merged else args.base_model
    tokenizer = get_tokenizer(tokenizer_path)
    print(f"[Step 2/6] Tokenizer loaded from '{tokenizer_path}'.")

    # ------------------------------------------------------------------ #
    # Step 3: Load test data
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3/6] Loading test data from '{args.test_data}' ...")
    test_data = load_jsonl(args.test_data)
    print(f"[Step 3/6] Loaded {len(test_data)} test samples.")

    # ------------------------------------------------------------------ #
    # Step 4: Load tools schema
    # ------------------------------------------------------------------ #
    print(f"\n[Step 4/6] Loading tools schema from '{args.tools_schema}' ...")
    with open(args.tools_schema, "r", encoding="utf-8") as f:
        tools_schema = json.load(f)

    available_tools = []
    for tool in tools_schema:
        if "function" in tool and "name" in tool["function"]:
            available_tools.append(tool["function"]["name"])
        elif "name" in tool:
            available_tools.append(tool["name"])
    print(f"[Step 4/6] Found {len(available_tools)} tool definitions.")

    # ------------------------------------------------------------------ #
    # Step 5: Run inference on test set
    # ------------------------------------------------------------------ #
    print(f"\n[Step 5/6] Running inference on {len(test_data)} test samples ...")
    predictions = []
    references = []
    inference_prompts = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, sample in enumerate(test_data):
        messages = sample["messages"]

        # Extract prompt messages (up to and including user turn)
        prompt_messages = extract_prompt_messages(messages)

        # Extract reference tool calls from assistant turn
        ref_tool_calls = extract_reference_tool_calls(messages)
        references.append(ref_tool_calls)

        # Parse tools from sample or use the loaded schema
        tools = None
        if "tools" in sample:
            tools_value = sample["tools"]
            tools = json.loads(tools_value) if isinstance(tools_value, str) else tools_value
        else:
            tools = tools_schema

        # Apply chat template to format the prompt
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Collect prompts for latency measurement later
        if i < 5:
            inference_prompts.append(prompt_text)

        # Tokenize and generate
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=512,
            )

        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_length:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        predictions.append(prediction)

        if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
            print(f"  Processed {i + 1}/{len(test_data)} samples ...")

    print(f"[Step 5/6] Inference complete.")

    # ------------------------------------------------------------------ #
    # Step 6: Compute metrics and save results
    # ------------------------------------------------------------------ #
    print(f"\n[Step 6/6] Computing evaluation metrics ...")

    # Core metrics
    eval_results = compute_full_evaluation(predictions, references, available_tools)

    # Latency measurement
    print("  Measuring latency ...")
    latency_results = measure_latency(model, tokenizer, inference_prompts, device=device)
    eval_results["latency"] = latency_results

    # VRAM measurement
    print("  Measuring VRAM usage ...")
    vram_results = measure_vram_usage()
    eval_results["vram"] = vram_results

    # Add metadata
    eval_results["metadata"] = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "test_data": args.test_data,
        "num_test_samples": len(test_data),
        "is_merged": args.is_merged,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to '{args.output}'.")

    # ------------------------------------------------------------------ #
    # Print formatted report
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  Model:               {args.model_path}")
    print(f"  Test samples:        {len(test_data)}")
    print(f"  Is merged:           {args.is_merged}")
    print(f"{'='*60}")
    print("  Tool-Calling Metrics:")
    print(f"    Tool name accuracy:  {eval_results['tool_name_accuracy']:.4f}")
    print(f"    Parameter accuracy:  {eval_results['parameter_accuracy']:.4f}")
    print(f"    JSON validity rate:  {eval_results['json_validity_rate']:.4f}")
    print(f"    Hallucination rate:  {eval_results['hallucination_rate']:.4f}")
    print(f"{'='*60}")
    print("  Latency:")
    print(f"    Mean:  {latency_results['mean_ms']:.1f} ms")
    print(f"    P50:   {latency_results['p50_ms']:.1f} ms")
    print(f"    P95:   {latency_results['p95_ms']:.1f} ms")
    print(f"{'='*60}")
    print("  VRAM Usage:")
    print(f"    Used:    {vram_results['used_mb']:.1f} MB")
    print(f"    Total:   {vram_results['total_mb']:.1f} MB")
    print(f"    Percent: {vram_results['percent']:.1f}%")
    print(f"{'='*60}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
