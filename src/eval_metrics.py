"""
Evaluation metrics module for QLoRA fine-tuning of Qwen2.5-1.5B-Instruct
for GEN NX API tool calling.

Provides metrics for tool name accuracy, parameter accuracy, JSON validity,
hallucination detection, latency measurement, and VRAM usage tracking.
"""

import json
import re
import statistics
import time
from typing import Any

import torch


def parse_tool_calls_from_output(text: str) -> list[dict]:
    """Parse tool calls from Qwen2.5 output format.

    Qwen2.5 wraps tool calls in <tool_call>...</tool_call> tags.
    Each match is parsed as JSON. Malformed entries are skipped.

    Args:
        text: Raw model output string.

    Returns:
        List of parsed tool call dicts.
    """
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            tool_calls.append(parsed)
        except (json.JSONDecodeError, ValueError):
            continue
    return tool_calls


def tool_name_accuracy(
    predictions: list[list[dict]], references: list[list[dict]]
) -> float:
    """Compare predicted vs reference tool call function names.

    For each sample, checks if the predicted tool names match the reference
    tool names in an order-independent manner.

    Args:
        predictions: List of samples, each a list of tool_call dicts with
            structure {"function": {"name": "...", "arguments": "..."}}.
        references: List of samples with the same structure as predictions.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    if not predictions or not references:
        return 0.0

    num_samples = min(len(predictions), len(references))
    correct = 0

    for i in range(num_samples):
        pred_names = sorted(
            tc.get("function", {}).get("name", "") for tc in predictions[i]
        )
        ref_names = sorted(
            tc.get("function", {}).get("name", "") for tc in references[i]
        )
        if pred_names == ref_names:
            correct += 1

    return correct / num_samples


def parameter_accuracy(
    predictions: list[list[dict]], references: list[list[dict]]
) -> float:
    """Compare predicted vs reference tool call arguments.

    For each sample where tool names match, compares the JSON-parsed arguments.
    Accuracy is the fraction of matching parameter keys and values across all
    samples that have matching tool names.

    Args:
        predictions: List of samples, each a list of tool_call dicts.
        references: List of samples with the same structure.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    if not predictions or not references:
        return 0.0

    num_samples = min(len(predictions), len(references))
    total_params = 0
    matching_params = 0

    for i in range(num_samples):
        pred_calls = predictions[i]
        ref_calls = references[i]

        # Build lookup by function name for order-independent comparison
        pred_by_name: dict[str, list[dict]] = {}
        for tc in pred_calls:
            name = tc.get("function", {}).get("name", "")
            pred_by_name.setdefault(name, []).append(tc)

        ref_by_name: dict[str, list[dict]] = {}
        for tc in ref_calls:
            name = tc.get("function", {}).get("name", "")
            ref_by_name.setdefault(name, []).append(tc)

        # Only compare where tool names match
        common_names = set(pred_by_name.keys()) & set(ref_by_name.keys())

        for name in common_names:
            pred_list = pred_by_name[name]
            ref_list = ref_by_name[name]

            # Compare pairwise up to the shorter list
            for j in range(min(len(pred_list), len(ref_list))):
                pred_args_str = pred_list[j].get("function", {}).get("arguments", "{}")
                ref_args_str = ref_list[j].get("function", {}).get("arguments", "{}")

                try:
                    pred_args = json.loads(pred_args_str) if isinstance(pred_args_str, str) else pred_args_str
                except (json.JSONDecodeError, ValueError):
                    pred_args = {}

                try:
                    ref_args = json.loads(ref_args_str) if isinstance(ref_args_str, str) else ref_args_str
                except (json.JSONDecodeError, ValueError):
                    ref_args = {}

                if not isinstance(pred_args, dict) or not isinstance(ref_args, dict):
                    continue

                all_keys = set(pred_args.keys()) | set(ref_args.keys())
                total_params += len(all_keys)

                for key in all_keys:
                    if key in pred_args and key in ref_args:
                        if pred_args[key] == ref_args[key]:
                            matching_params += 1

    if total_params == 0:
        return 0.0

    return matching_params / total_params


def json_validity_rate(predictions: list[list[dict]]) -> float:
    """Check what fraction of predicted tool calls have valid JSON arguments.

    Args:
        predictions: List of samples, each a list of tool_call dicts.

    Returns:
        Rate as a float between 0 and 1.
    """
    total = 0
    valid = 0

    for sample in predictions:
        for tc in sample:
            total += 1
            args = tc.get("function", {}).get("arguments", "")
            if isinstance(args, dict):
                # Already a parsed dict, counts as valid
                valid += 1
                continue
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    valid += 1
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    if total == 0:
        return 0.0

    return valid / total


def hallucination_rate(
    predictions: list[list[dict]], available_tools: list[str]
) -> float:
    """Check what fraction of predicted tool names are NOT in the available tools list.

    Args:
        predictions: List of samples, each a list of tool_call dicts.
        available_tools: List of valid tool name strings.

    Returns:
        Rate as a float between 0 and 1.
    """
    total = 0
    hallucinated = 0
    available_set = set(available_tools)

    for sample in predictions:
        for tc in sample:
            total += 1
            name = tc.get("function", {}).get("name", "")
            if name not in available_set:
                hallucinated += 1

    if total == 0:
        return 0.0

    return hallucinated / total


def measure_latency(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: str = "cuda",
) -> dict:
    """Measure inference latency across a set of prompts.

    Args:
        model: A HuggingFace model instance.
        tokenizer: The corresponding tokenizer.
        prompts: List of prompt strings to run inference on.
        device: Device string (default "cuda").

    Returns:
        Dict with mean_ms, p50_ms, and p95_ms latency values.
    """
    if not prompts:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

    latencies: list[float] = []

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    p50_idx = max(0, int(n * 0.50) - 1)
    p95_idx = max(0, int(n * 0.95) - 1)

    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": latencies_sorted[p50_idx],
        "p95_ms": latencies_sorted[p95_idx],
    }


def measure_vram_usage() -> dict:
    """Measure current GPU VRAM usage via pynvml.

    pynvml is imported inside this function to avoid import errors
    when a GPU is not available.

    Returns:
        Dict with used_mb, total_mb, and percent.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        used_mb = mem_info.used / (1024 * 1024)
        total_mb = mem_info.total / (1024 * 1024)
        percent = (mem_info.used / mem_info.total) * 100.0 if mem_info.total > 0 else 0.0

        return {
            "used_mb": round(used_mb, 2),
            "total_mb": round(total_mb, 2),
            "percent": round(percent, 2),
        }
    except Exception:
        return {
            "used_mb": 0.0,
            "total_mb": 0.0,
            "percent": 0.0,
        }


def compute_full_evaluation(
    predictions: list[str],
    references: list[list[dict]],
    available_tools: list[str],
) -> dict:
    """Run the full evaluation pipeline.

    Parses tool calls from each prediction string and computes all metrics:
    tool_name_accuracy, parameter_accuracy, json_validity_rate, and
    hallucination_rate.

    Args:
        predictions: List of raw model output strings.
        references: List of samples, each a list of reference tool_call dicts.
        available_tools: List of valid tool name strings.

    Returns:
        Dict containing all metric values.
    """
    parsed_predictions = [
        parse_tool_calls_from_output(pred) for pred in predictions
    ]

    return {
        "tool_name_accuracy": tool_name_accuracy(parsed_predictions, references),
        "parameter_accuracy": parameter_accuracy(parsed_predictions, references),
        "json_validity_rate": json_validity_rate(parsed_predictions),
        "hallucination_rate": hallucination_rate(parsed_predictions, available_tools),
    }
