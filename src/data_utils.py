"""
Data utilities for QLoRA fine-tuning of Qwen2.5-1.5B-Instruct for GEN NX API tool calling.

Provides helpers for loading configs, reading/writing JSONL datasets,
validating TRL tool-calling format samples, formatting for Qwen2.5 chat
templates, tokenizer setup, and token-length statistics.
"""

import copy
import json
import os
import statistics
from pathlib import Path
from typing import Any

import yaml

# Prevent tokenizers parallelism deadlock on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Parsed configuration as a dictionary.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, one JSON object per line.

    Args:
        path: Filesystem path to the JSONL file.

    Returns:
        List of dictionaries, one per line.
    """
    jsonl_path = Path(path)
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(data: list[dict], path: str) -> None:
    """Save a list of dicts as a JSONL file.

    Args:
        data: List of dictionaries to serialize.
        path: Filesystem path for the output JSONL file.
    """
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_data_format(sample: dict) -> tuple[bool, list[str]]:
    """Validate a single TRL tool-calling format sample.

    Checks performed:
      - Has ``"messages"`` key containing a list.
      - Each message has ``"role"`` and either ``"content"`` or ``"tool_calls"``.
      - For assistant messages with ``tool_calls``: each tool_call has
        ``type="function"``, ``function.name``, and ``function.arguments``
        (must be a JSON string).
      - For tool messages: must have ``"name"`` and ``"content"``.
      - Optionally has ``"tools"`` key (must be a JSON string if present).

    Args:
        sample: A single training sample dictionary.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: list[str] = []

    # --- "messages" key ---
    if "messages" not in sample:
        errors.append("Missing 'messages' key")
        return (False, errors)

    messages = sample["messages"]
    if not isinstance(messages, list):
        errors.append("'messages' must be a list")
        return (False, errors)

    # --- Per-message validation ---
    for idx, msg in enumerate(messages):
        prefix = f"messages[{idx}]"

        if not isinstance(msg, dict):
            errors.append(f"{prefix}: message must be a dict")
            continue

        # role is required
        if "role" not in msg:
            errors.append(f"{prefix}: missing 'role'")
            continue

        role = msg["role"]

        # Must have "content" or "tool_calls"
        if "content" not in msg and "tool_calls" not in msg:
            errors.append(f"{prefix}: must have 'content' or 'tool_calls'")

        # Assistant messages with tool_calls
        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            if not isinstance(tool_calls, list):
                errors.append(f"{prefix}: 'tool_calls' must be a list")
            else:
                for tc_idx, tc in enumerate(tool_calls):
                    tc_prefix = f"{prefix}.tool_calls[{tc_idx}]"
                    if not isinstance(tc, dict):
                        errors.append(f"{tc_prefix}: tool_call must be a dict")
                        continue

                    # type must be "function"
                    if tc.get("type") != "function":
                        errors.append(
                            f"{tc_prefix}: 'type' must be 'function', "
                            f"got {tc.get('type')!r}"
                        )

                    # function block
                    func = tc.get("function")
                    if not isinstance(func, dict):
                        errors.append(f"{tc_prefix}: missing or invalid 'function' dict")
                        continue

                    if "name" not in func:
                        errors.append(f"{tc_prefix}.function: missing 'name'")

                    if "arguments" not in func:
                        errors.append(f"{tc_prefix}.function: missing 'arguments'")
                    else:
                        args = func["arguments"]
                        if not isinstance(args, str):
                            errors.append(
                                f"{tc_prefix}.function: 'arguments' must be a "
                                f"JSON string, got {type(args).__name__}"
                            )
                        else:
                            try:
                                json.loads(args)
                            except json.JSONDecodeError as exc:
                                errors.append(
                                    f"{tc_prefix}.function: 'arguments' is not "
                                    f"valid JSON: {exc}"
                                )

        # Tool messages
        if role == "tool":
            if "name" not in msg:
                errors.append(f"{prefix}: tool message missing 'name'")
            if "content" not in msg:
                errors.append(f"{prefix}: tool message missing 'content'")

    # --- Optional "tools" key ---
    if "tools" in sample:
        tools = sample["tools"]
        if not isinstance(tools, str):
            errors.append("'tools' must be a JSON string")
        else:
            try:
                json.loads(tools)
            except json.JSONDecodeError as exc:
                errors.append(f"'tools' is not valid JSON: {exc}")

    is_valid = len(errors) == 0
    return (is_valid, errors)


def format_tool_calls_for_qwen(sample: dict) -> dict:
    """Ensure the sample is in the correct format for the Qwen2.5 chat template.

    Transformations applied:
      - ``arguments`` inside each ``tool_call`` is converted to a JSON string
        if it is currently a dict.
      - The top-level ``tools`` field is converted to a JSON string if it is
        currently a list.

    Args:
        sample: A single training sample dictionary.

    Returns:
        A deep-copied sample with corrections applied.
    """
    sample = copy.deepcopy(sample)

    # --- Fix tool_calls arguments ---
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                if "arguments" in func and not isinstance(func["arguments"], str):
                    func["arguments"] = json.dumps(
                        func["arguments"], ensure_ascii=False
                    )

    # --- Fix tools field ---
    if "tools" in sample and not isinstance(sample["tools"], str):
        sample["tools"] = json.dumps(sample["tools"], ensure_ascii=False)

    return sample


def get_tokenizer(model_name_or_path: str) -> Any:
    """Load and configure a tokenizer for Qwen2.5 fine-tuning.

    Sets:
      - ``eos_token`` to ``"<|im_end|>"``
      - ``pad_token`` to ``"<|endoftext|>"``
      - ``padding_side`` to ``"left"``

    Args:
        model_name_or_path: HuggingFace model identifier or local path.

    Returns:
        Configured ``PreTrainedTokenizerFast`` instance.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.padding_side = "left"
    return tokenizer


def compute_token_stats(samples: list[dict], tokenizer: Any) -> dict:
    """Compute token-length statistics over a list of chat samples.

    Each sample is rendered with ``tokenizer.apply_chat_template`` using the
    sample's ``messages`` and optional ``tools``.

    Args:
        samples: List of training sample dictionaries.
        tokenizer: A HuggingFace tokenizer with ``apply_chat_template``.

    Returns:
        Dictionary with keys: ``min``, ``max``, ``mean``, ``median``,
        ``p95``, and ``count``.
    """
    lengths: list[int] = []

    for sample in samples:
        tools = None
        if "tools" in sample:
            tools_value = sample["tools"]
            tools = json.loads(tools_value) if isinstance(tools_value, str) else tools_value

        token_ids = tokenizer.apply_chat_template(
            sample["messages"],
            tools=tools,
            tokenize=True,
        )
        lengths.append(len(token_ids))

    if not lengths:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "count": 0,
        }

    sorted_lengths = sorted(lengths)
    p95_index = int(len(sorted_lengths) * 0.95)
    # Clamp to valid index range
    p95_index = min(p95_index, len(sorted_lengths) - 1)

    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(statistics.mean(lengths), 2),
        "median": round(statistics.median(lengths), 2),
        "p95": sorted_lengths[p95_index],
        "count": len(lengths),
    }
