"""
06_serve.py

Serves the fine-tuned model in three modes:
  - cli:         Interactive REPL for tool-calling conversations.
  - gradio:      Web UI via Gradio ChatInterface.
  - ollama-info: Prints deployment instructions for Ollama.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.data_utils import get_tokenizer
from src.eval_metrics import parse_tool_calls_from_output


TOOLS_SCHEMA_PATH = "data/samples/gennx_tool_schemas_tier1.json"
SYSTEM_PROMPT = "You are a structural engineering assistant for GEN NX."


def load_model_and_tokenizer(model_path: str):
    """Load the model with 4-bit quantization and its tokenizer.

    Args:
        model_path: Path to the merged model directory.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading model from '{model_path}' ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    tokenizer = get_tokenizer(model_path)
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def load_tools_schema():
    """Load the GEN NX tool schemas from the default path.

    Returns:
        List of tool schema dicts.
    """
    with open(TOOLS_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_response(model, tokenizer, messages, tools):
    """Generate a response given conversation messages and available tools.

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Loaded tokenizer.
        messages: List of conversation message dicts.
        tools: List of tool schema dicts.

    Returns:
        Decoded response string (new tokens only).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
        )

    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


def format_tool_calls_display(response: str) -> str:
    """Parse and pretty-print any tool calls found in the response.

    Args:
        response: Raw model output string.

    Returns:
        Formatted string with tool calls displayed nicely, or the original
        response if no tool calls are found.
    """
    tool_calls = parse_tool_calls_from_output(response)
    if not tool_calls:
        return response

    lines = []
    lines.append("Tool Call(s) Detected:")
    for i, tc in enumerate(tool_calls, 1):
        name = tc.get("name", tc.get("function", {}).get("name", "unknown"))
        arguments = tc.get("arguments", tc.get("function", {}).get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                pass
        lines.append(f"  [{i}] Function: {name}")
        lines.append(f"      Arguments: {json.dumps(arguments, indent=6, ensure_ascii=False)}")
    return "\n".join(lines)


# ====================================================================== #
# Mode: CLI
# ====================================================================== #


def run_cli(model_path: str):
    """Run the interactive CLI REPL mode.

    Args:
        model_path: Path to the merged model directory.
    """
    model, tokenizer = load_model_and_tokenizer(model_path)
    tools = load_tools_schema()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print(f"\n{'='*60}")
    print("GEN NX Tool Calling Assistant - CLI Mode")
    print(f"{'='*60}")
    print("Type your message and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting ...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Exiting ...")
            break

        messages.append({"role": "user", "content": user_input})

        response = generate_response(model, tokenizer, messages, tools)

        # Display the response with pretty-printed tool calls
        display = format_tool_calls_display(response)
        print(f"\nAssistant: {display}\n")

        messages.append({"role": "assistant", "content": response})


# ====================================================================== #
# Mode: Gradio
# ====================================================================== #


def run_gradio(model_path: str):
    """Run the Gradio web UI mode.

    Args:
        model_path: Path to the merged model directory.
    """
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] Gradio is not installed. Install it with:")
        print("  pip install gradio")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(model_path)
    tools = load_tools_schema()

    def predict(message, history):
        """Generate a response for the Gradio ChatInterface.

        Args:
            message: Current user message string.
            history: List of [user, assistant] message pairs from Gradio.

        Returns:
            Model response string.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Rebuild conversation from history
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": message})

        response = generate_response(model, tokenizer, messages, tools)
        display = format_tool_calls_display(response)
        return display

    demo = gr.ChatInterface(
        predict,
        title="GEN NX Tool Calling Assistant",
        description="Structural engineering assistant with tool-calling capabilities.",
        examples=[
            "What tools do you have available?",
            "Create a node at coordinates (0, 0, 0).",
            "Add a beam element between nodes 1 and 2.",
        ],
    )

    print(f"\n{'='*60}")
    print("Launching Gradio interface ...")
    print(f"{'='*60}\n")

    demo.launch()


# ====================================================================== #
# Mode: Ollama Info
# ====================================================================== #


def run_ollama_info():
    """Print instructions for deploying the model with Ollama."""
    print(f"\n{'='*60}")
    print("Ollama Deployment Instructions")
    print(f"{'='*60}")
    print()
    print("Follow these steps to deploy the fine-tuned model with Ollama:")
    print()
    print("1. First, merge the adapter and export to GGUF format:")
    print("   python scripts/05_merge_adapters.py \\")
    print("       --adapter-path models/qlora-adapter \\")
    print("       --output-dir models/final \\")
    print("       --export-gguf")
    print()
    print("2. Convert to GGUF using llama.cpp:")
    print("   python llama.cpp/convert_hf_to_gguf.py models/final \\")
    print("       --outfile models/final/model.gguf \\")
    print("       --outtype q4_k_m")
    print()
    print("3. Create a Modelfile (save as 'models/final/Modelfile'):")
    print()
    print("   ------- Modelfile -------")
    print('   FROM ./model.gguf')
    print('   TEMPLATE "{{ .System }}\\n{{ .Prompt }}"')
    print('   PARAMETER stop "<|im_end|>"')
    print("   -------------------------")
    print()
    print("4. Create the Ollama model:")
    print("   cd models/final")
    print("   ollama create gennx-tool-calling -f Modelfile")
    print()
    print("5. Run the model:")
    print("   ollama run gennx-tool-calling")
    print()
    print("6. Test with a prompt:")
    print('   ollama run gennx-tool-calling "Create a node at coordinates (1, 2, 3)"')
    print()
    print("For API access, Ollama serves on http://localhost:11434 by default.")
    print("You can use the OpenAI-compatible endpoint:")
    print("   curl http://localhost:11434/v1/chat/completions \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"model\": \"gennx-tool-calling\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'")
    print()
    print(f"{'='*60}")


# ====================================================================== #
# Main
# ====================================================================== #


def main():
    parser = argparse.ArgumentParser(
        description="Serve the fine-tuned model in CLI, Gradio, or Ollama mode."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/final",
        help="Path to the merged model directory (default: models/final).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gradio", "cli", "ollama-info"],
        default="cli",
        help="Serving mode: 'cli', 'gradio', or 'ollama-info' (default: cli).",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli(args.model_path)
    elif args.mode == "gradio":
        run_gradio(args.model_path)
    elif args.mode == "ollama-info":
        run_ollama_info()


if __name__ == "__main__":
    main()
