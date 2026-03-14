"""
07_serve_vllm.py

Serves the fine-tuned model via vLLM with OpenAI-compatible API.

Modes:
  - server:  Launch vLLM OpenAI-compatible API server.
  - client:  Interactive CLI client that connects to the running server.
  - test:    Send a single test request to verify the server is working.

Prerequisites:
  - vLLM installed: pip install vllm
  - Merged model at models/final/ (run 05_merge_adapters.py first)
  - NOTE: vLLM only supports Linux. On Windows, use WSL2 or Docker.
    See --help or the bottom of this file for WSL2/Docker instructions.

Usage:
  # Terminal 1: Start the server
  python scripts/07_serve_vllm.py server

  # Terminal 2: Test or interact
  python scripts/07_serve_vllm.py test
  python scripts/07_serve_vllm.py client
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL_PATH = "models/final"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
TOOLS_SCHEMA_PATH = "data/samples/gennx_tool_schemas_tier1.json"
SYSTEM_PROMPT = "You are a structural engineering assistant for GEN NX."


# ====================================================================== #
# Mode: Server
# ====================================================================== #


def run_server(args):
    """Launch the vLLM OpenAI-compatible API server.

    Args:
        args: Parsed CLI arguments.
    """
    model_path = str(Path(args.model_path).resolve())

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--dtype", "float16",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]

    if args.enable_tool_calling:
        cmd.extend([
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ])

    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    print("=" * 60)
    print("Starting vLLM Server")
    print("=" * 60)
    print(f"  Model:        {model_path}")
    print(f"  Host:         {args.host}")
    print(f"  Port:         {args.port}")
    print(f"  Max seq len:  {args.max_model_len}")
    print(f"  Tool calling: {args.enable_tool_calling}")
    print(f"  Quantization: {args.quantization or 'none (fp16)'}")
    print(f"  GPU memory:   {args.gpu_memory_utilization}")
    print("=" * 60)
    print()
    print("Server will be available at:")
    print(f"  http://localhost:{args.port}/v1/chat/completions")
    print(f"  http://localhost:{args.port}/v1/models")
    print()
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except FileNotFoundError:
        print("[ERROR] vLLM is not installed. Install it with:")
        print("  pip install vllm")
        print()
        print("NOTE: vLLM only supports Linux.")
        print("On Windows, use WSL2 or Docker. Run with --help for details.")
        sys.exit(1)


# ====================================================================== #
# Mode: Client
# ====================================================================== #


def get_openai_client(host: str, port: int):
    """Create an OpenAI client pointing to the local vLLM server.

    Args:
        host: Server hostname.
        port: Server port number.

    Returns:
        OpenAI client instance.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai package is not installed. Install it with:")
        print("  pip install openai")
        sys.exit(1)

    base_url = f"http://{host}:{port}/v1"
    return OpenAI(base_url=base_url, api_key="unused")


def load_tools_schema():
    """Load GEN NX tool schemas for tool-calling requests.

    Returns:
        List of tool schema dicts.
    """
    schema_path = Path(TOOLS_SCHEMA_PATH)
    if not schema_path.exists():
        print(f"[WARN] Tool schema not found at {TOOLS_SCHEMA_PATH}")
        print("  Tool calling will be disabled.")
        return None

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_tool_calls(tool_calls) -> str:
    """Format tool calls from the API response for display.

    Args:
        tool_calls: List of tool call objects from the API response.

    Returns:
        Formatted string.
    """
    lines = ["Tool Call(s):"]
    for i, tc in enumerate(tool_calls, 1):
        name = tc.function.name
        try:
            arguments = json.loads(tc.function.arguments)
            args_str = json.dumps(arguments, indent=4, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            args_str = tc.function.arguments
        lines.append(f"  [{i}] {name}")
        lines.append(f"      {args_str}")
    return "\n".join(lines)


def run_client(args):
    """Run interactive CLI client connected to the vLLM server.

    Args:
        args: Parsed CLI arguments.
    """
    client = get_openai_client(args.host, args.port)
    tools = load_tools_schema()

    # Detect the model name from the server
    try:
        models = client.models.list()
        model_name = models.data[0].id if models.data else "default"
    except Exception as e:
        print(f"[ERROR] Cannot connect to server at {args.host}:{args.port}")
        print(f"  {e}")
        print("  Make sure the server is running first:")
        print(f"  python scripts/07_serve_vllm.py server")
        sys.exit(1)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 60)
    print("GEN NX Tool Calling Assistant - vLLM Client")
    print("=" * 60)
    print(f"  Connected to: http://{args.host}:{args.port}")
    print(f"  Model: {model_name}")
    print(f"  Tools: {len(tools) if tools else 0} schemas loaded")
    print("=" * 60)
    print("Type your message and press Enter.")
    print("Type 'quit' or 'exit' to stop. Type 'clear' to reset.")
    print("=" * 60)
    print()

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
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[Conversation cleared]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        request_kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 512,
        }
        if tools:
            request_kwargs["tools"] = tools

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            messages.pop()  # Remove failed user message
            continue

        choice = response.choices[0].message

        # Display response
        if choice.tool_calls:
            display = format_tool_calls(choice.tool_calls)
            print(f"\nAssistant: {display}\n")
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.tool_calls
                ],
            })
        else:
            content = choice.content or "(empty response)"
            print(f"\nAssistant: {content}\n")
            messages.append({"role": "assistant", "content": content})


# ====================================================================== #
# Mode: Test
# ====================================================================== #


def run_test(args):
    """Send a single test request to verify the server works.

    Args:
        args: Parsed CLI arguments.
    """
    client = get_openai_client(args.host, args.port)
    tools = load_tools_schema()

    # Check server connectivity
    print("=" * 60)
    print("Testing vLLM Server Connection")
    print("=" * 60)

    try:
        models = client.models.list()
        model_name = models.data[0].id if models.data else "default"
        print(f"  [OK] Connected to server at {args.host}:{args.port}")
        print(f"  [OK] Model: {model_name}")
    except Exception as e:
        print(f"  [FAIL] Cannot connect: {e}")
        sys.exit(1)

    # Test 1: Simple completion
    print("\n--- Test 1: Simple completion ---")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "안녕하세요"},
            ],
            max_tokens=128,
        )
        print(f"  Response: {response.choices[0].message.content}")
        print("  [OK] Simple completion works")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 2: Tool calling (use only first 2 tools to fit within context)
    if tools:
        print("\n--- Test 2: Tool calling ---")
        test_tools = tools[:2]
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "좌표 (0, 0, 0)에 노드를 생성해줘"},
                ],
                tools=test_tools,
                max_tokens=256,
            )
            choice = response.choices[0].message
            if choice.tool_calls:
                print(f"  {format_tool_calls(choice.tool_calls)}")
                print("  [OK] Tool calling works")
            else:
                print(f"  Response: {choice.content}")
                print("  [WARN] No tool call returned (model responded with text)")
        except Exception as e:
            print(f"  [FAIL] {e}")

    print("\n" + "=" * 60)
    print("Test complete.")
    print("=" * 60)


# ====================================================================== #
# Main
# ====================================================================== #


def main():
    parser = argparse.ArgumentParser(
        description="Serve the fine-tuned model via vLLM with OpenAI-compatible API.",
        epilog=WINDOWS_INSTRUCTIONS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["server", "client", "test"],
        help="'server' to launch vLLM, 'client' for interactive chat, 'test' for quick check.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the merged model directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Server host (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory to use (default: 0.85).",
    )
    parser.add_argument(
        "--enable-tool-calling",
        action="store_true",
        default=True,
        help="Enable automatic tool call parsing (default: True).",
    )
    parser.add_argument(
        "--no-tool-calling",
        action="store_false",
        dest="enable_tool_calling",
        help="Disable automatic tool call parsing.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "squeezellm", None],
        help="Quantization method (default: None, uses fp16 merged model).",
    )
    args = parser.parse_args()

    if args.mode == "server":
        model_dir = Path(args.model_path)
        if not model_dir.exists():
            print(f"[ERROR] Model directory not found: {args.model_path}")
            print("  Run the merge script first:")
            print("  python scripts/05_merge_adapters.py \\")
            print("      --adapter-path models/checkpoints/final_adapter \\")
            print("      --output-dir models/final")
            sys.exit(1)
        run_server(args)
    elif args.mode == "client":
        run_client(args)
    elif args.mode == "test":
        run_test(args)


WINDOWS_INSTRUCTIONS = """
===========================================================================
Windows 사용자 안내 (vLLM은 Linux 전용)
===========================================================================

vLLM은 Linux에서만 동작합니다. Windows에서는 아래 두 가지 방법을 사용하세요.

방법 1: WSL2 (권장)
  # WSL2 Ubuntu 설치 후 터미널에서:
  pip install vllm openai
  cd /mnt/c/MIDAS_Source/qlora-function-calling
  python scripts/07_serve_vllm.py server

  # Windows 쪽에서 client/test 실행:
  python scripts/07_serve_vllm.py test --host localhost
  python scripts/07_serve_vllm.py client --host localhost

방법 2: Docker
  docker run --gpus all -p 8000:8000 \\
      -v ./models/final:/model \\
      vllm/vllm-openai:latest \\
      --model /model \\
      --dtype float16 \\
      --max-model-len 2048 \\
      --enable-auto-tool-choice \\
      --tool-call-parser hermes

  # 그 후 client/test 실행:
  python scripts/07_serve_vllm.py test
  python scripts/07_serve_vllm.py client

===========================================================================
"""


if __name__ == "__main__":
    main()
