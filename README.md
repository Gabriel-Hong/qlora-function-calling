# QLoRA Fine-Tuning for GEN NX API Tool Calling

A project for fine-tuning the Qwen2.5-1.5B-Instruct model with QLoRA to learn tool calling for the GEN NX structural engineering API.

## Overview

| Item | Details |
|------|---------|
| Base Model | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Training Method | QLoRA (4-bit Quantization + LoRA) |
| Target API | GEN NX Structural Engineering API — 273 endpoints |
| Training Environment | RTX 3060 Laptop 6GB VRAM, Windows 11, CUDA 12.8, Python 3.11 |
| Training Framework | Hugging Face TRL (SFTTrainer) |

## Project Structure

```
qlora-function-calling/
├── config/                          # YAML configuration files
│   ├── model_config.yaml            #   Model, quantization, token settings
│   ├── lora_config.yaml             #   LoRA rank, target modules settings
│   └── training_config.yaml         #   Learning rate, batch, Early Stopping settings
│
├── scripts/                         # Pipeline scripts (run in order)
│   ├── 01_download_model.py         #   Download model
│   ├── 02_prepare_data.py           #   Data validation and splitting
│   ├── 03_train_qlora.py            #   QLoRA training
│   ├── 04_evaluate.py               #   Evaluation (accuracy, latency)
│   ├── 05_merge_adapters.py         #   Merge LoRA adapters
│   └── 06_serve.py                  #   Serving (CLI / Gradio / Ollama)
│
├── src/                             # Utility modules
│   ├── data_utils.py                #   Data loading, validation, transformation
│   └── eval_metrics.py              #   Evaluation metrics computation
│
├── notebooks/                       # Jupyter notebooks (step-by-step exploration)
│   ├── 01_inference_basics.ipynb    #   Model loading and tool calling verification
│   ├── 02_tokenizer_explore.ipynb   #   Tokenizer and chat template exploration
│   ├── 03_first_finetune.ipynb      #   Small-scale training trial
│   └── 04_eval_comparison.ipynb     #   Pre/post-training performance comparison
│
├── data/
│   ├── samples/                     # Sample data (git-tracked)
│   │   ├── gennx_tool_calling_samples.jsonl   # 10 training samples
│   │   └── gennx_tool_schemas_tier1.json      # 15 Tier-1 tool schemas
│   ├── raw/                         # Raw data (git-ignored)
│   ├── processed/                   # Preprocessed train/eval/test (git-ignored)
│   └── eval/                        # Evaluation results (git-ignored)
│
├── docs/                            # Documentation
│   ├── GETTING_STARTED.md           #   Getting started guide (full pipeline)
│   ├── DATA_FORMAT.md               #   Training data format rules
│   ├── CONFIG_REFERENCE.md          #   Configuration file quick reference
│   ├── CONFIG_DEEP_DIVE.md          #   Configuration file deep dive
│   ├── GEN_NX_API_분석.md           #   GEN NX API 273 endpoint analysis & Tier classification
│   ├── LLM_FineTuning_Plan.md       #   Fine-tuning execution plan
│   ├── LLM_FineTuning_핵심개념.md    #   QLoRA/LoRA core concepts
│   ├── LLM_FineTuning_학습과정_및_지표.md  #  Training process & evaluation metrics
│   ├── LLM_FineTuning_고도화_전략.md  #   Advanced optimization strategies
│   └── LLM_FineTuning_아키텍처_및_인프라.md  # Architecture & infrastructure design
│
├── models/                          # Model storage (git-ignored)
│   ├── base/                        #   Downloaded base model
│   ├── checkpoints/                 #   Training checkpoints
│   └── final/                       #   Merged final model
│
├── logs/                            # TensorBoard logs (git-ignored)
├── requirements.txt                 # Python package dependencies
└── .gitignore
```

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate    # Git Bash

# Install packages
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Step 1: Download model
python scripts/01_download_model.py

# Step 2: Prepare data
python scripts/02_prepare_data.py

# Step 3: QLoRA training
python scripts/03_train_qlora.py

# Step 4: Evaluate
python scripts/04_evaluate.py --model-path models/checkpoints/final_adapter

# Step 5: Merge adapters
python scripts/05_merge_adapters.py --adapter-path models/checkpoints/final_adapter

# Step 6: Serve
python scripts/06_serve.py --mode cli
```

For detailed instructions, see [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Training Data Format

JSONL file following TRL tool-calling format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a structural engineering assistant..."},
    {"role": "user", "content": "Add node 1 at the origin"},
    {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "POST /db/node", "arguments": "{\"Assign\":{\"1\":{\"X\":0,\"Y\":0,\"Z\":0}}}"}}]},
    {"role": "tool", "name": "POST /db/node", "content": "{\"NODE\":{\"1\":{\"X\":0,\"Y\":0,\"Z\":0}}}"},
    {"role": "assistant", "content": "Node 1 has been added at the origin (0, 0, 0)."}
  ],
  "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"POST /db/node\",...}}]"
}
```

For detailed format rules, see [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## GEN NX API Tier Classification

| Tier | Endpoints | Description | Training Data |
|------|-----------|-------------|---------------|
| Tier 1 (Core) | ~127 | Modeling, boundary conditions, loads, analysis | 15–20 per API |
| Tier 2 (Auxiliary) | ~102 | Design, load combinations, dynamic loads | 5–10 per API |
| Tier 3 (Specialized) | ~44 | Moving loads, heat of hydration, etc. | 0–2 per API |
| **Total** | **273** | | **~2,400–3,650** |

## Key Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| Quantization | 4-bit NF4 + double quantization | VRAM savings (~774MB) |
| LoRA rank | 8 (expandable to 16–32) | ~9.23M trainable parameters (0.6%) |
| target_modules | all-linear | All 7 linear layers |
| Learning rate | 2e-4, cosine scheduler | 5% warmup + cosine decay |
| Batch size | 1 x 8 (gradient accumulation) | Effective batch size 8 |
| Precision | fp16 | RTX 3060 hardware acceleration |
| Optimizer | paged_adamw_8bit | 8-bit Adam + VRAM paging |
| Early Stopping | patience 3, threshold 0.01 | Based on eval_loss |

For detailed configuration explanations, see [docs/CONFIG_DEEP_DIVE.md](docs/CONFIG_DEEP_DIVE.md).

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| tool_name_accuracy | Whether the correct tool was called |
| parameter_accuracy | Whether parameters are accurate |
| json_validity_rate | Whether output is valid JSON |
| hallucination_rate | Whether non-existent tools were called |

## Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Full pipeline getting started guide |
| [DATA_FORMAT.md](docs/DATA_FORMAT.md) | Training data format rules |
| [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | Configuration file quick reference |
| [CONFIG_DEEP_DIVE.md](docs/CONFIG_DEEP_DIVE.md) | Configuration file deep dive (quantization, LoRA, learning rate, etc.) |
| [GEN_NX_API_분석.md](docs/GEN_NX_API_분석.md) | GEN NX API 273 endpoint analysis & Tier classification |
| [LLM_FineTuning_Plan.md](docs/LLM_FineTuning_Plan.md) | Fine-tuning execution plan |

## Requirements

- **GPU**: NVIDIA GPU 6GB+ VRAM (RTX 3060 or above)
- **RAM**: 16GB+
- **OS**: Windows 10/11
- **Python**: 3.11
- **CUDA**: 12.x (12.8 recommended)
