# QLoRA Fine-Tuning 시작 가이드

> Qwen2.5-1.5B-Instruct 모델을 QLoRA로 fine-tuning하여
> GEN NX API tool calling을 학습시키는 전체 과정을 설명합니다.

---

## 목차

1. [사전 준비](#1-사전-준비)
2. [환경 설정](#2-환경-설정)
3. [Step 1: 모델 다운로드](#3-step-1-모델-다운로드)
4. [Step 2: 데이터 준비](#4-step-2-데이터-준비)
5. [Step 3: QLoRA 학습](#5-step-3-qlora-학습)
6. [Step 4: 평가](#6-step-4-평가)
7. [Step 5: 어댑터 병합](#7-step-5-어댑터-병합)
8. [Step 6: 서빙 (사용하기)](#8-step-6-서빙-사용하기)
9. [문제 해결 (Troubleshooting)](#9-문제-해결-troubleshooting)

---

## 1. 사전 준비

### 하드웨어 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | NVIDIA GPU 6GB VRAM | RTX 3060 이상 |
| RAM | 16GB | 32GB |
| 디스크 | 10GB 여유 공간 | 20GB 이상 |

### 소프트웨어 요구사항

| 항목 | 버전 |
|------|------|
| OS | Windows 10/11 |
| Python | 3.11 |
| CUDA | 12.x (12.8 권장) |
| Git | 최신 |

### CUDA 설치 확인

```bash
# CUDA 버전 확인
nvidia-smi
```

출력에서 `CUDA Version: 12.x`가 보이면 준비 완료입니다.

---

## 2. 환경 설정

### 2-1. 가상환경 생성

```bash
# 프로젝트 디렉토리로 이동
cd C:\MIDAS_Source\qlora-function-calling

# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 가상환경 활성화 (Windows CMD)
.\venv\Scripts\activate.bat

# 가상환경 활성화 (Git Bash)
source venv/Scripts/activate
```

### 2-2. 패키지 설치

```bash
pip install -r requirements.txt
```

> **설치에 5~10분 소요됩니다.** PyTorch가 가장 큰 패키지(~2.5GB)입니다.

### 2-3. 설치 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

다음과 같이 나오면 성공입니다:
```
PyTorch: 2.x.x+cu128
CUDA: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

> **CUDA: False** 가 나오면 → [문제 해결](#cuda-false가-나오는-경우) 참고

---

## 3. Step 1: 모델 다운로드

Hugging Face에서 Qwen2.5-1.5B-Instruct 모델을 다운로드합니다.

```bash
python scripts/01_download_model.py
```

### 무슨 일이 일어나나요?

1. Hugging Face Hub에서 모델 파일 다운로드 (~3GB)
2. 다운받은 모델을 4-bit 양자화로 로딩하여 정상 동작 확인
3. VRAM 사용량과 파라미터 수 출력

### 예상 출력

```
[Step 1/3] Downloading model: Qwen/Qwen2.5-1.5B-Instruct
  → Saving to: models/base/Qwen2.5-1.5B-Instruct
  Download complete!

[Step 2/3] Verifying 4-bit quantized loading...
  Model loaded successfully!

[Step 3/3] Model diagnostics:
  VRAM usage: ~1200 MB
  Total parameters: 1,500,000,000
  Tokenizer vocab size: 151,936
```

### 옵션

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model-name` | `Qwen/Qwen2.5-1.5B-Instruct` | 다른 모델을 쓰고 싶을 때 |
| `--output-dir` | `models/base/Qwen2.5-1.5B-Instruct` | 저장 경로 변경 |

> **Hugging Face 로그인이 필요한 경우:**
> ```bash
> pip install huggingface_hub
> huggingface-cli login
> ```

---

## 4. Step 2: 데이터 준비

학습용 JSONL 데이터를 검증하고 train/eval/test로 분할합니다.

### 4-1. 샘플 데이터로 테스트 (처음엔 이걸로)

```bash
python scripts/02_prepare_data.py
```

기본값으로 `data/samples/gennx_tool_calling_samples.jsonl` (10개 샘플)을 사용합니다.

### 4-2. 실제 데이터로 실행

자체 데이터를 만들었다면:

```bash
python scripts/02_prepare_data.py --input data/raw/my_data.jsonl
```

### 무슨 일이 일어나나요?

1. JSONL 파일 로딩
2. 각 샘플의 포맷 검증 (잘못된 데이터는 건너뜀)
3. Qwen2.5 chat template에 맞게 포맷 변환
4. 셔플 후 80/10/10으로 분할
5. `data/processed/`에 `train.jsonl`, `eval.jsonl`, `test.jsonl` 저장
6. 토큰 길이 통계 출력

### 예상 출력

```
[Step 1/6] Loading data from: data/samples/gennx_tool_calling_samples.jsonl
  Loaded 10 samples

[Step 2/6] Validating data format...
  Valid: 10 / Invalid: 0

[Step 4/6] Splitting data (seed=42)...
  Train: 8 / Eval: 1 / Test: 1

[Step 6/6] Token statistics:
  Count: 10
  Min: 180 / Max: 950 / Mean: 420
  Median: 380 / P95: 850
```

### 옵션

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input` | `data/samples/gennx_tool_calling_samples.jsonl` | 입력 JSONL 파일 |
| `--output-dir` | `data/processed` | 출력 디렉토리 |
| `--train-ratio` | `0.8` | 학습 데이터 비율 |
| `--eval-ratio` | `0.1` | 검증 데이터 비율 |
| `--test-ratio` | `0.1` | 테스트 데이터 비율 |
| `--model-name` | `Qwen/Qwen2.5-1.5B-Instruct` | 토큰 통계 계산용 토크나이저 |

### 데이터 포맷 (중요!)

학습 데이터는 **TRL tool-calling format**을 따라야 합니다.
자세한 포맷 설명은 [DATA_FORMAT.md](./DATA_FORMAT.md)를 참고하세요.

---

## 5. Step 3: QLoRA 학습

이 단계가 핵심입니다. 4-bit 양자화된 모델에 LoRA 어댑터를 붙여 학습합니다.

### 5-1. 학습 시작

```bash
python scripts/03_train_qlora.py
```

### 무슨 일이 일어나나요?

1. `config/` 디렉토리에서 설정 파일 3개 로딩
2. `data/processed/`에서 train/eval 데이터 로딩
3. 모델을 4-bit으로 로딩 + LoRA 어댑터 부착
4. SFTTrainer로 학습 시작
5. 매 epoch마다 체크포인트 저장
6. eval_loss가 3 epoch 동안 개선 안 되면 조기 종료 (Early Stopping)
7. 최종 어댑터를 `models/checkpoints/final_adapter/`에 저장

### 예상 소요 시간

| 데이터 수 | Epochs | 예상 시간 |
|-----------|--------|-----------|
| 10개 (샘플) | 2 | ~5분 |
| 100개 | 10 | ~30분 |
| 1,000개 | 10 | ~3시간 |
| 5,000개 | 10 | ~15시간 |

> 데이터 수와 max_seq_length에 따라 크게 달라집니다.

### 학습 중 모니터링

별도 터미널을 열어서 TensorBoard로 loss 곡선을 실시간 확인할 수 있습니다:

```bash
tensorboard --logdir logs
```

브라우저에서 `http://localhost:6006` 접속 → train_loss, eval_loss 그래프 확인

### 학습 재개 (중단된 경우)

학습 도중 중단되었다면 체크포인트에서 이어서 학습:

```bash
python scripts/03_train_qlora.py --resume
```

### 옵션

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--config-dir` | `config` | YAML 설정 파일 디렉토리 |
| `--data-dir` | `data/processed` | 학습 데이터 디렉토리 |
| `--output-dir` | `models/checkpoints` | 체크포인트 저장 경로 |
| `--resume` | (플래그) | 마지막 체크포인트에서 재개 |
| `--use-unsloth` | (플래그) | Unsloth 백엔드 사용 (더 빠름) |

### 설정 커스터마이징

`config/` 디렉토리의 YAML 파일을 수정하여 학습 설정을 바꿀 수 있습니다:

- **`config/training_config.yaml`** — 학습률, epoch 수, batch 크기 등
- **`config/lora_config.yaml`** — LoRA rank, target modules 등
- **`config/model_config.yaml`** — 모델, 양자화 설정 등

> 설정 파일 상세 설명은 [CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md)를 참고하세요.

---

## 6. Step 4: 평가

학습된 모델의 성능을 테스트 데이터로 측정합니다.

```bash
python scripts/04_evaluate.py --model-path models/checkpoints/final_adapter
```

### 무슨 일이 일어나나요?

1. Base 모델 + 학습된 LoRA 어댑터 로딩
2. 테스트 데이터의 각 샘플에 대해 추론 실행
3. 정답(reference)과 비교하여 지표 계산
4. 결과를 JSON 파일로 저장 + 리포트 출력

### 측정되는 지표

| 지표 | 설명 | 좋은 값 |
|------|------|---------|
| **tool_name_accuracy** | 올바른 도구를 호출했는가 | 높을수록 좋음 (1.0 = 100%) |
| **parameter_accuracy** | 파라미터가 정확한가 | 높을수록 좋음 |
| **json_validity_rate** | 출력이 유효한 JSON인가 | 높을수록 좋음 |
| **hallucination_rate** | 존재하지 않는 도구를 호출했는가 | 낮을수록 좋음 (0.0 = 최상) |
| **latency (mean_ms)** | 추론 속도 | 낮을수록 좋음 |

### 예상 출력

```
========== Evaluation Report ==========
  tool_name_accuracy:   0.8000
  parameter_accuracy:   0.6500
  json_validity_rate:   0.9500
  hallucination_rate:   0.0500

  Latency (mean):       450.2 ms
  VRAM usage:           2100 MB / 6144 MB (34.2%)

  Results saved to: data/eval/eval_results.json
========================================
```

### 옵션

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model-path` | (필수) | 어댑터 또는 병합된 모델 경로 |
| `--base-model` | `Qwen/Qwen2.5-1.5B-Instruct` | 베이스 모델 |
| `--test-data` | `data/processed/test.jsonl` | 테스트 데이터 |
| `--tools-schema` | `data/samples/gennx_tool_schemas_tier1.json` | 도구 스키마 |
| `--output` | `data/eval/eval_results.json` | 결과 저장 경로 |
| `--is-merged` | (플래그) | 이미 병합된 모델인 경우 |

---

## 7. Step 5: 어댑터 병합

학습된 LoRA 어댑터를 base 모델에 합쳐서 하나의 독립 모델로 만듭니다.

```bash
python scripts/05_merge_adapters.py --adapter-path models/checkpoints/final_adapter
```

### 왜 병합하나요?

- 학습 결과물은 **LoRA 어댑터**(~수십 MB)로, base 모델과 별도로 존재
- 배포 시에는 base + adapter를 매번 조합하는 것보다 **하나로 합친 모델**이 편리
- Ollama 등 외부 도구에서 사용하려면 병합이 필요

### 무슨 일이 일어나나요?

1. Base 모델을 float16 (전체 정밀도)로 CPU에 로딩
2. LoRA 어댑터 로딩
3. 어댑터 가중치를 base 모델에 합침 (merge_and_unload)
4. `models/final/`에 통합 모델 저장

### Ollama 배포용 GGUF 변환 안내

```bash
python scripts/05_merge_adapters.py --adapter-path models/checkpoints/final_adapter --export-gguf
```

`--export-gguf` 플래그를 추가하면 GGUF 변환 및 Ollama 배포 방법을 안내해줍니다.

### 옵션

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--adapter-path` | (필수) | LoRA 어댑터 경로 |
| `--base-model` | `Qwen/Qwen2.5-1.5B-Instruct` | 베이스 모델 |
| `--output-dir` | `models/final` | 병합된 모델 저장 경로 |
| `--export-gguf` | (플래그) | GGUF 변환 안내 출력 |

---

## 8. Step 6: 서빙 (사용하기)

학습+병합이 완료된 모델을 실제로 사용해봅니다.

### 8-1. CLI 모드 (터미널 대화)

```bash
python scripts/06_serve.py --mode cli
```

```
GEN NX Tool Calling Assistant (type 'quit' to exit)
================================================

You: 절점 1번을 원점에 추가해줘
Assistant: [Tool Call] POST /db/node
  {"Assign": {"1": {"X": 0, "Y": 0, "Z": 0}}}

  절점 1번이 원점(0, 0, 0)에 추가되었습니다.

You: quit
```

### 8-2. Gradio 모드 (웹 UI)

```bash
python scripts/06_serve.py --mode gradio
```

브라우저에서 `http://localhost:7860` 접속하면 채팅 UI가 열립니다.

### 8-3. Ollama 배포 안내

```bash
python scripts/06_serve.py --mode ollama-info
```

Ollama로 배포하는 방법을 안내해줍니다.

### 옵션

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model-path` | `models/final` | 모델 경로 |
| `--mode` | `cli` | `cli`, `gradio`, `ollama-info` 중 선택 |

---

## 9. 문제 해결 (Troubleshooting)

### CUDA: False가 나오는 경우

PyTorch가 CPU 버전으로 설치된 것입니다.

```bash
# 기존 PyTorch 제거 후 CUDA 버전 재설치
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CUDA Out of Memory (OOM)

6GB VRAM에서 OOM이 발생하면:

1. **LoRA fallback 사용** — `config/lora_config.yaml`에서 `lora` 섹션을 `lora_fallback` 값으로 교체:
   ```yaml
   lora:
     target_modules:
       - "q_proj"
       - "v_proj"
   ```

2. **max_seq_length 줄이기** — `config/model_config.yaml`에서:
   ```yaml
   model:
     max_seq_length: 1024  # 2048 → 1024
   ```

3. **gradient_accumulation 늘리기** — `config/training_config.yaml`에서:
   ```yaml
   training:
     gradient_accumulation_steps: 16  # 8 → 16
   ```

### ImportError: No module named 'bitsandbytes'

Windows에서 bitsandbytes 설치 문제:

```bash
pip install bitsandbytes --prefer-binary
```

### 학습 중 loss가 줄지 않는 경우

- 데이터 포맷이 올바른지 `02_prepare_data.py`로 재검증
- learning_rate를 `1e-4`로 낮춰보기
- 데이터 수가 너무 적은 경우 epoch 수를 늘려보기

### TensorBoard가 안 열리는 경우

```bash
pip install tensorboard
tensorboard --logdir logs --bind_all
```

---

## 전체 요약: 한눈에 보는 파이프라인

```
[환경 설정]     pip install -r requirements.txt
     |
[Step 1]        python scripts/01_download_model.py
모델 다운로드        → models/base/에 모델 저장
     |
[Step 2]        python scripts/02_prepare_data.py
데이터 준비          → data/processed/에 train/eval/test 저장
     |
[Step 3]        python scripts/03_train_qlora.py
QLoRA 학습          → models/checkpoints/final_adapter/ 생성
     |
[Step 4]        python scripts/04_evaluate.py --model-path models/checkpoints/final_adapter
평가                → data/eval/eval_results.json 생성
     |
[Step 5]        python scripts/05_merge_adapters.py --adapter-path models/checkpoints/final_adapter
어댑터 병합          → models/final/에 통합 모델 저장
     |
[Step 6]        python scripts/06_serve.py --mode cli
서빙                → 대화형으로 사용!
```

---

## 노트북으로 학습하기 (선택)

스크립트 실행 전에 노트북으로 각 단계를 하나씩 체험해볼 수 있습니다.
VS Code에서 `.ipynb` 파일을 열어 셀 단위로 실행하세요.

| 노트북 | 언제 사용하나요? |
|--------|-----------------|
| `notebooks/01_inference_basics.ipynb` | 모델이 잘 로딩되는지, tool calling이 되는지 확인할 때 |
| `notebooks/02_tokenizer_explore.ipynb` | 토크나이저와 chat template이 어떻게 동작하는지 이해할 때 |
| `notebooks/03_first_finetune.ipynb` | 본격 학습 전에 소규모로 먼저 시험해볼 때 |
| `notebooks/04_eval_comparison.ipynb` | 학습 전후 성능을 차트로 비교할 때 |

> 노트북 사용법: VS Code에서 `.ipynb` 파일 열기 → 셀 왼쪽의 재생 버튼 클릭 → 결과가 셀 아래에 표시됨
