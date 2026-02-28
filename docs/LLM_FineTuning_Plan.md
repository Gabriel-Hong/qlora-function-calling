# LLM Fine-Tuning 프로젝트 플랜

## Context

멀티에이전트 시스템과 RAG 구현 경험을 기반으로, **LLM 모델 자체를 핸들링하는 경험**을 쌓기 위한 프로젝트.
오픈소스 소형 LLM을 직접 fine-tuning하고, 학습된 모델에 질문하여 결과를 확인하는 것이 목표.

**환경 제약**: RTX 3060 Laptop GPU (6GB VRAM), Windows 11, CUDA 12.5

---

## Phase 0: 환경 구축 (사전 준비)

### 0-1. CUDA 업그레이드 (필수)
- **문제**: CUDA 12.5는 최신 PyTorch 2.10+에서 미지원
- **조치**: CUDA Toolkit 12.8 설치 (https://developer.nvidia.com/cuda-toolkit-archive)
- 환경변수 설정: `CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`

### 0-2. Python 환경 설정
```bash
conda create -n llm-ft python=3.11 -c conda-forge
conda activate llm-ft

# PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 핵심 라이브러리
pip install transformers peft trl datasets accelerate bitsandbytes

# Unsloth (2-5x 학습 가속, 60% VRAM 절감)
pip install unsloth
```

### 0-3. bitsandbytes Windows 이슈
- 공식 Windows 지원 불안정 → 설치 실패 시 WSL2 사용 권장
- WSL2 대안: Ubuntu 22.04에서 전체 학습 파이프라인 실행
- 또는 8-bit 양자화로 대체 (4-bit 대비 VRAM 조금 더 사용)

### 0-4. 검증
```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # RTX 3060 Laptop GPU
print(torch.version.cuda)               # 12.8
```

---

## Phase 1: 모델 로딩 & 추론 기초 (3~5일)

**목표**: HuggingFace 모델을 로컬에서 로딩하고 추론하는 과정을 이해

### 1-1. 기본 모델 로딩 & 텍스트 생성
- `Qwen2.5-1.5B-Instruct` 모델 다운로드
- FP16 → 4-bit 양자화 로딩 비교 (BitsAndBytesConfig)
- 간단한 채팅 추론 실행

### 1-2. 토크나이저 이해
- 토큰화 동작 원리 (encode/decode)
- 특수 토큰 (BOS, EOS, PAD)
- ChatML 템플릿 구조 이해 (`apply_chat_template`)

### 1-3. 4-bit 양자화 체험
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4가 FP4보다 우수
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,       # 이중 양자화로 추가 절감
)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", quantization_config=bnb_config)
```

### 1-4. VRAM 모니터링
- `nvidia-smi`로 실시간 VRAM 사용량 확인
- 모델 크기별 VRAM 비교표 직접 작성

---

## Phase 2: LoRA/QLoRA 이론 & 첫 Fine-Tuning (5~7일)

**목표**: LoRA의 원리를 이해하고, 공개 데이터셋으로 첫 fine-tuning 실행

### 2-1. LoRA 핵심 개념
- Low-Rank Adaptation: 거대 가중치 행렬을 작은 두 행렬의 곱으로 근사
- 원래 모델은 freeze → LoRA 어댑터만 학습 (전체 파라미터의 ~1%)
- QLoRA: 4-bit 양자화 + LoRA = 6GB에서도 3B 모델 학습 가능

### 2-2. 공개 데이터셋으로 첫 SFT
- 데이터셋: `tatsu-lab/alpaca` 또는 `Open-Orca/OpenOrca` (소규모 서브셋)
- SFTTrainer (trl 라이브러리) 사용

### 2-3. QLoRA 설정 (6GB 최적)
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                          # rank (6GB면 8이 안전, 16도 시도 가능)
    lora_alpha=16,                # alpha = 2 * r
    lora_dropout=0.05,
    target_modules="all-linear",  # 2025 best practice
    task_type="CAUSAL_LM",
)
```

### 2-4. 학습 하이퍼파라미터
```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,      # 6GB 필수 제약
    gradient_accumulation_steps=8,       # effective batch = 8
    learning_rate=2e-4,
    num_train_epochs=10,                 # 넉넉하게 설정 (Early Stopping이 자동 중단)
    warmup_steps=100,
    fp16=True,                           # consumer GPU는 fp16
    optim="paged_adamw_8bit",            # OOM 방지
    logging_steps=10,
    eval_strategy="epoch",               # 매 epoch마다 검증
    save_strategy="epoch",               # 매 epoch마다 체크포인트 저장
    load_best_model_at_end=True,         # 최적 모델 자동 선택
    metric_for_best_model="eval_loss",   # 검증 Loss 기준으로 판단
)
```

> **참고**: `max_steps` 대신 `num_train_epochs` + Early Stopping 조합 사용. 검증 Loss가 올라가기 시작하면 과적합 신호이며, `load_best_model_at_end=True`가 가장 좋았던 체크포인트를 자동 선택한다. 보통 epoch 2~4 사이에서 최적점이 결정된다.

### 2-5. 학습 전/후 비교
- 동일 프롬프트에 대한 응답 품질 비교
- loss 곡선 분석 (TensorBoard 또는 wandb)

---

## Phase 3: GEN NX 특화 데이터셋 구축 (5~7일)

**목표**: GEN NX Tool Calling + 건축구조 도메인 지식 학습용 데이터셋 구축

### 3-1. 공개 Function Calling 데이터셋 참고
| 데이터셋 | 크기 | 특징 |
|----------|------|------|
| **Salesforce xLAM-60k** | 60K | 가장 포괄적, BFCL 상위 |
| **glaive-function-calling-v2** | 2K+ | 다양한 시나리오 |
| **NousResearch hermes-function-calling-v1** | 다수 | 구조화된 출력 포함 |
| **Locutusque/function-calling-chatml** | 다수 | ChatML 포맷 (Qwen 호환) |

> 위 데이터셋은 포맷/패턴 참고용. 실제 학습 데이터는 GEN NX API 기반으로 직접 구축한다.

### 3-2. 커스텀 데이터 생성 프로세스

**Step 1: GEN NX 도구 스키마 정의 (1일)**
- GEN NX MCP tool 중 학습 대상 선별 (계층별 차등 분배)
  - Tier 1 (핵심 도구 30~50개): 도구당 30~50개 데이터 → 데이터의 60%
  - Tier 2 (보조 도구 100~150개): 도구당 5~10개 데이터 → 데이터의 30%
  - Tier 3 (나머지): 일반화에 의존, 도구 설명(description)을 명확히 작성 → 데이터의 10%
- 각 도구를 JSON Schema로 정의

**Step 2: 시드 데이터 수동 작성 (2~3일)**
- 각 카테고리별 20~40개씩 직접 작성 → 총 100~200개
- 이 시드 데이터가 전체 품질의 기준선이 된다

**Step 3: LLM으로 대량 확장 (1~2일)**
- 시드 데이터를 Claude/GPT에 보여주고 변형 생성
- tool 파라미터 값, 질문 표현, 건물 유형 등을 다양하게 변경
- 총 2,500개로 확장

**Step 4: 검수 (1~1.5일)**
- 자동 검증: JSON 파싱, tool name 유효성, required 파라미터 확인 (스크립트)
- 수동 검수: 무작위 200~300개 샘플링하여 내용 확인

### 3-3. 학습 데이터 포맷 (ChatML + Tool Calling)
```jsonl
{"messages": [
  {"role": "system", "content": "You have access to these tools:\n[{\"name\": \"create_model\", \"description\": \"Create structural model\", \"parameters\": {\"type\": \"object\", \"properties\": {\"type\": {\"type\": \"string\"}, \"stories\": {\"type\": \"integer\"}}, \"required\": [\"type\", \"stories\"]}}]"},
  {"role": "user", "content": "10층 RC 건물 모델을 생성해줘"},
  {"role": "assistant", "content": null, "tool_calls": [{"name": "create_model", "arguments": {"type": "RC", "stories": 10}}]},
  {"role": "tool", "content": "{\"status\": \"success\", \"model_id\": \"M001\"}"},
  {"role": "assistant", "content": "10층 RC 구조 모델이 생성되었습니다. (모델 ID: M001)"}
]}
```

### 3-4. 데이터 구성 전략 (GEN NX 특화)
- **목표 규모**: 2,500개 고품질 예시

| 카테고리 | 비율 | 개수 | 설명 |
|----------|:----:|:----:|------|
| GEN NX 단일 Tool Calling | 35% | ~875개 | 단일 API 호출 |
| GEN NX 다단계 Tool Calling | 20% | ~500개 | 순차/병렬 다중 호출 |
| 건축구조 도메인 Q&A | 25% | ~625개 | KBC, ACI, Eurocode 등 기준서 지식 |
| 기준서 비교/판단 질문 | 10% | ~250개 | "KBC vs Eurocode 차이" 등 |
| 부정 예시 (도구 호출 불필요) | 10% | ~250개 | 일반 대화, 인사 등 |

> **부정 예시가 중요한 이유**: 없으면 모델이 모든 질문에 tool을 호출하려는 버릇이 생긴다.
> **도메인 Q&A를 섞는 이유**: tool calling만 학습하면 Catastrophic Forgetting으로 범용 능력이 퇴화한다. 도메인 지식 + 범용 대화를 10~25% 섞어야 균형을 유지할 수 있다.

### 3-5. 데이터 품질 검증
- JSON 파싱 유효성 검사
- tool name이 정의된 GEN NX 도구 목록에 있는지 확인
- required 파라미터 누락 여부 검사
- 동일 질문에 다른 도구를 호출하는 모순 여부 확인
- train/val/test split (80/10/10) → 2,000 / 250 / 250

---

## Phase 4: Function Calling 특화 Fine-Tuning (5~7일)

**목표**: Phase 3 데이터셋으로 function calling 모델 학습

### 4-1. 모델 선택

| 모델 | 크기 | VRAM (QLoRA) | Function Calling |
|------|------|-------------|-----------------|
| **Qwen2.5-1.5B-Instruct** (1순위) | 1.5B | ~3GB | 네이티브 지원 |
| **Qwen2.5-3B-Instruct** (2순위) | 3B | ~4.5GB | 네이티브 지원 |
| **xLAM-1b-fc-r** (대안) | 1B | ~2.5GB | Function calling 전용 |

### 4-2. 학습 실행
- Phase 2의 QLoRA 설정 + Phase 3 데이터셋 결합
- Unsloth 사용 시 학습 속도 2-5x 향상
- **예상 시간**: 1.5B 모델 + 2,500개 데이터 + 3 epochs → **약 1.5시간**

### 4-3. Catastrophic Forgetting 대응
- Phase 3 데이터에 도메인 Q&A 25% + 부정 예시 10%를 섞어서 범용 능력 퇴화 방지
- LoRA rank를 과도하게 키우지 않기 (r=8~16 범위 유지)
- Epoch을 너무 오래 돌리지 않기 (Early Stopping으로 3~5 epoch 이내에서 자동 중단)
- 학습 후 base 모델 대비 일반 대화/추론 능력 퇴화 여부 확인

### 4-4. 평가 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| Tool Name Accuracy | 올바른 도구 이름 선택률 | >95% |
| Parameter Accuracy | 파라미터 정확도 | >90% |
| JSON Validity | 출력 형식 유효율 | 100% |
| Hallucination Rate | 없는 파라미터 생성률 | <5% |
| Latency (tokens/sec) | 추론 속도 | 서빙 가능성 확인 |
| VRAM Usage | nvidia-smi 피크 메모리 | 6GB 이내 유지 |

### 4-5. 반복 실험
```
실험 1: 500개 데이터, r=8 → 기준선 측정 (15~20분)
실험 2: 1,000개 데이터, r=8 → 개선 확인 (30분)
실험 3: 2,000개 데이터, r=16 → 최적 조합 탐색 (1시간)
실험 4: 3,000개 데이터, best config → 최종 모델 (1.5시간)
```

---

## Phase 5: 모델 서빙 & 결과 확인 (3~5일)

**목표**: Fine-tuned 모델을 로컬에서 서빙하여 직접 질문하고 답변 확인

### 5-1. LoRA 어댑터 병합
```python
# LoRA 어댑터를 base 모델에 병합
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")
```

### 5-2. 로컬 서빙 방법 (택 1)

**방법 A: Ollama (가장 간단 — 터미널 채팅)**
```bash
# GGUF 변환 후 Ollama에 등록
ollama create my-finetuned-model -f Modelfile
ollama run my-finetuned-model
# → 터미널에서 바로 대화 가능
```

**방법 B: Gradio 웹 UI (ChatGPT 같은 웹 채팅)**
```python
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="./merged-model", device="cuda")

def chat(message, history):
    response = pipe(message, max_new_tokens=256)
    return response[0]["generated_text"]

gr.ChatInterface(chat, title="Fine-Tuned Model").launch()
# → 브라우저에서 http://localhost:7860 접속하여 채팅
```

**방법 C: Python 스크립트로 직접 추론 (가장 기본)**
```python
# 학습 전/후 모델 비교
prompts = ["서울 날씨 알려줘", "주문 상태 확인해줘", ...]
for prompt in prompts:
    print(f"[Base 모델] {generate(base_model, prompt)}")
    print(f"[Fine-tuned] {generate(finetuned_model, prompt)}")
```

### 5-3. 결과 비교 확인
- **동일 질문**에 대한 base 모델 vs fine-tuned 모델 응답 나란히 비교
- function calling 정확도 수동 검증
- 다양한 질문 유형별 응답 품질 확인

---

## 프로젝트 디렉토리 구조

```
qlora-function-calling/
├── config/
│   ├── model_config.yaml       # 모델 설정
│   ├── training_config.yaml    # 학습 하이퍼파라미터
│   └── lora_config.yaml        # LoRA 설정
├── data/
│   ├── samples/                # 데이터 형식 예시 (5~10개, repo에 포함)
│   ├── raw/                    # 원본 데이터셋 (.gitignore)
│   ├── processed/              # 전처리된 학습 데이터 (.gitignore)
│   └── eval/                   # 평가 데이터 (.gitignore)
├── scripts/
│   ├── 01_download_model.py
│   ├── 02_prepare_data.py
│   ├── 03_train_qlora.py
│   ├── 04_evaluate.py
│   ├── 05_merge_adapters.py
│   └── 06_serve.py
├── notebooks/
│   ├── 01_inference_basics.ipynb
│   ├── 02_tokenizer_explore.ipynb
│   ├── 03_first_finetune.ipynb
│   └── 04_eval_comparison.ipynb
├── models/                     # (.gitignore)
│   ├── base/                   # 다운로드된 base 모델
│   ├── checkpoints/            # 학습 체크포인트
│   └── final/                  # 병합된 최종 모델
├── src/
│   ├── data_utils.py           # 데이터 처리 유틸
│   └── eval_metrics.py         # 평가 지표
├── logs/                       # 학습 로그 - TensorBoard/wandb (.gitignore)
├── .gitignore
└── requirements.txt
```

---

## 전체 타임라인

| Phase | 기간 | 핵심 산출물 |
|-------|------|------------|
| **Phase 0**: 환경 구축 | 1~2일 | CUDA 12.8 + conda 환경 |
| **Phase 1**: 모델 로딩 & 추론 | 3~5일 | 4-bit 추론 성공, 토크나이저 이해 |
| **Phase 2**: 첫 QLoRA Fine-Tuning | 5~7일 | Alpaca 데이터로 SFT 성공 |
| **Phase 3**: 데이터셋 구축 | 5~7일 | 2,000~3,000개 function calling 데이터 |
| **Phase 4**: Function Calling Fine-Tuning | 5~7일 | 평가 지표 달성 모델 |
| **Phase 5**: 서빙 & 결과 확인 | 3~5일 | 로컬에서 대화하며 결과 확인 |
| **총 기간** | **3.5~5주** | |

---

## 검증 방법

### 각 Phase별 성공 기준

1. **Phase 0**: `torch.cuda.is_available()` = True, CUDA 12.8 확인
2. **Phase 1**: Qwen2.5-1.5B 4-bit 로딩, 채팅 추론 정상 동작
3. **Phase 2**: loss 곡선 수렴 확인, 학습 전/후 응답 품질 차이 확인
4. **Phase 3**: 2,000개+ 데이터, JSON 유효성 100%, 시나리오 다양성 충족
5. **Phase 4**: Tool Name Accuracy >95%, Parameter Accuracy >90%
6. **Phase 5**: 로컬에서 fine-tuned 모델에 질문 → base 모델 대비 개선된 답변 확인

---

## 참고 자료 (조사 출처)

- [QLoRA: Efficient Finetuning of Quantized LLMs - NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023)
- [Unsloth - 2x Faster LLM Fine-tuning](https://github.com/unslothai/unsloth)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [Berkeley Function-Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Salesforce xLAM Function Calling Dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- [Fine-Tuning SLMs for Function Calling - Microsoft](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/fine-tuning-small-language-models-for-function-calling)
- [Qwen Function Calling Documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Profiling LoRA/QLoRA Fine-Tuning on Consumer GPUs - arXiv](https://arxiv.org/html/2509.12229v1)
- [LoRA Hyperparameters Guide - Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [How to Fine-tune Open LLMs in 2025 - Philipp Schmid](https://www.philschmid.de/fine-tune-llms-in-2025)
