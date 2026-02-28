# QLoRA Fine-Tuning for GEN NX API Tool Calling

Qwen2.5-1.5B-Instruct 모델을 QLoRA로 fine-tuning하여 GEN NX 구조공학 API의 tool calling을 학습시키는 프로젝트입니다.

## 개요

| 항목 | 내용 |
|------|------|
| Base Model | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| 학습 기법 | QLoRA (4-bit 양자화 + LoRA) |
| 대상 API | GEN NX 구조공학 API 273개 엔드포인트 |
| 학습 환경 | RTX 3060 Laptop 6GB VRAM, Windows 11, CUDA 12.8, Python 3.11 |
| 학습 프레임워크 | Hugging Face TRL (SFTTrainer) |

## 프로젝트 구조

```
qlora-function-calling/
├── config/                          # YAML 설정 파일
│   ├── model_config.yaml            #   모델, 양자화, 토큰 설정
│   ├── lora_config.yaml             #   LoRA rank, target modules 설정
│   └── training_config.yaml         #   학습률, 배치, Early Stopping 설정
│
├── scripts/                         # 파이프라인 스크립트 (순서대로 실행)
│   ├── 01_download_model.py         #   모델 다운로드
│   ├── 02_prepare_data.py           #   데이터 검증 및 분할
│   ├── 03_train_qlora.py            #   QLoRA 학습
│   ├── 04_evaluate.py               #   평가 (정확도, 지연시간)
│   ├── 05_merge_adapters.py         #   LoRA 어댑터 병합
│   └── 06_serve.py                  #   서빙 (CLI / Gradio / Ollama)
│
├── src/                             # 유틸리티 모듈
│   ├── data_utils.py                #   데이터 로딩, 검증, 변환
│   └── eval_metrics.py              #   평가 지표 계산
│
├── notebooks/                       # Jupyter 노트북 (학습 단계별 체험)
│   ├── 01_inference_basics.ipynb    #   모델 로딩 및 tool calling 확인
│   ├── 02_tokenizer_explore.ipynb   #   토크나이저와 chat template 탐색
│   ├── 03_first_finetune.ipynb      #   소규모 학습 시험
│   └── 04_eval_comparison.ipynb     #   학습 전후 성능 비교
│
├── data/
│   ├── samples/                     # 샘플 데이터 (git 추적)
│   │   ├── gennx_tool_calling_samples.jsonl   # 학습 샘플 10개
│   │   └── gennx_tool_schemas_tier1.json      # Tier-1 도구 스키마 15개
│   ├── raw/                         # 원본 데이터 (git 미추적)
│   ├── processed/                   # 전처리된 train/eval/test (git 미추적)
│   └── eval/                        # 평가 결과 (git 미추적)
│
├── docs/                            # 문서
│   ├── GETTING_STARTED.md           #   시작 가이드 (전체 파이프라인)
│   ├── DATA_FORMAT.md               #   학습 데이터 포맷 규칙
│   ├── CONFIG_REFERENCE.md          #   설정 파일 간단 레퍼런스
│   ├── CONFIG_DEEP_DIVE.md          #   설정 파일 심화 해설
│   ├── GEN_NX_API_분석.md           #   GEN NX API 273개 분석 및 Tier 분류
│   ├── LLM_FineTuning_Plan.md       #   Fine-Tuning 실행 계획
│   ├── LLM_FineTuning_핵심개념.md    #   QLoRA/LoRA 핵심 개념 정리
│   ├── LLM_FineTuning_학습과정_및_지표.md  #  학습 과정 및 평가 지표
│   ├── LLM_FineTuning_고도화_전략.md  #   고도화 전략
│   └── LLM_FineTuning_아키텍처_및_인프라.md  # 아키텍처 및 인프라 설계
│
├── models/                          # 모델 저장 (git 미추적)
│   ├── base/                        #   다운로드된 베이스 모델
│   ├── checkpoints/                 #   학습 체크포인트
│   └── final/                       #   병합된 최종 모델
│
├── logs/                            # TensorBoard 로그 (git 미추적)
├── requirements.txt                 # Python 패키지 의존성
└── .gitignore
```

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/Scripts/activate    # Git Bash

# 패키지 설치
pip install -r requirements.txt
```

### 2. 파이프라인 실행

```bash
# Step 1: 모델 다운로드
python scripts/01_download_model.py

# Step 2: 데이터 준비
python scripts/02_prepare_data.py

# Step 3: QLoRA 학습
python scripts/03_train_qlora.py

# Step 4: 평가
python scripts/04_evaluate.py --model-path models/checkpoints/final_adapter

# Step 5: 어댑터 병합
python scripts/05_merge_adapters.py --adapter-path models/checkpoints/final_adapter

# Step 6: 서빙
python scripts/06_serve.py --mode cli
```

자세한 설명은 [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)를 참고하세요.

## 학습 데이터 포맷

TRL tool-calling format을 따르는 JSONL 파일:

```json
{
  "messages": [
    {"role": "system", "content": "You are a structural engineering assistant..."},
    {"role": "user", "content": "절점 1번을 원점에 추가해줘"},
    {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "POST /db/node", "arguments": "{\"Assign\":{\"1\":{\"X\":0,\"Y\":0,\"Z\":0}}}"}}]},
    {"role": "tool", "name": "POST /db/node", "content": "{\"NODE\":{\"1\":{\"X\":0,\"Y\":0,\"Z\":0}}}"},
    {"role": "assistant", "content": "절점 1번이 원점(0, 0, 0)에 추가되었습니다."}
  ],
  "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"POST /db/node\",...}}]"
}
```

자세한 포맷 규칙은 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)를 참고하세요.

## GEN NX API Tier 분류

| Tier | 엔드포인트 수 | 설명 | 학습 데이터 |
|------|-------------|------|-----------|
| Tier 1 (핵심) | ~127개 | 모델링, 경계조건, 하중, 해석 | API당 15~20개 |
| Tier 2 (보조) | ~102개 | 설계, 하중조합, 동적하중 | API당 5~10개 |
| Tier 3 (특수) | ~44개 | 이동하중, 수화열 등 특수 기능 | API당 0~2개 |
| **합계** | **273개** | | **~2,400~3,650개** |

## 핵심 설정

| 설정 | 값 | 설명 |
|------|-----|------|
| 양자화 | 4-bit NF4 + 이중 양자화 | VRAM 절약 (~774MB) |
| LoRA rank | 8 (확장 시 16~32) | 학습 파라미터 ~923만 개 (0.6%) |
| target_modules | all-linear | 7개 선형 레이어 전부 |
| 학습률 | 2e-4, cosine scheduler | warmup 5% + cosine 감소 |
| 배치 | 1 x 8 (gradient accumulation) | 실효 배치 크기 8 |
| 정밀도 | fp16 | RTX 3060 하드웨어 가속 |
| 옵티마이저 | paged_adamw_8bit | 8-bit Adam + VRAM 페이징 |
| Early Stopping | patience 3, threshold 0.01 | eval_loss 기준 |

설정 상세 설명은 [docs/CONFIG_DEEP_DIVE.md](docs/CONFIG_DEEP_DIVE.md)를 참고하세요.

## 평가 지표

| 지표 | 설명 |
|------|------|
| tool_name_accuracy | 올바른 도구를 호출했는가 |
| parameter_accuracy | 파라미터가 정확한가 |
| json_validity_rate | 출력이 유효한 JSON인가 |
| hallucination_rate | 존재하지 않는 도구를 호출했는가 |

## 문서

| 문서 | 설명 |
|------|------|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | 전체 파이프라인 시작 가이드 |
| [DATA_FORMAT.md](docs/DATA_FORMAT.md) | 학습 데이터 포맷 규칙 |
| [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | 설정 파일 간단 레퍼런스 |
| [CONFIG_DEEP_DIVE.md](docs/CONFIG_DEEP_DIVE.md) | 설정 파일 심화 해설 (양자화, LoRA, 학습률 등) |
| [GEN_NX_API_분석.md](docs/GEN_NX_API_분석.md) | GEN NX API 273개 분석 및 Tier 분류 |
| [LLM_FineTuning_Plan.md](docs/LLM_FineTuning_Plan.md) | Fine-Tuning 실행 계획 |

## 요구사항

- **GPU**: NVIDIA GPU 6GB+ VRAM (RTX 3060 이상)
- **RAM**: 16GB+
- **OS**: Windows 10/11
- **Python**: 3.11
- **CUDA**: 12.x (12.8 권장)
