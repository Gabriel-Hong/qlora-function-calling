# 설정 파일 심화 해설

> `config/` 디렉토리의 YAML 파일 각 항목이 **왜 이 값인지, 내부에서 어떻게 동작하는지**를 설명합니다.
> 간단한 레퍼런스는 [CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md)를 참고하세요.

---

## 목차

1. [model_config.yaml](#1-model_configyaml)
   - [dtype (모델 연산 정밀도)](#dtype-모델-연산-정밀도)
   - [tokens (특수 토큰과 chat_template)](#tokens-특수-토큰과-chat_template)
   - [quantization (양자화)](#quantization-양자화)
   - [environment (환경 변수)](#environment-환경-변수)
2. [lora_config.yaml](#2-lora_configyaml)
   - [LoRA란 무엇인가](#lora란-무엇인가)
   - [LoRA vs QLoRA](#lora-vs-qlora)
   - [r (rank)](#r-rank)
   - [target_modules](#target_modules)
   - [bias](#bias)
   - [task_type](#task_type)
   - [gradient_checkpointing](#gradient_checkpointing)
3. [training_config.yaml](#3-training_configyaml)
   - [배치 설정](#배치-설정)
   - [학습률 설정](#학습률-설정)
   - [에폭 설정](#에폭-설정)
   - [fp16 vs bf16](#fp16-vs-bf16)
   - [옵티마이저 (paged_adamw_8bit)](#옵티마이저-paged_adamw_8bit)
   - [TensorBoard 로깅](#tensorboard-로깅)
   - [Early Stopping (조기 종료)](#early-stopping-조기-종료)
   - [Windows 호환성](#windows-호환성)
   - [SFT 설정 (packing)](#sft-설정-packing)

---

## 1. model_config.yaml

### dtype: 모델 연산 정밀도

```yaml
model:
  dtype: "float16"
```

float16은 숫자 하나를 **16비트(2바이트)**로 표현하는 부동소수점 형식이다. "16자리"가 아니라 "16비트"이다.

```
float32 (일반 정밀도):  1비트(부호) + 8비트(지수) + 23비트(가수) = 32비트
float16 (반정밀도):     1비트(부호) + 5비트(지수) + 10비트(가수) = 16비트
```

| | float32 | float16 |
|---|---------|---------|
| 메모리 | 4바이트 | **2바이트 (절반)** |
| 유효 자릿수 | ~7자리 | ~3.3자리 |

Qwen2.5-1.5B는 파라미터가 **15억 개**이므로:

```
float32: 15억 x 4바이트 = 6.0 GB
float16: 15억 x 2바이트 = 3.0 GB  ← 절반
4-bit:   15억 x 0.5바이트 = 0.75 GB  ← 현재 QLoRA가 쓰는 방식
```

정밀도가 조금 떨어져도 LLM의 출력 품질에는 거의 차이가 없다.

이 프로젝트에서 float16이 두 군데 나온다:

| 설정 | 역할 |
|------|------|
| `model.dtype: float16` | 모델 **연산** 시 float16 사용 |
| `quantization.bnb_4bit_compute_dtype: float16` | 4-bit 양자화된 가중치를 **계산할 때** float16으로 변환 |

```
저장: 4-bit (0.5바이트)   ← VRAM 절약
연산: float16 (2바이트)   ← 계산은 이걸로
```

---

### tokens: 특수 토큰과 chat_template

```yaml
tokens:
  eos_token: "<|im_end|>"
  pad_token: "<|endoftext|>"
  chat_template: "qwen"
```

#### chat_template이란

LLM에게 대화를 입력할 때 **"누가 말한 건지"를 구분하는 포맷 규칙**이다. 모델에게는 모든 입력이 하나의 텍스트 덩어리이므로, 특수 태그로 감싸서 역할을 알려줘야 한다.

**Qwen2.5의 ChatML 형식:**

```
<|im_start|>system
You are a structural engineering assistant for GEN NX.<|im_end|>
<|im_start|>user
절점 1번을 원점에 추가해줘<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "POST /db/node", "arguments": {"Assign": {"1": {"X": 0, "Y": 0, "Z": 0}}}}
</tool_call><|im_end|>
```

| 태그 | 의미 |
|------|------|
| `<\|im_start\|>system` | 시스템 메시지 시작 |
| `<\|im_start\|>user` | 사용자 메시지 시작 |
| `<\|im_start\|>assistant` | AI 응답 시작 |
| `<\|im_end\|>` | 해당 메시지 끝 (= eos_token) |

`im`은 **I**nstruction **M**essage의 약자이다.

**모델마다 템플릿이 다르다:**

```
[Qwen2.5 - ChatML]
<|im_start|>user
안녕<|im_end|>

[Llama 3]
<|start_header_id|>user<|end_header_id|>
안녕<|eot_id|>

[Gemma]
<start_of_turn>user
안녕<end_of_turn>
```

`chat_template: "qwen"`은 messages 배열을 자동으로 위의 `<|im_start|>...<|im_end|>` 형식으로 변환하라는 뜻이다. Qwen2.5 토크나이저에 이 변환 로직이 내장되어 있다. **잘못된 템플릿을 쓰면 모델이 역할 구분을 못해서 학습이 제대로 되지 않는다.**

---

### quantization: 양자화

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true
```

#### 양자화란

큰 숫자를 작은 숫자로 바꾸되, **원래 값으로 되돌릴 수 있는 힌트(스케일 팩터)를 같이 저장**하는 것이다.

```
원본 가중치 64개 (각각 float16, 2바이트):
[0.12, 0.45, -0.33, 0.78, -0.91, 0.56, ...]

→ 4-bit 양자화: 64개를 한 블록으로 묶고, 스케일 팩터 1개 생성

  스케일 팩터: 0.13 (float16, 2바이트)
  양자화된 값: [1, 3, -3, 6, -7, 4, ...] (각각 4-bit, 0.5바이트)

  복원: 양자화값 x 스케일 = 근사 원본
    1 x 0.13 = 0.13 (원래 0.12, 약간 오차)
```

#### 1회 양자화 메모리 계산

```
가중치:     15억 x 0.5바이트(4-bit)        = 750.0 MB
스케일 팩터: 15억 / 64 x 2바이트(float16)   =  46.9 MB
                  ↑ 64개당 스케일 1개    ↑ 스케일은 float16으로 저장
합계: 796.9 MB
```

#### 이중 양자화 (bnb_4bit_use_double_quant: true)

스케일 팩터(float16) 자체를 **다시 8-bit으로 양자화**하는 것이다.

```
1회 양자화로 생긴 스케일 팩터들 (각각 float16):
[0.13, 0.09, 0.21, ...]  ← 2,343만 개

이것들을 256개씩 묶어서 다시 양자화:
  양자화된 스케일: [43, 30, 70, ...] (각각 8-bit, 1바이트)
  슈퍼 스케일: 0.003 (float32, 4바이트) ← 1개

슈퍼 스케일이 float32(4바이트)인 이유:
  양자화의 양자화를 복원하는 최종 기준점이므로 정밀해야 한다.
  개수가 ~9만 개로 매우 적어서 메모리 부담이 없다.
```

```
                    스케일 팩터     총 메모리    절약량
이중 양자화 OFF:    46.9 MB        796.9 MB     -
이중 양자화 ON:     23.8 MB        773.8 MB     ~23 MB 절약
```

23MB가 적어 보이지만, RTX 3060 6GB에서는 학습 중 VRAM이 빡빡하기 때문에 이 23MB가 OOM과 정상 동작의 경계가 될 수 있다.

---

### environment: 환경 변수

#### pytorch_cuda_alloc_conf: "expandable_segments:True"

**GPU 메모리 단편화를 방지**한다.

GPU 메모리 단편화란, 텐서를 할당하고 해제하기를 반복하면 빈 공간이 조각나서, **총 여유 VRAM은 충분한데도 큰 텐서를 할당 못해서 OOM이 발생**하는 현상이다.

```
[단편화된 상태]
VRAM: [사용][빈][사용][빈][사용][빈][사용]
→ 빈 공간 합계는 충분한데, 연속된 공간이 없어서 큰 텐서 할당 불가

에러 메시지:
  "CUDA out of memory. Tried to allocate 500MB.
   GPU 0 has 6GB total of which 2.5GB is free."
  → "free가 있는데 왜 OOM?" 의 원인이 단편화
```

**기본 방식**: 고정 크기 블록 여러 개로 관리 → 블록 내/간 구멍 발생

**expandable_segments**: 확장 가능한 세그먼트로 관리 → 연속 공간 유지, OOM 위험 감소

QLoRA 학습은 Forward → Loss → Backward → Optimizer 과정에서 텐서 할당/해제가 매우 불규칙하므로 이 설정이 특히 중요하다.

#### tokenizers_parallelism: "false"

**Windows에서 토크나이저 deadlock을 방지**한다.

Deadlock(교착 상태)이란 두 개 이상의 작업이 서로를 기다리면서 **영원히 멈추는 상태**이다. OOM은 에러가 나면서 프로그램이 죽지만, deadlock은 에러도 없이 프로그램이 그냥 멈춘다.

```
Hugging Face 토크나이저: Rust 기반 멀티스레드 병렬 처리
PyTorch DataLoader: 데이터를 병렬로 불러오려고 자식 프로세스 생성

Windows에서 이 둘이 동시에 작동하면:
  프로세스 복제(spawn) 시 스레드 잠금 상태가 꼬임
  → 워커끼리 서로 자원을 기다림
  → 학습이 멈춤
```

`false`로 설정하면 토크나이저가 단일 스레드로 동작하여 충돌이 불가능해진다. 토큰화 속도가 약간 느려지지만, 학습 병목은 GPU 연산이지 토큰화가 아니므로 체감 차이는 거의 없다.

---

## 2. lora_config.yaml

### LoRA란 무엇인가

**Low-Rank Adaptation**. 원래 모델의 가중치를 직접 수정하지 않고, 변화량(ΔW)을 저랭크 분해로 근사한다.

```
원래 방식 (Full Fine-Tuning):
  W (1000x1000) 전체를 업데이트 → 1,000,000개 파라미터 학습

LoRA 방식 (r=8):
  W는 고정(freeze)
  ΔW = A x B 로 분해
  A (1000x8) x B (8x1000) → 16,000개 파라미터만 학습 (1/62.5로 축소)
```

원래 레이어의 계산 경로에 **새로운 우회 경로를 추가**하는 것이다:

```
[LoRA 어댑터를 붙인 후]

              원본 경로 (고정)
입력 → [ W (고정) ] ──────────→ (+) → 출력
  |        우회 경로 (학습)       ↑
  └→ [ A ] → [ B ] ─────────────┘

출력 = 입력 x W + 입력 x A x B
       ─────────   ──────────────
       원본 (고정)   우회 경로 (학습)
```

**A와 B는 항상 분리된 상태**로 존재하고, 역전파(Backward) 과정에서 각각 따로 업데이트된다. 학습이 끝난 후 배포할 때 합친다 (W_최종 = W_원본 + A x B). 이것이 `05_merge_adapters.py`가 하는 일이다.

초기 상태에서 A는 랜덤 작은 값, **B는 전부 0으로 초기화**된다. 따라서 학습 시작 시 A x B = 0이므로 원본 모델 그대로 동작하는 상태에서 출발한다.

### LoRA vs QLoRA

차이는 **베이스 모델을 양자화하느냐** 하나뿐이다.

```
LoRA:   베이스 모델 (float16, 3GB)  + LoRA 어댑터 학습
QLoRA:  베이스 모델 (4-bit, 0.75GB) + LoRA 어댑터 학습
```

| | Full Fine-Tuning | LoRA | QLoRA |
|---|-----------------|------|-------|
| 베이스 모델 | float16 (3GB) | float16 (3GB) | **4-bit (0.75GB)** |
| 학습 대상 | 전체 15억 개 | 어댑터만 ~923만 개 | 어댑터만 ~923만 개 |
| VRAM | ~14 GB | ~4 GB | **~1.5 GB** |
| 학습 품질 | 100% 기준 | ~98% | ~95-98% |

RTX 3060 6GB에서 1.5B 모델을 학습할 수 있는 이유가 바로 QLoRA 덕분이다.

---

### r (rank)

```yaml
lora:
  r: 8
  lora_alpha: 16
```

r은 A와 B 사이의 **정보 통로 너비**이다.

```
입력 (1536차원) → A → (r차원) → B → 출력 (1536차원)
                       ────────
                       여기가 병목!

r=2:   1536차원 → 2차원으로 압축  → 단순한 패턴만 표현 가능
r=8:   1536차원 → 8차원으로 압축  → 주요 패턴 대부분 담을 수 있음
r=64:  1536차원 → 64차원으로 압축 → 거의 모든 뉘앙스를 담을 수 있음
```

Qwen2.5-1.5B 기준 r별 학습 파라미터와 VRAM:

| r | 학습 파라미터 | VRAM 추가 | 표현력 |
|---|-------------|----------|--------|
| 4 | 4.6M | 55 MB | 단순 패턴만 |
| **8 (현재)** | **9.2M** | **110 MB** | **15개 도구에 충분** |
| 16 | 18.5M | 222 MB | 127개 도구 (Tier 1) 대응 |
| 32 | 36.9M | 443 MB | 273개 도구 (전체) 대응 |
| 64 | 73.9M | 887 MB | Full FT에 근접 |

r을 8→32로 4배 올려도 VRAM 추가는 333MB뿐이다. 6GB GPU에서 모두 여유 있게 동작한다. 성능 병목은 r이 아니라 **학습 데이터 양**이다.

`lora_alpha`는 LoRA 출력의 스케일링 계수로, 보통 r의 2배로 설정한다 (r=8이면 alpha=16).

---

### target_modules

```yaml
target_modules: "all-linear"    # 기본: 7개 레이어 전부
```

Qwen2.5-1.5B의 각 Transformer 블록(28개 층) 내부에는 **7개의 선형 레이어**가 있다:

```
Transformer 블록 하나 (x28 반복)
|
├── [Attention] "문장에서 어디를 봐야 하는가"
|   ├── q_proj (Query)  ── "무엇이 궁금한가"
|   ├── k_proj (Key)    ── "어떤 단어가 열쇠인가"
|   ├── v_proj (Value)  ── "실제로 전달할 정보"
|   └── o_proj (Output) ── "결과를 합쳐서 전달"
|
└── [MLP] "정보를 가공/변환"
    ├── gate_proj  ── 정보 통과량 결정
    ├── up_proj    ── 차원 확장 (1536 → 8960)
    └── down_proj  ── 차원 축소 (8960 → 1536)
```

**설정별 비교:**

| target_modules | LoRA 수 | VRAM | 표현력 | 용도 |
|----------------|---------|------|--------|------|
| `"all-linear"` | 196개 | 가장 높음 | 최대 | **기본 설정** |
| `["q_proj","k_proj","v_proj","o_proj"]` | 112개 | 중간 | 높음 | Attention 전체 |
| `["q_proj","v_proj"]` | 56개 | 낮음 | 낮음 | **OOM fallback** |
| `["gate_proj","up_proj","down_proj"]` | 84개 | 중간 | 중간 | MLP만 |

LoRA 원논문에서 Q와 V만 학습해도 full fine-tuning의 ~90% 성능을 달성했다. tool calling에서는 Q가 "절점 추가해줘"에서 핵심 단어에 집중하도록, V가 `POST /db/node`라는 정보를 전달하도록 변경한다.

---

### bias

```yaml
bias: "none"
```

모든 선형 레이어에는 가중치(W)와 함께 bias(편향)가 있을 수 있다:

```
출력 = 입력 x W + b
             ↑     ↑
          가중치   bias
```

| 설정 | 의미 |
|------|------|
| `"none"` | bias 학습 안 함 (권장) |
| `"all"` | 모든 bias 학습 |
| `"lora_only"` | LoRA 붙은 레이어의 bias만 학습 |

`"none"` 권장 이유: bias는 가중치 대비 0.06%밖에 안 되어 학습 효과가 미미하고, 오히려 학습을 불안정하게 만들 수 있다.

---

### task_type

```yaml
task_type: "CAUSAL_LM"
```

LoRA 어댑터를 만들 때 **어떤 구조의 모델에 붙일 것인지** 알려주는 설정이다.

```
CAUSAL_LM:    왼쪽에서 오른쪽으로만 예측 (GPT, Qwen, Llama 계열)
              각 토큰은 과거(왼쪽)만 볼 수 있음

SEQ2SEQ_LM:   입력 전체를 보고 출력 생성 (T5, BART 계열)
              번역처럼 입력과 출력이 분리된 구조

TOKEN_CLS:    토큰 분류 (개체명 인식)
SEQ_CLS:      문장 분류 (감성 분석)
```

Qwen2.5는 GPT 계열이므로 `CAUSAL_LM` 고정이다. 이 설정은 PEFT(Parameter-Efficient Fine-Tuning) 라이브러리에 전달되며, `from peft import LoraConfig`에서 사용된다.

---

### gradient_checkpointing

```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
```

학습 중 **중간 계산값(activation)을 메모리에서 버리고, 필요할 때 다시 계산**하는 기법이다. **메모리를 시간으로 교환**하는 것이다.

#### 왜 필요한가

학습은 Forward(순전파) → Backward(역전파) 순서로 진행된다. Backward에서 gradient를 계산하려면 Forward의 중간 계산값(activation)이 필요하므로 전부 VRAM에 저장해둬야 한다.

```
[Checkpointing OFF] 28층의 activation 전부 저장
  28개 x ~70MB = ~1,960 MB

[Checkpointing ON] 체크포인트 층만 저장 (예: 4층마다)
  7~8개 x ~70MB = ~490-560 MB
  버린 activation은 역전파 시 가장 가까운 체크포인트에서 다시 Forward로 재계산
```

| | Checkpointing OFF | Checkpointing ON |
|---|-------------------|------------------|
| Activation 메모리 | ~1,800 MB | ~600 MB |
| VRAM 절약 | - | **~40%** |
| 학습 속도 | 기준 | 20~30% 느림 |

#### use_reentrant: false

PyTorch 2.0+ 신규 구현 방식이다. 레거시 방식(true)은 QLoRA처럼 일부만 `requires_grad=True`인 경우 gradient가 누락될 수 있는 버그가 있다. **QLoRA와 함께 쓸 때 false는 필수**이다.

---

## 3. training_config.yaml

### 배치 설정

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

**배치(Batch) = 한 번에 묶어서 학습하는 데이터 개수**이다.

배치가 크면 여러 데이터의 평균을 보고 가중치를 수정하므로 안정적이지만, 동시에 VRAM에 올려야 하는 데이터가 늘어난다. RTX 3060 6GB에서 batch_size를 2 이상으로 올리면 OOM이 발생할 수 있다.

**gradient_accumulation_steps**는 VRAM에는 1개만 올리면서 큰 배치와 같은 효과를 내는 기법이다:

```
batch_size=1, gradient_accumulation_steps=8:

  데이터1 Forward → gradient 누적 (업데이트 안 함)
  데이터2 Forward → gradient 누적
  ...
  데이터8 Forward → gradient 누적 → 8개 평균으로 업데이트!

실효 배치 크기 = 1 x 8 = 8
  → VRAM은 1개 분량만 사용하면서 8개를 본 것과 같은 안정적인 학습
```

---

### 학습률 설정

```yaml
training:
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.01
```

#### learning_rate: 2e-4

**가중치를 한 번에 얼마나 수정할지** 정하는 값이다.

```
가중치 업데이트: W_new = W_old - lr x gradient

lr = 2e-4 = 0.0002 → 한 스텝에 아주 살짝만 수정
```

```
LoRA의 적정 범위: 1e-4 ~ 3e-4

  1e-4:  보수적 (안정적이지만 느림)
  2e-4:  가장 많이 쓰는 값 ← 현재 설정
  3e-4:  약간 공격적
```

LoRA가 Full Fine-Tuning보다 학습률이 높은 이유: 전체 15억 개가 아니라 923만 개만 수정하므로 작은 변화의 영향이 제한적이어서 과감하게 설정할 수 있다.

#### lr_scheduler_type: "cosine"

학습률을 처음부터 끝까지 같은 값으로 쓰지 않고, **cosine 곡선으로 점점 줄여나가는** 방식이다.

```
학습률
  |     2e-4 ●●●── 최대 학습률
  |        ●      ●●
  |      ●            ●●
  |    ●                  ●●●
  |  ●  warmup               ●●●●
  |●  (5%)                        ●●●●●●●
  |                                      ●●●
  └──────────────────────────────────────────→ 학습 진행
   0%                                   100%
```

초반에는 천천히 줄어서 충분히 학습하고, 후반에는 빠르게 줄어서 정밀하게 수렴한다.

#### warmup_ratio: 0.05

학습 시작 시 학습률을 **0에서 서서히 올리는** 구간이다. 전체 스텝의 5%를 워밍업에 사용한다.

학습 시작 직후에는 gradient 방향이 불안정한데, 바로 최대 학습률을 적용하면 loss가 폭발할 수 있다. 워밍업으로 gradient가 안정될 때까지 조심스럽게 진행한다.

#### weight_decay: 0.01

가중치가 **너무 커지지 않도록 매 스텝마다 가중치를 미세하게 줄이는** 정규화 기법이다.

```
W_new = W_old x (1 - lr x weight_decay) - lr x gradient
        ──────────────────────────────
        매 스텝마다 가중치를 0.01%씩 줄임
```

불필요하게 큰 가중치를 억제하여 과적합을 방지하고 일반화 성능을 향상시킨다.

---

### 에폭 설정

```yaml
training:
  num_train_epochs: 10
  max_steps: -1
```

- **epoch**: 전체 데이터를 한 바퀴 도는 것
- **step**: 가중치를 한 번 업데이트하는 것
- **max_steps: -1**: epoch 기반으로 학습 (max_steps 미사용)
- **max_steps: 100**: epoch과 무관하게 100 스텝에서 무조건 종료 (빠른 테스트용, num_train_epochs보다 우선)

```
데이터 100개, batch_size=1, gradient_accumulation=8 일 때:
  1 epoch = 100 / 8 = 12 스텝
  10 epochs = 120 스텝
```

---

### fp16 vs bf16

```yaml
training:
  fp16: true
  bf16: false
```

둘 다 16비트(2바이트)지만, **16비트를 지수와 가수에 배분하는 방식**이 다르다.

```
fp16:  1비트(부호) + 5비트(지수)  + 10비트(가수)  → 정밀도 우선
bf16:  1비트(부호) + 8비트(지수)  + 7비트(가수)   → 범위 우선
```

지수 비트는 "2의 몇 제곱"을 저장한다. 비트가 조금만 늘어도 **2의 거듭제곱이므로 범위가 기하급수적으로 커진다**:

```
fp16: 지수 5비트 → 2^15  = 65,504         범위 좁음
bf16: 지수 8비트 → 2^127 = 3.4 x 10^38   범위 넓음 (fp32와 동일)
```

가수 비트는 정밀도를 결정한다. `정밀도 = log10(2^가수비트)`:

```
fp16: 가수 10비트 → 2^10 = 1,024가지 구분 → ~3.3자리 정밀도
bf16: 가수 7비트  → 2^7  = 128가지 구분   → ~2.4자리 정밀도
```

| | fp16 | bf16 |
|---|------|------|
| 숫자 범위 | 65,504 (좁음) | 3.4 x 10^38 (넓음) |
| 정밀도 | ~3.3자리 (좋음) | ~2.4자리 (낮음) |
| overflow 위험 | 있음 (loss scaling으로 해결) | 없음 |
| **RTX 3060 하드웨어 가속** | **지원** | **미지원** |

**RTX 3060에서는 fp16이 정답**이다. bf16은 하드웨어 가속을 못 받아 더 느리고, 정밀도도 낮아서 이점이 없다. RTX 4090이나 A100 이상의 GPU에서는 bf16이 overflow 걱정 없이 안정적이다.

---

### 옵티마이저 (paged_adamw_8bit)

```yaml
training:
  optim: "paged_adamw_8bit"
```

이름이 세 가지 개념의 조합이다:

```
paged_adamw_8bit
  |     |    |
  |     |    └── 8-bit: optimizer state를 8비트로 압축
  |     └─────── AdamW: optimizer 알고리즘
  └───────────── paged: VRAM 부족 시 CPU RAM으로 넘기는 기능
```

#### AdamW란

가장 단순한 옵티마이저(SGD)는 gradient 방향으로 바로 이동하므로 불안정하다. Adam은 **2가지 기억**을 유지한다:

```
1. Momentum (관성):  gradient의 이동 평균 → "최근에 주로 어느 방향으로 갔는가"
2. Variance (분산):  gradient 제곱의 이동 평균 → "이 방향이 얼마나 흔들리는가"

W_new = W_old - lr x momentum / (sqrt(variance) + epsilon)
                      ────────   ──────────
                      관성 방향    흔들림 큰 방향은 축소
```

AdamW는 Adam에서 weight decay를 gradient와 분리하여 적용하는 개선 버전이다.

#### 8-bit

Adam은 파라미터마다 momentum과 variance를 저장해야 한다. 일반적으로 float32(4바이트 x 2 = 8바이트/파라미터)이지만, **8-bit으로 줄이면 2바이트/파라미터**가 된다.

```
LoRA 파라미터 923만 개 기준:
  일반 Adam (32-bit): 9.23M x 8바이트 = 73.8 MB
  8-bit Adam:         9.23M x 2바이트 = 18.5 MB  (55 MB 절약)
```

optimizer state는 "대략적인 경향"만 기억하면 되므로 8-bit으로도 학습 품질 차이가 거의 없다.

#### Paged

VRAM이 부족한 순간에 optimizer state를 **일시적으로 CPU RAM으로 옮기고**, 피크가 지나면 다시 복귀시킨다. 약간 느려지지만 OOM을 방지한다.

---

### TensorBoard 로깅

```yaml
training:
  logging_steps: 5
  logging_dir: "logs"
  report_to: "tensorboard"
```

**TensorBoard**는 학습 과정을 실시간 그래프로 보여주는 모니터링 도구이다.

```bash
# 별도 터미널에서 실행
tensorboard --logdir logs
# 브라우저에서 http://localhost:6006 접속
```

주요 그래프와 판단 방법:

| 그래프 | 의미 | 보는 법 |
|--------|------|---------|
| train/loss | 학습 데이터의 loss | 꾸준히 내려가면 정상 |
| eval/loss | 검증 데이터의 loss | train은 내리는데 이것만 올라가면 과적합 |
| learning_rate | 현재 학습률 | cosine 스케줄러 곡선이 보임 |

```
정상:     train_loss ↘  eval_loss ↘  (둘 다 하락)
과적합:   train_loss ↘  eval_loss ↗  (검증만 상승 → Early Stopping이 잡아줌)
학습 실패: train_loss →  또는 ↗      (학습률이 너무 크거나 데이터 오류)
```

---

### Early Stopping (조기 종료)

```yaml
training:
  early_stopping_patience: 3
  early_stopping_threshold: 0.01
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  load_best_model_at_end: true
```

#### patience: 3

eval_loss가 **3 에폭 동안 의미 있게 개선되지 않으면** 학습을 종료한다.

#### threshold: 0.01

**"개선"으로 인정하는 최소 변화량**이다. threshold 미만의 변화는 "의미 없는 개선"으로 취급하여 patience 카운트를 올린다. **학습 자체를 막는 것이 아니라, patience만 증가시킨다.**

```
epoch 5:  eval_loss = 1.50   ← 최고 기록
epoch 6:  eval_loss = 1.497  ← 변화 0.003 < 0.01 → patience 1/3 (학습 계속)
epoch 7:  eval_loss = 1.493  ← 변화 0.004 < 0.01 → patience 2/3 (학습 계속)
epoch 8:  eval_loss = 1.40   ← 변화 0.10  >= 0.01 → 개선 인정! patience 리셋
epoch 9:  eval_loss = 1.395  ← 변화 0.005 < 0.01 → patience 1/3
epoch 10: eval_loss = 1.391  ← 변화 0.004 < 0.01 → patience 2/3
epoch 11: eval_loss = 1.388  ← 변화 0.003 < 0.01 → patience 3/3 → 종료!
```

threshold가 없으면(=0) 0.0001 같은 미세한 변화에도 patience가 리셋되어 실질적으로 개선이 없는데도 계속 학습하게 된다.

#### metric_for_best_model: "eval_loss"

최적 모델을 판단하는 기준이다. eval_loss 외에 커스텀 지표도 사용 가능하다:

```python
# compute_metrics 함수를 정의하면 커스텀 지표 사용 가능
def compute_metrics(eval_pred):
    return {
        "tool_name_accuracy": 0.85,
        "parameter_accuracy": 0.72,
    }
```

| metric | 설명 | greater_is_better |
|--------|------|-------------------|
| `eval_loss` | 검증 loss (현재 설정) | false (낮을수록 좋음) |
| `tool_name_accuracy` | 도구 호출 정확도 (커스텀) | true (높을수록 좋음) |

현재 단계에서는 eval_loss가 적합하다. 추가 코드 없이 자동 계산되고, tool calling 정확도와 높은 상관관계가 있다.

---

### Windows 호환성

```yaml
training:
  dataloader_pin_memory: false
  dataloader_num_workers: 0
```

#### dataloader_num_workers: 0

DataLoader가 데이터를 읽는 **병렬 워커(일꾼) 수**이다.

```
num_workers=0: 메인 프로세스가 직접 읽음 (순차)
num_workers=4: 별도 프로세스 4개가 미리 읽어둠 (병렬, Linux에서 효율적)
```

Windows에서 0으로 해야 하는 이유: Windows의 멀티프로세싱(spawn)은 새 프로세스를 처음부터 만들어서 토크나이저/CUDA를 다시 로딩하므로 느리고 메모리를 많이 쓰며 간헐적으로 충돌한다.

0이어도 괜찮은 이유: 데이터 읽기(~1ms)가 GPU 연산(~500ms) 대비 0.4%밖에 안 되므로 GPU가 거의 대기하지 않는다.

#### dataloader_pin_memory: false

CPU RAM → VRAM 전송 시 **고정 메모리(pinned memory)**를 쓸지 여부이다.

```
pin_memory=false: CPU RAM → 임시 복사 → VRAM (2단계)
pin_memory=true:  CPU RAM(pinned) → VRAM 직접 전송 (1단계, 약간 빠름)
```

Windows에서 false로 해야 하는 이유:
- Windows CUDA 드라이버의 pinned memory 할당이 특정 상황에서 실패
- 프로세스당 pinned memory 제한 초과 시 시스템 불안정
- pin_memory의 속도 개선 효과가 전체 학습에서 ~0.08%로 무시할 수 있는 수준

---

### SFT 설정 (packing)

```yaml
sft:
  packing: false
  max_seq_length: 2048
```

#### packing: false

**여러 개의 짧은 대화를 하나의 시퀀스에 이어붙여서 패딩을 줄이는 기법**을 비활성화한다.

```
[packing = false] 각 대화를 개별 시퀀스로 (현재 설정)

시퀀스 1: [대화A 350토큰][패딩 1698토큰]   활용률 17%
시퀀스 2: [대화B 520토큰][패딩 1528토큰]   활용률 25%
→ 패딩 낭비가 크지만, 각 대화가 완전히 분리됨


[packing = true] 여러 대화를 이어붙임

시퀀스 1: [대화A 350][대화B 520][대화C 280][패딩 898]  활용률 56%
→ 효율적이지만, 대화 경계가 오염될 수 있음
```

packing을 끄는 이유: **대화 경계 오염** 때문이다.

```
packing 시 시퀀스 내부:
  ...대화A의 tool 응답...
  {"NODE": {"1": {"X": 0, "Y": 0, "Z": 0}}}
  <|im_end|>
  <|im_start|>user          ← 대화B 시작인데
  재료 SS400을 추가해줘       모델은 "위 tool 응답을 본 후
  <|im_end|>                 재료 추가를 요청한 것"으로 학습

  → tool 응답과 다음 요청 사이의 잘못된 인과관계 학습
  → 추론 시 엉뚱한 도구를 호출할 수 있음
```

| | packing = true | packing = false |
|---|---------------|-----------------|
| GPU 활용률 | ~70-80% | ~20-30% |
| 학습 속도 | 3~4배 빠름 | 느림 |
| 대화 경계 | 오염 가능 | 깨끗하게 분리 |
| 적합한 경우 | 일반 텍스트, 대량 데이터 | **tool calling, 구조화된 데이터** |

tool calling처럼 "어떤 도구를 호출할지"가 대화 맥락에 민감한 태스크에서는 **packing을 끄는 것이 정확도를 위해 올바른 선택**이다.

---

## 설정 변경 가이드

### VRAM이 부족할 때 (OOM)

```yaml
# config/lora_config.yaml — Attention만 학습
lora:
  target_modules:
    - "q_proj"
    - "v_proj"

# config/model_config.yaml — 시퀀스 길이 축소
model:
  max_seq_length: 1024

# config/training_config.yaml — gradient 누적 강화
training:
  gradient_accumulation_steps: 16
```

### 학습이 불안정할 때 (loss가 튐)

```yaml
# config/training_config.yaml
training:
  learning_rate: 1.0e-4    # 줄이기
  warmup_ratio: 0.1        # 워밍업 늘리기
  weight_decay: 0.05       # 정규화 강화
```

### 빠르게 테스트할 때

```yaml
# config/training_config.yaml
training:
  num_train_epochs: 2
  max_steps: 100           # 100 스텝만 돌려서 확인
  logging_steps: 1
  save_strategy: "no"
  eval_strategy: "no"
```

### API 도구 270개로 확장할 때

```yaml
# config/lora_config.yaml — rank 올리기
lora:
  r: 16
  lora_alpha: 32

# config/model_config.yaml — 필요 시 시퀀스 길이 확장
model:
  max_seq_length: 4096
```
