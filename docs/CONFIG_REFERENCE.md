# 설정 파일 상세 레퍼런스

> `config/` 디렉토리의 YAML 파일 각 항목에 대한 설명입니다.

---

## config/model_config.yaml

### model (모델 기본 설정)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `name` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face 모델 ID 또는 로컬 경로 |
| `local_path` | `models/base/Qwen2.5-1.5B-Instruct` | 로컬 다운로드 경로 |
| `max_seq_length` | `2048` | 최대 시퀀스 길이 (토큰 수). 줄이면 VRAM 절약, 늘이면 긴 대화 가능 |
| `dtype` | `float16` | 모델 연산 정밀도 |

### tokens (특수 토큰)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `eos_token` | `<\|im_end\|>` | 문장 끝 토큰. Qwen2.5 필수 설정 |
| `pad_token` | `<\|endoftext\|>` | 패딩 토큰 |
| `chat_template` | `qwen` | 채팅 템플릿 (Qwen 내장 템플릿 사용) |

### quantization (양자화 설정)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `load_in_4bit` | `true` | 4-bit 양자화 활성화 |
| `bnb_4bit_quant_type` | `nf4` | 양자화 타입. NF4가 QLoRA 논문 권장 |
| `bnb_4bit_compute_dtype` | `float16` | 연산 시 사용할 정밀도 |
| `bnb_4bit_use_double_quant` | `true` | 이중 양자화로 추가 메모리 절약 |

### environment (환경 변수)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `pytorch_cuda_alloc_conf` | `expandable_segments:True` | GPU 메모리 단편화 방지 |
| `tokenizers_parallelism` | `false` | Windows에서 토크나이저 deadlock 방지 |

---

## config/lora_config.yaml

### lora (기본 LoRA 설정)

| 항목 | 기본값 | 설명 | 조절 가이드 |
|------|--------|------|------------|
| `r` | `8` | LoRA rank. 학습 가능한 파라미터 수 결정 | 4→작고 빠름, 16→크고 표현력 높음 |
| `lora_alpha` | `16` | LoRA 스케일링 계수. 보통 r의 2배 | r과 같이 조절 |
| `lora_dropout` | `0.05` | 드롭아웃 비율 | 과적합 시 0.1로 올려볼 것 |
| `target_modules` | `all-linear` | LoRA를 적용할 레이어 | 모든 선형 레이어에 적용 |
| `bias` | `none` | bias 학습 여부 | none 권장 |
| `task_type` | `CAUSAL_LM` | 태스크 타입 | 변경 불필요 |

### lora_fallback (OOM 시 대안)

6GB VRAM에서 OOM이 발생하면 이 설정으로 교체하세요.
`target_modules`를 `["q_proj", "v_proj"]`로 바꿔서 attention 레이어만 학습합니다.
VRAM 사용량이 크게 줄어들지만, 학습 표현력도 줄어듭니다.

### gradient_checkpointing

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `gradient_checkpointing` | `true` | 활성화 시 VRAM ~40% 절약 (속도는 약간 느려짐) |
| `use_reentrant` | `false` | PyTorch 호환성 설정 (false 권장) |

---

## config/training_config.yaml

### training (학습 하이퍼파라미터)

#### 배치 설정

| 항목 | 기본값 | 설명 | 조절 가이드 |
|------|--------|------|------------|
| `per_device_train_batch_size` | `1` | GPU당 배치 크기 | 6GB에서는 1 고정 |
| `per_device_eval_batch_size` | `1` | 평가 배치 크기 | 1 고정 |
| `gradient_accumulation_steps` | `8` | 그래디언트 누적 | 실효 배치 = batch_size x 이 값 |

> 실효 배치 크기 = 1 x 8 = **8** (메모리 제약 없이 큰 배치 효과)

#### 학습률 설정

| 항목 | 기본값 | 설명 | 조절 가이드 |
|------|--------|------|------------|
| `learning_rate` | `2e-4` | 학습률 | LoRA의 일반적 범위: 1e-4 ~ 3e-4 |
| `lr_scheduler_type` | `cosine` | 학습률 스케줄러 | cosine이 안정적 |
| `warmup_ratio` | `0.05` | 워밍업 비율 | 전체 스텝의 5%를 워밍업 |
| `weight_decay` | `0.01` | 가중치 감쇠 | 과적합 방지용 |

#### 에폭 설정

| 항목 | 기본값 | 설명 | 조절 가이드 |
|------|--------|------|------------|
| `num_train_epochs` | `10` | 최대 학습 에폭 수 | Early stopping이 있으므로 넉넉하게 |
| `max_steps` | `-1` | 최대 스텝 (-1이면 epoch 기반) | 빠른 테스트시 100 등으로 설정 |

#### 정밀도

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `fp16` | `true` | 16-bit 부동소수점 사용 |
| `bf16` | `false` | bfloat16 비활성화 (RTX 3060은 fp16이 더 효율적) |

#### 옵티마이저

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `optim` | `paged_adamw_8bit` | 8-bit 페이징 AdamW. VRAM 절약 |

#### 저장 및 평가

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `save_strategy` | `epoch` | 에폭마다 체크포인트 저장 |
| `save_total_limit` | `3` | 최근 3개 체크포인트만 유지 (디스크 절약) |
| `eval_strategy` | `epoch` | 에폭마다 평가 실행 |
| `logging_steps` | `5` | 5 스텝마다 로그 출력 |
| `logging_dir` | `logs` | TensorBoard 로그 디렉토리 |
| `report_to` | `tensorboard` | 로깅 대상 |

#### Early Stopping (조기 종료)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `early_stopping_patience` | `3` | eval_loss가 3 에폭 동안 개선 안 되면 종료 |
| `early_stopping_threshold` | `0.01` | 최소 개선 기준값 |
| `metric_for_best_model` | `eval_loss` | 최적 모델 판단 기준 |
| `greater_is_better` | `false` | loss는 낮을수록 좋으므로 false |
| `load_best_model_at_end` | `true` | 학습 완료 시 최적 체크포인트 로딩 |

#### Windows 호환성

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `dataloader_pin_memory` | `false` | Windows CUDA 호환성 |
| `dataloader_num_workers` | `0` | Windows 멀티프로세싱 이슈 방지 |

### sft (SFT 학습 설정)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `packing` | `false` | 데이터 패킹 비활성화 (대화 경계 오염 방지) |
| `max_seq_length` | `2048` | 최대 시퀀스 길이 |

---

## 설정 변경 예시

### VRAM이 부족할 때

```yaml
# config/lora_config.yaml
lora:
  target_modules:
    - "q_proj"
    - "v_proj"

# config/model_config.yaml
model:
  max_seq_length: 1024

# config/training_config.yaml
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
  logging_steps: 1
  save_strategy: "no"
  eval_strategy: "no"
```
