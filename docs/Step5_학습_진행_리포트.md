# Step 5: QLoRA 학습 완료 리포트

## 개요

| 항목 | 값 |
|------|-----|
| 모델 | Qwen2.5-1.5B-Instruct |
| 학습 방식 | QLoRA (4-bit NF4 + LoRA r=8) |
| 학습 데이터 | 2,856 samples (train), 357 samples (eval) |
| GPU | NVIDIA RTX 3060 Laptop 6GB |
| 총 학습 시간 | **약 17시간** (1차 ~11시간 + 2차 ~5시간 52분) |
| 최종 Epoch | 8 / 10 (Early Stopping) |
| 최종 Step | 2,856 / 3,570 |
| Best Eval Loss | **0.3956** (Epoch 4) |
| Peak VRAM | 6,040 MiB / 6,144 MiB (98.3%) |

## 학습 실행 이력

| 구분 | 1차 학습 | 2차 학습 (재개) |
|------|---------|----------------|
| 시작 시각 | 2026-03-08 20:21 | 2026-03-09 21:25 |
| 종료 시각 | 2026-03-09 07:11 (수동 중단) | 2026-03-10 03:17 |
| Epoch 범위 | 1 ~ 5 | 5 ~ 8 (Early Stop) |
| Step 범위 | 0 ~ 1,785 | 1,785 ~ 2,856 |
| 소요 시간 | ~11시간 | 5시간 52분 |
| 재개 방법 | - | `--resume` (checkpoint-1785) |

## 학습 설정

| 파라미터 | 값 |
|---------|-----|
| LoRA r / alpha | 8 / 16 |
| LoRA target | all-linear |
| LoRA dropout | 0.05 |
| Batch size | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Learning rate | 2e-4 (cosine scheduler) |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Precision | bf16 |
| Optimizer | paged_adamw_8bit |
| Max epochs | 10 |
| Early stopping patience | 3 (threshold 0.01) |
| Gradient checkpointing | true |
| Max sequence length | 2048 |
| Packing | false |

## Epoch별 종합 지표

| Epoch | Step | Train Loss | Eval Loss | Eval 변화량 | Token Acc (Train) | Token Acc (Eval) | Grad Norm | LR |
|-------|------|------------|-----------|-------------|-------------------|------------------|-----------|-----|
| 1 | 357 | 0.5614 | 0.5233 | - | 87.3% | 87.8% | 0.064 | 1.99e-4 |
| 2 | 714 | 0.4340 | 0.4394 | -0.0839 | 89.6% | 89.3% | 0.074 | 1.88e-4 |
| 3 | 1,071 | 0.3658 | 0.4065 | -0.0329 | 90.9% | 90.0% | 0.062 | 1.68e-4 |
| **4** | **1,428** | **0.3055** | **0.3956** | **-0.0109** | **92.1%** | **90.3%** | **0.059** | **1.41e-4** |
| 5 | 1,785 | 0.2734 | 0.4017 | +0.0061 | 92.6% | 90.3% | 0.069 | 1.08e-4 |
| 6 | 2,142 | 0.2433 | 0.4146 | +0.0129 | 93.5% | 90.3% | 0.074 | 7.57e-5 |
| 7 | 2,499 | 0.1856 | 0.4425 | +0.0279 | 95.1% | 90.2% | 0.081 | 4.58e-5 |
| 8 | 2,856 | 0.1430 | 0.4751 | +0.0326 | 96.3% | 90.1% | 0.070 | 2.12e-5 |

## Train Loss vs Eval Loss 분석

| Epoch | Train Loss | Eval Loss | Gap | 판단 |
|-------|-----------|-----------|-----|------|
| 1 | 0.5614 | 0.5233 | -0.04 | 정상 (eval이 낮음) |
| 2 | 0.4340 | 0.4394 | +0.01 | 매우 건강 |
| 3 | 0.3658 | 0.4065 | +0.04 | 정상 |
| 4 | 0.3055 | 0.3956 | +0.09 | 주의 |
| 5 | 0.2734 | 0.4017 | +0.13 | 과적합 시작 |
| 6 | 0.2433 | 0.4146 | +0.17 | 과적합 진행 |
| 7 | 0.1856 | 0.4425 | +0.26 | 과적합 심화 |
| 8 | 0.1430 | 0.4751 | +0.33 | 과적합 확정 |

- Epoch 4를 기점으로 train loss는 계속 감소하나 eval loss는 상승 → **전형적인 과적합 패턴**
- Early Stopping이 적절히 작동하여 Epoch 4의 Best 모델을 보존

## Early Stopping 경과

| Epoch | Eval Loss | Best 대비 | Patience 카운터 |
|-------|-----------|----------|----------------|
| 4 | 0.3956 | - | 0 (Best 갱신) |
| 5 | 0.4017 | +0.006 | 1/3 |
| 6 | 0.4146 | +0.019 | 2/3 |
| 7 | 0.4425 | +0.047 | **3/3 → Early Stop 트리거** |
| 8 | 0.4751 | +0.080 | 종료 (마지막 epoch 실행) |

`early_stopping_threshold: 0.01` 기준, Epoch 5부터 개선 실패 → Epoch 7에서 patience 소진 → Epoch 8 eval 후 학습 종료.

## Learning Rate Schedule (Cosine)

```
LR 2.0e-4 ┤                     ╭─────╮
           │                   ╭╯     ╰╮
           │                 ╭╯        ╰╮
           │               ╭╯           ╰╮
           │             ╭╯              ╰╮
           │           ╭╯                 ╰╮
           │        ╭─╯                    ╰╮
           │     ╭─╯                        ╰─╮
LR 0.0   ─┤──╯                                ╰──
           └──┬────┬────┬────┬────┬────┬────┬────┬──
             E1   E2   E3   E4   E5   E6   E7   E8
```

| Epoch | Learning Rate |
|-------|--------------|
| 1 | 1.99e-4 (peak) |
| 2 | 1.88e-4 |
| 3 | 1.68e-4 |
| 4 | 1.41e-4 |
| 5 | 1.08e-4 |
| 6 | 7.57e-5 |
| 7 | 4.58e-5 |
| 8 | 2.12e-5 |

## Checkpoint 상태

| Checkpoint | Epoch | 저장 시각 | 상태 |
|-----------|-------|----------|------|
| checkpoint-1428 | 4 | 03/09 05:01 | **Best model** |
| checkpoint-2499 | 7 | 03/10 02:28 | 보존 |
| checkpoint-2856 | 8 | 03/10 03:12 | 최종 |
| final_adapter | 4 (best) | 03/10 03:17 | **최종 배포용** |

- `save_total_limit: 3` → 최대 3개 체크포인트 유지
- `load_best_model_at_end: true` → final_adapter에 Best 모델(Epoch 4) 자동 저장
- final_adapter 크기: ~36.9 MB (adapter_model.safetensors)

## 최종 모델 성능 요약

| 지표 | 값 | 평가 |
|------|-----|------|
| Best Eval Loss | 0.3956 | 양호 |
| Eval Token Accuracy | 90.3% | 우수 |
| Train-Eval Gap (Best) | 0.09 | 적정 수준 |
| Final Train Loss | 0.0666 | 수렴 완료 |
| Peak VRAM | 6,040 MiB (98.3%) | 6GB VRAM 한계 활용 |
| 총 학습 시간 | ~17시간 | RTX 3060 Laptop 기준 |
| Total FLOPs | 1.35e+17 | - |

## 분석 및 인사이트

### 잘 된 점
1. **Early Stopping이 적절히 작동**: Epoch 4에서 최적점을 포착하고, 과적합 진행 시 자동 종료
2. **VRAM 효율**: 6GB GPU에서 98.3% 활용하면서 OOM 없이 안정적 학습 완료
3. **Token Accuracy 90%+**: Eval 기준 90.3%로 함수 호출 학습이 잘 진행됨
4. **Checkpoint resume 정상 작동**: 수동 중단 후 `--resume`으로 무결하게 재개

### 개선 가능한 점
1. **과적합이 비교적 빠르게 시작** (Epoch 5~): 데이터 증강이나 dropout 조정 고려
2. **Train-Eval Gap 확대**: 정규화 강화 (weight_decay 증가, LoRA dropout 증가) 가능
3. **Eval Loss 정체**: 0.39대에서 수렴 → 더 큰 모델이나 데이터 추가로 개선 여지

## 산출물

| 파일 | 경로 | 설명 |
|------|------|------|
| Best Adapter | `models/checkpoints/final_adapter/` | Epoch 4 best 모델 (배포용) |
| Adapter weights | `models/checkpoints/final_adapter/adapter_model.safetensors` | 36.9 MB |
| Training logs | `models/checkpoints/runs/` | TensorBoard 로그 |
| Trainer state | `models/checkpoints/checkpoint-2856/trainer_state.json` | 전체 학습 이력 |

## 다음 단계

1. ~~학습 완료 대기~~ (완료)
2. ~~`final_adapter` 저장 확인~~ (완료)
3. **Step 6: 평가** (`scripts/04_evaluate.py`) — tool calling 정확도, 파라미터 매칭 등
4. Step 7: 어댑터 병합 (`scripts/05_merge_adapters.py`)
5. Step 8: 서빙 (`scripts/06_serve.py`)
