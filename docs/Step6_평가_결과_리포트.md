# Step 6: 평가 결과 리포트

## 개요

| 항목 | 값 |
|------|-----|
| 모델 | Qwen2.5-1.5B-Instruct + QLoRA Adapter (Epoch 4) |
| 어댑터 | `models/checkpoints/final_adapter/` (36.9 MB) |
| 양자화 | 4-bit NF4 + Double Quantization |
| 테스트 데이터 | `data/processed/test.jsonl` (358건) |
| 평가 환경 | RTX 3060 Laptop 6GB, Windows 11, CUDA 12.8 |
| 평가 일시 | 2026-03-12 |

---

## 평가 결과

### Tool-Calling 메트릭

| 메트릭 | 값 | 설명 |
|--------|-----|------|
| **Tool Name Accuracy** | **96.65%** | 예측한 API 이름이 정답과 일치하는 비율 |
| **Parameter Accuracy** | **16.18%** | 이름이 일치하는 tool call의 파라미터 키-값 일치율 |
| **JSON Validity Rate** | **95.95%** | 모델이 생성한 arguments가 유효한 JSON인 비율 |
| **Hallucination Rate** | **0.00%** | 존재하지 않는 tool을 호출한 비율 |

### 성능 메트릭

| 메트릭 | 값 |
|--------|-----|
| Latency (Mean) | 33,281.0 ms |
| Latency (P50) | 5,755.7 ms |
| Latency (P95) | 65,709.4 ms |
| VRAM 사용량 | 3,392.7 MB / 6,144.0 MB (55.2%) |

---

## 메트릭별 상세 분석

### Tool Name Accuracy: 96.65%

- 358건 중 346건에서 올바른 API를 선택
- 242종의 서로 다른 API에 대해 높은 정확도
- 1.5B 파라미터 모델로서 우수한 tool 선택 능력

### Parameter Accuracy: 16.18%

상대적으로 낮은 수치이며, 샘플 분석 결과 주요 원인은 다음과 같다:

| 원인 | 설명 | 예시 |
|------|------|------|
| **값 수준 불일치** | 키 구조는 유사하나 구체적인 값이 다름 | `"SECT_NAME": "PSC-I-Composite"` vs `"PSC-I-701"` |
| **불필요한 wrapper 키** | Feature 설명 텍스트를 JSON 키로 추가 | `{"Smart Graph (...)": {"Assign": {...}}}` vs `{"Assign": {...}}` |
| **GET 요청 인자 누락** | 조회 대상 ID를 빈 객체로 예측 | `{}` vs `{"Assign": {"1":{}, "2":{}}}` |

- 키 구조(Assign, SECTTYPE 등)는 대체로 정확하게 생성
- 구체적인 수치/이름 값은 사용자 질문에서 유추해야 하므로 불일치 발생
- 학습 데이터의 양과 다양성 확대로 개선 가능

### JSON Validity Rate: 95.95%

- 358건 중 346건에서 유효한 JSON arguments 생성
- 4.05%의 invalid JSON은 긴 arguments에서 max_new_tokens(512) 제한으로 잘린 경우로 추정

### Hallucination Rate: 0.00%

- 모델이 존재하지 않는 tool을 호출한 경우 없음
- 242종 API에 대해 hallucination 없이 정확한 tool name 생성

---

## Latency 분석

| 구간 | 값 | 비고 |
|------|-----|------|
| P50 | 5.8초 | 절반의 요청이 이 시간 내 완료 |
| Mean | 33.3초 | 일부 긴 프롬프트가 평균을 크게 올림 |
| P95 | 65.7초 | 상위 5%의 요청에서 매우 긴 생성 시간 |

- P50과 Mean의 큰 차이는 프롬프트 길이 편차가 크기 때문
- 긴 tool schema를 포함하는 프롬프트에서 latency가 급증
- 4-bit 양자화로 VRAM은 55%만 사용하여 6GB GPU에서 안정적 동작

---

## 종합 평가

### 강점
- **Tool 선택 능력 우수**: 96.65% 정확도로 올바른 API 선택
- **Hallucination 없음**: 존재하지 않는 API를 호출하지 않음
- **JSON 형식 안정**: 95.95% 유효한 JSON 생성
- **효율적인 리소스 사용**: 6GB VRAM의 55%만 사용

### 개선 필요 영역
- **Parameter Accuracy**: 16.18%로 실제 서비스 적용에는 부족
  - 학습 데이터 양 확대 (현재 2,856건 → 목표 10,000건+)
  - 파라미터 값에 대한 더 구체적인 few-shot 예시 추가
  - arguments 이중 string 인코딩 파싱 개선
- **Latency**: P95 65초로 일부 요청에서 느림
  - 프롬프트 길이 최적화 (tool schema 압축)
  - 또는 vLLM 등 추론 엔진 도입 검토
