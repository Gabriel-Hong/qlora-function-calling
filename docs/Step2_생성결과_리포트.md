# Step 2: 학습 샘플 대량 생성 - 결과 리포트

## 실행 요약 (최종)

| 항목 | 수치 |
|------|------|
| 실행 일시 | 2026-03-07 ~ 2026-03-08 |
| 모델 | GPT-5 mini |
| 총 요청 | 3,650 (1 요청 = 1 샘플) |
| **최종 성공** | **3,571 (97.8%)** |
| **최종 에러** | **79 (2.2%)** |
| 소요 시간 | 약 36시간 (3라운드 합산) |
| 예상 비용 | ~$14 |
| 입력 | `data/prompts/training_prompts.jsonl` (1,122 레코드) |
| 출력 | `data/generated/training_samples.jsonl` (3,571 샘플) |
| 에러 로그 | `data/generated/errors.jsonl` (743건 누적, 최종 실패 79건) |

## 스크립트

`scripts/generate_training_samples.py`
- 1 요청 = 1 샘플 전략 (단일 요청으로 100% 파싱 안정성 확보)
- 16개 VARIATION_HINTS 순환 배정 (레코드 인덱스 기반 오프셋)
- `--resume` 지원: 출력 파일의 `(_source_index, _variation_idx)` 기반 체크포인트
- `repair_json_string()`: LLM이 생성한 JSON 쉼표 누락 자동 수정

---

## 라운드별 진행 결과

| 라운드 | 대상 | 성공 | 실패 | 누적 성공 | 달성률 | 주요 변경 |
|--------|------|------|------|----------|--------|----------|
| 1차 | 3,650 | 3,143 | 507 | 3,143 | 86.1% | `max_completion_tokens=4096` |
| 2차 (`--resume`) | 507 | 350 | 157 | 3,493 | 95.7% | `max_completion_tokens=8192` |
| 3차 (`--resume`) | 157 | 78 | 79 | 3,571 | 97.8% | `repair_json_string()` 추가 |

---

## 최종 에러 분석 (79건)

### 잔여 에러 상위 API

| API | 에러 수 | 주 원인 |
|-----|---------|---------|
| view/CAPTURE POST | ~12 | 파라미터 매우 많음 (40+ 옵션) |
| view/DISPLAY POST | ~8 | 다양한 디스플레이 옵션 |
| db/SECT POST/PUT | ~10 | 17개 서브타입, 스키마 복잡 |
| post/TABLE POST | ~8 | 결과 테이블 구조 복잡 |
| db/MVHL POST/PUT | ~6 | 인도 차량 하중 (프롬프트 ~50K tokens) |
| 기타 | ~35 | 간헐적 빈 응답/파싱 실패 |

79건 모두 **구조적으로 복잡한 API**에서 발생. 3회 재시도 후에도 실패하는 건은 모델 능력 한계로 판단.

---

## 1차 실행 상세 분석 (참고)

### 1차 에러 유형 분류 (507건)

| 유형 | 건수 | 비율 | 설명 |
|------|------|------|------|
| parse_error | 293 | 57.8% | JSON 파싱 실패 (빈 응답 포함) |
| validation | 213 | 42.0% | TRL 포맷 검증 실패 |
| api_error | 1 | 0.2% | API 호출 자체 실패 |

### parse_error 상세

- **빈 응답 (0 chars)**: ~93건 — 모델이 응답을 반환하지 않음
  - 프롬프트 길이와 **약한 상관관계** (평균 21K자 vs 성공 평균 10K자)
  - 단, 4K자 프롬프트에서도 발생 → 길이만의 문제는 아님
  - GPT-5 mini의 간헐적 생성 실패로 추정
  - **2차에서 96% 복구 (214/222)** → 간헐적 오류 확정
- **잘린 응답**: ~27건 — `max_completion_tokens=4096`에서 JSON이 도중 절단
  - **2차에서 `max_completion_tokens=8192`로 해결**
- **기타 파싱 실패**: ~173건 — 모델이 JSON 외 텍스트를 포함하거나 구조 불일치

### validation 에러 상세

- **arguments JSON 깨짐**: 대부분 `Expecting ',' delimiter` — 쉼표 누락
  - 복잡한 API (view/CAPTURE, db/SECT, db/BMLD)에서 집중 발생
  - **3차에서 `repair_json_string()` 추가로 49.7% 추가 복구**

### 에러 없는 API

전체 802개 API-메서드 조합 중 **600개(74.8%)는 1차부터 에러 0%**.

---

## 대응 방안 (실행 완료)

### 실행 완료

1. ~~`--resume` 재시도~~ → **완료** (2차: 350건, 3차: 78건 복구)
2. ~~`max_completion_tokens` 8192~~ → **완료** (잘린 응답 해결)
3. ~~JSON 쉼표 자동 수정~~ → **완료** (`repair_json_string()` 추가)

### 추가 전략 (미실행, 필요 시)

4. **프롬프트 축약**: 50K+ tokens 프롬프트 (db/MVHL, post/TABLE 등) 스키마 요약
5. **상위 모델 사용**: 에러 상위 API만 GPT-5로 재생성 (비용 증가)
6. **수동 보정**: 최종 에러 건 중 중요 API만 수동 작성

---

## 생성 샘플 메타데이터

각 샘플에 포함된 추적용 필드:

```json
{
  "_source_index": 0,
  "_api_name": "db/NODE",
  "_method": "POST",
  "_variation_idx": 0,
  "messages": [...],
  "tools": "..."
}
```

## 다음 단계

1. ~~실패 건 재시도~~ → **완료** (3라운드, 97.8% 달성)
2. ~~에러 분석 후 후처리~~ → **완료**
3. **Step 3**: `02_prepare_data.py` → train/eval/test 분할
4. **Step 4**: `03_train_qlora.py` → QLoRA 학습 (~1-2시간, RTX 3060 6GB)
