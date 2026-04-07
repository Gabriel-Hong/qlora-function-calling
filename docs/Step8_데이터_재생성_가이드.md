# Step 8 — 학습 데이터 재생성 가이드

## 1. 배경 및 진단 요약

Step 5 QLoRA 학습 → Step 6 평가 결과, 모델은 다음과 같은 **비대칭 성능**을 보였습니다.

| 지표 | 결과 | 평가 |
|---|---|---|
| Tool name accuracy | **96.65%** | 양호 |
| Hallucination rate | **0%** | 양호 |
| JSON validity | 95.95% | 양호 |
| **Parameter accuracy** | **16.18%** | **심각** |

모델은 "어떤 API를 호출할지"는 거의 완벽히 학습했지만, **"인자를 어떻게 채울지"는 거의 학습하지 못했습니다.** Qwen2.5-1.5B는 tool name과 스키마 구조를 학습할 만큼 충분한 capacity를 보였으므로, parameter accuracy 부족은 **모델 크기가 아니라 학습 데이터 라벨의 불일관성** 문제로 진단됩니다. 더 큰 모델(예: Gemma 4 26B-A4B)로 교체해도 노이즈 라벨을 그대로 학습할 뿐이므로, **데이터 재생성이 우선 과제**입니다.

관련 문서:
- `docs/Step6_평가_결과_리포트.md`
- `docs/Step6_평가_트러블슈팅.md`
- `docs/Step2_생성결과_리포트.md`

---

## 2. 발견된 문제점

### 문제 1 — 함수명 형식 혼용 (영향 추정 ~20%)

**증상**: 동일 학습셋 안에 두 가지 함수명 표기가 공존합니다.

수기 샘플 (`data/samples/elem_training_samples.jsonl`):
```json
"function":{"name":"post_db_ELEM","arguments":"{\"Assign\":{\"1\":{\"TYPE\":\"BEAM\",...}}}"}
```

생성 샘플 (`data/generated/training_samples.jsonl`):
```json
"function":{"name":"POST /db/node","arguments":"{\"Assign\":{\"1\":{\"X\":0,...}}}"}
```

**원인**: `scripts/build_training_prompts.py`의 few-shot 템플릿이 두 형식을 정리하지 않은 채 GPT-5 mini에 전달, 모델이 자유롭게 생성. 수기 샘플은 snake_case, 생성 샘플은 HTTP 메서드+경로 형식으로 굳어졌고, 학습 시 두 형식이 섞여 들어감.

**영향**: 모델은 "tool name" 자체는 학습하지만, 동일 의미의 두 형식이 다른 토큰 시퀀스로 보이므로 인자 학습 신호가 분산됨.

---

### 문제 2 — Arguments 구조 흔들림 (영향 추정 ~50%, 가장 큼)

**증상 A** — wrapper 키 변동: 같은 API에 대해서도 응답 wrapper가 들쑥날쑥합니다. 위 db/NODE 예시에서 tool 응답은 `{"NODE":{...}}` wrapper를 쓰지만, 다른 API들에서는 wrapper 없이 `{"Assign":{...}}`로 바로 시작하는 경우가 섞여 있습니다.

**증상 B** — GET 요청 인자 누락: GET 메서드인데 조회 대상 ID가 빈 객체(`{}`)로 생성된 케이스가 다수 발견됩니다. 평가 시 모델은 학습한 대로 빈 인자를 생성하지만, 정답은 ID 명시이므로 점수가 0이 됩니다.

**증상 C** — 복잡 API의 생성 실패:
- `view/CAPTURE` (POST): 40+ 옵션 → 프롬프트 50K+ 토큰 → 12건 에러
- `db/SECT` (POST/PUT): 17개 서브타입 → 10건 에러
- `post/TABLE` (POST): 8건 에러
- `db/MVHL` (POST/PUT): 6건 에러

**원인**: GPT-5 mini는 프롬프트가 길어질수록 JSON 출력이 불안정해집니다 (쉼표 누락, 잘림, 키 순서 바뀜). 후처리 `repair_json_string()`이 문법 오류는 복구하지만, **의미적 흔들림(wrapper 추가/누락, 인자 누락)은 복구 불가**.

---

### 문제 3 — Few-shot 예시 일관성 부족 (영향 추정 ~20%)

**증상**: `scripts/build_training_prompts.py`의 `FEW_SHOT_SECTION_TEMPLATE`이 메서드별 분리 없이 예시를 일괄 제공하고, 함수명 표기 규칙을 명시하지 않습니다.

**원인**:
- POST/GET/PUT/DELETE 메서드별로 인자 구성 방식이 다른데(POST는 Assign 객체에 신규 데이터, GET은 조회 키, PUT은 ID+변경 필드, DELETE는 ID), few-shot 예시가 이를 구분하지 않음
- 결과적으로 GPT-5 mini가 GET 메서드에 대해서도 POST 패턴(빈 Assign)을 모방

---

### 문제 4 — 조합별 샘플 분포 불균형 (영향 추정 ~10%)

**증상**: 총 3,571 샘플 / 242 endpoint×method 조합 = **평균 14.8 샘플/조합**.
- 단순 API(db/NODE 등): 256+ 샘플 (16 variation × 16 hint)
- 복잡 API(db/SECT, view/CAPTURE 등): 한 자릿수
- test 분할 358개는 조합당 평균 1.5개 → 통계적 유의성 부족

**원인**: 모든 API에 동일한 16 VARIATION_HINTS를 적용했지만, 복잡 API는 생성 실패율이 높아 실제 살아남는 샘플 수가 적음.

---

## 3. 데이터 재생성 원칙 (체크리스트)

재생성된 모든 샘플은 다음을 만족해야 합니다.

- [ ] **함수명 형식 통일**: `{method}_{path_segments}` snake_case 형식으로 통일 (예: `post_db_ELEM`, `get_db_node`, `put_db_SECT`, `delete_db_MVHL`). 수기 샘플 형식과 정렬.
- [ ] **Arguments 루트 구조**: 항상 API 스키마 정의대로. wrapper 키(`Assign` 등)는 스키마에 명시된 경우만 사용, 임의 추가/제거 금지.
- [ ] **GET 요청**: 반드시 조회 키(`Assign` 객체 또는 path parameter) 명시. 빈 객체 금지.
- [ ] **DELETE 요청**: 대상 ID 명시. `key` 또는 `Assign` 객체 사용.
- [ ] **수기 샘플 ↔ 생성 샘플 형식 100% 일치**: regex 검증 통과.
- [ ] **tools 필드 일관성**: 학습용 `tools` 필드의 함수명도 위 규칙과 동일.

---

## 4. 파이프라인 개선 사항

### 4.1 `scripts/build_training_prompts.py`

수정 포인트:
1. **함수명 규칙 명시**: 시스템 프롬프트에 다음 추가
   ```
   함수명은 반드시 snake_case 형식 {method}_{path}로 작성하세요.
   예: post_db_ELEM, get_db_node, put_db_SECT, delete_db_MVHL.
   ```
2. **few-shot 분리**: `FEW_SHOT_SECTION_TEMPLATE`을 메서드별(POST/GET/PUT/DELETE)로 4개 섹션 분할. 각 섹션에 메서드 특화 가이드 문장 포함:
   - POST: "신규 데이터를 Assign 객체에 키-값으로 작성"
   - GET: "조회 대상 ID/키를 Assign 객체 또는 path param에 명시"
   - PUT: "변경 대상 ID와 갱신 필드를 함께 명시"
   - DELETE: "대상 ID만 명시, 데이터 필드 금지"
3. **복잡 API 스키마 압축**: properties 30+ 또는 oneOf/anyOf 5+ 인 API는 필수 인자만 추출한 요약 스키마를 별도 첨부 → 50K 토큰 초과 방지.

### 4.2 `scripts/generate_training_samples.py`

수정 포인트:
1. **사후 검증 강화** (실패 시 폐기 후 다른 시드로 재요청):
   - 함수명 정규식 검사: `^(post|get|put|delete)_[a-z]+_[A-Za-z]+$`
   - JSON schema validation: 생성된 arguments가 해당 API 스키마를 통과하는지 (`jsonschema` 라이브러리)
   - wrapper 키 화이트리스트: 스키마에 정의된 키만 허용
   - GET/DELETE는 인자 비어있으면 폐기
2. **VARIATION_HINTS 차등 적용**:
   - 단순 API: 16 hints × 2 variation = 최대 32 샘플
   - 중간 API: 16 hints × 3 = 48 샘플
   - 복잡 API: 16 hints × 5 + 서브타입별 추가 = 80+ 샘플
3. **재시도 정책 변경**: 동일 prompt + repair 대신 → 폐기 후 다른 VARIATION_HINT로 재생성. 노이즈 라벨이 학습셋에 들어가지 않게.
4. **로깅 강화**: 폐기된 샘플의 사유(검증 종류)를 별도 로그로 남겨 가이드 5절 검증에 활용.

---

## 5. 적정 데이터 양 가이드

### 범위
사용자가 사전에 선별한 **Core API 리스트 = 현재 학습 대상 242 endpoint×method 조합** (Tier 재분할 없이 그대로 사용).
- 이 242 조합은 이미 "해석 전까지 모델링에 꼭 필요한 API"로 선별됨
- `data/prompts/training_prompts.jsonl`의 unique `(_api_name, _method)` 집합이 곧 Core 리스트
- 카운트 검증 명령:
  ```bash
  jq -r '[._api_name, ._method] | @tsv' data/prompts/training_prompts.jsonl | sort -u | wc -l
  ```

### 권장 분포

| 복잡도 카테고리 | 조합 수 (추정) | 샘플/조합 | 총 샘플 |
|---|---|---|---|
| 단순 (인자 ≤5, 서브타입 없음) | ~140 | 25~30 | 3,500~4,200 |
| 중간 (인자 6~15) | ~80 | 35~45 | 2,800~3,600 |
| 복잡 (서브타입 多, 옵션 40+) | ~22 | 60~80 | 1,320~1,760 |
| **합계** | **242** | 평균 32~40 | **약 7,500~9,500** |

### 근거
- 현재: 14.8 샘플/조합 → param acc 16%
- 목표: 32+ 샘플/조합 (약 2배 밀도) → param acc 70%+ 기대
- function calling fine-tuning 일반치: 조합당 30~50 샘플에서 90% 도달 (참고치)
- 복잡 API(db/SECT 17 서브타입, view/CAPTURE 40+ 옵션, db/MVHL)는 **서브타입별 최소 5 샘플 보장** → 자동으로 60~80 도달
- **다양성 4축 강제**: 각 조합마다 다음 4가지 카테고리 최소 1개씩 포함
  - (a) 최소 인자만 사용한 호출
  - (b) 선택 인자를 다수 포함한 호출
  - (c) 경계값/특수문자 포함 (예: 큰 노드 번호, 음수 좌표)
  - (d) 자연어 표현 변형 (예: "만들어줘" / "추가해줘" / "생성" / "create")

### 복잡도 분류 기준
가이드만 명시하고 실제 자동 분류는 재생성 시 스크립트로 산출:
- **단순**: properties 5개 이하, oneOf/anyOf 없음
- **중간**: properties 6~15개, oneOf/anyOf ≤2
- **복잡**: properties 16+ 또는 oneOf/anyOf 3+ 또는 서브타입 다수

분류 소스: `data/samples/gennx_tool_schemas_tier1.json` 및 외부 GENNX_API_Schema 폴더.

### 학습 분할
- train 80% / eval 10% / test 10%
- **test는 조합당 최소 3개 보장** (총 test ≥ 726) — Step 6의 통계 유의성 부족 문제 해결
- 약 train 6,000~7,600, eval 750~950, test 750~950

### 1차 재생성 권장 규모
**약 8,000 샘플** (조합당 평균 ~33개)을 1차 목표로 생성. 1차 학습+평가 결과:
- param acc 70%+ → 완료
- param acc 50~70% → 복잡 API에만 추가 생성하여 ~9,500까지 확장
- param acc <50% → 검증 절차 점검 (4절 재실행)

---

## 6. 검증 단계

재생성 후, **학습 시작 전에 반드시 다음 5개 체크를 통과**해야 합니다.

1. **함수명 형식 검사** (regex)
   - `^(post|get|put|delete)_[a-z]+_[A-Za-z]+$` 패턴에 맞지 않는 샘플 0건
2. **조합별 샘플 수 분포**
   - 모든 242 조합이 최소 10개 이상
   - 최대/최소 비율 ≤ 8:1
3. **동일 API 내 arguments 키 일관성**
   - 같은 API 내 모든 샘플의 arguments root 키 집합이 스키마와 일치
4. **JSON Schema 검증**
   - 모든 arguments가 해당 API의 정의 스키마(`gennx_tool_schemas_tier1.json` 또는 GENNX_API_Schema)를 통과
5. **수기 샘플 25개와의 형식 매칭**
   - 수기 샘플과 동일 API의 생성 샘플을 비교하여 함수명·wrapper·키 형식 100% 일치

검증 스크립트는 `scripts/validate_training_data.py`로 신규 작성 권장 (이번 가이드 범위 외).

---

## 7. 단계별 실행 순서

1. **함수명 통일 규칙 결정** → 수기 샘플 25개와 정렬 확인
2. **`build_training_prompts.py` 수정** (함수명 규칙 명시, few-shot 메서드별 분리, 복잡 API 스키마 압축)
3. **`generate_training_samples.py` 수정** (사후 검증 강화, VARIATION_HINTS 차등화)
4. **소규모 시범 생성** (단순 API 5개 × 30 샘플 = 150 샘플)으로 형식 검증
5. **검증 5개 통과** 후 → 전체 재생성 약 8,000 샘플
6. **데이터 분할** (`02_prepare_data.py`에 조합당 test 3개 보장 로직 추가)
7. **QLoRA 재학습** (`03_train_qlora.py` 그대로) → Step 5 리포트 갱신
8. **평가** (`04_evaluate.py`) → param accuracy 목표 **70%+**

---

## 8. 모델 교체 시점 가이드

데이터 재생성 후에도 부족한 경우의 단계적 에스컬레이션:

| 재학습 후 param acc | 권장 조치 |
|---|---|
| 70%+ | 완료. 필요 시 에지 케이스 추가 학습 |
| 50~70% | 복잡 API 샘플만 추가 생성 (9,500까지 확장) |
| 30~50% | **Qwen2.5-7B QLoRA**로 모델 업그레이드 (RTX 5090 32GB면 충분) |
| <30% | 검증 절차 재점검. 데이터 자체에 구조적 결함 의심 |

**중요**: 데이터 재생성 전에 모델 교체는 권장하지 않습니다. Gemma 4 26B-A4B나 Qwen2.5-7B로 갈아타도 노이즈 라벨을 더 잘 외워버려 평가 점수만 오르고 실사용에서는 똑같이 틀릴 수 있습니다 (overfitting to noisy labels). **데이터를 먼저 고치고, 그래도 부족할 때 모델을 키우는 것이 비용 대비 효과가 가장 좋습니다.**
