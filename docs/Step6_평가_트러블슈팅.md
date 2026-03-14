# Step 6: 평가 트러블슈팅 기록

## Issue #1: 전 메트릭 0 — 파싱 구조 불일치 (2026-03-11)

### 증상
| 지표 | 값 |
|------|-----|
| Tool name accuracy | 0.0000 |
| Parameter accuracy | 0.0000 |
| JSON validity rate | 0.0000 |
| Hallucination rate | 1.0000 |

### 원인

모델 출력은 정상이었으나, `parse_tool_calls_from_output()`의 반환 구조와 metric 함수의 기대 구조가 불일치.

```
모델 출력:    {"name": "GET /db/hahs", "arguments": "{}"}
metric 기대:  {"function": {"name": "GET /db/hahs", "arguments": "..."}}
```

metric 함수가 `tc.get("function", {}).get("name", "")`로 접근하므로, `"function"` 키가 없어 모든 tool name이 빈 문자열로 처리됨.

### 수정

`src/eval_metrics.py`의 `parse_tool_calls_from_output()`에서 모델 출력을 reference와 동일한 구조로 정규화:

```python
if "function" not in parsed and "name" in parsed:
    parsed = {
        "type": "function",
        "function": {
            "name": parsed["name"],
            "arguments": parsed.get("arguments", "{}"),
        },
    }
```

---

## Issue #2: Hallucination Rate 95.4% — available_tools 리스트 불일치 (2026-03-12)

### 증상

Issue #1 수정 후 재평가 결과:
| 지표 | 값 | 해석 |
|------|-----|------|
| Tool name accuracy | 96.6% | 정상 |
| Hallucination rate | **95.4%** | **비정상** — name accuracy와 모순 |

Tool name accuracy가 96.6%인데 hallucination이 95.4%라는 것은 논리적으로 모순.

### 원인 분석

`hallucination_rate()`는 모델이 예측한 tool name이 `available_tools` 리스트에 있는지 확인하는 메트릭.

**`available_tools` 구축 방식이 문제였다:**

```python
# 04_evaluate.py (수정 전)
# --tools-schema 인자로 gennx_tool_schemas_tier1.json 로드 → 15개 tool만 포함
with open(args.tools_schema, "r", encoding="utf-8") as f:
    tools_schema = json.load(f)
```

| 항목 | 개수 | 내용 |
|------|------|------|
| `available_tools` (스키마 파일) | **15개** | Tier-1 API만 포함 |
| test 데이터의 reference tool names | **242개** | 전체 API |
| 겹치는 것 | **7개** | - |

학습 데이터는 Tier-1~4 전체 API(242종)로 생성되었으나, `--tools-schema`에는 Tier-1 스키마 파일(15개)만 지정되어 있었다. 모델이 `POST /db/bmld` 같은 올바른 tool을 출력해도 15개 리스트에 없으므로 hallucination으로 판정됨.

### 수정

`04_evaluate.py`의 Step 4를 변경하여 test 데이터의 `tools` 필드에서 available_tools를 직접 추출:

**Before:**
```python
# Step 4: 외부 스키마 파일에서 로드 (15개만)
with open(args.tools_schema, "r", encoding="utf-8") as f:
    tools_schema = json.load(f)
available_tools = [tool["function"]["name"] for tool in tools_schema ...]
```

**After:**
```python
# Step 4: test 데이터의 tools 필드에서 추출 (242개)
available_tools_set = set()
for sample in test_data:
    if "tools" in sample:
        tools_value = sample["tools"]
        tools_list = json.loads(tools_value) if isinstance(tools_value, str) else tools_value
        for tool in tools_list:
            if "function" in tool and "name" in tool["function"]:
                available_tools_set.add(tool["function"]["name"])
            elif "name" in tool:
                available_tools_set.add(tool["name"])
available_tools = sorted(available_tools_set)
```

추가로 `--tools-schema` CLI 인자를 제거하고, 추론 루프의 `tools_schema` fallback도 정리.

### 수정 결과

| 지표 | 수정 전 | 수정 후 |
|------|---------|---------|
| Hallucination rate | 95.4% | **0.0%** |
| 기타 메트릭 | 동일 | 동일 |

### 교훈

- 학습 데이터와 평가 설정의 **스코프 일치** 확인 필요
- 학습 데이터가 전체 API로 생성되었다면, 평가 시 available_tools도 전체 API를 포함해야 함
- test 데이터 자체에 `tools` 필드가 있으므로, 외부 파일 의존 없이 test 데이터에서 추출하는 것이 가장 안전한 방법
