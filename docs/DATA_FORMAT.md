# 학습 데이터 포맷 가이드

> 자체 학습 데이터를 만들 때 이 포맷을 반드시 따라야 합니다.

---

## 파일 형식

- 파일 확장자: `.jsonl` (JSON Lines)
- 인코딩: UTF-8
- 한 줄 = 하나의 완전한 대화 (JSON 객체)

---

## 기본 구조

```jsonl
{"messages": [...], "tools": "<tools JSON 문자열>"}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `messages` | array | O | 대화 메시지 배열 |
| `tools` | string | O | 사용 가능한 도구 스키마 (JSON **문자열**) |

---

## messages 배열 구조

하나의 대화는 다음 순서로 구성됩니다:

```
system → user → assistant(tool_calls) → tool(결과) → assistant(확인)
```

### 1. System 메시지

```json
{
  "role": "system",
  "content": "You are a structural engineering assistant for GEN NX. Use the provided tools to execute user requests on the structural model."
}
```

### 2. User 메시지

```json
{
  "role": "user",
  "content": "절점 1번을 원점에 추가해줘"
}
```

### 3. Assistant 메시지 (tool_calls)

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "POST /db/node",
        "arguments": "{\"Assign\":{\"1\":{\"X\":0,\"Y\":0,\"Z\":0}}}"
      }
    }
  ]
}
```

**주의:** `arguments` 값은 반드시 **JSON 문자열**이어야 합니다 (dict 아님).

### 4. Tool 응답 메시지

```json
{
  "role": "tool",
  "name": "POST /db/node",
  "content": "{\"NODE\":{\"1\":{\"X\":0,\"Y\":0,\"Z\":0}}}"
}
```

### 5. Assistant 확인 메시지

```json
{
  "role": "assistant",
  "content": "절점 1번이 원점(0, 0, 0)에 추가되었습니다."
}
```

---

## tools 필드

`tools`는 사용 가능한 도구 목록을 **JSON 문자열**로 담습니다.

```json
"tools": "[{\"type\":\"function\",\"function\":{\"name\":\"POST /db/node\",\"description\":\"Create nodes\",\"parameters\":{\"type\":\"object\",\"properties\":{\"Assign\":{\"type\":\"object\"}},\"required\":[\"Assign\"]}}}]"
```

**주의:** list가 아니라 **문자열**이어야 합니다. `json.dumps(tools_list)`로 변환하세요.

---

## 완성 예시 (한 줄)

```jsonl
{"messages":[{"role":"system","content":"You are a structural engineering assistant for GEN NX. Use the provided tools to execute user requests on the structural model."},{"role":"user","content":"절점 1번을 원점에 추가해줘"},{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"POST /db/node","arguments":"{\"Assign\":{\"1\":{\"X\":0,\"Y\":0,\"Z\":0}}}"}}]},{"role":"tool","name":"POST /db/node","content":"{\"NODE\":{\"1\":{\"X\":0,\"Y\":0,\"Z\":0}}}"},{"role":"assistant","content":"절점 1번이 원점(0, 0, 0)에 추가되었습니다."}],"tools":"[{\"type\":\"function\",\"function\":{\"name\":\"POST /db/node\",\"description\":\"Create or add nodes to the structural model\",\"parameters\":{\"type\":\"object\",\"properties\":{\"Assign\":{\"type\":\"object\",\"description\":\"Node assignments with node number as key\"}},\"required\":[\"Assign\"]}}}]"}
```

---

## 다중 도구 호출

하나의 요청에서 여러 도구를 동시에 호출하는 경우:

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "POST /db/matl",
        "arguments": "{\"Assign\":{\"1\":{\"TYPE\":\"STEEL\",\"NAME\":\"SS400\"}}}"
      }
    },
    {
      "type": "function",
      "function": {
        "name": "POST /db/sect",
        "arguments": "{\"Assign\":{\"1\":{\"SHAPE\":\"H\",\"NAME\":\"H-300x150\"}}}"
      }
    }
  ]
}
```

이 경우 tool 응답도 **각각** 있어야 합니다:

```json
{"role": "tool", "name": "POST /db/matl", "content": "{\"MATL\":{\"1\":{...}}}"},
{"role": "tool", "name": "POST /db/sect", "content": "{\"SECT\":{\"1\":{...}}}"}
```

---

## 자주 하는 실수

| 실수 | 올바른 방법 |
|------|------------|
| `arguments`가 dict | `arguments`는 반드시 **JSON 문자열** |
| `tools`가 list | `tools`는 반드시 **JSON 문자열** |
| tool 응답 누락 | tool_calls가 있으면 반드시 tool 응답이 뒤따라야 함 |
| role 오타 | `system`, `user`, `assistant`, `tool`만 허용 |
| 마지막 확인 메시지 누락 | tool 응답 후 assistant 확인 메시지 권장 |

---

## 데이터 검증 방법

```bash
python scripts/02_prepare_data.py --input data/raw/my_data.jsonl
```

검증에 실패한 샘플이 있으면 오류 메시지가 출력됩니다.

또는 Python에서 직접:

```python
from src.data_utils import load_jsonl, validate_data_format

samples = load_jsonl("data/raw/my_data.jsonl")
for i, sample in enumerate(samples):
    valid, errors = validate_data_format(sample)
    if not valid:
        print(f"Sample {i}: {errors}")
```
