"""
build_training_prompts.py

Assembles LLM prompts for generating TRL-format training samples.
Reads API schemas, feature docs, tier assignments, and existing samples,
then produces one prompt record per API × method (× subtype) combination.
"""

import argparse
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "data_utils",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "data_utils.py"),
)
_data_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data_utils)
load_jsonl = _data_utils.load_jsonl
save_jsonl = _data_utils.save_jsonl

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

DEFAULT_API_DATA_DIR = (
    r"C:\Users\hjm0830\OneDrive - MIDAS"
    r"\바탕 화면\MIDAS\01_Study\PyTorch\API_Data"
)

TIER_RANGES = {1: (30, 50), 2: (15, 25), 3: (5, 10), 4: (2, 5)}
METHOD_WEIGHTS = {"POST": 0.40, "PUT": 0.30, "GET": 0.15, "DELETE": 0.15}

PROMPT_TEMPLATE = """\
당신은 구조공학 소프트웨어 GEN NX의 API 학습 데이터를 생성하는 전문가입니다.

아래 정보를 참고하여 **{target_samples}개**의 TRL 형식 학습 샘플을 생성하세요.

## API 정보
- 엔드포인트: `{endpoint}`
- HTTP 메서드: `{method}`
- 도구 이름: `"{tool_name}"`
{subtype_info}

## 메서드 지침
{method_instruction}

## 스키마 속성 정의
{schema_properties}

## 스키마 예시 페이로드
```json
{schema_example}
```

## 도구 정의 (tools 필드)
```json
{tool_schema_json}
```
{feature_section}
{few_shot_section}

## 출력 포맷 규칙
1. 각 샘플은 독립적인 JSON 객체 (JSONL 형식, 한 줄에 하나)
2. 각 샘플 구조:
   - `messages`: 대화 배열 (system, user, assistant with tool_calls, tool, assistant)
   - `tools`: **JSON 문자열** (리스트를 문자열로 직렬화)
3. `tool_calls` 내 `arguments`는 반드시 **JSON 문자열**이어야 함 (객체가 아님)
4. system 메시지: "You are a structural engineering assistant for GEN NX. Use the provided tools to execute user requests on the structural model."
5. tool 응답 메시지의 `content`도 JSON 문자열

## 다양성 요구사항
- 사용자 메시지는 **한국어**로 작성
- 다양한 시나리오와 맥락 (초보자 질문, 전문가 요청, 구체적 수치, 일반적 요청 등)
- 파라미터 조합을 다양하게 (필수만, 필수+선택, 전체 등)
- 복잡도 변화 (단일 항목, 다중 항목, 조건부 요청 등)
- 자연스럽고 현실적인 구조공학 시나리오
- assistant의 최종 응답도 한국어로, 수행 결과를 친절하게 설명
"""

METHOD_INSTRUCTIONS = {
    "POST": "POST (생성): 새로운 데이터를 생성합니다. Assign 객체에 ID를 키로, 속성을 값으로 지정합니다.",
    "PUT": "PUT (수정): 기존 데이터를 수정합니다. 수정할 항목의 ID와 변경할 속성을 지정합니다.",
    "GET": "GET (조회): 데이터를 조회합니다. 파라미터 없이 호출하거나 특정 조건으로 필터링합니다.",
    "DELETE": "DELETE (삭제): 기존 데이터를 삭제합니다. 삭제할 항목의 ID를 지정합니다.",
}

FEATURE_SECTION_TEMPLATE = """
## Feature 컨텍스트 (기능 설명)
- 기능명: {feature_name}
- 기능 설명: {function_desc}
- 호출 방법: {call_desc}
- 입력 정보: {input_desc}
{note_desc}"""

FEW_SHOT_SECTION_TEMPLATE = """
## 참고 예시 (기존 학습 샘플)
아래 예시의 포맷과 스타일을 참고하세요:

{examples}"""


# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_data(api_data_dir: str, few_shot_path: str) -> dict:
    """Load all data sources at once."""
    base = Path(api_data_dir)

    tier_assignment = load_json(base / "api_tier_assignment.json")
    schema_index = load_json(base / "GENNX_API_Schema" / "_index.json")
    feature_mapping = load_json(base / "api_to_feature_mapping_v3.json")
    feature_index = load_json(base / "GENNX_Feature" / "_index.json")

    few_shot_examples = []
    if os.path.exists(few_shot_path):
        few_shot_examples = load_jsonl(few_shot_path)
        logger.info("Loaded %d few-shot examples from %s", len(few_shot_examples), few_shot_path)
    else:
        logger.warning("Few-shot file not found: %s", few_shot_path)

    return {
        "tier_assignment": tier_assignment,
        "schema_index": schema_index,
        "schema_dir": base / "GENNX_API_Schema",
        "feature_mapping": feature_mapping,
        "feature_index": feature_index,
        "feature_dir": base / "GENNX_Feature",
        "few_shot_examples": few_shot_examples,
    }


# ────────────────────────────────────────────────────────────────────
# Schema resolution
# ────────────────────────────────────────────────────────────────────

def resolve_api_schemas(api_name: str, schema_index: dict, schema_dir: Path) -> list:
    """Return [(subtype_name|None, schema_data)] for an API.

    Simple APIs have a string ID; subtype APIs have a dict of name->ID.
    """
    entry = schema_index.get(api_name)
    if entry is None:
        logger.warning("No schema index entry for %s", api_name)
        return []

    results = []
    if isinstance(entry, str):
        # Simple API — single schema file
        schema_path = schema_dir / f"{entry}.json"
        if schema_path.exists():
            results.append((None, load_json(schema_path)))
        else:
            logger.warning("Schema file not found: %s", schema_path)
    elif isinstance(entry, dict):
        # Subtype API — multiple schema files
        for sub_name, article_id in entry.items():
            schema_path = schema_dir / f"{article_id}.json"
            if schema_path.exists():
                results.append((sub_name, load_json(schema_path)))
            else:
                logger.warning("Schema file not found for %s/%s: %s", api_name, sub_name, schema_path)
    return results


# ────────────────────────────────────────────────────────────────────
# Feature resolution
# ────────────────────────────────────────────────────────────────────

def resolve_feature(api_name: str, feature_mapping: dict, feature_index: dict, feature_dir: Path) -> dict | None:
    """Look up feature documentation for an API. Returns feature dict or None."""
    feature_name = feature_mapping.get(api_name)
    if not feature_name:
        logger.debug("No feature mapping for %s", api_name)
        return None

    # Feature index keys have " ↗" suffix
    index_key = f"{feature_name} ↗"
    article_id = feature_index.get(index_key)
    if article_id is None:
        logger.debug("Feature '%s' not found in feature index", index_key)
        return None

    feature_path = feature_dir / f"{article_id}.json"
    if not feature_path.exists():
        logger.warning("Feature file not found: %s", feature_path)
        return None

    return load_json(feature_path)


# ────────────────────────────────────────────────────────────────────
# Schema context extraction
# ────────────────────────────────────────────────────────────────────

def extract_schema_context(schema_data: dict, method: str) -> dict:
    """Extract properties, tables, and examples from schema data."""
    context = {
        "endpoint": schema_data.get("endpoint", ""),
        "title": schema_data.get("title", ""),
        "active_methods": schema_data.get("active_methods", []),
        "properties": {},
        "tables": [],
        "examples": {},
    }

    # Extract properties from json_schema
    json_schema = schema_data.get("json_schema", {})
    if json_schema:
        # json_schema has a top-level key (e.g. "NODE") containing the actual schema
        for key, val in json_schema.items():
            if isinstance(val, dict) and "properties" in val:
                context["properties"] = val["properties"]
                break

    # Extract tables (parameter metadata)
    tables = schema_data.get("tables", [])
    if tables:
        context["tables"] = tables

    # Extract examples
    examples = schema_data.get("examples", {})
    if examples:
        context["examples"] = examples

    return context


def format_schema_properties(schema_context: dict) -> str:
    """Format schema properties and tables into readable text."""
    lines = []

    # Use tables if available (richer metadata)
    tables = schema_context.get("tables", [])
    if tables:
        for table_idx, table in enumerate(tables):
            if not table:
                continue
            if len(tables) > 1:
                lines.append(f"### 테이블 {table_idx + 1}")
            lines.append("| Key | Description | Type | Default | Required |")
            lines.append("|-----|-------------|------|---------|----------|")
            for row in table:
                if isinstance(row, dict):
                    key = row.get("Key", "")
                    desc = row.get("Description", "")
                    vtype = row.get("Value Type", "")
                    default = row.get("Default", "")
                    required = row.get("Required", "")
                    lines.append(f"| {key} | {desc} | {vtype} | {default} | {required} |")
        return "\n".join(lines)

    # Fallback to json_schema properties
    props = schema_context.get("properties", {})
    if props:
        lines.append("| Key | Description | Type |")
        lines.append("|-----|-------------|------|")
        for key, val in props.items():
            desc = val.get("description", "") if isinstance(val, dict) else ""
            vtype = val.get("type", "") if isinstance(val, dict) else ""
            lines.append(f"| {key} | {desc} | {vtype} |")
        return "\n".join(lines)

    return "(스키마 속성 정보 없음)"


def format_schema_example(schema_context: dict) -> str:
    """Format schema examples as JSON string."""
    examples = schema_context.get("examples", {})
    if examples:
        return json.dumps(examples, indent=2, ensure_ascii=False)
    return "{}"


# ────────────────────────────────────────────────────────────────────
# Tool schema builder
# ────────────────────────────────────────────────────────────────────

def build_tool_schema(api_name: str, method: str, schema_context: dict) -> dict:
    """Build a TRL-format tool definition."""
    tool_name = f"{method} /{api_name.lower()}"
    title = schema_context.get("title", api_name)

    method_verbs = {
        "POST": "Create or add",
        "PUT": "Update existing",
        "GET": "Retrieve",
        "DELETE": "Delete",
    }
    verb = method_verbs.get(method, "Operate on")
    description = f"{verb} {title.lower()} in the structural model"

    # Build parameters from properties
    properties = {}
    props = schema_context.get("properties", {})
    if props:
        # Wrap in Assign for consistency with existing samples
        properties = {
            "Assign": {
                "type": "object",
                "description": f"{title} assignments",
            }
        }
        required = ["Assign"]
    else:
        required = []

    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    return tool


# ────────────────────────────────────────────────────────────────────
# Target samples computation
# ────────────────────────────────────────────────────────────────────

def compute_target_samples(tier: int, method: str, methods_list: list, sub_count: int) -> int:
    """Compute target sample count for a specific method/subtype."""
    low, high = TIER_RANGES.get(tier, (2, 5))
    midpoint = (low + high) / 2

    weight = METHOD_WEIGHTS.get(method, 0.15)
    # Normalize weights to methods actually present
    total_weight = sum(METHOD_WEIGHTS.get(m, 0.15) for m in methods_list)
    normalized_weight = weight / total_weight if total_weight > 0 else weight

    target = midpoint * normalized_weight
    if sub_count > 1:
        target = target / sub_count

    return max(2, round(target))


# ────────────────────────────────────────────────────────────────────
# Few-shot example selection
# ────────────────────────────────────────────────────────────────────

def _extract_tool_name(sample: dict) -> str | None:
    """Extract the tool name from a sample's first tool_call."""
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tcs = msg["tool_calls"]
            if tcs:
                return tcs[0].get("function", {}).get("name", "")
    return None


def select_few_shot_examples(api_name: str, method: str, all_examples: list) -> list:
    """Select 1-2 relevant few-shot examples from existing samples."""
    tool_name = f"{method} /{api_name.lower()}"

    # Priority 1: exact match on tool name
    exact = [s for s in all_examples if _extract_tool_name(s) == tool_name]
    if exact:
        return exact[:2]

    # Priority 2: same API, different method
    api_lower = api_name.lower()
    same_api = [s for s in all_examples if _extract_tool_name(s) and api_lower in (_extract_tool_name(s) or "").lower()]
    if same_api:
        return same_api[:2]

    # Priority 3: same method, different API
    same_method = [s for s in all_examples if (_extract_tool_name(s) or "").upper().startswith(method)]
    if same_method:
        return same_method[:1]

    # Priority 4: any example
    if all_examples:
        return all_examples[:1]

    return []


def format_few_shot(examples: list) -> str:
    """Format few-shot examples as readable JSON."""
    if not examples:
        return ""
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"### 예시 {i}")
        parts.append("```json")
        parts.append(json.dumps(ex, indent=2, ensure_ascii=False))
        parts.append("```")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# Prompt builder
# ────────────────────────────────────────────────────────────────────

def build_prompt(
    api_name: str,
    method: str,
    sub_type: str | None,
    target_samples: int,
    schema_context: dict,
    tool_schema: dict,
    feature_data: dict | None,
    few_shot_examples: list,
) -> str:
    """Assemble the final LLM prompt from all context."""
    tool_name = f"{method} /{api_name.lower()}"

    subtype_info = ""
    if sub_type:
        subtype_info = f"- 서브타입: `{sub_type}`"

    # Feature section
    feature_section = ""
    if feature_data:
        sections = feature_data.get("sections", {})
        note_line = ""
        if sections.get("note"):
            note_line = f"- 주의사항: {sections['note'][:500]}"
        feature_section = FEATURE_SECTION_TEMPLATE.format(
            feature_name=feature_data.get("title", ""),
            function_desc=sections.get("function", "(없음)")[:500],
            call_desc=sections.get("call", "(없음)")[:300],
            input_desc=sections.get("input", "(없음)")[:800],
            note_desc=note_line,
        )

    # Few-shot section
    few_shot_text = format_few_shot(few_shot_examples)
    few_shot_section = ""
    if few_shot_text:
        few_shot_section = FEW_SHOT_SECTION_TEMPLATE.format(examples=few_shot_text)
    else:
        few_shot_section = "\n## 참고 예시\n(기존 예시 없음 — 위 스키마와 포맷 규칙을 기반으로 생성하세요)"

    prompt = PROMPT_TEMPLATE.format(
        target_samples=target_samples,
        endpoint=f"/{api_name}",
        method=method,
        tool_name=tool_name,
        subtype_info=subtype_info,
        method_instruction=METHOD_INSTRUCTIONS.get(method, ""),
        schema_properties=format_schema_properties(schema_context),
        schema_example=format_schema_example(schema_context),
        tool_schema_json=json.dumps(tool_schema, indent=2, ensure_ascii=False),
        feature_section=feature_section,
        few_shot_section=few_shot_section,
    )
    return prompt


# ────────────────────────────────────────────────────────────────────
# Feature context extraction (for record output)
# ────────────────────────────────────────────────────────────────────

def extract_feature_context(feature_data: dict | None) -> dict | None:
    if feature_data is None:
        return None
    sections = feature_data.get("sections", {})
    return {
        "title": feature_data.get("title", ""),
        "function": sections.get("function", ""),
        "call": sections.get("call", ""),
        "input": sections.get("input", ""),
        "note": sections.get("note", ""),
    }


# ────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Assemble LLM prompts for training sample generation."
    )
    parser.add_argument(
        "--api-data-dir",
        type=str,
        default=DEFAULT_API_DATA_DIR,
        help="Path to API_Data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/prompts/training_prompts.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--few-shot-path",
        type=str,
        default="data/samples/gennx_tool_calling_samples.jsonl",
        help="Path to few-shot examples JSONL",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="1,2,3,4",
        help="Comma-separated tier numbers to process",
    )
    parser.add_argument(
        "--apis",
        type=str,
        default=None,
        help="Comma-separated API names to filter (e.g. db/NODE,db/SECT)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics only, don't save file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    tiers_to_process = [int(t.strip()) for t in args.tiers.split(",")]
    api_filter = None
    if args.apis:
        api_filter = set(a.strip() for a in args.apis.split(","))

    # ── Step 1: Load all data ──
    logger.info("Loading data sources from %s ...", args.api_data_dir)
    data = load_all_data(args.api_data_dir, args.few_shot_path)
    tier_assignment = data["tier_assignment"]
    schema_index = data["schema_index"]
    schema_dir = data["schema_dir"]
    feature_mapping = data["feature_mapping"]
    feature_index = data["feature_index"]
    feature_dir = data["feature_dir"]
    few_shot_examples = data["few_shot_examples"]

    # ── Step 2: Iterate tiers and APIs ──
    records = []
    tier_stats = {}

    for tier_num in tiers_to_process:
        tier_key = f"tier_{tier_num}"
        tier_data = tier_assignment.get(tier_key)
        if not tier_data:
            logger.warning("Tier %d not found in assignment file", tier_num)
            continue

        apis = tier_data.get("apis", {})
        tier_record_count = 0
        tier_sample_total = 0

        for api_name, api_info in apis.items():
            if api_filter and api_name not in api_filter:
                continue

            methods_list = api_info.get("methods", [])
            sub_count = api_info.get("sub_count", 1)

            # Resolve schemas (handles subtypes)
            schema_list = resolve_api_schemas(api_name, schema_index, schema_dir)
            if not schema_list:
                logger.warning("Skipping %s: no schemas resolved", api_name)
                continue

            # Resolve feature (same for all subtypes of an API)
            feature_data = resolve_feature(api_name, feature_mapping, feature_index, feature_dir)
            feature_ctx = extract_feature_context(feature_data)

            for sub_type, schema_data in schema_list:
                schema_context = extract_schema_context(schema_data, "")
                active_methods = schema_context.get("active_methods", [])

                # Intersect tier-assigned methods with schema active methods
                effective_methods = [m for m in methods_list if m in active_methods]
                if not effective_methods:
                    # Fall back to tier methods if active_methods is empty
                    effective_methods = methods_list
                    logger.debug(
                        "%s (sub=%s): no method intersection, using tier methods",
                        api_name, sub_type,
                    )

                for method in effective_methods:
                    target = compute_target_samples(tier_num, method, effective_methods, sub_count)
                    tool_schema = build_tool_schema(api_name, method, schema_context)
                    few_shots = select_few_shot_examples(api_name, method, few_shot_examples)

                    prompt = build_prompt(
                        api_name=api_name,
                        method=method,
                        sub_type=sub_type,
                        target_samples=target,
                        schema_context=schema_context,
                        tool_schema=tool_schema,
                        feature_data=feature_data,
                        few_shot_examples=few_shots,
                    )

                    record = {
                        "api_name": api_name,
                        "method": method,
                        "sub_type": sub_type,
                        "tier": tier_num,
                        "target_samples": target,
                        "prompt": prompt,
                        "tool_schema": tool_schema,
                        "schema_context": {
                            "endpoint": schema_context["endpoint"],
                            "title": schema_context["title"],
                            "properties": schema_context["properties"],
                            "examples": schema_context["examples"],
                        },
                        "feature_context": feature_ctx,
                    }
                    records.append(record)
                    tier_record_count += 1
                    tier_sample_total += target

        tier_stats[tier_num] = {
            "records": tier_record_count,
            "target_samples": tier_sample_total,
            "apis": len(apis),
        }

    # ── Step 3: Save or report ──
    print(f"\n{'='*60}")
    print("Training Prompt Assembly Summary")
    print(f"{'='*60}")
    print(f"  Total records: {len(records)}")
    total_samples = sum(r["target_samples"] for r in records)
    print(f"  Total target samples: {total_samples}")
    print()
    for tier_num in sorted(tier_stats):
        s = tier_stats[tier_num]
        print(f"  Tier {tier_num}: {s['records']} records, {s['target_samples']} target samples ({s['apis']} APIs in tier)")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN] No file saved.")
    else:
        save_jsonl(records, args.output)
        print(f"\nSaved {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
