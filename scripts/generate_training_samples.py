"""
generate_training_samples.py

Step 2: OpenAI API를 호출하여 TRL 형식 학습 샘플을 대량 생성.
training_prompts.jsonl의 각 프롬프트에 대해 1개씩 요청하되,
변형 힌트를 순환 배정하여 다양성을 보장한다.
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "data_utils",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "data_utils.py"),
)
_data_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data_utils)
load_jsonl = _data_utils.load_jsonl
save_jsonl = _data_utils.save_jsonl
validate_data_format = _data_utils.validate_data_format
format_tool_calls_for_qwen = _data_utils.format_tool_calls_for_qwen

from openai import OpenAI

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# API 키: 프로젝트 루트의 .env 파일에서 로드
#   .env 파일 형식: OPENAI_API_KEY=sk-proj-xxxxx
# ────────────────────────────────────────────────────────────────────

def _load_dotenv():
    """프로젝트 루트의 .env 파일을 읽어 환경변수에 설정."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()


def get_api_key() -> str:
    """환경변수에서 API 키 로드 (.env → 시스템 환경변수)."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    print("[ERROR] OpenAI API 키가 설정되지 않았습니다.")
    print("  프로젝트 루트의 .env 파일에 다음을 추가하세요:")
    print("  OPENAI_API_KEY=sk-proj-xxxxx")
    sys.exit(1)


# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a training data generator for structural engineering AI. "
    "Output ONLY a single valid JSON object (one complete training sample). "
    "Do NOT wrap output in markdown code fences. "
    "Do NOT add any commentary or explanation before or after the JSON."
)

VARIATION_HINTS = [
    "초보자가 간단하게 요청하는 시나리오 (예: '~해줘', '~추가해줘')",
    "전문가가 구체적 수치와 단위를 명시하는 시나리오",
    "여러 항목을 한번에 처리하는 복잡한 시나리오 (다중 ID, 다중 속성)",
    "모호하게 요청해서 기본값이 많이 사용되는 시나리오",
    "실무 프로젝트 맥락이 포함된 시나리오 (예: '3층 건물의 기둥을...')",
    "기존 데이터를 확인/검토하는 맥락의 시나리오",
    "조건이나 제약이 포함된 시나리오 (예: '~이면 ~하고, 아니면 ~')",
    "다른 API 작업과 연계되는 맥락의 시나리오 (예: '절점 추가하고 요소 연결')",
    "단일 항목만 간결하게 처리하는 시나리오",
    "큰 숫자의 ID나 좌표를 사용하는 시나리오 (예: 절점 100번, 좌표 50m)",
    "음수 좌표나 특수한 값을 사용하는 시나리오",
    "구어체로 자연스럽게 대화하는 시나리오 (예: '이거 좀 바꿔줄래?')",
    "정중한 존댓말로 요청하는 시나리오 (예: '~해주시겠습니까?')",
    "여러 속성 중 일부만 선택적으로 지정하는 시나리오",
    "반복 작업을 한 문장으로 요청하는 시나리오 (예: '1~10번까지 절점을...')",
    "오류 수정이나 잘못된 입력을 고치는 맥락의 시나리오",
]

SINGLE_SAMPLE_INSTRUCTION = """
위 API 정보를 참고하여 TRL 형식 학습 샘플을 **정확히 1개만** 생성하세요.

### 시나리오 지시
{variation_hint}

### 출력 규칙
1. 출력은 단일 JSON 객체 (한 줄 또는 여러 줄 모두 가능)
2. 구조: messages 배열 (system, user, assistant with tool_calls, tool, assistant) + tools 필드
3. `tool_calls` 내 `arguments`는 반드시 **JSON 문자열**
4. `tools`는 반드시 **JSON 문자열** (리스트를 문자열로 직렬화)
5. system 메시지: "You are a structural engineering assistant for GEN NX. Use the provided tools to execute user requests on the structural model."
6. tool 응답 `content`도 JSON 문자열
7. 사용자 메시지와 assistant 최종 응답은 **한국어**로 작성
"""


# ────────────────────────────────────────────────────────────────────
# Prompt builder for single-sample requests
# ────────────────────────────────────────────────────────────────────

def build_single_prompt(original_prompt: str, variation_hint: str) -> str:
    """Modify the original N-sample prompt to request exactly 1 sample
    with a specific variation hint."""
    # Remove the original target count instruction (e.g. "**16개**의")
    # and replace with single-sample instruction at the end
    prompt = original_prompt

    # Remove the "다양성 요구사항" section (we replace it with variation hint)
    diversity_idx = prompt.find("## 다양성 요구사항")
    if diversity_idx != -1:
        prompt = prompt[:diversity_idx].rstrip()

    # Also remove the "출력 포맷 규칙" section (we replace it)
    format_idx = prompt.find("## 출력 포맷 규칙")
    if format_idx != -1:
        prompt = prompt[:format_idx].rstrip()

    # Append single-sample instruction with variation hint
    prompt += "\n\n" + SINGLE_SAMPLE_INSTRUCTION.format(variation_hint=variation_hint)

    return prompt


# ────────────────────────────────────────────────────────────────────
# OpenAI API call
# ────────────────────────────────────────────────────────────────────

def call_openai(client: OpenAI, prompt: str, model: str, temperature: float,
                max_retries: int) -> str | None:
    """Call OpenAI API with retry logic. Returns response text or None."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(max_retries):
        try:
            # GPT-5 계열: max_completion_tokens, temperature 고정(1)
            extra = {}
            if model.startswith("gpt-5") or model.startswith("o"):
                extra["max_completion_tokens"] = 8192
            else:
                extra["max_tokens"] = 4096
                extra["temperature"] = temperature

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **extra,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_name = type(e).__name__
            if "RateLimitError" in error_name or "429" in str(e):
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limit hit, waiting %ds (attempt %d/%d)",
                               wait, attempt + 1, max_retries)
                time.sleep(wait)
            elif "APIConnectionError" in error_name or "500" in str(e) or "502" in str(e) or "503" in str(e):
                wait = 2 ** attempt
                logger.warning("API error: %s, retrying in %ds (attempt %d/%d)",
                               e, wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                logger.error("API call failed: %s", e)
                return None

    logger.error("All %d retries exhausted", max_retries)
    return None


# ────────────────────────────────────────────────────────────────────
# Response parsing
# ────────────────────────────────────────────────────────────────────

def parse_llm_response(response_text: str) -> list[dict]:
    """Parse LLM response text into individual JSON objects."""
    if not response_text:
        return []

    text = response_text.strip()

    # Strip markdown code fences
    text = re.sub(r"```(?:json|jsonl)?\s*\n?", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Try parsing as a single JSON object first (most common for 1-sample requests)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "messages" in obj:
            return [obj]
    except json.JSONDecodeError:
        pass

    # Try line-by-line JSONL parsing
    samples = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "messages" in obj:
                samples.append(obj)
        except json.JSONDecodeError:
            continue

    if samples:
        return samples

    # Fallback: use raw_decode to extract JSON objects with "messages" key
    decoder = json.JSONDecoder()
    samples = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx] != "{":
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            if isinstance(obj, dict) and "messages" in obj:
                samples.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            idx += 1

    return samples


# ────────────────────────────────────────────────────────────────────
# Sample format fixing
# ────────────────────────────────────────────────────────────────────

def repair_json_string(s: str) -> str:
    """Attempt to fix missing commas in a JSON string.

    Common LLM mistake: omitting commas between key-value pairs.
    e.g. '"value" "key"' → '"value", "key"'
    """
    # Already valid
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass

    # Pattern: missing comma before a new key
    # Matches: value<whitespace>"key" where value ends with ", number, }, ], true, false, null
    fixed = re.sub(
        r'(")\s*\n?\s*(")',         # "value" "nextKey"
        r'\1, \2', s
    )
    fixed = re.sub(
        r'(\d)\s*\n?\s*(")',         # 123 "nextKey"
        r'\1, \2', fixed
    )
    fixed = re.sub(
        r'(\})\s*\n?\s*(")',         # } "nextKey"
        r'\1, \2', fixed
    )
    fixed = re.sub(
        r'(\])\s*\n?\s*(")',         # ] "nextKey"
        r'\1, \2', fixed
    )
    fixed = re.sub(
        r'(true|false|null)\s*\n?\s*(")',  # true/false/null "nextKey"
        r'\1, \2', fixed
    )
    # Missing comma between array elements: }{
    fixed = re.sub(
        r'(\})\s*\n?\s*(\{)',
        r'\1, \2', fixed
    )

    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        return s  # Return original if still broken


def fix_sample_format(sample: dict, tool_schema: dict) -> dict:
    """Fix common LLM formatting mistakes in generated samples."""
    sample = copy.deepcopy(sample)

    # Fix tool_calls arguments: dict → JSON string, and repair broken JSON
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                if "arguments" in func and not isinstance(func["arguments"], str):
                    func["arguments"] = json.dumps(func["arguments"], ensure_ascii=False)
                elif "arguments" in func and isinstance(func["arguments"], str):
                    func["arguments"] = repair_json_string(func["arguments"])
                if "type" not in tc:
                    tc["type"] = "function"

        # Fix tool message content: dict → JSON string
        if msg.get("role") == "tool" and isinstance(msg.get("content"), dict):
            msg["content"] = json.dumps(msg["content"], ensure_ascii=False)

    # Fix tools field: list → JSON string
    if "tools" in sample:
        if isinstance(sample["tools"], list):
            sample["tools"] = json.dumps(sample["tools"], ensure_ascii=False)
    else:
        sample["tools"] = json.dumps([tool_schema], ensure_ascii=False)

    # Final normalization
    sample = format_tool_calls_for_qwen(sample)
    return sample


# ────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────

def validate_and_collect(samples: list[dict], tool_schema: dict,
                         source_info: dict) -> tuple[list[dict], list[dict]]:
    """Validate samples, attempt fixes, return (valid_samples, error_records)."""
    valid = []
    errors = []

    for i, sample in enumerate(samples):
        is_valid, errs = validate_data_format(sample)
        if is_valid:
            valid.append(sample)
            continue

        fixed = fix_sample_format(sample, tool_schema)
        is_valid, errs = validate_data_format(fixed)
        if is_valid:
            valid.append(fixed)
            continue

        errors.append({
            "source_index": source_info.get("_index", -1),
            "api_name": source_info.get("api_name", ""),
            "method": source_info.get("method", ""),
            "variation": source_info.get("variation_idx", -1),
            "error_type": "validation",
            "error_detail": "; ".join(errs),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    return valid, errors


# ────────────────────────────────────────────────────────────────────
# Resume support
# ────────────────────────────────────────────────────────────────────

def load_completed_keys(output_path: str) -> set[str]:
    """Scan existing output to find completed (source_index, variation_idx) pairs."""
    completed = set()
    path = Path(output_path)
    if not path.exists():
        return completed
    try:
        samples = load_jsonl(output_path)
        for s in samples:
            idx = s.get("_source_index")
            var = s.get("_variation_idx")
            if idx is not None and var is not None:
                completed.add(f"{idx}_{var}")
    except Exception as e:
        logger.warning("Could not read existing output for resume: %s", e)
    return completed


# ────────────────────────────────────────────────────────────────────
# File append helpers
# ────────────────────────────────────────────────────────────────────

def append_jsonl(records: list[dict], path: str):
    """Append records to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────────────────────────
# Job expansion: 1 record → N single-sample jobs
# ────────────────────────────────────────────────────────────────────

def expand_to_jobs(records: list[dict]) -> list[dict]:
    """Expand each prompt record into individual single-sample jobs."""
    jobs = []
    for record in records:
        target = record.get("target_samples", 1)
        for var_idx in range(target):
            hint = VARIATION_HINTS[(record["_index"] + var_idx) % len(VARIATION_HINTS)]
            prompt = build_single_prompt(record["prompt"], hint)
            jobs.append({
                "_index": record["_index"],
                "api_name": record.get("api_name", ""),
                "method": record.get("method", ""),
                "sub_type": record.get("sub_type"),
                "tier": record.get("tier"),
                "variation_idx": var_idx,
                "variation_hint": hint,
                "prompt": prompt,
                "tool_schema": record.get("tool_schema", {}),
            })
    return jobs


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate TRL training samples via OpenAI API (1 sample per request)."
    )
    parser.add_argument("--input", type=str,
                        default="data/prompts/training_prompts.jsonl")
    parser.add_argument("--output", type=str,
                        default="data/generated/training_samples.jsonl")
    parser.add_argument("--error-log", type=str,
                        default="data/generated/errors.jsonl")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only first 5 jobs")
    parser.add_argument("--tiers", type=str, default=None)
    parser.add_argument("--apis", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # ── Initialize client ──
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    logger.info("API 키 확인됨: %s...%s", api_key[:12], api_key[-4:])
    logger.info("모델: %s", args.model)

    # ── Load prompt records ──
    logger.info("프롬프트 로드: %s", args.input)
    records = load_jsonl(args.input)
    logger.info("총 %d개 프롬프트 레코드 로드됨", len(records))

    for i, rec in enumerate(records):
        rec["_index"] = i

    # ── Apply filters ──
    if args.tiers:
        tier_set = {int(t.strip()) for t in args.tiers.split(",")}
        records = [r for r in records if r.get("tier") in tier_set]
        logger.info("티어 필터 적용: %s → %d개 레코드", args.tiers, len(records))

    if args.apis:
        api_set = {a.strip() for a in args.apis.split(",")}
        records = [r for r in records if r.get("api_name") in api_set]
        logger.info("API 필터 적용: %s → %d개 레코드", args.apis, len(records))

    # ── Expand to individual jobs ──
    jobs = expand_to_jobs(records)
    logger.info("총 %d개 개별 요청으로 확장됨 (1 요청 = 1 샘플)", len(jobs))

    if args.dry_run:
        jobs = jobs[:5]
        logger.info("[DRY RUN] 첫 %d개 요청만 처리", len(jobs))

    # ── Resume: skip completed ──
    if args.resume:
        completed = load_completed_keys(args.output)
        before = len(jobs)
        jobs = [j for j in jobs if f"{j['_index']}_{j['variation_idx']}" not in completed]
        logger.info("Resume: %d개 완료됨, %d개 남음", before - len(jobs), len(jobs))

    if not jobs:
        print("처리할 요청이 없습니다.")
        return

    # ── Process jobs ──
    total_jobs = len(jobs)
    total_valid = 0
    total_errors = 0
    start_time = time.time()

    print(f"\n{'='*60}")
    print("학습 샘플 생성 시작 (1 요청 = 1 샘플)")
    print(f"{'='*60}")
    print(f"  총 요청: {total_jobs}개")
    print(f"  모델: {args.model}")
    print(f"{'='*60}\n")

    for i, job in enumerate(jobs):
        api_name = job["api_name"]
        method = job["method"]
        var_idx = job["variation_idx"]
        sub_type = job.get("sub_type")

        sub_label = f" ({sub_type})" if sub_type else ""
        short_hint = job["variation_hint"][:20]
        print(f"[{i+1}/{total_jobs}] {api_name} {method}{sub_label} "
              f"v{var_idx} [{short_hint}...] ", end="", flush=True)

        # Call LLM
        raw_response = call_openai(client, job["prompt"], args.model,
                                   args.temperature, args.max_retries)

        if raw_response is None:
            append_jsonl([{
                "source_index": job["_index"],
                "api_name": api_name,
                "method": method,
                "variation_idx": var_idx,
                "error_type": "api_error",
                "error_detail": "All retries exhausted",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }], args.error_log)
            total_errors += 1
            print(f"FAIL (API)  [누적: {total_valid}/{i+1}]")
            continue

        # Parse
        parsed = parse_llm_response(raw_response)
        if not parsed:
            append_jsonl([{
                "source_index": job["_index"],
                "api_name": api_name,
                "method": method,
                "variation_idx": var_idx,
                "error_type": "parse_error",
                "error_detail": f"No JSON parsed ({len(raw_response)} chars)",
                "raw_response_preview": raw_response[:2000],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }], args.error_log)
            total_errors += 1
            print(f"FAIL (parse)  [누적: {total_valid}/{i+1}]")
            continue

        # Validate
        source_info = {"_index": job["_index"], "api_name": api_name,
                       "method": method, "variation_idx": var_idx}
        valid_samples, validation_errors = validate_and_collect(
            parsed, job["tool_schema"], source_info)

        # Add metadata and save
        for sample in valid_samples:
            sample["_source_index"] = job["_index"]
            sample["_api_name"] = api_name
            sample["_method"] = method
            sample["_variation_idx"] = var_idx

        if valid_samples:
            append_jsonl(valid_samples, args.output)
            total_valid += len(valid_samples)

        if validation_errors:
            append_jsonl(validation_errors, args.error_log)
            total_errors += len(validation_errors)

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (total_jobs - i - 1) / rate if rate > 0 else 0
        eta_min = eta / 60

        ok_str = "OK" if valid_samples else "FAIL (valid)"
        print(f"{ok_str}  [누적: {total_valid}/{i+1} | ETA: {eta_min:.1f}min]")

    # ── Summary ──
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("생성 완료 요약")
    print(f"{'='*60}")
    print(f"  총 요청:         {total_jobs}")
    print(f"  생성 샘플:       {total_valid}")
    print(f"  달성률:          {total_valid/total_jobs*100:.1f}%")
    print(f"  에러:            {total_errors}")
    print(f"  소요 시간:       {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  출력 파일:       {args.output}")
    if total_errors > 0:
        print(f"  에러 로그:       {args.error_log}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
