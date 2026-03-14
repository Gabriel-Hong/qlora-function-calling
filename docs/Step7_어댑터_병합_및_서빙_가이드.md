# Step 7-8: 어댑터 병합 및 vLLM 서빙 가이드

## 개요

| 항목 | 값 |
|------|-----|
| 이전 단계 | Step 6: 평가 완료 (tool name acc 96.65%, param acc 16.18%) |
| 본 단계 목표 | LoRA 어댑터를 base 모델에 병합 → vLLM으로 OpenAI 호환 API 서빙 |
| 환경 | RTX 3060 Laptop 6GB, Windows 11 + WSL2 Ubuntu, CUDA 12.8 |
| 작성일 | 2026-03-12 |

---

## 파이프라인 전체 흐름

```
[Base 모델]  +  [LoRA Adapter]     (Step 7: 병합)
Qwen2.5-1.5B     final_adapter/          ↓
  (~3 GB)          (~37 MB)        [Merged 모델]
                                    models/final/
                                      (~3 GB)
                                         ↓
                                   (Step 8: 서빙)
                                         ↓
                                   [vLLM 서버]
                                   localhost:8000
                                         ↓
                              OpenAI 호환 REST API
                          /v1/chat/completions
                          /v1/models
```

---

## Step 7: 어댑터 병합

### 7.1 개념

학습 결과물은 두 조각으로 나뉘어 있다:

| 구성 요소 | 경로 | 크기 | 설명 |
|-----------|------|------|------|
| Base 모델 | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | HuggingFace 원본 (frozen) |
| LoRA 어댑터 | `models/checkpoints/final_adapter/` | 36.9 MB | 학습된 delta 가중치 (Epoch 4 Best) |

추론 시 매번 두 조각을 조합하는 것은 비효율적이므로, **하나의 독립 모델로 병합(merge)**하여 저장한다.

```
병합 원리: merged_weight = base_weight + (lora_A × lora_B) × (alpha / r)
```

- `merge_and_unload()`: LoRA 가중치를 base에 흡수한 뒤 LoRA 레이어를 제거
- 결과물은 원본 Qwen2.5와 동일한 구조지만, fine-tuned 가중치가 반영된 모델

### 7.2 실행

```bash
# Windows 또는 WSL2에서 실행
python scripts/05_merge_adapters.py \
    --adapter-path models/checkpoints/final_adapter \
    --output-dir models/final
```

### 7.3 스크립트 동작 (05_merge_adapters.py)

| 단계 | 동작 | 비고 |
|------|------|------|
| Step 1/4 | Base 모델을 CPU에 float16으로 로드 | GPU 메모리 절약을 위해 CPU 로드 |
| Step 2/4 | LoRA 어댑터 로드 및 적용 | `PeftModel.from_pretrained()` |
| Step 3/4 | 어댑터 가중치를 base에 병합 | `model.merge_and_unload()` |
| Step 4/4 | 병합된 모델 + 토크나이저 저장 | `models/final/` |

### 7.4 산출물

| 파일 | 설명 |
|------|------|
| `models/final/config.json` | 모델 아키텍처 설정 |
| `models/final/model.safetensors` | 병합된 가중치 (~3 GB) |
| `models/final/tokenizer.json` | 토크나이저 |
| `models/final/tokenizer_config.json` | 토크나이저 설정 |

---

## Step 8: vLLM 서빙

### 8.1 서빙이란?

모델 파일은 그 자체로는 가중치 덩어리일 뿐이다. "서빙"이란 이것을 **HTTP API 서버로 띄워서 외부에서 요청을 보내고 응답을 받을 수 있게 하는 것**을 의미한다.

서빙 엔진이 담당하는 역할:
1. GPU 메모리에 모델 로드
2. 토큰화 → 추론 → 디토큰화 파이프라인
3. 여러 요청의 효율적인 배치 처리
4. HTTP API 엔드포인트 제공

### 8.2 서빙 엔진 비교

| 엔진 | 특징 | 장점 | 단점 |
|------|------|------|------|
| **vLLM** | PagedAttention, OpenAI 호환 API | 프로덕션급, tool calling 지원, 높은 처리량 | Linux 전용 |
| **Transformers 직접** | `model.generate()` | 간단, 크로스 플랫폼 | 배치 처리 없음, API 직접 구현 필요 |
| **llama.cpp** | GGUF 포맷, CPU 지원 | 경량, CPU에서도 동작 | GGUF 변환 필요 |
| **TGI** | HuggingFace 공식 | Docker 기반 배포 용이 | 설정 복잡 |
| **Ollama** | GGUF 기반, CLI 간편 | 설치/실행 매우 쉬움 | tool calling 제한적 |

### 8.3 vLLM 선택 이유

1. **OpenAI 호환 API 내장**: 서버 실행만으로 `/v1/chat/completions` 엔드포인트 자동 생성 → 기존 OpenAI SDK 코드 재사용 가능
2. **Tool Calling 자동 파싱**: Qwen2.5의 `<tool_call>` 포맷을 자동 감지하여 `tool_calls` 필드로 구조화 반환 (Hermes parser)
3. **PagedAttention**: GPU 메모리를 페이지 단위로 관리하여 동일 GPU에서 더 많은 동시 요청 처리
4. **프로덕션 확장성**: GEN NX 제품에 직접 붙일 수 있는 표준 API 인터페이스

### 8.4 Windows 환경 대응

vLLM은 **Linux 전용**이므로 Windows에서는 아래 두 가지 방법을 사용한다:

#### 방법 1: WSL2 (권장)

WSL2 Ubuntu에서 vLLM 서버를 실행하고, Windows에서 API를 호출한다.

```bash
# WSL2 Ubuntu 터미널
pip install vllm openai
cd /mnt/c/MIDAS_Source/qlora-function-calling

# 서버 시작
python scripts/07_serve_vllm.py server
```

```bash
# Windows 터미널 (또는 다른 WSL 터미널)
python scripts/07_serve_vllm.py test --host localhost
python scripts/07_serve_vllm.py client --host localhost
```

WSL2는 Windows와 localhost를 공유하므로, Windows 앱에서 `http://localhost:8000`으로 바로 접근 가능하다.

#### 방법 2: Docker

```bash
docker run --gpus all -p 8000:8000 \
    -v ./models/final:/model \
    vllm/vllm-openai:latest \
    --model /model \
    --dtype float16 \
    --max-model-len 2048 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### 8.5 스크립트 사용법 (07_serve_vllm.py)

3가지 모드를 제공한다:

| 모드 | 명령 | 설명 |
|------|------|------|
| **server** | `python scripts/07_serve_vllm.py server` | vLLM API 서버 실행 |
| **test** | `python scripts/07_serve_vllm.py test` | 서버 연결 확인 + tool call 테스트 |
| **client** | `python scripts/07_serve_vllm.py client` | 대화형 CLI 클라이언트 |

#### 서버 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model-path` | `models/final` | 병합된 모델 경로 |
| `--host` | `0.0.0.0` | 바인딩 호스트 |
| `--port` | `8000` | 서버 포트 |
| `--max-model-len` | `2048` | 최대 시퀀스 길이 |
| `--gpu-memory-utilization` | `0.85` | GPU 메모리 사용 비율 |
| `--enable-tool-calling` | `True` | tool call 자동 파싱 (Hermes parser) |
| `--quantization` | `None` | 양자화 방식 (awq, gptq 등) |

### 8.6 외부 앱에서 호출 예시

서버가 실행되면 어떤 언어/프레임워크에서든 OpenAI SDK로 연결 가능하다:

#### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="models/final",
    messages=[
        {"role": "system", "content": "You are a structural engineering assistant for GEN NX."},
        {"role": "user", "content": "1층 기둥 단면 조회해줘"},
    ],
    tools=[...],  # GEN NX API 스키마
)

# tool_calls 필드에 파싱된 결과
for tc in response.choices[0].message.tool_calls:
    print(tc.function.name)       # "GET /db/sect"
    print(tc.function.arguments)  # '{"Assign": {"1": {}}}'
```

#### curl
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/final",
    "messages": [
      {"role": "user", "content": "좌표 (0, 0, 0)에 노드를 생성해줘"}
    ],
    "tools": [...]
  }'
```

#### JavaScript/TypeScript
```typescript
import OpenAI from "openai";

const client = new OpenAI({
    baseURL: "http://localhost:8000/v1",
    apiKey: "unused",
});

const response = await client.chat.completions.create({
    model: "models/final",
    messages: [{ role: "user", content: "노드 1에 고정 지점 추가해줘" }],
    tools: [...],
});
```

### 8.7 vLLM 핵심 개념

#### PagedAttention
- 기존 방식: 추론 시 KV Cache가 연속 메모리 블록을 차지 → 메모리 낭비
- vLLM: OS의 가상 메모리처럼 KV Cache를 **페이지 단위로 분할** 관리
- 효과: 동일 GPU에서 2~4배 더 많은 동시 요청 처리 가능

#### Continuous Batching
- 기존 방식: 한 배치의 모든 요청이 끝날 때까지 대기
- vLLM: 요청이 끝나는 즉시 새 요청을 배치에 추가
- 효과: GPU 유휴 시간 최소화, 처리량(throughput) 향상

#### Hermes Tool Call Parser
- Qwen2.5 계열이 사용하는 `<tool_call>{"name": ..., "arguments": ...}</tool_call>` 포맷을 자동 감지
- API 응답의 `tool_calls` 필드로 구조화하여 반환
- 클라이언트는 별도 파싱 없이 `response.choices[0].message.tool_calls`로 접근

---

## 실행 순서 요약

```bash
# === Step 7: 어댑터 병합 (Windows에서 실행) ===
python scripts/05_merge_adapters.py \
    --adapter-path models/checkpoints/final_adapter \
    --output-dir models/final

# === Step 8: vLLM 서빙 ===

# (WSL2 Ubuntu 터미널)
pip install vllm openai
cd /mnt/c/MIDAS_Source/qlora-function-calling
python scripts/07_serve_vllm.py server

# (다른 터미널 — Windows 또는 WSL2)
python scripts/07_serve_vllm.py test
python scripts/07_serve_vllm.py client
```

---

## 기존 서빙 방식과의 비교 (06_serve.py vs 07_serve_vllm.py)

| 항목 | 06_serve.py (Transformers) | 07_serve_vllm.py (vLLM) |
|------|---------------------------|------------------------|
| 추론 방식 | `model.generate()` 직접 호출 | vLLM 서버 프로세스 |
| API 제공 | 없음 (CLI/Gradio만) | OpenAI 호환 REST API |
| 동시 요청 | 1개 | 여러 개 (Continuous Batching) |
| Tool Call 파싱 | 수동 (`parse_tool_calls_from_output`) | 자동 (Hermes parser) |
| 메모리 효율 | 기본 | PagedAttention |
| 제품 연동 | 어려움 | `base_url` 변경만으로 연동 |
| 플랫폼 | Windows/Linux | Linux 전용 (WSL2/Docker) |
| 적합한 용도 | 빠른 테스트, 데모 | 프로덕션, 제품 연동 |

---

## 리소스 예상

| 항목 | 예상 값 | 비고 |
|------|---------|------|
| 모델 크기 (fp16) | 2.89 GiB | 1.5B 파라미터 |
| GPU 메모리 사용 | 5.4 / 6.0 GiB (90%) | gpu-memory-utilization=0.70 기준 |
| 서버 시작 시간 | ~90초 | 모델 로드 + torch.compile + CUDA graph |
| KV cache | 0.74 GiB (27,888 토큰) | max-model-len=2048 기준 |
| 비용 | 0원 (로컬) | 로컬 GPU 사용 시 |

---

## 트러블슈팅 기록

### Issue #1: tokenizer_config.json — extra_special_tokens 포맷 불일치

- **증상**: vLLM 서버 시작 시 `AttributeError: 'list' object has no attribute 'keys'`
- **원인**: 병합 시 Windows의 transformers 버전이 `extra_special_tokens`를 리스트로 저장했으나, WSL2의 transformers 4.57.6은 딕셔너리를 기대
- **수정**: `models/final/tokenizer_config.json`에서 리스트를 딕셔너리로 변환
  ```json
  // Before (리스트)
  "extra_special_tokens": ["<|im_start|>", "<|im_end|>", ...]
  // After (딕셔너리)
  "extra_special_tokens": {"im_start": "<|im_start|>", "im_end": "<|im_end|>", ...}
  ```

### Issue #2: GPU 메모리 부족 — gpu-memory-utilization 0.85

- **증상**: `ValueError: Free memory on device cuda:0 (4.99/6.0 GiB) ... less than desired GPU memory utilization (0.85, 5.1 GiB)`
- **원인**: Windows 디스플레이 드라이버가 ~1GB를 이미 점유하여 여유 메모리가 4.99GB뿐
- **수정**: `--gpu-memory-utilization 0.70`으로 낮춤 (4.2GB 요청). 1.5B 모델에 충분

### Issue #3: Triton 컴파일 실패 — gcc, Python.h 누락

- **증상**: `CalledProcessError: Command '/usr/bin/gcc' ... returned non-zero exit status 1` 및 `Python.h: No such file or directory`
- **원인**: WSL2 Ubuntu에 개발 도구가 미설치. vLLM의 Triton이 런타임에 C 코드를 컴파일하려면 gcc + Python 헤더 + CUDA Toolkit이 필요
- **수정**:
  ```bash
  sudo apt-get install -y gcc python3.10-dev
  # CUDA Toolkit (드라이버 제외)
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8
  # PATH 설정
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
  ```
- **대안**: `--enforce-eager` 옵션으로 Triton 컴파일을 건너뛸 수 있음 (성능 약간 저하, 1.5B 모델에서는 큰 차이 없음)

### Issue #4: Tool calling 시 context length 초과

- **증상**: `Error code: 400 - You passed 1793 input tokens and requested 256 output tokens. However, the model's context length is only 2048 tokens`
- **원인**: tool 스키마 15개를 모두 보내면 입력 토큰만 ~1,793개 소비 → 출력 여유가 없음
- **수정**: 요청 시 필요한 tool만 선별하여 전송 (2~5개). RTX 3060 6GB에서는 `--max-model-len 2048`이 한계이므로 4096 확장은 메모리 부족으로 불가
- **프로덕션 권장**: 사용자 의도 분류 → 관련 tool 선별 → 모델 호출의 2단계 파이프라인 구성

### 최종 동작 확인 서버 실행 명령

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/c/MIDAS_Source/qlora-function-calling/models/final \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 2048 \
    --dtype float16 \
    --gpu-memory-utilization 0.70 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### 최종 리소스 사용 (실측)

| 항목 | 실측 값 |
|------|---------|
| 모델 로드 | 2.89 GiB, 28초 |
| KV cache | 0.74 GiB (27,888 토큰) |
| CUDA graphs | 0.48 GiB |
| 전체 GPU 사용 | 5.4 / 6.0 GiB (90%) |
| 서버 시작 총 시간 | ~90초 (로드 28초 + 컴파일 26초 + 그래프 캡처 7초 + 기타) |
| 최대 동시 요청 | 13.62x (2048 토큰 기준) |

---

## 향후 고려사항

### 프로덕션 배포 시
- **전용 GPU 서버**: 사내 서버 또는 클라우드 GPU 인스턴스 (AWS g5, GCP A100 등)
- **AWQ/GPTQ 양자화**: 4-bit 양자화로 메모리 절반 이하, 속도 향상
- **로드 밸런싱**: 여러 vLLM 인스턴스 + Nginx/HAProxy
- **모니터링**: Prometheus + Grafana로 latency, throughput, GPU 사용률 추적

### 모델 개선 시
- Parameter accuracy 개선 후 새 어댑터 병합 → 모델 교체만으로 서버 업데이트 가능
- vLLM은 `--model` 경로만 변경하면 되므로 배포 파이프라인이 단순
