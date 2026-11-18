# OpenAI-compatible PointLLM service

The service exposes Chat Completions plus a custom point-cloud upload API.
It does not render point clouds to images: the uploaded XYZRGB tensor goes
through PointLLM's point encoder.

## Run

### Remote launcher (recommended)

Activate the repository's model environment first (not the LangChain client
environment). The launcher includes dependency installation as an explicit
option, selects the cached checkpoint, enables authentication, and runs Uvicorn
with one worker:

```bash
cd /workspace/PointLLM
conda activate pointllm  # If this is the environment created using the README.

# First setup: install serving extras; use the already downloaded 7B weights.
bash scripts/serve_pointllm.sh --model 7b --install --offline

# Later starts do not need --install.
bash scripts/serve_pointllm.sh --model 7b --offline
```

The script runs in the foreground. To switch models, stop the running server
with **Ctrl+C**, wait for shutdown, then run:

```bash
bash scripts/serve_pointllm.sh --model 13b --offline
```

Only one instance of this launcher per checkout can run at a time; the lock is
held before checkpoint loading and released when the process exits. This does
not manage or stop servers launched separately. Cloud IDs disappear on restart.
If necessary, select a Python interpreter instead of activating an environment:

```bash
bash scripts/serve_pointllm.sh --model 7b --offline \
  --python /path/to/model/environment/bin/python
```

`7b` and `13b` select the pinned v1.2 snapshots tested with this service. Cached
snapshots are preferred without network access; omitting `--offline` permits
downloading a missing snapshot. You can also use `--model /absolute/checkpoint`.
`--install` still needs package-network access; `--offline` only applies to
checkpoint loading.

The default bind is `0.0.0.0:8000`, device `cuda:0`, and dtype `float16`.
Override with `--host`, `--port`, `--device`, and `--dtype`. Run
`bash scripts/serve_pointllm.sh --help` for all options.

If `POINTLLM_API_KEY` is unset, the script creates and reuses a random secret in
`serving_workdirs/api-key` (mode 600, ignored by Git). It never prints the secret.
To use an existing private key file, pass `--api-key-file /path/to/key`. The
launcher requires a key of at least 32 characters without whitespace. An
explicit key-file option takes precedence over `POINTLLM_API_KEY`.

Once startup finishes, verify locally:

```bash
curl --fail http://127.0.0.1:8000/ready
export POINTLLM_API_KEY="$(cat serving_workdirs/api-key)"
curl --fail http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $POINTLLM_API_KEY"
```

Your remote application uses `http://SERVER_IP:8000/v1`, model alias `pointllm`,
and the same API key. Transfer the key securely to the application's secret
configuration; do not commit it. `0.0.0.0` is a bind address, not a client URL.
Allow inbound traffic only from trusted client IPs using your host firewall
and cloud security-group/network rules.

**Plain HTTP does not encrypt the API key or point-cloud data.** Use HTTP only
on a trusted network/VPN; for public access use an HTTPS reverse proxy or supply
your own certificate and private key:

```bash
bash scripts/serve_pointllm.sh --model 7b --offline --port 8443 \
  --ssl-certfile /path/to/fullchain.pem \
  --ssl-keyfile /path/to/privkey.pem
```

Clients then use `https://YOUR_CERTIFICATE_HOSTNAME:8443/v1`. Do not disable
certificate verification. The launcher does not provision DNS, certificates,
firewall rules, or background-service supervision.

### Direct Uvicorn command

Use the repository's model environment (Python 3.10, PyTorch 2.0.1, and the
Transformers Git revision pinned in `pyproject.toml`). From the repository root:

```bash
python -m pip install -e '.[serve,test]'
export POINTLLM_MODEL_PATH=RunsenXu/PointLLM_7B_v1.2
export POINTLLM_DEVICE=cuda:0
export POINTLLM_DTYPE=float16
python -m uvicorn pointllm.serve.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

The model downloads on first startup unless `POINTLLM_MODEL_PATH` points to a
local trained checkpoint. Do not use an `_init` training checkpoint. The README
estimates 14 GB GPU memory for 7B FP16; leave additional memory for inference.
The serving extra constrains NumPy to `<2` for PyTorch 2.0.1 compatibility and
uses a compatible `plyfile` version.

**Use one worker.** Each process loads its own model and has its own cloud
store. Do not use reload mode in production. Startup fails if model loading
fails; `/ready` returns success only after loading. `/health` is unauthenticated.

Set `POINTLLM_API_KEY` to require `Authorization: Bearer <key>` on all `/v1`
routes. Without it, the API is unauthenticated; keep the default localhost
binding. If exposing the service, configure authentication, TLS, request limits,
and buffering/timeouts at your reverse proxy. Disable proxy buffering for SSE.

## Upload and infer

Upload **raw `.npy` bytes**, not multipart form data. The array must:

- Have shape `(N, 6)` with XYZ followed by RGB in `[0, 1]`.
- Contain 8,192–32,768 points by default, real numeric values, no NaN/Inf,
  and nonzero spatial extent.
- Use NPY v1/v2 without pickle objects, structured values, or trailing data.

Larger-than-8,192 clouds are downsampled using farthest-point sampling.
The sampled XYZ coordinates are centered and normalized to a unit sphere;
RGB is unchanged. XYZ-only clouds, PLY, byte-range RGB, and clouds with too
few points are explicitly rejected.

```bash
curl --fail http://localhost:8000/v1/point-clouds \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @chair.npy
```

The response contains an opaque `id` such as `pc-...`, `created`, `expires_at`,
and `point_count`. Use that ID as `point_cloud_id`:

```bash
curl --fail http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "pointllm",
    "point_cloud_id": "pc-REPLACE_WITH_UPLOAD_ID",
    "messages": [{"role": "user", "content": "Describe this object."}],
    "max_completion_tokens": 256,
    "temperature": 0
  }'
```

Clouds are held in bounded process memory, expire after one hour by default,
and disappear on restart. Expired entries are removed on store access.
Delete explicitly with `DELETE /v1/point-clouds/{id}`. Queued/running requests
retain their cloud reference even if it is deleted or expires.

## Python clients

Install current clients **in a separate environment** rather than upgrading
the legacy model dependencies:

```bash
python -m venv /tmp/pointllm-client
/tmp/pointllm-client/bin/python -m pip install openai langchain-openai httpx
```

Both clients use the same upload:

```python
from pathlib import Path
import httpx

base_url = "http://localhost:8000"
api_key = "local"  # Replace if POINTLLM_API_KEY is configured.
headers = {"Authorization": f"Bearer {api_key}"}

response = httpx.post(
    f"{base_url}/v1/point-clouds",
    headers={**headers, "Content-Type": "application/octet-stream"},
    content=Path("chair.npy").read_bytes(),
    timeout=120,
)
response.raise_for_status()
cloud_id = response.json()["id"]
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key, timeout=180)
messages = [{"role": "user", "content": "Describe this object."}]
answer = client.chat.completions.create(
    model="pointllm",
    messages=messages,
    max_completion_tokens=256,
    extra_body={"point_cloud_id": cloud_id},
)
print(answer.choices[0].message.content)
```

For streaming, add `stream=True` and optionally
`stream_options={"include_usage": True}`. Each chunk uses the standard Chat
Completions SSE format; the last event is `[DONE]`.

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="pointllm",
    base_url=f"{base_url}/v1",
    api_key=api_key,
    temperature=0,
    max_tokens=256,
    timeout=180,
    use_responses_api=False,
)
cloud_llm = llm.bind(extra_body={"point_cloud_id": cloud_id})
answer = cloud_llm.invoke("Describe this object.")
print(answer.content)

# Resend the same cloud ID and the complete history for follow-up questions.
history = [
    ("human", "Describe this object."),
    answer,
    ("human", "What color is it?"),
]
for chunk in cloud_llm.stream(history):
    print(chunk.content, end="", flush=True)
```

## Supported contract and limits

- Routes: `POST /v1/point-clouds`, `DELETE /v1/point-clouds/{id}`,
  `GET /v1/models`, `POST /v1/chat/completions`, `GET /health`, `GET /ready`.
- The chat model name is the configured alias, `pointllm` by default, not the
  checkpoint path.
- `point_cloud_id` is a custom extension sent via `extra_body`. There is no
  standard OpenAI point-cloud content type.
- Messages support an optional leading text `system` message, followed by
  alternating `user`/`assistant` text messages ending in `user`. One cloud is
  attached to the first user message when reconstructing the prompt. Histories
  are request-local; start fresh history when switching clouds.
- Supported controls: `temperature` (zero means greedy), `top_p`,
  `max_tokens`/`max_completion_tokens`, `stream`, `stream_options.include_usage`,
  and `n=1`. Token-limit aliases must agree if both are present; the default
  output limit is 256.
- Prompt plus output must fit the checkpoint's context limit, capped at 2048
  by default. The prompt count includes the point prefix and conversation
  template. Requests are rejected rather than silently truncating history.
- Usage counts model tokens, including a generated EOS where applicable.
  Finish reasons distinguish `stop` from `length`.
- Responses API, tool calls, JSON-schema enforcement, custom `stop`, multimodal
  message blocks, and unknown parameters are not supported and are rejected.
- Streaming publishes complete words (or CJK characters) during generation,
  not a post-generation animation. Errors after streaming starts arrive as an
  SSE `error` object followed by `[DONE]`.
- One generation runs at a time, with bounded admission including queued jobs.
  Queue saturation returns 429. Generation deadlines include queueing; timed-out
  or disconnected jobs retain capacity until their worker actually stops.
  Cancellation is cooperative between decoding steps: a running GPU kernel or
  point-encoder prefill cannot be interrupted immediately.
- Upload processing is serialized; concurrent uploads return 429. Preprocessing
  and GPU work run outside the async event loop.

Every setting below is an environment variable with prefix `POINTLLM_`:

| Name | Default |
|---|---|
| `MODEL_PATH` | `RunsenXu/PointLLM_7B_v1.2` |
| `MODEL_ALIAS` | `pointllm` |
| `DEVICE` | `cuda:0` |
| `DTYPE` | `float16` (`float32` and `bfloat16` also supported) |
| `API_KEY` | unset |
| `MAX_CONTEXT_TOKENS` | `2048` (also capped by checkpoint capacity) |
| `MAX_UPLOAD_BYTES` | `16777216` |
| `MAX_REQUEST_BYTES` | `1048576` (chat JSON) |
| `MAX_INPUT_POINTS` | `32768` |
| `MAX_CLOUDS` | `32` |
| `CLOUD_TTL_SECONDS` | `3600` |
| `MAX_PENDING_REQUESTS` | `4` (running plus queued) |
| `REQUEST_TIMEOUT_SECONDS` | `120` (generation; also body read deadline) |

Errors use `{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`.
Validation errors return 400, unknown IDs/models 404, body limits 413,
unsupported upload content types 415, admission/storage limits 429, body-read
timeouts 408, and generation timeouts 504. Internal details stay in server logs.

## Verification

From the repository root:

```bash
# CPU/API tests; no weights, CUDA, Torch, or Transformers imports are required.
python -m pytest -q -m 'not gpu'

# Include real LangChain HTTP contract tests using the separate client environment.
POINTLLM_CLIENT_PYTHON=/tmp/pointllm-client/bin/python python -m pytest -q -m 'not gpu'

# Launcher validation (no model is loaded or network listener exposed).
python -m pytest -q tests/test_serve_script.py

# Execute the real encoder + LLaMA on CUDA with a tiny random local checkpoint.
# This tests FP32/FP16/BF16 backend wiring, NOT pretrained model quality.
POINTLLM_TEST_TINY_GPU=1 python -m pytest -q -m gpu

# Test a downloaded trained checkpoint instead; no automatic test-time download.
POINTLLM_TEST_MODEL=/absolute/path/to/PointLLM_7B_v1.2 python -m pytest -q -m gpu

# Live HTTP test: load a trained checkpoint once, defaulting to FP16.
# Runs uploads, OpenAI SDK chat/follow-up/streaming, and optional LangChain clients.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
POINTLLM_TEST_MODEL=/absolute/path/to/cached/snapshot \
POINTLLM_CLIENT_PYTHON=/tmp/pointllm-client/bin/python \
POINTLLM_TEST_REPORT=/tmp/pointllm-inference-report.json \
python -m pytest -q -s tests/test_pretrained_endpoint.py -m gpu
```

The OpenAI SDK contract test runs when `openai` is installed. The LangChain test
is skipped unless `POINTLLM_CLIENT_PYTHON` is supplied. GPU tests are opt-in and
skip without the relevant environment variable or CUDA. They use a synthetic
valid colored cloud to check execution, not semantic accuracy.

The live pretrained test uses a reproducible synthetic red-chair surface cloud
and records actual answers, usage, startup time, streaming chunk count, and
time to first content. `POINTLLM_TEST_REPORT` optionally writes those results
as JSON. `POINTLLM_TEST_DEVICE` and `POINTLLM_TEST_DTYPE` override `cuda:0` and
`float16` for this test. A successful test establishes inference/client wiring,
not object-recognition accuracy on a real-world dataset.

For a quality smoke test, run the service with trained weights and a known real
object cloud, then try both clients above and check the initial answer, follow-up
grounding, and streaming. `pointllm/eval/evaluator.py` is not used: it calls paid
external GPT APIs.
