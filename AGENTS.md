# PointLLM endpoint implementation

## Runtime and existing entrypoints
- The HTTP entrypoint is `pointllm.serve.app:create_app` (Uvicorn `--factory --workers 1`); `docs/serving.md` defines the upload/API contract and configuration. `pointllm/eval/utils.py` is an external OpenAI evaluation client, not a serving backend.
- Remote launcher: `bash scripts/serve_pointllm.sh --model 7b|13b --offline` (choose one value). It pins tested snapshots, requires authentication, defaults to `0.0.0.0:8000`, and holds a per-checkout `flock` across Uvicorn exec; do not remove single-process protection. Generated secrets live in ignored `serving_workdirs/api-key`, never logs.
- Install from the repository root with `python -m pip install -e .`. Prefer the README's tested Python 3.10/PyTorch 2.0.1/CUDA 11.7 baseline; preserve the pinned Transformers Git revision `cae78c46` and `tokenizers==0.12.1` unless deliberately migrating the custom model.
- Keep modern LangChain client dependencies separate from the legacy model environment. FlashAttention installation and the training scripts are not inference prerequisites.
- Install serving/test extras with `python -m pip install -e '.[serve,test]'`; serving constrains NumPy `<2` and `plyfile<1.1` for the PyTorch 2.0.1 stack. Preserve lazy imports in `pointllm/__init__.py` and the backend boundary: CPU API tests must not import the model stack.
- Use `RunsenXu/PointLLM_7B_v1.2` (or a local trained checkpoint), not an `_init` training checkpoint. Weights and point-cloud data are not bundled. README estimates 14 GB GPU memory for 7B FP16; allow additional serving headroom.
- `pointllm/eval/PointLLM_chat.py` demonstrates inference but currently reads nonexistent `args.model_path` while its parser defines `--model_name`; do not reuse its initializer unchanged. The actual data option is `--data_path`, not the README example's `--data_name`.
- `pointllm/eval/chat_gradio.py` contains upload preprocessing and dialogue handling, but imports UI/Objaverse dependencies; extract the inference logic rather than importing the demo as the server.

## Model invocation invariants
- Load `PointLLMLlamaForCausalLM` from `pointllm.model` and its `AutoTokenizer` once at startup; call `initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)` before reading point-token metadata. The similarly named method without `_wo_embedding` resizes embeddings for training.
- Use `model.eval()` and `torch.inference_mode()`. Existing inference uses CUDA; move cloud tensors to the model's device and floating dtype, leaving text token IDs integer.
- Build prompts with a fresh `conv_templates["vicuna_v1_1"].copy()` per request/conversation, not `default_conversation` or shared mutable history. Map API roles to the template's roles and handle system text separately.
- Prepend the point prefix only to the first user message in the reconstructed history: `default_point_patch_token * point_token_len`, wrapped in configured start/end tokens when `mm_use_point_start_end` is true. Read these values from `model.get_model().point_backbone_config`; do not hardcode patch counts or use a literal `<point>` as the final prompt.
- Append the assistant placeholder (`None`) before `get_prompt()`. Pass the actual batched cloud through `model.generate(..., point_clouds=...)` on every request, including follow-ups; point-token text alone does not encode geometry.
- The shipped demos budget 2048 total tokens, including point tokens and history. Check prompt plus requested output against the checkpoint's context limit; `max_length` is total length, unlike `max_new_tokens`.
- Decode only `output_ids[:, input_ids.shape[1]:]`, skip special tokens, and remove the conversation stop suffix. Recreate `KeywordsStoppingCriteria` per generation; it stores mutable state and inspects batch element zero.

## Point-cloud boundary
- Reference preprocessing: `pointllm/data/utils.py` (`pc_norm`, `load_objaverse_point_cloud`, `farthest_point_sample`). Objaverse lookup expects `<object_id>_8192.npy`; that loader normalizes XYZ but does not sample or rescale RGB.
- Colored v1.2 inputs are normally `(N, 6)` arrays (`xyz`, then RGB in `[0, 1]`), batched to `(1, N, 6)`; training uses 8192 points. Explicitly define sampling/short-cloud behavior and any XYZ-only fallback instead of relying on demo filename heuristics.
- Validate numeric shape, nonempty input, finite values, color range, and nonzero spatial extent before `pc_norm`: its unit-sphere normalization divides by the maximum centered radius without guarding zero.
- Load uploaded NumPy arrays with `allow_pickle=False`; serving uses opaque in-memory cloud IDs, never client-supplied filesystem paths. Bound upload size and point count: the NumPy farthest-point sampler loops over every requested sample.

## OpenAI/LangChain contract and verification
- `/v1/point-clouds` accepts raw NPY bytes (not multipart); its expiring in-memory ID is passed to `/v1/chat/completions` as `point_cloud_id` via LangChain's `extra_body`. Clients should use Chat Completions, not the Responses API.
- Accept/map both `max_tokens` and `max_completion_tokens`; return OpenAI-shaped assistant choices, finish reasons, usage, and errors. Reject unsupported options explicitly rather than pretending tool calling or structured-output enforcement exists.
- Gradio's character-by-character output is not token streaming. Serving uses a stopping-criteria callback because pinned Transformers lacks the modern `streamer` API; preserve genuine incremental SSE and disconnect/backpressure cancellation.
- Keep GPU execution behind a backend boundary so API validation/serialization tests can run without loading weights. Load once per serving process, avoid accidental multi-worker model duplication, and isolate request histories.
- CPU tests: `python -m pytest -q -m 'not gpu'`; add `POINTLLM_CLIENT_PYTHON=/path/to/client/venv/bin/python` to include the live LangChain test. OpenAI client tests skip when that SDK is unavailable.
- GPU tests: `POINTLLM_TEST_TINY_GPU=1 python -m pytest -q -m gpu` executes the real architecture with random tiny weights in FP32/FP16/BF16; `POINTLLM_TEST_MODEL=/local/trained/checkpoint` selects trained weights instead. Tests do not download checkpoints, and tiny-weight passes are not evidence of pretrained answer quality.
- For a single-load pretrained HTTP smoke test: `POINTLLM_TEST_MODEL=/local/snapshot POINTLLM_CLIENT_PYTHON=/client/venv/bin/python python -m pytest -q -s tests/test_pretrained_endpoint.py -m gpu`. It defaults to FP16 and uses a synthetic red chair; `POINTLLM_TEST_REPORT=/path/report.json` saves answers and timings.
- Real-model verification needs downloaded weights, CUDA, and a valid cloud: exercise initial and follow-up requests through the OpenAI client and LangChain, plus malformed/degenerate clouds, context overflow, and whichever streaming behavior is supported. Do not use `pointllm/eval/evaluator.py` as a local smoke test: it calls paid external GPT APIs.
