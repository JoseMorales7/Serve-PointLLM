#!/usr/bin/env bash
# Run in the PointLLM model environment, not the LangChain client environment.
set -Eeuo pipefail
umask 077

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${POINTLLM_MODEL_PATH:-7b}"
host="${POINTLLM_HOST:-0.0.0.0}"
port="${POINTLLM_PORT:-8000}"
python_bin="${POINTLLM_PYTHON:-python}"
device="${POINTLLM_DEVICE:-cuda:0}"
dtype="${POINTLLM_DTYPE:-float16}"
key_file="$repo_root/serving_workdirs/api-key"
explicit_key_file=0
offline=0
install=0
certfile=""
tls_keyfile=""

usage() {
    cat <<'HELP'
Usage: bash scripts/serve_pointllm.sh [options]

  --model 7b|13b|PATH    Select one checkpoint (default: 7b).
  --host ADDRESS       Bind address (default: 0.0.0.0, accessible remotely).
  --port PORT          Listen port (default: 8000).
  --device DEVICE      Torch device (default: cuda:0).
  --dtype DTYPE        float16, bfloat16, or float32 (default: float16).
  --python EXECUTABLE  Python from the model environment (default: python).
  --api-key-file FILE  Read/create a private persistent API-key file.
  --offline            Never contact Hugging Face; require cached weights.
  --install            Install editable serving extras before starting.
  --ssl-certfile FILE  Serve HTTPS using this certificate.
  --ssl-keyfile FILE   Private key for the HTTPS certificate.
  -h, --help           Show this help.

Activate the repository's Python 3.10 model environment first, or use --python.
Serving dependencies are installed only with --install, not on every restart.
POINTLLM_API_KEY overrides the default key file unless --api-key-file is given.
Otherwise a key is created at serving_workdirs/api-key with mode 600, and reused.
The secret is never printed. The client model alias remains "pointllm".
Ctrl+C stops the server; restart with --model 13b/7b to switch models.
Only one instance of this launcher per checkout may run at a time.
HELP
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --offline) offline=1; shift ;;
        --install) install=1; shift ;;
        --model|--host|--port|--device|--dtype|--python|--api-key-file|--ssl-certfile|--ssl-keyfile)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "Missing value for $1"
            case "$1" in
                --model) model="$2" ;;
                --host) host="$2" ;;
                --port) port="$2" ;;
                --device) device="$2" ;;
                --dtype) dtype="$2" ;;
                --python) python_bin="$2" ;;
                --api-key-file) key_file="$2"; explicit_key_file=1 ;;
                --ssl-certfile) certfile="$2" ;;
                --ssl-keyfile) tls_keyfile="$2" ;;
            esac
            shift 2
            ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done

[[ "$port" =~ ^[0-9]{1,5}$ ]] || die "Port must be an integer from 1 to 65535"
((10#$port >= 1 && 10#$port <= 65535)) || die "Port must be from 1 to 65535"
case "$dtype" in float16|bfloat16|float32) ;; *) die "Invalid dtype: $dtype" ;; esac
command -v "$python_bin" >/dev/null || die "Python not found: $python_bin"
python_bin="$(command -v "$python_bin")"
[[ "$python_bin" == /* ]] || python_bin="$(pwd)/$python_bin"
command -v flock >/dev/null || die "Install util-linux (flock) to use this Linux launcher"
if [[ -n "$certfile" || -n "$tls_keyfile" ]]; then
    [[ -r "$certfile" && -r "$tls_keyfile" ]] || die "Provide readable --ssl-certfile and --ssl-keyfile together"
    certfile="$("$python_bin" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$certfile")"
    tls_keyfile="$("$python_bin" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$tls_keyfile")"
fi

# Resolve local paths relative to the caller, before changing to the repo root.
revision=""
case "$model" in
    7b|7B|RunsenXu/PointLLM_7B_v1.2)
        model="RunsenXu/PointLLM_7B_v1.2"
        revision="37d8c15aaff8e7e05f04729dcb6960d5758e9f86"
        ;;
    13b|13B|RunsenXu/PointLLM_13B_v1.2)
        model="RunsenXu/PointLLM_13B_v1.2"
        revision="6aafe45d6931f235617799fa7ba29580a794338a"
        ;;
    *)
        [[ -d "$model" ]] || die "Use --model 7b, --model 13b, or a local checkpoint directory"
        model="$(cd -- "$model" && pwd)"
        ;;
esac

if [[ "$explicit_key_file" == 1 || -z "${POINTLLM_API_KEY:-}" ]]; then
    # Relative key paths are also interpreted from the caller's directory.
    export POINTLLM_API_KEY
    POINTLLM_API_KEY="$("$python_bin" - "$key_file" <<'PY'
import os
import secrets
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().absolute()
path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    pass
else:
    with os.fdopen(fd, "w") as output:
        output.write(secrets.token_urlsafe(32) + "\n")
if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
    sys.exit(f"API-key file must be private: chmod 600 '{path}'")
print(path.read_text().strip())
PY
)"
    printf 'API-key file: %s (secret not displayed)\n' "$key_file"
fi
[[ ${#POINTLLM_API_KEY} -ge 32 && ! "$POINTLLM_API_KEY" =~ [[:space:]] ]] \
    || die "Use an API key of at least 32 characters without whitespace"
export POINTLLM_API_KEY

cd -- "$repo_root"
mkdir -p serving_workdirs
# Keep the lock FD open across exec, including all checkpoint loading time.
exec 9>>"${POINTLLM_SERVE_LOCK:-$repo_root/serving_workdirs/server.lock}"
flock --exclusive --nonblock 9 || die "Another launcher is running; stop it before switching models"

if ((install)); then
    "$python_bin" -m pip install -e '.[serve]'
fi
if ! "$python_bin" -c 'import uvicorn; from pointllm.serve.app import create_app'; then
    die "Serving dependencies are missing/incompatible; use the model environment and --install"
fi
if ((offline)); then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

if [[ -n "$revision" ]]; then
    printf 'Resolving %s at %s (cached weights preferred)\n' "$model" "$revision"
    model="$("$python_bin" - "$model" "$revision" "$offline" <<'PY'
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

repo, revision, offline = sys.argv[1:]
try:
    directory = snapshot_download(repo, revision=revision, local_files_only=True)
except LocalEntryNotFoundError:
    if offline == "1":
        sys.exit("Checkpoint is not cached; omit --offline to download it")
    directory = snapshot_download(repo, revision=revision)
print(directory)
PY
)"
fi
[[ -f "$model/config.json" && -f "$model/tokenizer_config.json" ]] \
    || die "Checkpoint directory must contain config.json and tokenizer_config.json"
export POINTLLM_MODEL_PATH="$model" POINTLLM_DEVICE="$device" POINTLLM_DTYPE="$dtype"

args=(-m uvicorn pointllm.serve.app:create_app --factory --host "$host" --port "$port" --workers 1)
scheme=http
if [[ -n "$certfile" ]]; then
    scheme=https
    args+=(--ssl-certfile "$certfile" --ssl-keyfile "$tls_keyfile")
else
    printf 'WARNING: HTTP is unencrypted. Use a trusted network/VPN or an HTTPS reverse proxy.\n' >&2
fi
printf 'Checkpoint: %s\nDevice: %s; dtype: %s\n' "$model" "$device" "$dtype"
printf 'Binding %s://%s:%s; clients use model alias "%s".\n' \
    "$scheme" "$host" "$port" "${POINTLLM_MODEL_ALIAS:-pointllm}"
printf 'Use this server'\''s reachable hostname/IP (not 0.0.0.0) in your application URL.\n'
printf 'Startup may take several minutes; wait for "Application startup complete".\n'
exec "$python_bin" "${args[@]}"
