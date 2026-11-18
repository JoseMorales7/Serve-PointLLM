import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_pointllm.sh"


@pytest.fixture
def launch_env(tmp_path):
    checkpoint = tmp_path / "checkpoint with spaces"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "tokenizer_config.json").write_text("{}")
    output = tmp_path / "launch.json"
    wrapper = tmp_path / "python"
    wrapper.write_text(f"""#!{sys.executable}
import json, os, sys
if sys.argv[1:3] == ["-m", "uvicorn"]:
    with open(os.environ["TEST_LAUNCH_OUTPUT"], "w") as output:
        json.dump({{"args": sys.argv[1:], "model": os.environ["POINTLLM_MODEL_PATH"],
                   "dtype": os.environ["POINTLLM_DTYPE"], "device": os.environ["POINTLLM_DEVICE"],
                   "authenticated": len(os.environ["POINTLLM_API_KEY"]) >= 32,
                   "offline": os.environ.get("HF_HUB_OFFLINE"),
                   "lock_inherited": bool(os.fstat(9))}}, output)
    if os.getenv("TEST_HOLD_LOCK"):
        sys.stdin.read()
    sys.exit(0)
if sys.argv[1:4] == ["-m", "pip", "install"]:
    with open(os.environ["TEST_LAUNCH_OUTPUT"] + ".install", "w") as output:
        json.dump(sys.argv[1:], output)
    sys.exit(0)
if len(sys.argv) == 5 and sys.argv[1] == "-":
    print(os.environ["TEST_CACHED_CHECKPOINT"])
    sys.exit(0)
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
""")
    wrapper.chmod(0o700)
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("POINTLLM_")
    }
    env.update(
        POINTLLM_PYTHON=str(wrapper),
        POINTLLM_API_KEY="test-secret-" + "x" * 32,
        POINTLLM_SERVE_LOCK=str(tmp_path / "server.lock"),
        TEST_LAUNCH_OUTPUT=str(output),
        TEST_CACHED_CHECKPOINT=str(checkpoint),
    )
    return env, checkpoint, output


def run_script(env, *args, **kwargs):
    return subprocess.run(
        ["bash", str(SCRIPT), *map(str, args)], env=env,
        capture_output=True, text=True, timeout=20, **kwargs,
    )


def test_help_and_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = run_script(os.environ.copy(), "--help")
    assert result.returncode == 0
    assert "--model 7b|13b|PATH" in result.stdout


@pytest.mark.parametrize("model", ["7b", "13b"])
def test_model_alias_remote_defaults(launch_env, model):
    env, checkpoint, output = launch_env
    result = run_script(env, "--model", model, "--offline")
    assert result.returncode == 0, result.stderr
    launched = json.loads(output.read_text())
    assert launched["model"] == str(checkpoint)
    assert launched["authenticated"] and launched["lock_inherited"]
    assert launched["offline"] == "1"
    assert launched["args"][-6:] == ["--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
    assert env["POINTLLM_API_KEY"] not in result.stdout + result.stderr
    assert f"RunsenXu/PointLLM_{model.upper()}_v1.2" in result.stdout


def test_local_checkpoint_and_overrides(launch_env):
    env, checkpoint, output = launch_env
    result = run_script(
        env, "--model", checkpoint, "--host", "127.0.0.1", "--port", "9000",
        "--device", "cuda:1", "--dtype", "bfloat16",
    )
    assert result.returncode == 0, result.stderr
    launched = json.loads(output.read_text())
    assert launched["dtype"] == "bfloat16"
    assert launched["device"] == "cuda:1"
    assert launched["args"][-6:] == ["--host", "127.0.0.1", "--port", "9000", "--workers", "1"]


def test_persistent_key_generation_and_reuse(launch_env, tmp_path):
    env, checkpoint, output = launch_env
    env.pop("POINTLLM_API_KEY")
    key_file = tmp_path / "private" / "api-key"
    first = None
    for _ in range(2):
        result = run_script(env, "--model", checkpoint, "--api-key-file", key_file)
        assert result.returncode == 0, result.stderr
        key = key_file.read_text().strip()
        assert len(key) >= 32
        assert key not in result.stdout + result.stderr
        assert key_file.stat().st_mode & 0o777 == 0o600
        if first is None:
            first = key
        assert key == first


def test_install_is_explicit(launch_env):
    env, checkpoint, output = launch_env
    marker = Path(str(output) + ".install")
    assert run_script(env, "--model", checkpoint).returncode == 0
    assert not marker.exists()
    result = run_script(env, "--model", checkpoint, "--install")
    assert result.returncode == 0, result.stderr
    assert json.loads(marker.read_text()) == ["-m", "pip", "install", "-e", ".[serve]"]


@pytest.mark.parametrize("args", [
    ["--model"], ["--model", "unknown"], ["--port", "0"],
    ["--port", "65536"], ["--port", "abc"], ["--dtype", "int8"], ["--unknown"],
])
def test_invalid_arguments(launch_env, args):
    env, _, output = launch_env
    result = run_script(env, *args)
    assert result.returncode != 0
    assert not output.exists()


def test_rejects_weak_or_public_key(launch_env, tmp_path):
    env, checkpoint, output = launch_env
    env["POINTLLM_API_KEY"] = "weak"
    assert run_script(env, "--model", checkpoint).returncode != 0
    key_file = tmp_path / "key"
    key_file.write_text("x" * 40)
    key_file.chmod(0o644)
    assert run_script(env, "--model", checkpoint, "--api-key-file", key_file).returncode != 0
    assert not output.exists()


def test_tls_flags_are_forwarded(launch_env, tmp_path):
    env, checkpoint, output = launch_env
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.touch()
    key.touch()
    result = run_script(env, "--model", checkpoint, "--ssl-certfile", cert)
    assert result.returncode != 0
    result = run_script(
        env, "--model", checkpoint, "--ssl-certfile", cert, "--ssl-keyfile", key,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["args"][-4:] == [
        "--ssl-certfile", str(cert), "--ssl-keyfile", str(key),
    ]


def test_second_launcher_refuses_before_loading(launch_env):
    env, checkpoint, output = launch_env
    env["TEST_HOLD_LOCK"] = "1"
    first = subprocess.Popen(
        ["bash", str(SCRIPT), "--model", str(checkpoint)], env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        # A line from the final Python wrapper confirms it inherited the lock.
        # Wait on a pipe rather than relying on GPU timing or a fixed sleep.
        assert first.stdout
        for line in first.stdout:
            if "Startup may take" in line:
                break
        result = run_script(env, "--model", checkpoint)
        assert result.returncode != 0
        assert "Another launcher is running" in result.stderr
    finally:
        first.communicate(input="", timeout=20)
    env.pop("TEST_HOLD_LOCK")
    assert run_script(env, "--model", checkpoint).returncode == 0
