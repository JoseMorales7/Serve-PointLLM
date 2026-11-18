import io
from dataclasses import replace

import numpy as np
import pytest

from pointllm.serve.config import Settings
from pointllm.serve.errors import APIError
from pointllm.serve.point_clouds import CloudStore, preprocess_cloud
from conftest import npy_bytes


def test_normalize_preserves_colors(cloud_array):
    points = preprocess_cloud(npy_bytes(cloud_array), Settings())
    np.testing.assert_allclose(points[:, :3].mean(axis=0), 0, atol=1e-6)
    assert np.linalg.norm(points[:, :3], axis=1).max() == pytest.approx(1)
    np.testing.assert_array_equal(points[:, 3:], cloud_array[:, 3:])
    assert points.dtype == np.float32


@pytest.mark.parametrize("points", [
    np.zeros((8192, 3)), np.zeros((8191, 6)), np.zeros((0, 6)),
    np.zeros((32769, 6)), np.zeros((8192, 6)),
    np.ones((8192, 6)), np.full((8192, 6), np.nan),
    np.full((8192, 6), np.inf), np.ones((8192, 6), dtype=object),
    np.ones((8192, 6), dtype=complex),
])
def test_invalid_arrays(points):
    with pytest.raises(APIError):
        preprocess_cloud(npy_bytes(points), Settings())


def test_bad_colors(cloud_array):
    for color in (-0.1, 255):
        cloud_array[0, 3] = color
        with pytest.raises(APIError, match="RGB"):
            preprocess_cloud(npy_bytes(cloud_array), Settings())


def test_non_npy_truncated_and_forged_header(cloud_bytes):
    for payload in (b"not numpy", cloud_bytes[:-10], cloud_bytes + b"junk"):
        with pytest.raises(APIError):
            preprocess_cloud(payload, Settings())
    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        payload, {"descr": "<f8", "fortran_order": False, "shape": (2**40, 6)}
    )
    with pytest.raises(APIError):
        preprocess_cloud(payload.getvalue(), Settings())


def test_downsample(cloud_array):
    larger = np.concatenate([cloud_array, cloud_array[:1]], axis=0)
    result = preprocess_cloud(npy_bytes(larger), Settings())
    assert result.shape == (8192, 6)
    assert np.isfinite(result).all()


def test_storage_expiry_and_capacity(cloud_array):
    now = [0]
    store = CloudStore(Settings(max_clouds=1, cloud_ttl_seconds=10), lambda: now[0])
    first = store.add(cloud_array)
    assert store.get(first.id) is first
    with pytest.raises(APIError, match="full"):
        store.add(cloud_array)
    now[0] = 10
    with pytest.raises(APIError, match="expired"):
        store.get(first.id)
    assert store.add(cloud_array).id != first.id


def test_settings_environment(monkeypatch):
    monkeypatch.setenv("POINTLLM_MAX_CLOUDS", "3")
    monkeypatch.setenv("POINTLLM_API_KEY", "key")
    monkeypatch.setenv("POINTLLM_DTYPE", "bfloat16")
    settings = Settings.from_env()
    assert settings.max_clouds == 3
    assert settings.api_key == "key"
    assert settings.dtype == "bfloat16"
    for options in ({"max_clouds": 0}, {"request_timeout_seconds": float("nan")}, {"dtype": "int8"}):
        with pytest.raises(ValueError):
            replace(settings, **options)
