import io
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

from .errors import APIError


def preprocess_cloud(payload, settings):
    """Check the NPY header before numpy can allocate a forged array shape."""
    if len(payload) > settings.max_upload_bytes:
        raise APIError("Point cloud exceeds upload byte limit", 413, "upload_too_large")
    source = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(source)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError("Only NPY v1/v2 are supported")
        if len(shape) != 2 or shape[1] != 6:
            raise ValueError("Expected an (N, 6) XYZRGB array")
        if not 8192 <= shape[0] <= settings.max_input_points:
            raise ValueError(f"Expected 8192 to {settings.max_input_points} points")
        if dtype.kind not in "fiu" or dtype.itemsize > 8:
            raise ValueError("Expected real numeric values, not objects or structured data")
        if len(payload) - source.tell() != shape[0] * 6 * dtype.itemsize:
            raise ValueError("NPY data size does not match its header")
        source.seek(0)
        points = np.load(source, allow_pickle=False).astype(np.float64)
        if not np.isfinite(points).all():
            raise ValueError("Point cloud contains non-finite values")
        if np.any(points[:, 3:] < 0) or np.any(points[:, 3:] > 1):
            raise ValueError("RGB must be in [0, 1]")
        # Scale before centering to avoid overflow for extreme but finite XYZ.
        xyz = points[:, :3]
        scale = np.max(np.abs(xyz))
        if scale == 0:
            raise ValueError("Point cloud has zero spatial extent")
        xyz = xyz / scale
        xyz -= xyz.mean(axis=0)
        radius = np.linalg.norm(xyz, axis=1).max()
        if radius == 0:
            raise ValueError("Point cloud has zero spatial extent")
        points[:, :3] = xyz / radius
    except (ValueError, TypeError, EOFError, OverflowError) as exc:
        raise APIError(f"Invalid point cloud: {exc}", param="point_cloud") from exc

    if len(points) > 8192:
        # Same NumPy FPS algorithm as data/utils.py, kept independent of its
        # package initializer (which imports training/model dependencies).
        indices = np.empty(8192, dtype=np.int64)
        distance = np.full(len(points), np.inf)
        farthest = 0
        for index in range(8192):
            indices[index] = farthest
            delta = points[:, :3] - points[farthest, :3]
            distance = np.minimum(distance, np.sum(delta * delta, axis=1))
            farthest = np.argmax(distance)
        points = points[indices]
        # Match the model's normalization on the sampled cloud, not its source.
        points[:, :3] -= points[:, :3].mean(axis=0)
        points[:, :3] /= np.linalg.norm(points[:, :3], axis=1).max()
    points = np.ascontiguousarray(points, dtype=np.float32)
    points.setflags(write=False)
    return points


@dataclass
class StoredCloud:
    id: str
    points: np.ndarray
    created: int
    expires_at: float
    deadline: float

    def metadata(self):
        return {
            "id": self.id,
            "object": "point_cloud",
            "created": self.created,
            "expires_at": self.expires_at,
            "point_count": len(self.points),
        }


class CloudStore:
    def __init__(self, settings, clock=time.monotonic):
        self.settings = settings
        self.clock = clock
        self.clouds = {}
        self.lock = threading.Lock()

    def _prune(self):
        for key in list(self.clouds):
            if self.clouds[key].deadline <= self.clock():
                del self.clouds[key]

    def add(self, points):
        with self.lock:
            self._prune()
            if len(self.clouds) >= self.settings.max_clouds:
                raise APIError("Cloud storage is full; delete a cloud or wait for expiry", 429, "storage_full")
            created = time.time()
            cloud = StoredCloud(
                "pc-" + uuid.uuid4().hex, points, int(created),
                created + self.settings.cloud_ttl_seconds,
                self.clock() + self.settings.cloud_ttl_seconds,
            )
            self.clouds[cloud.id] = cloud
            return cloud

    def get(self, cloud_id):
        with self.lock:
            self._prune()
            if cloud_id not in self.clouds:
                raise APIError("Unknown or expired point_cloud_id", 404, "point_cloud_not_found", "point_cloud_id")
            return self.clouds[cloud_id]

    def delete(self, cloud_id):
        with self.lock:
            self._prune()
            if cloud_id not in self.clouds:
                raise APIError("Unknown or expired point_cloud_id", 404, "point_cloud_not_found", "point_cloud_id")
            del self.clouds[cloud_id]
