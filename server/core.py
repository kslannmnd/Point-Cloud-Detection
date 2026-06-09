from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import io
import zipfile

import numpy as np

from .config import IGNORE_INDEX, TMP_DIR

@dataclass
class SceneData:
    coord: np.ndarray
    color: np.ndarray
    normal: np.ndarray
    segment: Optional[np.ndarray] = None
    instance: Optional[np.ndarray] = None
    scene_name: str = "scene"
    source_path: Optional[str] = None
    input_format: str = "unknown"
    frame_index: Optional[int] = None
    frame_id: Optional[str] = None

    @property
    def n_points(self) -> int:
        return int(self.coord.shape[0])

def _first_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in d:
            return key
    return None

def _normalize_color(color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.float32).reshape(-1, 3)
    if color.size == 0:
        return color
    if np.nanmax(color) <= 1.0:
        color = color * 255.0
    return np.clip(color, 0, 255).astype(np.float32)

def _safe_unit_normal(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1, 3)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return (v / norm).astype(np.float32)

def estimate_normals_fallback(coord: np.ndarray) -> np.ndarray:
    # Lightweight fallback: use centered coordinates as pseudo-normal direction.
    centered = coord.astype(np.float32) - coord.astype(np.float32).mean(axis=0, keepdims=True)
    return _safe_unit_normal(centered)

def _load_ply_with_open3d(path: Path) -> SceneData:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "PLY loading needs open3d. Install it or use the primary .r3d inference path."
        ) from exc

    pcd = o3d.io.read_point_cloud(str(path))
    coord = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        color = np.asarray(pcd.colors, dtype=np.float32)
        color = _normalize_color(color)
    else:
        color = np.full_like(coord, 127, dtype=np.float32)

    if pcd.has_normals():
        normal = np.asarray(pcd.normals, dtype=np.float32)
        normal = _safe_unit_normal(normal)
    else:
        normal = estimate_normals_fallback(coord)

    return SceneData(coord=coord, color=color, normal=normal, scene_name=path.stem, source_path=str(path), input_format="ply")

def _load_npz(path: Path) -> SceneData:
    arr = np.load(path, allow_pickle=True)

    coord_key = _first_key(arr, ["coord", "xyz", "points"])
    color_key = _first_key(arr, ["color", "rgb", "colors"])
    normal_key = _first_key(arr, ["normal", "normals"])
    segment_key = _first_key(arr, ["segment", "label", "labels", "semantic", "semantics"])
    instance_key = _first_key(arr, ["instance", "instances", "instance_id", "instance_ids"])

    if coord_key is None:
        raise KeyError(f"{path}: no coord/xyz/points key found")

    coord = np.asarray(arr[coord_key], dtype=np.float32).reshape(-1, 3)
    color = _normalize_color(np.asarray(arr[color_key], dtype=np.float32)) if color_key is not None else np.full_like(coord, 127, dtype=np.float32)

    if normal_key is not None:
        normal = np.asarray(arr[normal_key], dtype=np.float32).reshape(-1, 3)
        normal = _safe_unit_normal(normal)
    else:
        normal = estimate_normals_fallback(coord)

    segment = None
    instance = None
    if segment_key is not None:
        segment = np.asarray(arr[segment_key], dtype=np.int32).reshape(-1)
    if instance_key is not None:
        instance = np.asarray(arr[instance_key], dtype=np.int32).reshape(-1)

    # Optionally support compressed `meta` dict
    if "meta" in arr and isinstance(arr["meta"], np.ndarray) and arr["meta"].dtype == object and arr["meta"].size == 1:
        try:
            meta = arr["meta"].item()
            if isinstance(meta, dict) and "scene_name" in meta:
                scene_name = str(meta["scene_name"])
            else:
                scene_name = path.stem
        except Exception:
            scene_name = path.stem
    else:
        scene_name = path.stem

    return SceneData(
        coord=coord,
        color=color,
        normal=normal,
        segment=segment,
        instance=instance,
        scene_name=scene_name,
        source_path=str(path),
        input_format="npz",
    )

def _load_npy(path: Path) -> SceneData:
    data = np.load(path, allow_pickle=True)
    if data.ndim != 2 or data.shape[1] not in (3, 4, 6, 9):
        raise ValueError(f"{path}: expected Nx3/Nx4/Nx6/Nx9 array, got {data.shape}")
    coord = np.asarray(data[:, :3], dtype=np.float32)
    if data.shape[1] >= 6:
        color = _normalize_color(np.asarray(data[:, 3:6], dtype=np.float32))
    else:
        color = np.full_like(coord, 127, dtype=np.float32)
    normal = estimate_normals_fallback(coord)
    return SceneData(coord=coord, color=color, normal=normal, scene_name=path.stem, source_path=str(path), input_format="npy")

def _load_bin(path: Path) -> SceneData:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path}: .bin must contain float32 tuples of size 4 (x,y,z,i)")
    pts = raw.reshape(-1, 4)
    coord = pts[:, :3].astype(np.float32)
    color = np.full_like(coord, 127, dtype=np.float32)
    normal = estimate_normals_fallback(coord)
    return SceneData(coord=coord, color=color, normal=normal, scene_name=path.stem, source_path=str(path), input_format="bin")

def _load_r3d(
    path: Path,
    frame_index: int = 0,
    conf_threshold: int = 1,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    sample_step: int = 1,
) -> SceneData:
    from common.r3d import load_r3d_frame

    frame = load_r3d_frame(
        path,
        frame_index=frame_index,
        conf_threshold=conf_threshold,
        z_min=z_min,
        z_max=z_max,
        sample_step=sample_step,
        use_rgb=True,
    )
    coord = np.asarray(frame.coord, dtype=np.float32).reshape(-1, 3)
    color = _normalize_color(frame.color) if frame.color is not None else np.full_like(coord, 127, dtype=np.float32)
    normal = estimate_normals_fallback(coord)
    return SceneData(
        coord=coord,
        color=color,
        normal=normal,
        scene_name=frame.scene_name,
        source_path=str(path),
        input_format="r3d",
        frame_index=frame.frame_index,
        frame_id=frame.frame_id,
    )

def load_scene_file(
    path_like: str | Path,
    frame_index: int = 0,
    conf_threshold: int = 1,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    sample_step: int = 1,
) -> SceneData:
    path = Path(path_like)
    suffix = path.suffix.lower()

    if suffix == ".r3d":
        return _load_r3d(
            path,
            frame_index=frame_index,
            conf_threshold=conf_threshold,
            z_min=z_min,
            z_max=z_max,
            sample_step=sample_step,
        )
    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".ply":
        return _load_ply_with_open3d(path)
    if suffix == ".npy":
        return _load_npy(path)
    if suffix == ".bin":
        return _load_bin(path)
    if suffix == ".zip":
        # Optional: expect zip containing a single npz.
        with zipfile.ZipFile(path, "r") as zf:
            npz_names = [n for n in zf.namelist() if n.lower().endswith(".npz")]
            if not npz_names:
                raise ValueError(f"{path}: zip does not contain npz")
            first = npz_names[0]
            target = TMP_DIR / f"{path.stem}_{Path(first).name}"
            with zf.open(first) as src, open(target, "wb") as dst:
                dst.write(src.read())
            return _load_npz(target)
    raise ValueError(f"Unsupported file format: {suffix}")

def quantile_bbox(points: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> Dict[str, float]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError("Cannot build bbox from empty point set")
    lo = np.quantile(pts, q_low, axis=0)
    hi = np.quantile(pts, q_high, axis=0)
    # Guarantee non-zero sizes for visualization
    hi = np.maximum(hi, lo + 1e-4)
    return {
        "x_min": float(lo[0]),
        "y_min": float(lo[1]),
        "z_min": float(lo[2]),
        "x_max": float(hi[0]),
        "y_max": float(hi[1]),
        "z_max": float(hi[2]),
    }

def bbox_center_size(bbox: Dict[str, float]) -> Dict[str, float]:
    return {
        "cx": (bbox["x_min"] + bbox["x_max"]) / 2.0,
        "cy": (bbox["y_min"] + bbox["y_max"]) / 2.0,
        "cz": (bbox["z_min"] + bbox["z_max"]) / 2.0,
        "dx": bbox["x_max"] - bbox["x_min"],
        "dy": bbox["y_max"] - bbox["y_min"],
        "dz": bbox["z_max"] - bbox["z_min"],
    }

def hash_scene(coord: np.ndarray) -> int:
    h = hashlib.blake2b(digest_size=8)
    arr = np.ascontiguousarray(coord.astype(np.float32))
    h.update(arr[: min(len(arr), 5000)].tobytes())
    return int.from_bytes(h.digest(), "little", signed=False)
