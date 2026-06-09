from __future__ import annotations

from dataclasses import dataclass
import io
import json
import plistlib
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


@dataclass
class R3DFrame:
    coord: np.ndarray
    color: Optional[np.ndarray]
    frame_id: str
    frame_index: int
    scene_name: str
    metadata: Dict[str, Any]
    depth_shape: tuple[int, int]
    rgb_shape: Optional[tuple[int, int]]
    k_depth: np.ndarray


def _decode_metadata_bytes(raw: bytes) -> Dict[str, Any]:
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception:
        decoded = None
    if isinstance(decoded, dict):
        return decoded

    try:
        decoded = plistlib.loads(raw)
    except Exception:
        decoded = None
    if isinstance(decoded, dict):
        return decoded

    try:
        decoded = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception as exc:
        raise ValueError("Could not parse r3d metadata as JSON or plist.") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected r3d metadata dict, got {type(decoded).__name__}.")
    return decoded


def _numeric_sort_key(value: str):
    try:
        return (0, int(value))
    except Exception:
        return (1, value)


def _find_metadata_member(zf: zipfile.ZipFile) -> str:
    candidates = []
    for name in zf.namelist():
        lower = Path(name).name.lower()
        if lower in {"metadata", "metadata.json"} or "metadata" in lower:
            candidates.append(name)
    if not candidates:
        raise FileNotFoundError("No metadata member found in r3d archive.")
    candidates = sorted(candidates, key=lambda x: (Path(x).name.lower() != "metadata", len(x)))
    return candidates[0]


def scan_r3d_archive(r3d_path: str | Path):
    zf = zipfile.ZipFile(r3d_path, "r")
    metadata_name = _find_metadata_member(zf)
    metadata = _decode_metadata_bytes(zf.read(metadata_name))

    frame_map: Dict[str, Dict[str, str]] = {}
    for name in zf.namelist():
        path = Path(name)
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".depth", ".conf", ".png"}:
            frame_map.setdefault(path.stem, {})
            frame_map[path.stem][suffix] = name

    frame_ids = sorted(
        [frame_id for frame_id, members in frame_map.items() if ".depth" in members],
        key=_numeric_sort_key,
    )
    return zf, metadata_name, metadata, frame_map, frame_ids


def get_r3d_info(r3d_path: str | Path) -> Dict[str, Any]:
    r3d_path = Path(r3d_path)
    zf, metadata_name, metadata, _frame_map, frame_ids = scan_r3d_archive(r3d_path)
    try:
        depth_shape = guess_depth_shape(metadata)
        rgb_shape = guess_rgb_shape(metadata)
        return {
            "path": str(r3d_path),
            "scene_name": r3d_path.stem,
            "metadata_name": metadata_name,
            "num_depth_frames": len(frame_ids),
            "first_frame_id": frame_ids[0] if frame_ids else None,
            "middle_frame_index": len(frame_ids) // 2 if frame_ids else None,
            "middle_frame_id": frame_ids[len(frame_ids) // 2] if frame_ids else None,
            "last_frame_index": len(frame_ids) - 1 if frame_ids else None,
            "last_frame_id": frame_ids[-1] if frame_ids else None,
            "depth_shape": depth_shape,
            "rgb_shape": rgb_shape,
            "has_poses": isinstance(metadata.get("poses"), list),
            "num_poses": len(metadata.get("poses", [])) if isinstance(metadata.get("poses"), list) else 0,
        }
    finally:
        zf.close()


def _require_lzfse():
    try:
        import lzfse
    except Exception as exc:
        raise ImportError("Reading .r3d depth/conf files requires `lzfse`. Install it with `pip install lzfse`.") from exc
    return lzfse


def _decompress_lzfse(raw_bytes: bytes) -> bytes:
    return _require_lzfse().decompress(raw_bytes)


def guess_depth_shape(metadata: Dict[str, Any]) -> tuple[int, int]:
    dh = metadata.get("dh") or metadata.get("depthH")
    dw = metadata.get("dw") or metadata.get("depthW")
    if dh is None or dw is None:
        raise KeyError("r3d metadata does not contain depth size dh/dw.")
    return int(dh), int(dw)


def guess_rgb_shape(metadata: Dict[str, Any]) -> Optional[tuple[int, int]]:
    h = metadata.get("h") or metadata.get("rgbHeight")
    w = metadata.get("w") or metadata.get("rgbWidth")
    if h is None or w is None:
        return None
    return int(h), int(w)


def choose_intrinsics_matrix(k_flat, rgb_w: Optional[int] = None, rgb_h: Optional[int] = None):
    k_flat = np.asarray(k_flat, dtype=np.float64).reshape(-1)
    if k_flat.size != 9:
        raise ValueError(f"Expected 9 camera intrinsic values, got {k_flat.size}.")

    candidates = {
        "row_major": k_flat.reshape(3, 3),
        "column_major": k_flat.reshape(3, 3, order="F"),
    }
    scored = []
    for name, k in candidates.items():
        score = 0.0
        if np.isfinite(k).all():
            score += 1.0
        if abs(k[2, 2] - 1.0) < 1e-3:
            score += 2.0
        if abs(k[2, 0]) < 1e-3 and abs(k[2, 1]) < 1e-3:
            score += 2.0
        if abs(k[0, 1]) < 1e-3 and abs(k[1, 0]) < 1e-3:
            score += 1.0
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        if fx > 0 and fy > 0:
            score += 2.0
        if rgb_w is not None and 0 <= cx <= rgb_w * 1.2:
            score += 2.0
        if rgb_h is not None and 0 <= cy <= rgb_h * 1.2:
            score += 2.0
        scored.append((score, name, k))

    scored = sorted(scored, reverse=True, key=lambda item: item[0])
    return scored[0][2], {"chosen": scored[0][1], "score": float(scored[0][0])}


def scale_intrinsics_to_depth(k_rgb: np.ndarray, rgb_shape_hw: tuple[int, int], depth_shape_hw: tuple[int, int]):
    rgb_h, rgb_w = rgb_shape_hw
    depth_h, depth_w = depth_shape_hw
    sx = depth_w / rgb_w
    sy = depth_h / rgb_h
    k_depth = k_rgb.copy().astype(np.float64)
    k_depth[0, 0] *= sx
    k_depth[1, 1] *= sy
    k_depth[0, 2] *= sx
    k_depth[1, 2] *= sy
    return k_depth, {"sx": sx, "sy": sy}


def _load_rgb_frame(zf: zipfile.ZipFile, member_name: str) -> np.ndarray:
    raw = zf.read(member_name)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def _load_depth_frame(zf: zipfile.ZipFile, member_name: str, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    raw = _decompress_lzfse(zf.read(member_name))
    arr = np.frombuffer(raw, dtype="<f4")
    h, w = depth_shape_hw
    if arr.size != h * w:
        raise ValueError(f"Depth payload has {arr.size} float32 values, expected {h * w}.")
    return arr.reshape(h, w)


def _load_conf_frame(zf: zipfile.ZipFile, member_name: str, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    raw = _decompress_lzfse(zf.read(member_name))
    arr = np.frombuffer(raw, dtype=np.uint8)
    h, w = depth_shape_hw
    if arr.size != h * w:
        raise ValueError(f"Confidence payload has {arr.size} bytes, expected {h * w}.")
    return arr.reshape(h, w)


def _resize_rgb_to_depth(rgb: np.ndarray, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = depth_shape_hw
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))


def depth_to_point_cloud(
    depth: np.ndarray,
    k_depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    conf: Optional[np.ndarray] = None,
    conf_threshold: int = 1,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    sample_step: int = 1,
):
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    vv, uu = np.indices((height, width))

    mask = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        mask &= np.asarray(conf) >= int(conf_threshold)
    if z_min is not None:
        mask &= depth >= float(z_min)
    if z_max is not None:
        mask &= depth <= float(z_max)
    if sample_step > 1:
        mask &= (uu % int(sample_step) == 0) & (vv % int(sample_step) == 0)

    u = uu[mask].astype(np.float32)
    v = vv[mask].astype(np.float32)
    z = depth[mask].astype(np.float32)

    fx = float(k_depth[0, 0])
    fy = float(k_depth[1, 1])
    cx = float(k_depth[0, 2])
    cy = float(k_depth[1, 2])
    if fx == 0 or fy == 0:
        raise ValueError("Invalid intrinsics: fx/fy must be non-zero.")

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    coord = np.stack([x, y, z], axis=1).astype(np.float32)

    color = None
    if rgb is not None:
        rgb_small = _resize_rgb_to_depth(rgb, depth.shape)
        color = rgb_small[mask].astype(np.float32)

    return coord, color


def load_r3d_frame(
    r3d_path: str | Path,
    frame_index: int = 0,
    conf_threshold: int = 1,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    sample_step: int = 1,
    use_rgb: bool = True,
) -> R3DFrame:
    r3d_path = Path(r3d_path)
    zf, _metadata_name, metadata, frame_map, frame_ids = scan_r3d_archive(r3d_path)
    try:
        if not frame_ids:
            raise ValueError("r3d archive contains no depth frames.")
        if frame_index < 0 or frame_index >= len(frame_ids):
            raise IndexError(f"frame_index={frame_index} is out of range 0..{len(frame_ids) - 1}.")

        depth_shape = guess_depth_shape(metadata)
        rgb_shape = guess_rgb_shape(metadata)
        k_rgb, _k_info = choose_intrinsics_matrix(
            metadata["K"],
            rgb_w=None if rgb_shape is None else rgb_shape[1],
            rgb_h=None if rgb_shape is None else rgb_shape[0],
        )
        if rgb_shape is None:
            rgb_shape = depth_shape
        k_depth, _scale_info = scale_intrinsics_to_depth(k_rgb, rgb_shape, depth_shape)

        frame_id = frame_ids[frame_index]
        members = frame_map[frame_id]
        depth = _load_depth_frame(zf, members[".depth"], depth_shape)
        conf = _load_conf_frame(zf, members[".conf"], depth_shape) if ".conf" in members else None

        rgb = None
        if use_rgb:
            if ".jpg" in members:
                rgb = _load_rgb_frame(zf, members[".jpg"])
            elif ".jpeg" in members:
                rgb = _load_rgb_frame(zf, members[".jpeg"])
            elif ".png" in members:
                rgb = _load_rgb_frame(zf, members[".png"])

        coord, color = depth_to_point_cloud(
            depth=depth,
            k_depth=k_depth,
            rgb=rgb,
            conf=conf,
            conf_threshold=conf_threshold,
            z_min=z_min,
            z_max=z_max,
            sample_step=sample_step,
        )

        return R3DFrame(
            coord=coord,
            color=color,
            frame_id=frame_id,
            frame_index=frame_index,
            scene_name=f"{r3d_path.stem}_frame_{frame_id}",
            metadata=metadata,
            depth_shape=depth_shape,
            rgb_shape=rgb_shape,
            k_depth=k_depth,
        )
    finally:
        zf.close()
