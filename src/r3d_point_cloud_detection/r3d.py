from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import json
import plistlib
import zipfile

import lzfse
import numpy as np
from PIL import Image


@dataclass
class SingleFrameConfig:
    r3d_path: str | Path
    frame_index: int = 0
    confidence_threshold: int = 1
    z_min: float | None = None
    z_max: float | None = None
    sample_step: int = 1
    use_rgb: bool = True


def parse_metadata(raw: bytes):
    stripped = raw.lstrip()
    if stripped.startswith((b"{", b"[")):
        return json.loads(raw.decode("utf-8", errors="ignore"))
    return plistlib.loads(raw)


def scan_r3d_file(r3d_path: str | Path):
    zf = zipfile.ZipFile(r3d_path, "r")
    names = zf.namelist()
    metadata_candidates = [name for name in names if "metadata" in Path(name).name.lower()]
    if not metadata_candidates:
        zf.close()
        raise FileNotFoundError(f"No metadata file found inside {r3d_path}")
    metadata_name = sorted(metadata_candidates, key=len)[0]
    metadata = parse_metadata(zf.read(metadata_name))

    frame_map: dict[str, dict[str, str]] = {}
    for name in names:
        path = Path(name)
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".depth", ".conf"}:
            frame_map.setdefault(path.stem, {})[suffix] = name

    frame_ids = sorted(
        [frame_id for frame_id, members in frame_map.items() if ".depth" in members],
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if not frame_ids:
        zf.close()
        raise FileNotFoundError(f"No depth frames found inside {r3d_path}")
    return zf, metadata, frame_map, frame_ids


def select_cam_matrix(cam_matrix_values, rgb_w: int, rgb_h: int) -> np.ndarray:
    values = np.asarray(cam_matrix_values, dtype=np.float64).reshape(-1)
    candidates = [values.reshape(3, 3), values.reshape(3, 3, order="F")]

    def score(cam_matrix: np.ndarray) -> float:
        fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
        cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
        return (
            float(np.isfinite(cam_matrix).all())
            + 2 * (abs(cam_matrix[2, 2] - 1.0) < 1e-3)
            + 2 * (abs(cam_matrix[2, 0]) < 1e-3 and abs(cam_matrix[2, 1]) < 1e-3)
            + 1 * (abs(cam_matrix[0, 1]) < 1e-3 and abs(cam_matrix[1, 0]) < 1e-3)
            + 2 * (fx > 0 and fy > 0)
            + 2 * (0 <= cx <= rgb_w * 1.2)
            + 2 * (0 <= cy <= rgb_h * 1.2)
        )

    return max(candidates, key=score)


def cam_matrix_to_depth(
    rgb_cam_matrix: np.ndarray,
    rgb_shape_hw: tuple[int, int],
    depth_shape_hw: tuple[int, int],
) -> np.ndarray:
    rgb_h, rgb_w = rgb_shape_hw
    depth_h, depth_w = depth_shape_hw
    sx = depth_w / rgb_w
    sy = depth_h / rgb_h
    depth_cam_matrix = rgb_cam_matrix.copy().astype(np.float64)
    depth_cam_matrix[0, 0] *= sx
    depth_cam_matrix[1, 1] *= sy
    depth_cam_matrix[0, 2] *= sx
    depth_cam_matrix[1, 2] *= sy
    return depth_cam_matrix


def load_rgb_frame(zf: zipfile.ZipFile, member_name: str) -> np.ndarray:
    image = Image.open(io.BytesIO(zf.read(member_name))).convert("RGB")
    return np.asarray(image)


def load_depth_frame(zf: zipfile.ZipFile, member_name: str, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    raw = lzfse.decompress(zf.read(member_name))
    return np.frombuffer(raw, dtype="<f4").reshape(depth_shape_hw)


def load_conf_frame(zf: zipfile.ZipFile, member_name: str, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    raw = lzfse.decompress(zf.read(member_name))
    return np.frombuffer(raw, dtype=np.uint8).reshape(depth_shape_hw)


def resize_rgb_to_depth(rgb: np.ndarray, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = depth_shape_hw
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))


def depth_to_point_cloud(
    depth: np.ndarray,
    depth_cam_matrix: np.ndarray,
    rgb: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    confidence_threshold: int = 0,
    z_min: float | None = None,
    z_max: float | None = None,
    sample_step: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = depth.shape
    v, u = np.indices((h, w))
    z = depth.astype(np.float32)

    mask = np.isfinite(z) & (z > 0)
    if confidence is not None:
        mask &= confidence >= confidence_threshold
    if z_min is not None:
        mask &= z >= z_min
    if z_max is not None:
        mask &= z <= z_max
    if sample_step > 1:
        mask &= (u % sample_step == 0) & (v % sample_step == 0)

    u = u[mask].astype(np.float32)
    v = v[mask].astype(np.float32)
    z = z[mask].astype(np.float32)

    fx, fy = float(depth_cam_matrix[0, 0]), float(depth_cam_matrix[1, 1])
    cx, cy = float(depth_cam_matrix[0, 2]), float(depth_cam_matrix[1, 2])
    points = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)

    colors = None
    if rgb is not None:
        colors = resize_rgb_to_depth(rgb, depth.shape)[mask].astype(np.uint8)
    return points, colors


def r3d_to_point_cloud(config: SingleFrameConfig) -> dict[str, object]:
    zf, metadata, frame_map, frame_ids = scan_r3d_file(config.r3d_path)
    try:
        depth_shape = (int(metadata["dh"]), int(metadata["dw"]))
        rgb_shape = (int(metadata["h"]), int(metadata["w"]))
        rgb_cam_matrix = select_cam_matrix(metadata["K"], rgb_w=rgb_shape[1], rgb_h=rgb_shape[0])
        depth_cam_matrix = cam_matrix_to_depth(rgb_cam_matrix, rgb_shape, depth_shape)

        frame_id = frame_ids[config.frame_index]
        members = frame_map[frame_id]
        rgb_name = members.get(".jpg") or members.get(".jpeg")
        rgb = load_rgb_frame(zf, rgb_name) if rgb_name else None
        depth = load_depth_frame(zf, members[".depth"], depth_shape)
        confidence = load_conf_frame(zf, members[".conf"], depth_shape) if ".conf" in members else None

        points, colors = depth_to_point_cloud(
            depth=depth,
            depth_cam_matrix=depth_cam_matrix,
            rgb=rgb if config.use_rgb else None,
            confidence=confidence,
            confidence_threshold=config.confidence_threshold,
            z_min=config.z_min,
            z_max=config.z_max,
            sample_step=config.sample_step,
        )
        return {
            "frame_id": frame_id,
            "points": points,
            "colors": colors,
            "metadata": metadata,
            "depth_camera_matrix": depth_cam_matrix,
        }
    finally:
        zf.close()


def save_point_cloud_npz(result: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "points": np.asarray(result["points"], dtype=np.float32),
        "frame_id": np.asarray(str(result["frame_id"])),
    }
    colors = result.get("colors")
    if colors is not None:
        arrays["colors"] = np.asarray(colors, dtype=np.uint8)
    np.savez_compressed(output_path, **arrays)
    return output_path
