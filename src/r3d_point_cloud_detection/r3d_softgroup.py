from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig

from .classes import TARGET_CLASS_NAME_SET
from .s3dis_visualization import softgroup_instance_label_to_name
from .utils import ensure_dir, project_path, python_executable, repo_env, run_command, safe_remove_path, softgroup_repo_dir


INSTANCE_COLOR_PALETTE = np.array(
    [
        [31, 119, 180],
        [44, 160, 44],
        [255, 127, 14],
        [148, 103, 189],
        [23, 190, 207],
        [227, 119, 194],
        [188, 189, 34],
        [140, 86, 75],
        [127, 127, 127],
        [17, 141, 255],
        [42, 184, 121],
        [255, 187, 120],
    ],
    dtype=np.uint8,
)


def make_r3d_scene_name(prefix: str, r3d_stem: str, frame_id: str) -> str:
    raw = f"{prefix}_{r3d_stem}_frame_{frame_id}"
    scene = re.sub(r"[^A-Za-z0-9_]+", "_", raw)
    scene = re.sub(r"_+", "_", scene).strip("_")
    if not scene:
        raise ValueError("Could not build a valid SoftGroup scene name for R3D input.")
    return scene


def save_r3d_softgroup_preprocess(cfg: DictConfig, point_cloud: dict[str, Any], scene: str) -> Path:
    import torch

    repo_dir = softgroup_repo_dir(cfg)
    dataset_dir = repo_dir / "dataset" / "s3dis"
    preprocess_dir = ensure_dir(dataset_dir / "preprocess")
    val_gt_dir = ensure_dir(dataset_dir / "val_gt")
    ensure_dir(dataset_dir / "preprocess_sample")

    points = np.asarray(point_cloud["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"R3D points must have shape (N, 3), got {points.shape}")

    colors = point_cloud.get("colors")
    if colors is None:
        colors = np.full((len(points), 3), 127, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.shape != (len(points), 3):
        raise ValueError(f"R3D colors must have shape ({len(points)}, 3), got {colors.shape}")
    colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)

    semantic = np.full(len(points), int(cfg.inference.dummy_semantic_label), dtype=np.int64)
    instance = np.full(len(points), int(cfg.inference.dummy_instance_label), dtype=np.int64)
    room_label = np.zeros(len(points), dtype=np.int64)

    preprocess_path = preprocess_dir / f"{scene}_inst_nostuff.pth"
    if preprocess_path.exists():
        preprocess_path.unlink()
    torch.save((points, colors, semantic, instance, room_label, scene), preprocess_path)

    val_gt_path = val_gt_dir / f"{scene}.txt"
    np.savetxt(val_gt_path, semantic, fmt="%d")
    return preprocess_path


def resolve_r3d_softgroup_base_config(cfg: DictConfig) -> Path:
    if cfg.inference.get("softgroup_config"):
        path = project_path(cfg, cfg.inference.softgroup_config)
        if path.exists():
            return path
        raise FileNotFoundError(f"Configured R3D SoftGroup config not found: {path}")

    generated_dir = project_path(cfg, cfg.softgroup.generated_config_dir)
    if generated_dir.exists():
        candidates = sorted(generated_dir.glob("*_infer.yaml"))
        if candidates:
            return candidates[0]

    official = softgroup_repo_dir(cfg) / "configs" / "softgroup" / "softgroup_s3dis_fold5.yaml"
    if official.exists():
        return official
    raise FileNotFoundError(
        "No SoftGroup inference config found. Run scripts/generate_softgroup_configs.py or set inference.softgroup_config."
    )


def run_r3d_softgroup_inference(cfg: DictConfig, scene: str, run_dir: Path) -> Path:
    repo_dir = softgroup_repo_dir(cfg)
    checkpoint = project_path(cfg, cfg.inference.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"R3D inference checkpoint not found: {checkpoint}")

    base_config = resolve_r3d_softgroup_base_config(cfg)
    with base_config.open("r", encoding="utf-8") as file:
        softgroup_cfg = yaml.safe_load(file)

    raw_dir = ensure_dir(run_dir / "softgroup_raw")
    softgroup_cfg.setdefault("data", {}).setdefault("test", {})
    softgroup_cfg["data"]["test"]["prefix"] = scene
    softgroup_cfg["data"]["test"]["data_root"] = str(cfg.inference.softgroup_data_root)
    softgroup_cfg.setdefault("dataloader", {}).setdefault("test", {})
    softgroup_cfg["dataloader"]["test"]["batch_size"] = 1
    softgroup_cfg["dataloader"]["test"]["num_workers"] = 1
    softgroup_cfg.setdefault("model", {}).setdefault("test_cfg", {})
    softgroup_cfg["model"]["test_cfg"]["eval_tasks"] = ["semantic", "instance"]
    softgroup_cfg["work_dir"] = str(raw_dir)

    infer_config_path = run_dir / f"{scene}_softgroup_infer.yaml"
    with infer_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(softgroup_cfg, file, sort_keys=False)

    run_command(
        [python_executable(), "tools/test.py", str(infer_config_path), str(checkpoint), "--out", str(raw_dir)],
        cwd=repo_dir,
        env=repo_env(cfg),
    )
    return raw_dir


def load_r3d_pred_instances(scene: str, results_dir: Path, n_points: int) -> list[dict[str, Any]]:
    pred_root = Path(results_dir) / "pred_instance"
    txt_path = pred_root / f"{scene}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"R3D SoftGroup instance prediction file not found: {txt_path}")

    instances: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    with txt_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    if not lines:
        return []

    for proposal_id, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 3:
            continue
        rel_mask_path, label_str, score_str = parts[:3]
        mask_path = pred_root / rel_mask_path
        if not mask_path.exists():
            raise FileNotFoundError(f"R3D SoftGroup mask file not found: {mask_path}")
        mask = np.loadtxt(mask_path, dtype=np.int64).astype(bool)
        if len(mask) != n_points:
            raise ValueError(f"Mask length {len(mask)} does not match R3D point count {n_points}: {mask_path}")
        class_name = softgroup_instance_label_to_name(label_str)
        if class_name not in TARGET_CLASS_NAME_SET:
            continue
        counters[class_name] = counters.get(class_name, 0) + 1
        instance_name = f"{class_name}_{counters[class_name]}"
        instances.append(
            {
                "proposal_id": proposal_id,
                "class_name": class_name,
                "instance_name": instance_name,
                "raw_label_id": int(float(label_str)),
                "score": float(score_str),
                "num_points": int(mask.sum()),
                "mask_path": str(mask_path),
                "mask": mask,
            }
        )
    return sorted(instances, key=lambda item: (-item["score"], -item["num_points"], item["proposal_id"]))


def instances_to_point_predictions(
    n_points: int,
    instances: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    labels = np.full(n_points, "unassigned", dtype=object)
    colors = np.full((n_points, 3), 190, dtype=np.uint8)
    instance_ids = np.full(n_points, -1, dtype=np.int64)
    scores = np.full(n_points, np.nan, dtype=np.float32)

    for instance_id, inst in enumerate(sorted(instances, key=lambda item: (-item["score"], -item["num_points"], item["proposal_id"]))):
        mask = np.asarray(inst["mask"], dtype=bool)
        assign = mask & (instance_ids < 0)
        if not np.any(assign):
            continue
        label = str(inst["instance_name"])
        labels[assign] = label
        color = INSTANCE_COLOR_PALETTE[instance_id % len(INSTANCE_COLOR_PALETTE)]
        colors[assign] = color
        instance_ids[assign] = instance_id
        scores[assign] = float(inst["score"])

    assigned = np.flatnonzero(instance_ids >= 0)
    assignments = pd.DataFrame(
        {
            "point_index": assigned,
            "pred_instance_id": instance_ids[assigned],
            "pred_object": labels[assigned],
            "pred_score": scores[assigned],
        }
    )
    return colors, labels, instance_ids, scores, assignments


def instances_to_bboxes(
    points: np.ndarray,
    instances: list[dict[str, Any]],
    min_points: int,
    outlier_quantile: float | None = 0.02,
) -> tuple[list[dict], pd.DataFrame]:
    boxes: list[dict] = []
    rows: list[dict[str, Any]] = []
    for idx, inst in enumerate(instances):
        mask = np.asarray(inst["mask"], dtype=bool)
        if int(mask.sum()) < int(min_points):
            continue
        pts = points[mask]
        if outlier_quantile is not None and 0.0 < float(outlier_quantile) < 0.5 and len(pts) >= 10:
            mins = np.quantile(pts, float(outlier_quantile), axis=0)
            maxs = np.quantile(pts, 1.0 - float(outlier_quantile), axis=0)
        else:
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)

        x0, y0, z0 = mins
        x1, y1, z1 = maxs
        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=np.float32,
        )
        rgb = INSTANCE_COLOR_PALETTE[idx % len(INSTANCE_COLOR_PALETTE)]
        color = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
        label = str(inst["instance_name"])
        boxes.append({"points": corners, "label": label, "color": color})
        rows.append(
            {
                "pred_object": label,
                "class_name": inst["class_name"],
                "score": inst["score"],
                "num_mask_points": inst["num_points"],
                "x_min": float(x0),
                "y_min": float(y0),
                "z_min": float(z0),
                "x_max": float(x1),
                "y_max": float(y1),
                "z_max": float(z1),
                "color": color,
                "mask_path": inst["mask_path"],
            }
        )
    return boxes, pd.DataFrame(rows)
