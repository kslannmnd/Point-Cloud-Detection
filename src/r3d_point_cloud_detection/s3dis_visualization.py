from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import html
import re
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from omegaconf import DictConfig

from .classes import NON_TARGET_CLASS_NAMES, S3DIS_CLASSES, TARGET_CLASS_IDS, TARGET_CLASS_NAME_SET
from .utils import ensure_dir, project_path, python_executable, repo_env, run_command, softgroup_repo_dir

try:
    from IPython.display import HTML, display
except Exception:  # pragma: no cover - used outside notebooks
    HTML = None

    def display(value):
        print(value)


S3DIS_IGNORED_CLASS_NAMES = NON_TARGET_CLASS_NAMES
S3DIS_IGNORED_CLASS_IDS = set(range(len(S3DIS_CLASSES))) - set(TARGET_CLASS_IDS)

NAMED_COLORS = [
    "yellow",
    "pink",
    "cyan",
    "orange",
    "purple",
    "lime",
    "red",
    "blue",
    "green",
    "brown",
    "magenta",
    "gold",
    "navy",
    "teal",
    "olive",
    "coral",
    "violet",
    "turquoise",
    "salmon",
    "plum",
    "khaki",
    "orchid",
    "tomato",
    "deepskyblue",
    "chartreuse",
    "darkorange",
    "mediumvioletred",
    "darkcyan",
    "indigo",
    "springgreen",
    "crimson",
    "slateblue",
    "peru",
    "aquamarine",
]


@dataclass
class S3DISVisualizationContext:
    preprocess_dir: Path
    results_root: Path
    html_dir: Path
    softgroup_repo_dir: Path | None = None
    checkpoint_path: Path | None = None
    infer_config_path: Path | None = None
    save_html: bool = True
    show_inline: bool = True
    display_tables: bool = True
    renderer: str | None = None
    cfg: DictConfig | None = None


def context_from_cfg(cfg: DictConfig) -> S3DISVisualizationContext:
    infer_config_path = None
    generated_config_dir = project_path(cfg, cfg.softgroup.generated_config_dir)
    existing_infer_configs = sorted(generated_config_dir.glob("*_infer.yaml")) if generated_config_dir.exists() else []
    if existing_infer_configs:
        infer_config_path = existing_infer_configs[0]

    return S3DISVisualizationContext(
        preprocess_dir=project_path(cfg, cfg.visualization.preprocess_dir),
        results_root=project_path(cfg, cfg.visualization.results_root),
        html_dir=project_path(cfg, cfg.visualization.html_dir),
        softgroup_repo_dir=softgroup_repo_dir(cfg),
        checkpoint_path=project_path(cfg, cfg.inference.checkpoint),
        infer_config_path=infer_config_path,
        save_html=bool(cfg.visualization.save_html),
        show_inline=bool(cfg.visualization.show_inline),
        display_tables=bool(cfg.visualization.display_tables),
        renderer=cfg.visualization.renderer,
        cfg=cfg,
    )


def _normalize_area(area_i: str | int) -> str:
    area = str(area_i).strip()
    if area.isdigit():
        return f"Area_{area}"
    match = re.fullmatch(r"area[_\- ]?(\d+)", area, flags=re.IGNORECASE)
    if match:
        return f"Area_{match.group(1)}"
    if re.fullmatch(r"Area_\d+", area):
        return area
    raise ValueError("area must look like 1, Area_1, or area_1.")


def _normalize_room(room_i: str) -> str:
    room = str(room_i).strip()
    if not room:
        raise ValueError("room must be a non-empty string.")
    return room


def scene_id(area_i: str | int, room_i: str) -> tuple[str, str, str]:
    area = _normalize_area(area_i)
    room = _normalize_room(room_i)
    return area, room, f"{area}_{room}"


def _natural_key(value: Any):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(value))]


def semantic_label_to_name(label_id: int) -> str:
    label_id = int(label_id)
    if 0 <= label_id < len(S3DIS_CLASSES):
        return S3DIS_CLASSES[label_id]
    return f"class_{label_id}"


def softgroup_instance_label_to_name(label_id: int | float | str) -> str:
    label_id = int(float(label_id))
    if 1 <= label_id <= len(S3DIS_CLASSES):
        return S3DIS_CLASSES[label_id - 1]
    if 0 <= label_id < len(S3DIS_CLASSES):
        return S3DIS_CLASSES[label_id]
    return f"class_{label_id}"


def _torch_load_any(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.size == 0:
        return rgb.astype(np.uint8)
    rgb_float = rgb.astype(np.float32)
    mn, mx = np.nanmin(rgb_float), np.nanmax(rgb_float)
    if mn >= -1.1 and mx <= 1.1:
        rgb_float = (rgb_float + 1.0) * 127.5
    elif mn >= 0 and mx <= 1.1:
        rgb_float = rgb_float * 255.0
    return np.clip(np.rint(rgb_float), 0, 255).astype(np.uint8)


def _rgb_strings(rgb_uint8: np.ndarray) -> list[str]:
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb_uint8]


def _get_preprocess_path(area_i, room_i, preprocess_dir: Path) -> Path:
    area, room, scene = scene_id(area_i, room_i)
    path = preprocess_dir / f"{scene}_inst_nostuff.pth"
    if path.exists():
        return path
    candidates = sorted(preprocess_dir.glob(f"{scene}*_inst_nostuff.pth"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No SoftGroup preprocess file found for {scene}: {path}")


def _build_gt_objects(semantic_labels, instance_labels):
    semantic_labels = np.asarray(semantic_labels, dtype=np.int64)
    instance_labels = np.asarray(instance_labels, dtype=np.int64)
    n_points = len(semantic_labels)
    class_names = np.array([semantic_label_to_name(x) for x in semantic_labels], dtype=object)
    valid_mask = np.array(
        [
            int(label) not in S3DIS_IGNORED_CLASS_IDS and semantic_label_to_name(label) not in S3DIS_IGNORED_CLASS_NAMES
            for label in semantic_labels
        ],
        dtype=bool,
    )

    gt_object_names = np.full(n_points, "ignored", dtype=object)
    gt_objects = []
    counters = defaultdict(int)
    keys = sorted(
        {(int(semantic_labels[i]), int(instance_labels[i])) for i in np.flatnonzero(valid_mask)},
        key=lambda item: (semantic_label_to_name(item[0]), item[1]),
    )
    for sem_id, inst_id in keys:
        cls = semantic_label_to_name(sem_id)
        if cls in S3DIS_IGNORED_CLASS_NAMES:
            continue
        mask = valid_mask & (semantic_labels == sem_id) & (instance_labels == inst_id)
        if not np.any(mask):
            continue
        counters[cls] += 1
        obj_name = f"{cls}_{counters[cls]}"
        gt_object_names[mask] = obj_name
        gt_objects.append(
            {
                "gt_object": obj_name,
                "class_id": sem_id,
                "class_name": cls,
                "instance_label": inst_id,
                "num_points": int(mask.sum()),
                "mask": mask,
            }
        )

    gt_objects_df = pd.DataFrame([{k: v for k, v in obj.items() if k != "mask"} for obj in gt_objects])
    if len(gt_objects_df):
        gt_objects_df = gt_objects_df.sort_values(
            "gt_object", key=lambda col: col.map(_natural_key)
        ).reset_index(drop=True)
    return class_names, valid_mask, gt_object_names, gt_objects, gt_objects_df


def load_room_preprocess(area_i, room_i, context: S3DISVisualizationContext):
    area, room, requested_scene = scene_id(area_i, room_i)
    path = _get_preprocess_path(area, room, preprocess_dir=context.preprocess_dir)
    data = _torch_load_any(path)
    if len(data) < 4:
        raise ValueError(f"Unexpected preprocess file format: {path}")

    xyz = np.asarray(data[0], dtype=np.float32)
    rgb_raw = np.asarray(data[1])
    semantic_labels = np.asarray(data[2], dtype=np.int64)
    instance_labels = np.asarray(data[3], dtype=np.int64)
    scene_from_file = data[5] if len(data) >= 6 else requested_scene
    rgb_uint8 = _rgb_to_uint8(rgb_raw)
    gt_class_names, valid_mask, gt_object_names, gt_objects, gt_objects_df = _build_gt_objects(
        semantic_labels, instance_labels
    )
    return {
        "area": area,
        "room": room,
        "scene": str(scene_from_file),
        "requested_scene": requested_scene,
        "path": path,
        "xyz": xyz,
        "rgb_raw": rgb_raw,
        "rgb": rgb_uint8,
        "semantic_labels": semantic_labels,
        "instance_labels": instance_labels,
        "gt_class_names": gt_class_names,
        "gt_object_names": gt_object_names,
        "valid_mask": valid_mask,
        "gt_objects": gt_objects,
        "gt_objects_df": gt_objects_df,
    }


def _sample_global_indices(mask, max_points=None, random_state=42):
    idx = np.flatnonzero(mask)
    if max_points is not None and len(idx) > int(max_points):
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(idx, size=int(max_points), replace=False))
    return idx


def _make_base_hover(room_data, idx, include_gt_object=True):
    rgb = room_data["rgb"]
    rgb_txt = [f"{int(rgb[i, 0])},{int(rgb[i, 1])},{int(rgb[i, 2])}" for i in idx]
    if include_gt_object:
        return np.array(
            [[rgb_txt[j], room_data["gt_class_names"][i], room_data["gt_object_names"][i]] for j, i in enumerate(idx)],
            dtype=object,
        )
    return np.array([[rgb_txt[j], room_data["gt_class_names"][i]] for j, i in enumerate(idx)], dtype=object)


def _base_hovertemplate(include_gt_object=True):
    if include_gt_object:
        return (
            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
            "<br>rgb=%{customdata[0]}"
            "<br>class=%{customdata[1]}"
            "<br>gt_object=%{customdata[2]}"
            "<extra></extra>"
        )
    return "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>rgb=%{customdata[0]}<br>class=%{customdata[1]}<extra></extra>"


def _plot_layout(title, width=1050, height=800):
    return dict(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=width,
        height=height,
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=45),
    )


def _display_color_legend(color_map, title="Color legend"):
    if HTML is None:
        print(title)
        for name in sorted(color_map, key=_natural_key):
            print(f"{color_map[name]}: {name}")
        return
    rows = []
    for name in sorted(color_map, key=_natural_key):
        color = color_map[name]
        rows.append(
            "<tr>"
            f"<td style='padding:4px 10px;border:1px solid #ddd'>{html.escape(str(color))}</td>"
            f"<td style='padding:4px 10px;border:1px solid #ddd'>"
            f"<span style='display:inline-block;width:42px;height:14px;background:{html.escape(str(color))};border:1px solid #444'></span>"
            "</td>"
            f"<td style='padding:4px 10px;border:1px solid #ddd'>{html.escape(str(name))}</td>"
            "</tr>"
        )
    table = (
        f"<div style='font-weight:600;margin:8px 0 4px 0'>{html.escape(title)}</div>"
        "<table style='border-collapse:collapse;font-size:13px'>"
        "<tr><th style='padding:4px 10px;border:1px solid #ddd'>color</th>"
        "<th style='padding:4px 10px;border:1px solid #ddd'>sample</th>"
        "<th style='padding:4px 10px;border:1px solid #ddd'>object</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    display(HTML(table))


def _finalize_figure(fig: go.Figure, context: S3DISVisualizationContext, name: str) -> Path | None:
    if context.renderer:
        import plotly.io as pio

        pio.renderers.default = str(context.renderer)
    if context.show_inline:
        fig.show()
    if context.save_html:
        ensure_dir(context.html_dir)
        path = context.html_dir / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"Saved visualization: {path}")
        return path
    return None


def _has_prediction(results_dir: Path, scene: str) -> bool:
    return (Path(results_dir) / "pred_instance" / f"{scene}.txt").exists()


def _single_room_results_dir(context: S3DISVisualizationContext, scene: str) -> Path:
    return context.results_root / str(scene)


def _find_results_dir_for_scene(
    context: S3DISVisualizationContext,
    scene: str,
    results_dir: Path | None = None,
    allow_precomputed_results: bool = False,
) -> Path | None:
    candidates = [Path(results_dir)] if results_dir is not None else [_single_room_results_dir(context, scene)]
    if allow_precomputed_results:
        candidates.extend(sorted(context.results_root.glob("*")))
    seen = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if _has_prediction(root, scene):
            return root
    return None


def _run_softgroup_inference_for_room(
    context: S3DISVisualizationContext,
    area_i,
    room_i,
    out_dir: Path | None = None,
    cfg_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> Path:
    area, room, scene = scene_id(area_i, room_i)
    repo_dir = context.softgroup_repo_dir
    if repo_dir is None:
        raise ValueError("softgroup_repo_dir is required to run SoftGroup inference.")
    out_dir = Path(out_dir) if out_dir is not None else _single_room_results_dir(context, scene)
    ensure_dir(out_dir)

    cfg_path = Path(cfg_path or context.infer_config_path or repo_dir / "configs" / "softgroup" / "softgroup_s3dis_fold5.yaml")
    checkpoint_path = Path(checkpoint_path or context.checkpoint_path or repo_dir / "checkpoint" / "softgroup_s3dis_spconv2.pth")
    if not cfg_path.exists():
        raise FileNotFoundError(f"SoftGroup inference config not found: {cfg_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SoftGroup checkpoint not found: {checkpoint_path}")

    with cfg_path.open("r", encoding="utf-8") as file:
        softgroup_cfg = yaml.safe_load(file)
    softgroup_cfg.setdefault("data", {}).setdefault("test", {})
    softgroup_cfg["data"]["test"]["prefix"] = scene
    softgroup_cfg["data"]["test"]["data_root"] = "dataset/s3dis/preprocess"
    softgroup_cfg.setdefault("dataloader", {}).setdefault("test", {})
    softgroup_cfg["dataloader"]["test"]["batch_size"] = 1
    softgroup_cfg["dataloader"]["test"]["num_workers"] = 1
    softgroup_cfg.setdefault("model", {}).setdefault("test_cfg", {})
    softgroup_cfg["model"]["test_cfg"]["eval_tasks"] = ["semantic", "instance"]
    softgroup_cfg["work_dir"] = str(out_dir)

    generated_dir = out_dir / "generated_config"
    ensure_dir(generated_dir)
    room_cfg_path = generated_dir / f"softgroup_s3dis_{scene}_single_infer.yaml"
    with room_cfg_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(softgroup_cfg, file, sort_keys=False)

    env = repo_env(context.cfg) if context.cfg is not None else None
    run_command([python_executable(), "tools/test.py", str(room_cfg_path), str(checkpoint_path), "--out", str(out_dir)], cwd=repo_dir, env=env)
    if not _has_prediction(out_dir, scene):
        raise RuntimeError(f"SoftGroup inference finished but prediction file was not found: {out_dir / 'pred_instance' / (scene + '.txt')}")
    return out_dir


def _load_pred_instances(scene: str, results_dir: Path, n_points: int):
    pred_root = Path(results_dir) / "pred_instance"
    txt_path = pred_root / f"{scene}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Instance prediction file not found: {txt_path}")

    raw_instances = []
    with txt_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    for proposal_id, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 3:
            continue
        rel_mask_path, label_str, score_str = parts[:3]
        mask_path = pred_root / rel_mask_path
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = np.loadtxt(mask_path, dtype=np.int64).astype(bool)
        if len(mask) != n_points:
            raise ValueError(f"Mask length {len(mask)} does not match point count {n_points}: {mask_path}")
        label_id = int(float(label_str))
        class_name = softgroup_instance_label_to_name(label_id)
        if class_name not in TARGET_CLASS_NAME_SET:
            continue
        raw_instances.append(
            {
                "proposal_id": int(proposal_id),
                "raw_label_id": label_id,
                "class_name": class_name,
                "score": float(score_str),
                "num_points": int(mask.sum()),
                "mask_path": str(mask_path),
                "mask": mask,
            }
        )

    counters = defaultdict(int)
    instances = []
    for instance_id, inst in enumerate(sorted(raw_instances, key=lambda x: (-x["score"], -x["num_points"], x["proposal_id"]))):
        counters[inst["class_name"]] += 1
        item = dict(inst)
        item["instance_id"] = instance_id
        item["id_object_after_inference"] = f"{item['class_name']}_{counters[item['class_name']]}"
        item["pred_object"] = item["id_object_after_inference"]
        item["matched_gt_object"] = None
        item["matched_intersection"] = 0
        item["matched_iou"] = 0.0
        instances.append(item)
    return instances


def _match_predictions_to_gt(instances, gt_objects, min_iou=0.0):
    for inst in instances:
        best = None
        pmask = inst["mask"]
        for gt in gt_objects:
            if inst["class_name"] != gt["class_name"]:
                continue
            gmask = gt["mask"]
            inter = int(np.count_nonzero(pmask & gmask))
            if inter == 0:
                continue
            union = int(np.count_nonzero(pmask | gmask))
            iou = inter / union if union else 0.0
            candidate = (inter, iou, gt["gt_object"])
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is not None and best[1] >= min_iou:
            inter, iou, gt_object = best
            inst["matched_gt_object"] = gt_object
            inst["matched_intersection"] = int(inter)
            inst["matched_iou"] = float(iou)
            inst["pred_object"] = gt_object
    return instances


def _build_point_prediction_arrays(room_data, instances):
    n_points = len(room_data["xyz"])
    pred_class = np.full(n_points, "unassigned", dtype=object)
    pred_object = np.full(n_points, "unassigned", dtype=object)
    id_object_after_inference = np.full(n_points, "unassigned", dtype=object)
    pred_score = np.full(n_points, np.nan, dtype=np.float32)
    pred_instance_id = np.full(n_points, -1, dtype=np.int64)
    for inst in sorted(instances, key=lambda x: (-x["score"], -x["num_points"])):
        assign = inst["mask"] & (pred_instance_id < 0)
        if not np.any(assign):
            continue
        pred_class[assign] = inst["class_name"]
        pred_object[assign] = inst["pred_object"]
        id_object_after_inference[assign] = inst["id_object_after_inference"]
        pred_score[assign] = inst["score"]
        pred_instance_id[assign] = inst["instance_id"]
    return pred_class, pred_object, id_object_after_inference, pred_score, pred_instance_id


def soft_group_get_instance_seg(
    area_i,
    room_i,
    context: S3DISVisualizationContext,
    results_dir: Path | None = None,
    force_run: bool = False,
    display_tables: bool | None = None,
    allow_precomputed_results: bool = False,
):
    area, room, scene = scene_id(area_i, room_i)
    room_data = load_room_preprocess(area, room, context)
    n_points = len(room_data["xyz"])
    target_results_dir = Path(results_dir) if results_dir is not None else _single_room_results_dir(context, scene)
    found_results_dir = _find_results_dir_for_scene(
        context,
        scene,
        results_dir=target_results_dir,
        allow_precomputed_results=allow_precomputed_results,
    )
    display_tables = context.display_tables if display_tables is None else display_tables

    if force_run or found_results_dir is None:
        if display_tables:
            print(f"Running SoftGroup inference for one room only: {scene}")
            print(f"Output dir: {target_results_dir}")
        found_results_dir = _run_softgroup_inference_for_room(context, area, room, out_dir=target_results_dir)
    elif display_tables:
        print(f"Using cached room-level prediction for {scene}: {found_results_dir}")

    found_results_dir = Path(found_results_dir)
    instances = _load_pred_instances(scene, found_results_dir, n_points=n_points)
    instances = _match_predictions_to_gt(instances, room_data["gt_objects"], min_iou=0.0)
    pred_class, pred_object, id_object_after_inference, pred_score, pred_instance_id = _build_point_prediction_arrays(
        room_data, instances
    )

    idx = np.flatnonzero(room_data["valid_mask"])
    xyz = room_data["xyz"]
    rgb = room_data["rgb"]
    points_df = pd.DataFrame(
        {
            "point_index": idx,
            "x": xyz[idx, 0],
            "y": xyz[idx, 1],
            "z": xyz[idx, 2],
            "r": rgb[idx, 0],
            "g": rgb[idx, 1],
            "b": rgb[idx, 2],
            "gt_class": room_data["gt_class_names"][idx],
            "gt_object": room_data["gt_object_names"][idx],
            "pred_class": pred_class[idx],
            "pred_object": pred_object[idx],
            "id_object_after_inference": id_object_after_inference[idx],
            "pred_score": pred_score[idx],
            "pred_instance_id": pred_instance_id[idx],
        }
    )
    instances_df = pd.DataFrame([{k: v for k, v in inst.items() if k != "mask"} for inst in instances])
    if len(instances_df):
        instances_df = instances_df.sort_values(["score", "num_points"], ascending=[False, False]).reset_index(drop=True)

    result = {
        "area": area,
        "room": room,
        "scene": scene,
        "results_dir": found_results_dir,
        "room_data": room_data,
        "instances": instances,
        "instances_df": instances_df,
        "points_df": points_df,
        "pred_class": pred_class,
        "pred_object": pred_object,
        "id_object_after_inference": id_object_after_inference,
        "pred_object_aligned": pred_object,
        "pred_score": pred_score,
        "pred_instance_id": pred_instance_id,
    }
    if display_tables:
        print(f"Scene: {scene}")
        print(f"Results dir: {found_results_dir}")
        print(f"Valid non-ignored points: {len(points_df)} / {n_points}")
        print(f"Predicted non-ignored instances: {len(instances_df)}")
        display(instances_df)
        display(points_df.head(20))
    return result


def soft_group_inference_room(*args, **kwargs):
    return soft_group_get_instance_seg(*args, **kwargs)


def list_available_s3dis_rooms(context: S3DISVisualizationContext, area_i=None) -> pd.DataFrame:
    if not context.preprocess_dir.exists():
        raise FileNotFoundError(f"Preprocess directory not found: {context.preprocess_dir}")
    area_filter = _normalize_area(area_i) if area_i is not None else None
    rows = []
    for path in sorted(context.preprocess_dir.glob("*_inst_nostuff.pth")):
        scene = path.name.replace("_inst_nostuff.pth", "")
        match = re.fullmatch(r"(Area_\d+)_(.+)", scene)
        if not match:
            continue
        area, room = match.group(1), match.group(2)
        if area_filter is not None and area != area_filter:
            continue
        rows.append({"area": area, "room": room, "scene": scene, "preprocess_path": path})
    rooms_df = pd.DataFrame(rows)
    if len(rooms_df):
        rooms_df = rooms_df.sort_values(["area", "room"], key=lambda col: col.map(_natural_key)).reset_index(drop=True)
    return rooms_df


def run_softgroup_inference_for_all_rooms(
    context: S3DISVisualizationContext,
    area_i=None,
    rooms: Iterable[str] | None = None,
    results_root: Path | None = None,
    force_run: bool = False,
    stop_on_error: bool = True,
    display_progress: bool = True,
) -> pd.DataFrame:
    rooms_df = list_available_s3dis_rooms(context, area_i=area_i)
    if rooms is not None:
        allowed = {str(x).strip() for x in rooms}
        rooms_df = rooms_df[
            rooms_df["room"].astype(str).isin(allowed) | rooms_df["scene"].astype(str).isin(allowed)
        ].reset_index(drop=True)
    if len(rooms_df) == 0:
        raise ValueError("No rooms found for the requested area/room filter.")

    results_root = Path(results_root) if results_root is not None else context.results_root
    ensure_dir(results_root)
    summary_rows = []
    for row in rooms_df.to_dict("records"):
        area, room, scene = row["area"], row["room"], row["scene"]
        out_dir = results_root / scene
        already_done = _has_prediction(out_dir, scene)
        try:
            if force_run or not already_done:
                if display_progress:
                    print(f"[{len(summary_rows) + 1}/{len(rooms_df)}] running {scene}")
                _run_softgroup_inference_for_room(context, area, room, out_dir=out_dir)
                status = "computed"
            else:
                if display_progress:
                    print(f"[{len(summary_rows) + 1}/{len(rooms_df)}] cached {scene}")
                status = "cached"
            summary_rows.append({"area": area, "room": room, "scene": scene, "status": status, "results_dir": str(out_dir), "error": None})
        except Exception as exc:
            summary_rows.append({"area": area, "room": room, "scene": scene, "status": "error", "results_dir": str(out_dir), "error": repr(exc)})
            if stop_on_error:
                raise
    summary_df = pd.DataFrame(summary_rows)
    if display_progress:
        display(summary_df)
    return summary_df


def precompute_softgroup_inference_for_rooms(*args, **kwargs):
    return run_softgroup_inference_for_all_rooms(*args, **kwargs)


def _ensure_inference_data(area_i, room_i, context, inference_data=None):
    if inference_data is None:
        return soft_group_get_instance_seg(area_i, room_i, context=context, display_tables=False)
    return inference_data


def _object_color_map(objects, include_unassigned=True):
    objects = [obj for obj in sorted(set(map(str, objects)), key=_natural_key) if obj and obj != "ignored"]
    color_map = {}
    if include_unassigned:
        color_map["unassigned"] = "lightgray"
    for i, obj in enumerate([obj for obj in objects if obj != "unassigned"]):
        color_map[obj] = NAMED_COLORS[i % len(NAMED_COLORS)]
    return color_map


def _inference_object_color_map(inference_data, include_unassigned=True):
    objects = [str(inst["pred_object"]) for inst in inference_data.get("instances", [])]
    objects.extend(str(inst["id_object_after_inference"]) for inst in inference_data.get("instances", []))
    objects.extend(np.asarray(inference_data.get("pred_object", []), dtype=object).astype(str).tolist())
    return _object_color_map(objects, include_unassigned=include_unassigned)


def show_room(
    area_i,
    room_i,
    context: S3DISVisualizationContext,
    max_points=60000,
    point_size=2,
    opacity=0.85,
    random_state=42,
):
    room_data = load_room_preprocess(area_i, room_i, context)
    idx = _sample_global_indices(room_data["valid_mask"], max_points=max_points, random_state=random_state)
    xyz = room_data["xyz"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xyz[idx, 0],
            y=xyz[idx, 1],
            z=xyz[idx, 2],
            mode="markers",
            name=f"{room_data['requested_scene']} points",
            marker=dict(size=point_size, color=_rgb_strings(room_data["rgb"][idx]), opacity=opacity),
            customdata=_make_base_hover(room_data, idx, include_gt_object=True),
            hovertemplate=_base_hovertemplate(include_gt_object=True),
        )
    )
    fig.update_layout(**_plot_layout(f"{room_data['requested_scene']}: point cloud from SoftGroup preprocess"))
    _finalize_figure(fig, context, f"{room_data['requested_scene']}_room")
    return fig


def show_room_instance_seg(
    area_i,
    room_i,
    context: S3DISVisualizationContext,
    inference_data=None,
    max_points=70000,
    point_size=2.4,
    random_state=42,
):
    inference_data = _ensure_inference_data(area_i, room_i, context, inference_data)
    room_data = inference_data["room_data"]
    idx_all = _sample_global_indices(room_data["valid_mask"], max_points=max_points, random_state=random_state)
    xyz = room_data["xyz"]
    labels = inference_data["pred_object"][idx_all]
    color_map = _inference_object_color_map(inference_data, include_unassigned=True)
    fig = go.Figure()
    for obj in sorted(set(labels), key=_natural_key):
        idx = idx_all[labels == obj]
        if len(idx) == 0:
            continue
        customdata = np.array(
            [
                [
                    f"{int(room_data['rgb'][i, 0])},{int(room_data['rgb'][i, 1])},{int(room_data['rgb'][i, 2])}",
                    room_data["gt_class_names"][i],
                    room_data["gt_object_names"][i],
                    inference_data["pred_object"][i],
                    inference_data["id_object_after_inference"][i],
                    inference_data["pred_score"][i] if not np.isnan(inference_data["pred_score"][i]) else "",
                ]
                for i in idx
            ],
            dtype=object,
        )
        fig.add_trace(
            go.Scatter3d(
                x=xyz[idx, 0],
                y=xyz[idx, 1],
                z=xyz[idx, 2],
                mode="markers",
                name=str(obj),
                marker=dict(size=point_size, color=color_map.get(str(obj), "gray"), opacity=0.85),
                customdata=customdata,
                hovertemplate=(
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                    "<br>rgb=%{customdata[0]}"
                    "<br>gt_class=%{customdata[1]}"
                    "<br>gt_object=%{customdata[2]}"
                    "<br>pred_object=%{customdata[3]}"
                    "<br>id_object_after_inference=%{customdata[4]}"
                    "<br>score=%{customdata[5]}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(**_plot_layout(f"{inference_data['scene']}: SoftGroup instance segmentation"))
    _finalize_figure(fig, context, f"{inference_data['scene']}_instance_seg")
    _display_color_legend({k: v for k, v in color_map.items() if k in set(labels)}, title="Instance colors")
    return fig


def _bbox_edge_points(mins, maxs, points_per_edge=80):
    x0, y0, z0 = mins
    x1, y1, z1 = maxs
    corners = np.array(
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        dtype=np.float32,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    t = np.linspace(0.0, 1.0, int(points_per_edge), dtype=np.float32)[:, None]
    return np.vstack([corners[i] * (1.0 - t) + corners[j] * t for i, j in edges])


def _robust_bbox(points, outlier_quantile=0.02):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("BBox needs a non-empty array with shape (N, 3).")
    if outlier_quantile is not None and 0.0 < float(outlier_quantile) < 0.5 and points.shape[0] >= 10:
        mins = np.quantile(points, float(outlier_quantile), axis=0)
        maxs = np.quantile(points, 1.0 - float(outlier_quantile), axis=0)
    else:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
    return mins.astype(np.float32), maxs.astype(np.float32)


def show_room_bbox(
    area_i,
    room_i,
    context: S3DISVisualizationContext,
    inference_data=None,
    max_points=60000,
    point_size=1.8,
    bbox_point_size=3.0,
    bbox_points_per_edge=90,
    outlier_quantile=0.02,
    random_state=42,
):
    inference_data = _ensure_inference_data(area_i, room_i, context, inference_data)
    room_data = inference_data["room_data"]
    xyz = room_data["xyz"]
    rgb = room_data["rgb"]
    valid = room_data["valid_mask"]
    pred_object_arr = np.asarray(inference_data["pred_object"], dtype=object)
    pred_class_arr = np.asarray(inference_data.get("pred_class", np.array([""] * len(pred_object_arr))), dtype=object)
    id_after_arr = np.asarray(inference_data.get("id_object_after_inference", np.array([""] * len(pred_object_arr))), dtype=object)
    pred_score_arr = np.asarray(inference_data.get("pred_score", np.full(len(pred_object_arr), np.nan)), dtype=float)

    def is_valid_pred_object(obj):
        return str(obj) not in {"", "ignored", "unassigned", "None", "nan"}

    idx_base = _sample_global_indices(valid, max_points=max_points, random_state=random_state)
    base_customdata = np.array(
        [
            [
                f"{int(rgb[i, 0])},{int(rgb[i, 1])},{int(rgb[i, 2])}",
                pred_object_arr[i],
                f"{pred_object_arr[i]}_bbox" if is_valid_pred_object(pred_object_arr[i]) else "",
                room_data["gt_object_names"][i],
                room_data["gt_class_names"][i],
                id_after_arr[i],
                pred_score_arr[i] if not np.isnan(pred_score_arr[i]) else "",
            ]
            for i in idx_base
        ],
        dtype=object,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xyz[idx_base, 0],
            y=xyz[idx_base, 1],
            z=xyz[idx_base, 2],
            mode="markers",
            name="room points",
            marker=dict(size=point_size, color=_rgb_strings(rgb[idx_base]), opacity=0.42),
            customdata=base_customdata,
            hovertemplate=(
                "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                "<br>rgb=%{customdata[0]}"
                "<br>pred_object=%{customdata[1]}"
                "<br>pred_object_bbox=%{customdata[2]}"
                "<br>ground_truth=%{customdata[3]}"
                "<br>ground_truth_class=%{customdata[4]}"
                "<br>id_object_after_inference=%{customdata[5]}"
                "<br>score=%{customdata[6]}"
                "<extra></extra>"
            ),
        )
    )

    pred_objects = sorted({str(obj) for obj in pred_object_arr[valid] if is_valid_pred_object(obj)}, key=_natural_key)
    color_map = _inference_object_color_map(inference_data, include_unassigned=False)
    bbox_rows = []
    for obj in pred_objects:
        mask = (pred_object_arr == obj) & valid
        if int(mask.sum()) < 3:
            continue
        pts = xyz[mask]
        mins, maxs = _robust_bbox(pts, outlier_quantile=outlier_quantile)
        bbox_pts = _bbox_edge_points(mins, maxs, points_per_edge=bbox_points_per_edge)
        bbox_name = f"{obj}_bbox"
        color = color_map.get(obj, "red")
        id_after_values = sorted({str(x) for x in id_after_arr[mask] if str(x) not in {"", "ignored", "unassigned", "None", "nan"}}, key=_natural_key)
        class_values = sorted({str(x) for x in pred_class_arr[mask] if str(x) not in {"", "None", "nan"}}, key=_natural_key)
        score_values = pred_score_arr[mask]
        score_values = score_values[~np.isnan(score_values)]
        bbox_rows.append(
            {
                "pred_object": obj,
                "pred_object_bbox": bbox_name,
                "class_name": class_values[0] if len(class_values) == 1 else ", ".join(class_values),
                "id_object_after_inference": ", ".join(id_after_values),
                "score_max": float(score_values.max()) if len(score_values) else np.nan,
                "num_mask_points": int(mask.sum()),
                "x_min": float(mins[0]),
                "y_min": float(mins[1]),
                "z_min": float(mins[2]),
                "x_max": float(maxs[0]),
                "y_max": float(maxs[1]),
                "z_max": float(maxs[2]),
                "color": color,
            }
        )
        fig.add_trace(
            go.Scatter3d(
                x=bbox_pts[:, 0],
                y=bbox_pts[:, 1],
                z=bbox_pts[:, 2],
                mode="markers",
                name=bbox_name,
                marker=dict(size=bbox_point_size, color=color, opacity=0.98),
                customdata=np.array([[obj, bbox_name, bbox_rows[-1]["class_name"], bbox_rows[-1]["score_max"]]] * len(bbox_pts), dtype=object),
                hovertemplate=(
                    "pred_object=%{customdata[0]}"
                    "<br>pred_object_bbox=%{customdata[1]}"
                    "<br>class=%{customdata[2]}"
                    "<br>score_max=%{customdata[3]}"
                    "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(**_plot_layout(f"{inference_data['scene']}: room with SoftGroup bounding boxes"))
    _finalize_figure(fig, context, f"{inference_data['scene']}_bbox")
    bbox_df = pd.DataFrame(bbox_rows)
    if len(bbox_df):
        bbox_df = bbox_df.sort_values(["pred_object"], key=lambda col: col.map(_natural_key)).reset_index(drop=True)
        _display_color_legend({row["pred_object_bbox"]: row["color"] for _, row in bbox_df.iterrows()}, title="BBox colors")
        display(bbox_df)
    return fig, bbox_df


def compare_inference_with_ground_truth_table(area_i, room_i, context: S3DISVisualizationContext, inference_data=None, display_table=True):
    inference_data = _ensure_inference_data(area_i, room_i, context, inference_data)
    room_data = inference_data["room_data"]
    rows = []
    for gt in sorted(room_data["gt_objects"], key=lambda x: _natural_key(x["gt_object"])):
        obj = gt["gt_object"]
        gt_mask = gt["mask"] & room_data["valid_mask"]
        total = int(gt_mask.sum())
        if total == 0:
            continue
        correct_mask = gt_mask & (inference_data["pred_object"] == obj)
        correct = int(correct_mask.sum())
        matched = [inst for inst in inference_data["instances"] if inst.get("matched_gt_object") == obj]
        rows.append(
            {
                "object": obj,
                "class_name": gt["class_name"],
                "correct_points_percent": 100.0 * correct / total,
                "false_points_count": total - correct,
                "total_gt_points": total,
                "correct_points": correct,
                "false_positive_points": int(np.count_nonzero((inference_data["pred_object"] == obj) & (~gt_mask) & room_data["valid_mask"])),
                "matched_pred_object": matched[0]["pred_object"] if matched else None,
                "matched_id_object_after_inference": matched[0]["id_object_after_inference"] if matched else None,
                "matched_intersection": matched[0].get("matched_intersection", 0) if matched else 0,
                "matched_iou": matched[0]["matched_iou"] if matched else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values("object", key=lambda col: col.map(_natural_key)).reset_index(drop=True)
    if display_table:
        display(table)
    return table


def _resolve_gt_object_name(room_data, object_name):
    object_name = str(object_name)
    all_objects = [obj["gt_object"] for obj in room_data["gt_objects"]]
    if object_name in all_objects:
        return object_name
    matches = [obj for obj in all_objects if obj.startswith(object_name + "_") or obj == object_name]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Object {object_name!r} was not found unambiguously. Available: {', '.join(sorted(all_objects, key=_natural_key))}")


def compare_inference_with_ground_truth_plot(
    area_i,
    room_i,
    object_name,
    context: S3DISVisualizationContext,
    inference_data=None,
    max_background_points=70000,
    background_point_size=1.5,
    highlight_point_size=4.0,
    random_state=42,
):
    inference_data = _ensure_inference_data(area_i, room_i, context, inference_data)
    room_data = inference_data["room_data"]
    target = _resolve_gt_object_name(room_data, object_name)
    xyz = room_data["xyz"]
    valid = room_data["valid_mask"]
    target_gt = (room_data["gt_object_names"] == target) & valid
    pred_as_target = (inference_data["pred_object"] == target) & valid
    correct = target_gt & pred_as_target
    false_to_target = pred_as_target & (~target_gt)
    background_idx = _sample_global_indices(valid, max_points=max_background_points, random_state=random_state)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xyz[background_idx, 0],
            y=xyz[background_idx, 1],
            z=xyz[background_idx, 2],
            mode="markers",
            name="room points",
            marker=dict(size=background_point_size, color="lightgray", opacity=0.22),
            customdata=_make_base_hover(room_data, background_idx, include_gt_object=True),
            hovertemplate=_base_hovertemplate(include_gt_object=True),
        )
    )

    def add_highlight(mask, name, color):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", name=name, marker=dict(size=highlight_point_size, color=color)))
            return
        customdata = np.array(
            [
                [
                    f"{int(room_data['rgb'][i, 0])},{int(room_data['rgb'][i, 1])},{int(room_data['rgb'][i, 2])}",
                    room_data["gt_object_names"][i],
                    inference_data["pred_object"][i],
                    inference_data["id_object_after_inference"][i],
                    inference_data["pred_score"][i] if not np.isnan(inference_data["pred_score"][i]) else "",
                ]
                for i in idx
            ],
            dtype=object,
        )
        fig.add_trace(
            go.Scatter3d(
                x=xyz[idx, 0],
                y=xyz[idx, 1],
                z=xyz[idx, 2],
                mode="markers",
                name=name,
                marker=dict(size=highlight_point_size, color=color, opacity=0.95),
                customdata=customdata,
                hovertemplate=(
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                    "<br>rgb=%{customdata[0]}"
                    "<br>gt_object=%{customdata[1]}"
                    "<br>pred_object=%{customdata[2]}"
                    "<br>id_object_after_inference=%{customdata[3]}"
                    "<br>score=%{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    add_highlight(correct, f"green: correctly predicted {target}", "green")
    add_highlight(false_to_target, f"red: falsely attributed to {target}", "red")
    fig.update_layout(**_plot_layout(f"{inference_data['scene']}: GT vs SoftGroup for {target}"))
    _finalize_figure(fig, context, f"{inference_data['scene']}_{target}_compare")
    print(f"{target}: correct green points = {int(correct.sum())}, false red points = {int(false_to_target.sum())}")
    return fig


def save_room_bbox_csv(bbox_df: pd.DataFrame, area_i, room_i, context: S3DISVisualizationContext, out_dir: Path | None = None) -> Path:
    _, _, scene = scene_id(area_i, room_i)
    base_dir = ensure_dir(Path(out_dir) if out_dir is not None else context.html_dir / "bbox_csv")
    csv_path = base_dir / f"{scene}_aabb.csv"
    bbox_df.to_csv(csv_path, index=False)
    print(f"Saved bbox CSV: {csv_path}")
    return csv_path


def apply_visualization_defaults(cfg: DictConfig, command: str) -> dict[str, Any]:
    vis = cfg.visualization
    common = {"random_state": int(vis.random_state)}
    if command == "room":
        return {
            **common,
            "max_points": int(vis.room.max_points),
            "point_size": float(vis.room.point_size),
            "opacity": float(vis.room.opacity),
        }
    if command == "instance":
        return {**common, "max_points": int(vis.instance.max_points), "point_size": float(vis.instance.point_size)}
    if command == "bbox":
        return {
            **common,
            "max_points": int(vis.bbox.max_points),
            "point_size": float(vis.bbox.point_size),
            "bbox_point_size": float(vis.bbox.bbox_point_size),
            "bbox_points_per_edge": int(vis.bbox.bbox_points_per_edge),
            "outlier_quantile": vis.bbox.outlier_quantile,
        }
    if command == "compare":
        return {
            **common,
            "max_background_points": int(vis.compare.max_background_points),
            "background_point_size": float(vis.compare.background_point_size),
            "highlight_point_size": float(vis.compare.highlight_point_size),
        }
    return common
