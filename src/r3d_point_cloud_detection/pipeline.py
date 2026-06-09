from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .r3d import SingleFrameConfig, r3d_to_point_cloud, save_point_cloud_npz
from .r3d_softgroup import (
    instances_to_point_predictions,
    instances_to_bboxes,
    load_r3d_pred_instances,
    make_r3d_scene_name,
    run_r3d_softgroup_inference,
    save_r3d_softgroup_preprocess,
)
from .softgroup import (
    download_softgroup_checkpoints,
    generate_softgroup_configs,
    install_dependencies_and_build_softgroup,
    patch_softgroup_for_kaggle,
    prepare_kaggle_env,
    prepare_s3dis,
    run_s3dis_inference,
    train_softgroup,
)
from .utils import dump_effective_config, dump_json, ensure_dir, project_path, resolve_path
from .visualization import make_point_cloud_figure, save_plotly_html


def setup_all(cfg: DictConfig) -> None:
    if cfg.setup.run_prepare_kaggle_env:
        prepare_kaggle_env(cfg)
    if cfg.setup.run_install_dependencies:
        install_dependencies_and_build_softgroup(cfg)
    if cfg.setup.run_prepare_s3dis and cfg.dataset.type == "s3dis":
        prepare_s3dis(cfg)
    if cfg.setup.run_download_checkpoints:
        download_softgroup_checkpoints(cfg)
    if cfg.setup.run_patch_softgroup:
        patch_softgroup_for_kaggle(cfg)
    if cfg.setup.run_generate_configs and cfg.dataset.type == "s3dis":
        generate_softgroup_configs(cfg)


def train(cfg: DictConfig):
    return train_softgroup(cfg)


def infer(cfg: DictConfig):
    if cfg.inference.mode == "s3dis":
        return run_s3dis_inference(cfg)
    if cfg.inference.mode == "r3d":
        return run_r3d_inference(cfg)
    raise ValueError(f"Unsupported inference mode: {cfg.inference.mode}")


def resolve_r3d_input(cfg: DictConfig) -> Path:
    file_name = cfg.inference.target.file
    if not file_name:
        raise ValueError("R3D inference needs inference.target.file or positional run_r3d_inference.py argument.")
    direct_path = resolve_path(file_name)
    if direct_path.exists():
        return direct_path
    candidate = project_path(cfg, Path(cfg.dataset.data_dir) / str(file_name))
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"R3D file not found as {direct_path} or {candidate}")


def run_r3d_inference(cfg: DictConfig) -> Path:
    r3d_path = resolve_r3d_input(cfg)
    output_root = project_path(cfg, cfg.inference.output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_root / f"{run_id}_{r3d_path.stem}")

    frame_cfg = SingleFrameConfig(
        r3d_path=r3d_path,
        frame_index=int(cfg.dataset.frame_index),
        confidence_threshold=int(cfg.dataset.confidence_threshold),
        z_min=cfg.dataset.z_min,
        z_max=cfg.dataset.z_max,
        sample_step=int(cfg.dataset.sample_step),
        use_rgb=bool(cfg.dataset.use_rgb),
    )
    result = r3d_to_point_cloud(frame_cfg)
    point_cloud_path = save_point_cloud_npz(result, run_dir / f"{r3d_path.stem}_frame_{result['frame_id']}.npz")
    boxes = load_r3d_bbox_overlay(cfg)
    bbox_df = pd.DataFrame()
    softgroup_raw_dir = None
    softgroup_preprocess_path = None
    instance_colors = None
    instance_labels = None
    instance_ids = np.full(len(result["points"]), -1, dtype=np.int64)
    instances_df = pd.DataFrame()
    point_instances_df = pd.DataFrame()
    instance_summary_csv = None
    point_instance_csv = None
    scene = make_r3d_scene_name(str(cfg.inference.scene_prefix), r3d_path.stem, str(result["frame_id"]))

    if bool(cfg.inference.run_softgroup):
        softgroup_preprocess_path = save_r3d_softgroup_preprocess(cfg, result, scene)
        softgroup_raw_dir = run_r3d_softgroup_inference(cfg, scene, run_dir)
        instances = load_r3d_pred_instances(scene, softgroup_raw_dir, n_points=len(result["points"]))
        instance_colors, instance_labels, instance_ids, _, point_instances_df = instances_to_point_predictions(
            len(result["points"]),
            instances,
        )
        if instances and int(np.count_nonzero(instance_ids >= 0)) == 0:
            raise RuntimeError(f"R3D SoftGroup inference produced no assigned instance points for scene {scene}.")
        instances_df = pd.DataFrame([{key: value for key, value in inst.items() if key != "mask"} for inst in instances])
        instance_summary_csv = run_dir / "r3d_softgroup_instances.csv"
        point_instance_csv = run_dir / "r3d_softgroup_point_instances.csv"
        instances_df.to_csv(instance_summary_csv, index=False)
        point_instances_df.to_csv(point_instance_csv, index=False)
        predicted_boxes, bbox_df = instances_to_bboxes(
            np.asarray(result["points"], dtype=np.float32),
            instances,
            min_points=int(cfg.inference.min_points_for_box),
            outlier_quantile=cfg.dataset.bbox.get("outlier_quantile", 0.02),
        )
        boxes.extend(predicted_boxes)
        if len(bbox_df):
            bbox_df.to_csv(run_dir / "r3d_softgroup_bboxes.csv", index=False)

    title_suffix = " SoftGroup instance segmentation" if bool(cfg.inference.run_softgroup) else ""
    fig = make_point_cloud_figure(
        result["points"],
        instance_colors if instance_colors is not None else result.get("colors"),
        labels=instance_labels,
        boxes=boxes,
        max_points=int(cfg.dataset.max_points_plot),
        point_size=float(cfg.dataset.point_size),
        height=int(cfg.dataset.plot_height),
        title=f"{r3d_path.name}: frame {result['frame_id']}{title_suffix}",
    )
    html_path = save_plotly_html(fig, run_dir / "point_cloud.html") if cfg.inference.save_visualization else None

    adapter_status = "not_run"
    adapter_note = str(cfg.dataset.softgroup_adapter.note)
    if bool(cfg.inference.run_softgroup):
        adapter_status = "completed"
        adapter_note = "R3D point cloud was converted to a temporary SoftGroup S3DIS-style test scene."

    effective_config = dump_effective_config(cfg, run_dir)
    checkpoint = project_path(cfg, cfg.inference.checkpoint)
    dump_json(
        {
            "input_r3d_path": str(r3d_path),
            "point_cloud_path": str(point_cloud_path),
            "visualization_html": str(html_path) if html_path else None,
            "checkpoint_path": str(checkpoint),
            "config_path": str(effective_config),
            "softgroup_scene": scene,
            "softgroup_preprocess_path": str(softgroup_preprocess_path) if softgroup_preprocess_path else None,
            "softgroup_raw_output_dir": str(softgroup_raw_dir) if softgroup_raw_dir else None,
            "softgroup_adapter_status": adapter_status,
            "softgroup_adapter_note": adapter_note,
            "predicted_instance_count": int(len(instances_df)),
            "assigned_instance_points": int(np.count_nonzero(instance_ids >= 0)),
            "instance_summary_csv": str(instance_summary_csv) if instance_summary_csv else None,
            "point_instance_csv": str(point_instance_csv) if point_instance_csv else None,
            "bbox_overlay_count": len(boxes),
            "predicted_bbox_count": int(len(bbox_df)),
            "softgroup_bbox_csv": str(run_dir / "r3d_softgroup_bboxes.csv") if len(bbox_df) else None,
            "target": OmegaConf.to_container(cfg.inference.target, resolve=True),
            "num_points": int(len(result["points"])),
            "frame_id": str(result["frame_id"]),
        },
        run_dir / "provenance.json",
    )
    print(f"R3D point cloud saved to: {point_cloud_path}")
    if html_path:
        print(f"Interactive visualization saved to: {html_path}")
    print(f"Run provenance saved to: {run_dir / 'provenance.json'}")
    return run_dir


def load_r3d_bbox_overlay(cfg: DictConfig) -> list[dict]:
    bbox_cfg = cfg.dataset.get("bbox")
    if not bbox_cfg or not bbox_cfg.get("csv_path"):
        return []
    csv_path = resolve_path(bbox_cfg.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"R3D bbox overlay CSV not found: {csv_path}")
    table = pd.read_csv(csv_path)
    required = {"x_min", "y_min", "z_min", "x_max", "y_max", "z_max"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"R3D bbox CSV missing required columns: {sorted(missing)}")

    boxes: list[dict] = []
    label_column = str(bbox_cfg.get("label_column", "label"))
    color_column = str(bbox_cfg.get("color_column", "color"))
    for idx, row in table.iterrows():
        x0, y0, z0 = float(row.x_min), float(row.y_min), float(row.z_min)
        x1, y1, z1 = float(row.x_max), float(row.y_max), float(row.z_max)
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
        label = str(row[label_column]) if label_column in table.columns else f"bbox_{idx}"
        color = str(row[color_column]) if color_column in table.columns else str(bbox_cfg.get("default_color", "red"))
        boxes.append({"points": corners, "label": label, "color": color})
    return boxes
