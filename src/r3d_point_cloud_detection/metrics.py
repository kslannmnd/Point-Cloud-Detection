from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from .s3dis_visualization import (
    compare_inference_with_ground_truth_table,
    context_from_cfg,
    soft_group_get_instance_seg,
)
from .softgroup import resolve_s3dis_room_selection
from .utils import dump_effective_config, dump_json, ensure_dir, project_path


def compute_s3dis_metrics(cfg: DictConfig) -> Path:
    context = context_from_cfg(cfg)
    context.checkpoint_path = project_path(cfg, cfg.metrics.checkpoint)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(project_path(cfg, cfg.metrics.output_dir) / run_id)
    rooms_df = resolve_metric_rooms(cfg, context)

    per_object_tables: list[pd.DataFrame] = []
    room_rows: list[dict[str, Any]] = []
    for row in rooms_df.to_dict("records"):
        area, room, scene = row["area"], row["room"], row["scene"]
        print(f"Computing S3DIS metrics for {scene}")
        inference_data = soft_group_get_instance_seg(
            area,
            room,
            context=context,
            force_run=bool(cfg.metrics.force_run),
            display_tables=False,
            allow_precomputed_results=bool(cfg.metrics.allow_precomputed_results),
        )
        table = compare_inference_with_ground_truth_table(
            area,
            room,
            context=context,
            inference_data=inference_data,
            display_table=bool(cfg.metrics.display_tables),
        )
        if len(table):
            table.insert(0, "scene", scene)
            table.insert(0, "room", room)
            table.insert(0, "area", area)
            per_object_tables.append(table)
            total_gt = int(table["total_gt_points"].sum())
            total_correct = int(table["correct_points"].sum())
            total_false = int(table["false_points_count"].sum())
            total_fp = int(table["false_positive_points"].sum())
            room_rows.append(
                {
                    "area": area,
                    "room": room,
                    "scene": scene,
                    "objects": int(len(table)),
                    "total_gt_points": total_gt,
                    "correct_points": total_correct,
                    "false_points_count": total_false,
                    "false_positive_points": total_fp,
                    "weighted_correct_points_percent": 100.0 * total_correct / total_gt if total_gt else 0.0,
                    "mean_object_correct_points_percent": float(table["correct_points_percent"].mean()),
                    "mean_matched_iou": float(table["matched_iou"].mean()),
                }
            )
        else:
            room_rows.append(
                {
                    "area": area,
                    "room": room,
                    "scene": scene,
                    "objects": 0,
                    "total_gt_points": 0,
                    "correct_points": 0,
                    "false_points_count": 0,
                    "false_positive_points": 0,
                    "weighted_correct_points_percent": 0.0,
                    "mean_object_correct_points_percent": 0.0,
                    "mean_matched_iou": 0.0,
                }
            )

    per_object_df = pd.concat(per_object_tables, ignore_index=True) if per_object_tables else pd.DataFrame()
    room_df = pd.DataFrame(room_rows)
    summary = build_metrics_summary(room_df)

    if cfg.metrics.save_per_object_csv:
        per_object_df.to_csv(run_dir / "per_object_metrics.csv", index=False)
    if cfg.metrics.save_room_csv:
        room_df.to_csv(run_dir / "room_metrics.csv", index=False)
    if cfg.metrics.save_summary_json:
        dump_json(summary, run_dir / "summary.json")
    effective_config = dump_effective_config(cfg, run_dir)
    dump_json(
        {
            "checkpoint_path": str(context.checkpoint_path),
            "config_path": str(effective_config),
            "rooms": room_df["scene"].tolist() if "scene" in room_df else [],
            "summary_path": str(run_dir / "summary.json") if cfg.metrics.save_summary_json else None,
            "room_metrics_path": str(run_dir / "room_metrics.csv") if cfg.metrics.save_room_csv else None,
            "per_object_metrics_path": str(run_dir / "per_object_metrics.csv") if cfg.metrics.save_per_object_csv else None,
        },
        run_dir / "provenance.json",
    )

    print("Room metrics:")
    print(room_df)
    print("Summary:")
    print(summary)
    print(f"Metrics saved to: {run_dir}")
    return run_dir


def resolve_metric_rooms(cfg: DictConfig, context) -> pd.DataFrame:
    rows = resolve_s3dis_room_selection(
        cfg,
        rooms_value=cfg.metrics.rooms,
        selected_rooms=cfg.metrics.selected_rooms,
        max_rooms=cfg.metrics.max_rooms,
        default_scope="test",
        enforce_scope=bool(cfg.metrics.get("enforce_test_area", True)),
    )
    rooms_df = pd.DataFrame(rows)[["area", "room", "scene"]]
    if len(rooms_df) == 0:
        raise ValueError("No S3DIS metric rooms selected.")
    return rooms_df


def build_metrics_summary(room_df: pd.DataFrame) -> dict[str, Any]:
    if len(room_df) == 0:
        return {"rooms": 0}
    total_gt = int(room_df["total_gt_points"].sum())
    total_correct = int(room_df["correct_points"].sum())
    return {
        "rooms": int(len(room_df)),
        "objects": int(room_df["objects"].sum()),
        "total_gt_points": total_gt,
        "correct_points": total_correct,
        "false_points_count": int(room_df["false_points_count"].sum()),
        "false_positive_points": int(room_df["false_positive_points"].sum()),
        "weighted_correct_points_percent": 100.0 * total_correct / total_gt if total_gt else 0.0,
        "mean_room_correct_points_percent": float(room_df["weighted_correct_points_percent"].mean()),
        "mean_room_matched_iou": float(room_df["mean_matched_iou"].mean()),
    }
