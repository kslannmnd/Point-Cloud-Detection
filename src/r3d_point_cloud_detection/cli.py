from __future__ import annotations

from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from .s3dis_visualization import (
    apply_visualization_defaults,
    compare_inference_with_ground_truth_plot,
    compare_inference_with_ground_truth_table,
    context_from_cfg,
    list_available_s3dis_rooms,
    run_softgroup_inference_for_all_rooms,
    save_room_bbox_csv,
    show_room,
    show_room_bbox,
    show_room_instance_seg,
    soft_group_get_instance_seg,
)
from .metrics import compute_s3dis_metrics


def _project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "configs" / "config.yaml").exists():
        return cwd
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find configs/config.yaml. Run from the project root or install from this checkout.")


def _compose_cfg(overrides: list[str]):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = str((_project_root() / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides)


def _split_positionals(argv: list[str], count: int) -> tuple[list[str], list[str]]:
    positionals: list[str] = []
    overrides: list[str] = []
    for arg in argv:
        if len(positionals) < count and "=" not in arg and not arg.startswith("-"):
            positionals.append(arg)
        else:
            overrides.append(arg)
    if len(positionals) < count:
        raise SystemExit(f"Expected {count} positional arguments, got {len(positionals)}.")
    return positionals, overrides


def _ctx(overrides: list[str]):
    cfg = _compose_cfg(overrides)
    return cfg, context_from_cfg(cfg)


def show_s3dis_rooms_command(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    area = None
    overrides = []
    if argv and "=" not in argv[0] and not argv[0].startswith("-"):
        area = argv.pop(0)
    overrides = argv
    _, context = _ctx(overrides)
    print(list_available_s3dis_rooms(context, area_i=area))


def show_room_command(argv: list[str] | None = None) -> None:
    positionals, overrides = _split_positionals(list(sys.argv[1:] if argv is None else argv), 2)
    cfg, context = _ctx(overrides)
    show_room(positionals[0], positionals[1], context=context, **apply_visualization_defaults(cfg, "room"))


def show_room_instance_seg_command(argv: list[str] | None = None) -> None:
    positionals, overrides = _split_positionals(list(sys.argv[1:] if argv is None else argv), 2)
    cfg, context = _ctx(overrides)
    inference_data = soft_group_get_instance_seg(
        positionals[0],
        positionals[1],
        context=context,
        force_run=bool(cfg.visualization.force_run),
        allow_precomputed_results=bool(cfg.visualization.allow_precomputed_results),
    )
    show_room_instance_seg(positionals[0], positionals[1], context=context, inference_data=inference_data, **apply_visualization_defaults(cfg, "instance"))


def show_room_bbox_command(argv: list[str] | None = None) -> None:
    positionals, overrides = _split_positionals(list(sys.argv[1:] if argv is None else argv), 2)
    cfg, context = _ctx(overrides)
    inference_data = soft_group_get_instance_seg(
        positionals[0],
        positionals[1],
        context=context,
        force_run=bool(cfg.visualization.force_run),
        allow_precomputed_results=bool(cfg.visualization.allow_precomputed_results),
    )
    _, bbox_df = show_room_bbox(positionals[0], positionals[1], context=context, inference_data=inference_data, **apply_visualization_defaults(cfg, "bbox"))
    if len(bbox_df):
        save_room_bbox_csv(bbox_df, positionals[0], positionals[1], context=context)


def show_gt_compare_table_command(argv: list[str] | None = None) -> None:
    positionals, overrides = _split_positionals(list(sys.argv[1:] if argv is None else argv), 2)
    cfg, context = _ctx(overrides)
    inference_data = soft_group_get_instance_seg(
        positionals[0],
        positionals[1],
        context=context,
        force_run=bool(cfg.visualization.force_run),
        allow_precomputed_results=bool(cfg.visualization.allow_precomputed_results),
    )
    compare_inference_with_ground_truth_table(positionals[0], positionals[1], context=context, inference_data=inference_data)


def show_gt_compare_plot_command(argv: list[str] | None = None) -> None:
    positionals, overrides = _split_positionals(list(sys.argv[1:] if argv is None else argv), 3)
    cfg, context = _ctx(overrides)
    inference_data = soft_group_get_instance_seg(
        positionals[0],
        positionals[1],
        context=context,
        force_run=bool(cfg.visualization.force_run),
        allow_precomputed_results=bool(cfg.visualization.allow_precomputed_results),
    )
    compare_inference_with_ground_truth_plot(
        positionals[0],
        positionals[1],
        positionals[2],
        context=context,
        inference_data=inference_data,
        **apply_visualization_defaults(cfg, "compare"),
    )


def precompute_s3dis_inference_command(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    area = None
    rooms = None
    overrides = []
    if argv and "=" not in argv[0] and not argv[0].startswith("-"):
        area = argv.pop(0)
    if argv and "=" not in argv[0] and not argv[0].startswith("-"):
        rooms = [room.strip() for room in argv.pop(0).split(",") if room.strip()]
    overrides = argv
    cfg, context = _ctx(overrides)
    summary = run_softgroup_inference_for_all_rooms(
        context,
        area_i=area,
        rooms=rooms,
        force_run=bool(cfg.visualization.force_run),
        display_progress=bool(cfg.visualization.display_tables),
    )
    print(summary)


def compute_s3dis_metrics_command(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = _compose_cfg(argv)
    compute_s3dis_metrics(cfg)
