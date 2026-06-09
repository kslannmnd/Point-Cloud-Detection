from __future__ import annotations

import argparse
from pathlib import Path
import json

if __package__:
    from .api import send_scene
    from .visualize import visualize_scene
else:
    from api import send_scene
    from visualize import visualize_scene

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "r3d"


def default_r3d_scene() -> Path:
    scenes = sorted(DEFAULT_DATA_DIR.glob("*.r3d"))
    if not scenes:
        raise FileNotFoundError(f"No .r3d files found in {DEFAULT_DATA_DIR}")
    return scenes[0]


def load_scene_for_visualization(
    scene_path: Path,
    frame_index: int = 0,
    conf_threshold: int = 1,
    sample_step: int = 1,
    z_min: float | None = None,
    z_max: float | None = None,
):
    suffix = scene_path.suffix.lower()
    if suffix != ".r3d":
        raise ValueError(f"The MVP client expects a .r3d Record3D file, got: {scene_path}")

    from common.r3d import load_r3d_frame

    frame = load_r3d_frame(
        scene_path,
        frame_index=frame_index,
        conf_threshold=conf_threshold,
        sample_step=sample_step,
        z_min=z_min,
        z_max=z_max,
        use_rgb=True,
    )
    return frame.coord, frame.color

def main():
    parser = argparse.ArgumentParser(description="Point cloud MVP client")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="FastAPI server URL")
    parser.add_argument("--scene", type=str, default=None, help="Scene file, normally an .r3d recording")
    parser.add_argument("--frame-index", type=int, default=0, help="Zero-based depth frame index inside the .r3d recording")
    parser.add_argument("--conf-threshold", type=int, default=1, choices=[0, 1, 2], help="Record3D confidence threshold")
    parser.add_argument("--sample-step", type=int, default=1, help="Use every Nth depth pixel before inference")
    parser.add_argument("--z-min", type=float, default=None, help="Optional minimum depth in meters")
    parser.add_argument("--z-max", type=float, default=None, help="Optional maximum depth in meters")
    parser.add_argument("--out-html", type=str, default=str(Path(__file__).resolve().parent / "demo_result.html"))
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    scene_path = Path(args.scene) if args.scene else default_r3d_scene()
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)

    result = send_scene(
        args.server,
        scene_path,
        frame_index=args.frame_index,
        conf_threshold=args.conf_threshold,
        sample_step=args.sample_step,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"Server mode: {result['mode']}")
    print(f"Input: {result.get('input_format')} {scene_path}")
    if result.get("frame_index") is not None:
        print(f"Frame: index={result.get('frame_index')} id={result.get('frame_id')}")
    print(f"Points: {result['n_points']}")
    print(f"Objects: {len(result['objects'])}")

    for obj in result["objects"]:
        bbox = obj["bbox"]
        cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
        cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
        cz = (bbox["z_min"] + bbox["z_max"]) / 2.0
        print(
            f"- {obj['object_name']:<12} "
            f"class={obj['class_name']:<8} score={obj['score']:.2f} "
            f"center=({cx:.2f}, {cy:.2f}, {cz:.2f})"
        )

    coord, color = load_scene_for_visualization(
        scene_path,
        frame_index=args.frame_index,
        conf_threshold=args.conf_threshold,
        sample_step=args.sample_step,
        z_min=args.z_min,
        z_max=args.z_max,
    )

    fig = visualize_scene(coord=coord, objects=result["objects"], color=color, title=f"{scene_path.name} → {result['mode']}")
    out_html = Path(args.out_html)
    fig.write_html(out_html)
    print(f"Saved visualization: {out_html}")

if __name__ == "__main__":
    main()
