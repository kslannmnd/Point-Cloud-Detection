from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r3d_point_cloud_detection.cli import show_room_instance_seg_command


if __name__ == "__main__":
    show_room_instance_seg_command()
