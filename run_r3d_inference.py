from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import hydra
from omegaconf import DictConfig

from r3d_point_cloud_detection.pipeline import run_r3d_inference


def _extract_positional_r3d_arg() -> str | None:
    if len(sys.argv) <= 1:
        return None
    first = sys.argv[1]
    if first.startswith("-") or "=" in first:
        return None
    sys.argv.pop(1)
    return first


R3D_FILE_ARG = _extract_positional_r3d_arg()
if R3D_FILE_ARG:
    sys.argv.extend(["dataset=r3d", "inference=r3d", f"inference.target.file={R3D_FILE_ARG}"])


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_r3d_inference(cfg)


if __name__ == "__main__":
    main()
