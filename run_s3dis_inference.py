from __future__ import annotations

import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from r3d_point_cloud_detection.softgroup import run_s3dis_inference


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_s3dis_inference(cfg)


if __name__ == "__main__":
    main()
