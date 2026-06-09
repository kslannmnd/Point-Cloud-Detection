from __future__ import annotations

import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r3d_point_cloud_detection.softgroup import generate_softgroup_configs


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    generate_softgroup_configs(cfg)


if __name__ == "__main__":
    main()
