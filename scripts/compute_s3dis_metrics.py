from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hydra
from omegaconf import DictConfig

from r3d_point_cloud_detection.metrics import compute_s3dis_metrics


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    compute_s3dis_metrics(cfg)


if __name__ == "__main__":
    main()
