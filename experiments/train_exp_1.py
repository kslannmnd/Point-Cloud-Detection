from __future__ import annotations

from pathlib import Path
import sys

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r3d_point_cloud_detection.softgroup import train_softgroup


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiments/train_exp_1.py path/to/effective_config.yaml")
    cfg = OmegaConf.load(sys.argv[1])
    cfg.training.mode = "experiment_file"
    train_softgroup(cfg)


if __name__ == "__main__":
    main()
