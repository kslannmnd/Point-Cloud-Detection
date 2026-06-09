from __future__ import annotations

import os
from pathlib import Path

ENGINE_MODE = os.getenv("ENGINE_MODE", "model").strip().lower()
if ENGINE_MODE not in {"oracle", "stub", "model"}:
    ENGINE_MODE = "model"

CLASS_NAMES = os.getenv("CLASS_NAMES", "chair,table,sofa").split(",")
CLASS_NAMES = [c.strip() for c in CLASS_NAMES if c.strip()]
if not CLASS_NAMES:
    CLASS_NAMES = ["chair", "table", "sofa"]

IGNORE_INDEX = int(os.getenv("IGNORE_INDEX", "-1"))
MIN_POINTS_PER_OBJECT = int(os.getenv("MIN_POINTS_PER_OBJECT", "80"))
N_CLUSTERS_MAX = int(os.getenv("N_CLUSTERS_MAX", "6"))
N_CLUSTERS_MIN = int(os.getenv("N_CLUSTERS_MIN", "1"))
KMEANS_ITERS = int(os.getenv("KMEANS_ITERS", "15"))

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

BASE_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = Path(os.getenv("TMP_DIR", str(BASE_DIR / "tmp")))
TMP_DIR.mkdir(parents=True, exist_ok=True)
