from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import requests

def send_scene(
    server_url: str,
    scene_path: str | Path,
    timeout: int = 60,
    frame_index: int = 0,
    conf_threshold: int = 1,
    sample_step: int = 1,
    z_min: float | None = None,
    z_max: float | None = None,
) -> Dict[str, Any]:
    path = Path(scene_path)
    data = {
        "frame_index": str(frame_index),
        "conf_threshold": str(conf_threshold),
        "sample_step": str(sample_step),
    }
    if z_min is not None:
        data["z_min"] = str(z_min)
    if z_max is not None:
        data["z_max"] = str(z_max)

    with open(path, "rb") as f:
        files = {"file": (path.name, f, "application/octet-stream")}
        resp = requests.post(f"{server_url.rstrip('/')}/detect", files=files, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
