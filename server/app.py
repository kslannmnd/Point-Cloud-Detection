from __future__ import annotations

from typing import Optional, List, Dict, Any
import json
from pathlib import Path
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from .config import ENGINE_MODE, CLASS_NAMES, SERVER_HOST, SERVER_PORT, TMP_DIR
from .schemas import DetectResponse, DetectedObject, BBox
from .core import load_scene_file, SceneData
from .engines.oracle import OracleEngine
from .engines.stub import StubEngine
from .engines.softgroup import SoftGroupEngine

app = FastAPI(title="Point Cloud MVP API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    if ENGINE_MODE == "stub":
        _engine = StubEngine(class_names=CLASS_NAMES)
    elif ENGINE_MODE == "model":
        _engine = SoftGroupEngine(class_names=CLASS_NAMES)
    else:
        _engine = OracleEngine(class_names=CLASS_NAMES)
    return _engine

def _to_detect_response(scene: SceneData, objects: List[Dict[str, Any]], warnings: List[str] | None = None) -> DetectResponse:
    out_objects = []
    for obj in objects:
        bbox = obj["bbox"]
        out_objects.append(
            DetectedObject(
                object_id=int(obj["object_id"]),
                object_name=str(obj["object_name"]),
                class_id=int(obj["class_id"]),
                class_name=str(obj["class_name"]),
                score=float(obj["score"]),
                num_points=int(obj["num_points"]),
                bbox=BBox(**bbox),
                source=str(obj.get("source", ENGINE_MODE)),
                proposal_id=obj.get("proposal_id"),
            )
        )
    return DetectResponse(
        mode=ENGINE_MODE,
        n_points=scene.n_points,
        objects=out_objects,
        scene_name=scene.scene_name,
        input_format=scene.input_format,
        frame_index=scene.frame_index,
        frame_id=scene.frame_id,
        has_ground_truth=bool(scene.segment is not None and scene.instance is not None),
        warnings=warnings or [],
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine_mode": ENGINE_MODE,
        "class_names": CLASS_NAMES,
    }

@app.get("/")
def root():
    return {
        "message": "Point Cloud MVP API",
        "health": "/health",
        "detect": "/detect",
    }

@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    frame_index: int = Form(0),
    conf_threshold: int = Form(1),
    sample_step: int = Form(1),
    z_min: Optional[float] = Form(None),
    z_max: Optional[float] = Form(None),
):
    suffix = Path(file.filename).suffix.lower()
    raw = await file.read()
    original_stem = Path(file.filename).stem or "upload"
    tmp_path = TMP_DIR / f"{uuid.uuid4().hex}_{original_stem}{suffix}"
    tmp_path.write_bytes(raw)

    try:
        scene = load_scene_file(
            tmp_path,
            frame_index=frame_index,
            conf_threshold=conf_threshold,
            z_min=z_min,
            z_max=z_max,
            sample_step=sample_step,
        )
        if scene.input_format == "r3d":
            scene.scene_name = f"{original_stem}_frame_{scene.frame_id}"
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load scene: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    engine = get_engine()
    warnings: List[str] = []
    try:
        objects = engine.predict(scene)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    if scene.segment is None or scene.instance is None:
        if ENGINE_MODE == "oracle":
            warnings.append("Scene has no segment/instance labels; oracle mode fell back to stub-like clustering.")
    return _to_detect_response(scene, objects, warnings)

@app.post("/detect_json")
async def detect_json(payload: Dict[str, Any]):
    """Alternative endpoint for direct JSON payloads."""
    try:
        coord = payload["coord"]
        color = payload.get("color")
        normal = payload.get("normal")
        segment = payload.get("segment")
        instance = payload.get("instance")
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field: {exc}") from exc

    import numpy as np
    scene = SceneData(
        coord=np.asarray(coord, dtype=np.float32),
        color=np.asarray(color, dtype=np.float32) if color is not None else np.full((len(coord), 3), 127, dtype=np.float32),
        normal=np.asarray(normal, dtype=np.float32) if normal is not None else np.zeros((len(coord), 3), dtype=np.float32),
        segment=np.asarray(segment, dtype=np.int32) if segment is not None else None,
        instance=np.asarray(instance, dtype=np.int32) if instance is not None else None,
        scene_name=str(payload.get("scene_name", "json_scene")),
        input_format="json",
    )
    objects = get_engine().predict(scene)
    return _to_detect_response(scene, objects)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
