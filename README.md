# Point Cloud R3D MVP

The inference path for Record3D/LiDAR captures:

```text
.r3d file + frame_index
  -> FastAPI /detect
  -> server reconstructs selected depth frame into a point cloud
  -> inference engine returns 3D boxes
  -> client renders point cloud + boxes in Plotly
```

## Important files

- `common/r3d.py` — parser for `.r3d` files
- `data/r3d/` — uploaded Record3D/LiDAR recordings
- `server/app.py` — FastAPI server
- `server/core.py` — converts `.r3d` frames into `SceneData`
- `server/engines/stub.py` — model-free demo engine
- `server/engines/softgroup.py` — adapter boundary for the production model
- `client/demo_client.py` — uploads `.r3d` + `frame_index` and saves Plotly HTML

## API

`POST /detect` accepts multipart form data:

- `file` — required `.r3d`
- `frame_index` — zero-based frame index inside the `.r3d`
- `conf_threshold` — Record3D confidence threshold, default `1`
- `sample_step` — optional downsampling stride, default `1`
- `z_min`, `z_max` — optional depth clipping in meters

## Engine modes

Set `ENGINE_MODE` before starting the server:

- `model` — default; call an external model adapter via `MODEL_ADAPTER_MODULE`
- `stub` — model-free demo on Record3D `.r3d` point clouds
- `oracle` — legacy mode for labeled arrays

Model adapter:

```bash
MODEL_ADAPTER_MODULE=your_module_name uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The client command does not change when switching from `stub` to `model`.
