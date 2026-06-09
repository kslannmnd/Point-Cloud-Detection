# Client

The client uploads a Record3D `.r3d` file and the selected `frame_index` to the FastAPI server, then renders the same local frame with returned 3D boxes.

Useful parameters:

```text
--frame-index      zero-based frame inside the .r3d sequence
--conf-threshold   0, 1, or 2; higher means fewer, cleaner points
--sample-step      1 keeps all depth pixels, 2 keeps every second pixel
--z-min / --z-max  optional depth clipping in meters
```

If `--scene` is omitted, the client uses the first `.r3d` file in `data/r3d/`.
