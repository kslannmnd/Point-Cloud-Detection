from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import base64
import html
import io
import json

import numpy as np
from PIL import Image

from .r3d import (
    cam_matrix_to_depth,
    depth_to_point_cloud,
    load_conf_frame,
    load_depth_frame,
    load_rgb_frame,
    scan_r3d_file,
    select_cam_matrix,
)


@dataclass
class R3DAnnotationAppConfig:
    r3d_path: str | Path
    frame_index: int = 0
    output_path: str | Path | None = None
    confidence_threshold: int = 1
    z_min: float | None = None
    z_max: float | None = None
    sample_step: int = 1
    max_points_plot: int = 35000
    point_size: float = 2.5
    use_rgb: bool = True
    seed: int = 42


def _sample_for_plot(
    points: np.ndarray,
    colors: np.ndarray | None,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[idx], None if colors is None else colors[idx]


def _image_to_data_url(image: np.ndarray, image_format: str = "JPEG") -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def _depth_to_data_url(depth: np.ndarray) -> str:
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if len(finite) == 0:
        preview = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.quantile(finite, [0.02, 0.98])
        if hi <= lo:
            hi = lo + 1.0
        normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        preview = (255 - normalized * 255).astype(np.uint8)
        preview[~np.isfinite(depth) | (depth <= 0)] = 0
    return _image_to_data_url(preview, image_format="PNG")


def _float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def build_r3d_annotation_payload(config: R3DAnnotationAppConfig) -> dict:
    r3d_path = Path(config.r3d_path)
    zf, metadata, frame_map, frame_ids = scan_r3d_file(r3d_path)
    try:
        if not 0 <= config.frame_index < len(frame_ids):
            raise IndexError(f"frame_index must be between 0 and {len(frame_ids) - 1}, got {config.frame_index}")

        depth_shape = (int(metadata["dh"]), int(metadata["dw"]))
        rgb_shape = (int(metadata["h"]), int(metadata["w"]))
        rgb_cam_matrix = select_cam_matrix(metadata["K"], rgb_w=rgb_shape[1], rgb_h=rgb_shape[0])
        depth_cam_matrix = cam_matrix_to_depth(rgb_cam_matrix, rgb_shape, depth_shape)

        frame_id = frame_ids[config.frame_index]
        members = frame_map[frame_id]
        rgb_name = members.get(".jpg") or members.get(".jpeg")
        rgb = load_rgb_frame(zf, rgb_name) if rgb_name else None
        depth = load_depth_frame(zf, members[".depth"], depth_shape)
        confidence = load_conf_frame(zf, members[".conf"], depth_shape) if ".conf" in members else None
        points, colors = depth_to_point_cloud(
            depth=depth,
            depth_cam_matrix=depth_cam_matrix,
            rgb=rgb if config.use_rgb else None,
            confidence=confidence,
            confidence_threshold=config.confidence_threshold,
            z_min=config.z_min,
            z_max=config.z_max,
            sample_step=config.sample_step,
        )
        plot_points, plot_colors = _sample_for_plot(points, colors, config.max_points_plot, config.seed)
        if len(points) == 0:
            bounds = {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}
        else:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            bounds = {
                "x": [float(mins[0]), float(maxs[0])],
                "y": [float(mins[1]), float(maxs[1])],
                "z": [float(mins[2]), float(maxs[2])],
            }

        if plot_colors is None:
            color_values: list[str | float] = [float(value) for value in plot_points[:, 2]]
            color_mode = "depth"
        else:
            color_values = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in plot_colors.tolist()]
            color_mode = "rgb"

        return {
            "source": {
                "r3d_path": str(r3d_path),
                "r3d_name": r3d_path.name,
                "frame_index": int(config.frame_index),
                "frame_id": str(frame_id),
                "frame_count": int(len(frame_ids)),
            },
            "filters": {
                "confidence_threshold": int(config.confidence_threshold),
                "z_min": _float_or_none(config.z_min),
                "z_max": _float_or_none(config.z_max),
                "sample_step": int(config.sample_step),
                "use_rgb": bool(config.use_rgb),
                "max_points_plot": int(config.max_points_plot),
                "point_size": float(config.point_size),
            },
            "frame": {
                "rgb_shape": list(rgb.shape[:2]) if rgb is not None else None,
                "depth_shape": list(depth.shape),
                "depth_camera_matrix": depth_cam_matrix.astype(float).tolist(),
                "point_count": int(len(points)),
                "displayed_point_count": int(len(plot_points)),
                "bounds": bounds,
                "rgb_data_url": _image_to_data_url(rgb) if rgb is not None else None,
                "depth_data_url": _depth_to_data_url(depth),
            },
            "points": {
                "x": [float(value) for value in plot_points[:, 0]],
                "y": [float(value) for value in plot_points[:, 1]],
                "z": [float(value) for value in plot_points[:, 2]],
                "color": color_values,
                "color_mode": color_mode,
            },
            "annotations": [],
        }
    finally:
        zf.close()


def _json_for_script(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _default_output_path(config: R3DAnnotationAppConfig, payload: dict) -> Path:
    if config.output_path is not None:
        return Path(config.output_path)
    stem = Path(config.r3d_path).stem
    frame_id = payload["source"]["frame_id"]
    return Path("outputs") / "annotations" / f"{stem}_frame_{frame_id}_annotator.html"


def _annotation_filename(payload: dict) -> str:
    stem = Path(payload["source"]["r3d_name"]).stem
    frame_id = payload["source"]["frame_id"]
    return f"{stem}_frame_{frame_id}_annotations.json"


def render_r3d_annotation_html(payload: dict) -> str:
    title = f"{payload['source']['r3d_name']} frame {payload['source']['frame_id']}"
    safe_title = html.escape(title)
    data_json = _json_for_script(payload)
    download_name = _annotation_filename(payload)
    point_size = float(payload["filters"]["point_size"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  color-scheme: light;
  --bg: #f5f7fb;
  --panel: #ffffff;
  --line: #d9dee8;
  --text: #1d2530;
  --muted: #667085;
  --accent: #0f766e;
  --danger: #b42318;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, Segoe UI, Arial, sans-serif;
}}
header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 14px 18px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}}
h1 {{
  margin: 0;
  font-size: 18px;
  font-weight: 650;
}}
.meta {{
  color: var(--muted);
  font-size: 13px;
}}
main {{
  display: grid;
  grid-template-columns: minmax(520px, 1fr) 380px;
  min-height: calc(100vh - 58px);
}}
#plot {{
  width: 100%;
  height: calc(100vh - 58px);
}}
aside {{
  border-left: 1px solid var(--line);
  background: var(--panel);
  overflow: auto;
  height: calc(100vh - 58px);
}}
section {{
  padding: 14px;
  border-bottom: 1px solid var(--line);
}}
h2 {{
  margin: 0 0 10px;
  font-size: 14px;
  font-weight: 650;
}}
.preview-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}}
.preview {{
  display: grid;
  gap: 6px;
  color: var(--muted);
  font-size: 12px;
}}
.preview img {{
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: contain;
  border: 1px solid var(--line);
  background: #101828;
}}
label {{
  display: grid;
  gap: 5px;
  color: var(--muted);
  font-size: 12px;
}}
input, select, button {{
  min-height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  font: inherit;
  background: #fff;
  color: var(--text);
}}
input[type="color"] {{
  padding: 2px;
}}
.grid-2 {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}}
.grid-3 {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 8px;
}}
.stack {{
  display: grid;
  gap: 10px;
}}
.actions {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}}
button.primary {{
  border-color: var(--accent);
  background: var(--accent);
  color: #fff;
}}
button.danger {{
  border-color: var(--danger);
  color: var(--danger);
}}
button:disabled {{
  opacity: 0.5;
}}
.annotation-list {{
  display: grid;
  gap: 8px;
}}
.annotation-item {{
  width: 100%;
  text-align: left;
  background: #f8fafc;
}}
.annotation-item.active {{
  outline: 2px solid var(--accent);
  outline-offset: 0;
}}
pre {{
  margin: 0;
  max-height: 260px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  background: #101828;
  color: #e6edf7;
  font-size: 12px;
}}
.empty {{
  color: var(--muted);
  font-size: 13px;
}}
@media (max-width: 980px) {{
  main {{
    grid-template-columns: 1fr;
  }}
  #plot, aside {{
    height: auto;
    min-height: 520px;
  }}
  aside {{
    border-left: 0;
    border-top: 1px solid var(--line);
  }}
}}
</style>
</head>
<body>
<header>
  <div>
    <h1>{safe_title}</h1>
    <div class="meta" id="summary"></div>
  </div>
  <div class="actions">
    <button id="downloadTop" class="primary">Download JSON</button>
  </div>
</header>
<main>
  <div id="plot"></div>
  <aside>
    <section id="previews"></section>
    <section>
      <h2>Annotation</h2>
      <div class="stack">
        <div class="grid-2">
          <label>Label<input id="label" value="object"></label>
          <label>Instance<input id="instance" value="object_1"></label>
        </div>
        <div class="grid-2">
          <label>Color<input id="color" type="color" value="#f97316"></label>
          <label>Selected<select id="selected"></select></label>
        </div>
        <div class="grid-2">
          <label>X min<input id="xMin" type="number" step="0.001"></label>
          <label>X max<input id="xMax" type="number" step="0.001"></label>
          <label>Y min<input id="yMin" type="number" step="0.001"></label>
          <label>Y max<input id="yMax" type="number" step="0.001"></label>
          <label>Z min<input id="zMin" type="number" step="0.001"></label>
          <label>Z max<input id="zMax" type="number" step="0.001"></label>
        </div>
        <div class="actions">
          <button id="fitCloud">Fit Cloud</button>
          <button id="newBox">New</button>
          <button id="saveBox" class="primary">Add</button>
          <button id="deleteBox" class="danger">Delete</button>
        </div>
      </div>
    </section>
    <section>
      <h2>Annotations</h2>
      <div id="annotationList" class="annotation-list"></div>
    </section>
    <section>
      <h2>JSON</h2>
      <div class="stack">
        <div class="actions">
          <button id="downloadJson" class="primary">Download JSON</button>
          <label>Load JSON<input id="loadJson" type="file" accept="application/json"></label>
        </div>
        <pre id="jsonPreview"></pre>
      </div>
    </section>
  </aside>
</main>
<script type="application/json" id="frame-data">{data_json}</script>
<script>
const frame = JSON.parse(document.getElementById("frame-data").textContent);
const state = {{ annotations: [], selectedId: null }};
const downloadName = "{download_name}";
const fields = {{
  label: document.getElementById("label"),
  instance: document.getElementById("instance"),
  color: document.getElementById("color"),
  selected: document.getElementById("selected"),
  xMin: document.getElementById("xMin"),
  xMax: document.getElementById("xMax"),
  yMin: document.getElementById("yMin"),
  yMax: document.getElementById("yMax"),
  zMin: document.getElementById("zMin"),
  zMax: document.getElementById("zMax")
}};
function fixed(value) {{
  return Number(value).toFixed(4);
}}
function boundsAnnotation() {{
  const b = frame.frame.bounds;
  return {{
    bbox: {{
      x_min: b.x[0], x_max: b.x[1],
      y_min: b.y[0], y_max: b.y[1],
      z_min: b.z[0], z_max: b.z[1]
    }}
  }};
}}
function setInputs(annotation) {{
  const box = annotation.bbox;
  fields.label.value = annotation.label || "object";
  fields.instance.value = annotation.instance_id || "object_1";
  fields.color.value = annotation.color || "#f97316";
  fields.xMin.value = fixed(box.x_min);
  fields.xMax.value = fixed(box.x_max);
  fields.yMin.value = fixed(box.y_min);
  fields.yMax.value = fixed(box.y_max);
  fields.zMin.value = fixed(box.z_min);
  fields.zMax.value = fixed(box.z_max);
}}
function readInputs() {{
  return {{
    label: fields.label.value.trim() || "object",
    instance_id: fields.instance.value.trim() || "object_1",
    color: fields.color.value,
    bbox: {{
      x_min: Number(fields.xMin.value),
      x_max: Number(fields.xMax.value),
      y_min: Number(fields.yMin.value),
      y_max: Number(fields.yMax.value),
      z_min: Number(fields.zMin.value),
      z_max: Number(fields.zMax.value)
    }}
  }};
}}
function normalizeBox(annotation) {{
  const box = annotation.bbox;
  const pairs = [["x_min", "x_max"], ["y_min", "y_max"], ["z_min", "z_max"]];
  for (const [minKey, maxKey] of pairs) {{
    if (box[minKey] > box[maxKey]) {{
      const tmp = box[minKey];
      box[minKey] = box[maxKey];
      box[maxKey] = tmp;
    }}
  }}
  return annotation;
}}
function bboxTrace(annotation) {{
  const b = annotation.bbox;
  const corners = [
    [b.x_min, b.y_min, b.z_min], [b.x_max, b.y_min, b.z_min],
    [b.x_max, b.y_max, b.z_min], [b.x_min, b.y_max, b.z_min],
    [b.x_min, b.y_min, b.z_max], [b.x_max, b.y_min, b.z_max],
    [b.x_max, b.y_max, b.z_max], [b.x_min, b.y_max, b.z_max]
  ];
  const edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]];
  const x = [], y = [], z = [];
  for (const [a, bIdx] of edges) {{
    x.push(corners[a][0], corners[bIdx][0], null);
    y.push(corners[a][1], corners[bIdx][1], null);
    z.push(corners[a][2], corners[bIdx][2], null);
  }}
  return {{
    type: "scatter3d",
    mode: "lines",
    x, y, z,
    name: annotation.instance_id,
    line: {{ color: annotation.color, width: 7 }},
    hoverinfo: "skip"
  }};
}}
function plotData() {{
  const marker = frame.points.color_mode === "rgb"
    ? {{ size: {point_size}, color: frame.points.color, opacity: 0.86 }}
    : {{ size: {point_size}, color: frame.points.color, colorscale: "Viridis", opacity: 0.86, colorbar: {{ title: "Z" }} }};
  return [{{
    type: "scatter3d",
    mode: "markers",
    x: frame.points.x,
    y: frame.points.y,
    z: frame.points.z,
    name: "points",
    marker,
    hovertemplate: "x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>"
  }}, ...state.annotations.map(bboxTrace)];
}}
function renderPlot() {{
  Plotly.react("plot", plotData(), {{
    scene: {{
      xaxis: {{ title: "X, m" }},
      yaxis: {{ title: "Y, m" }},
      zaxis: {{ title: "Z, m" }},
      aspectmode: "data",
      dragmode: "orbit"
    }},
    margin: {{ l: 0, r: 0, t: 0, b: 0 }},
    showlegend: true
  }}, {{ scrollZoom: true, responsive: true }});
}}
function exportPayload() {{
  return {{
    source: frame.source,
    filters: frame.filters,
    frame: {{
      rgb_shape: frame.frame.rgb_shape,
      depth_shape: frame.frame.depth_shape,
      depth_camera_matrix: frame.frame.depth_camera_matrix,
      point_count: frame.frame.point_count
    }},
    annotation_type: "axis_aligned_3d_bbox",
    annotations: state.annotations.map((item, index) => ({{ ...item, index }}))
  }};
}}
function refreshJson() {{
  document.getElementById("jsonPreview").textContent = JSON.stringify(exportPayload(), null, 2);
}}
function refreshSelection() {{
  fields.selected.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "None";
  fields.selected.appendChild(empty);
  for (const annotation of state.annotations) {{
    const option = document.createElement("option");
    option.value = annotation.id;
    option.textContent = `${{annotation.instance_id}} (${{annotation.label}})`;
    fields.selected.appendChild(option);
  }}
  fields.selected.value = state.selectedId || "";
}}
function refreshList() {{
  const list = document.getElementById("annotationList");
  list.innerHTML = "";
  if (!state.annotations.length) {{
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No annotations yet.";
    list.appendChild(empty);
    return;
  }}
  for (const annotation of state.annotations) {{
    const button = document.createElement("button");
    button.className = "annotation-item" + (annotation.id === state.selectedId ? " active" : "");
    button.textContent = `${{annotation.instance_id}}: ${{annotation.label}}`;
    button.addEventListener("click", () => selectAnnotation(annotation.id));
    list.appendChild(button);
  }}
}}
function refresh() {{
  refreshSelection();
  refreshList();
  refreshJson();
  renderPlot();
}}
function selectAnnotation(id) {{
  state.selectedId = id || null;
  const annotation = state.annotations.find(item => item.id === state.selectedId);
  if (annotation) setInputs(annotation);
  refreshSelection();
  refreshList();
}}
function saveCurrent() {{
  const annotation = normalizeBox(readInputs());
  if ([annotation.bbox.x_min, annotation.bbox.x_max, annotation.bbox.y_min, annotation.bbox.y_max, annotation.bbox.z_min, annotation.bbox.z_max].some(Number.isNaN)) {{
    alert("BBox coordinates must be valid numbers.");
    return;
  }}
  if (state.selectedId) {{
    const index = state.annotations.findIndex(item => item.id === state.selectedId);
    if (index >= 0) state.annotations[index] = {{ ...state.annotations[index], ...annotation }};
  }} else {{
    state.selectedId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
    state.annotations.push({{ id: state.selectedId, ...annotation }});
  }}
  refresh();
}}
function newBox() {{
  state.selectedId = null;
  const fit = boundsAnnotation();
  setInputs({{
    label: "object",
    instance_id: `object_${{state.annotations.length + 1}}`,
    color: "#f97316",
    bbox: fit.bbox
  }});
  refreshSelection();
  refreshList();
}}
function deleteBox() {{
  if (!state.selectedId) return;
  state.annotations = state.annotations.filter(item => item.id !== state.selectedId);
  state.selectedId = null;
  newBox();
  refresh();
}}
function downloadJson() {{
  const blob = new Blob([JSON.stringify(exportPayload(), null, 2)], {{ type: "application/json" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = downloadName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}}
function loadJson(file) {{
  const reader = new FileReader();
  reader.onload = () => {{
    const payload = JSON.parse(reader.result);
    state.annotations = (payload.annotations || []).map(item => ({{
      id: item.id || (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random())),
      label: item.label,
      instance_id: item.instance_id,
      color: item.color || "#f97316",
      bbox: item.bbox
    }}));
    state.selectedId = state.annotations[0]?.id || null;
    if (state.selectedId) selectAnnotation(state.selectedId);
    refresh();
  }};
  reader.readAsText(file);
}}
function renderPreviews() {{
  const root = document.getElementById("previews");
  const rgb = frame.frame.rgb_data_url
    ? `<div class="preview"><img src="${{frame.frame.rgb_data_url}}" alt="RGB frame"><span>RGB</span></div>`
    : "";
  root.innerHTML = `<h2>Frame preview</h2><div class="preview-grid">${{rgb}}<div class="preview"><img src="${{frame.frame.depth_data_url}}" alt="Depth preview"><span>Depth</span></div></div>`;
}}
document.getElementById("summary").textContent = `${{frame.frame.point_count}} points, ${{frame.frame.displayed_point_count}} displayed, frame ${{frame.source.frame_index + 1}} of ${{frame.source.frame_count}}`;
document.getElementById("fitCloud").addEventListener("click", () => setInputs({{ label: fields.label.value, instance_id: fields.instance.value, color: fields.color.value, bbox: boundsAnnotation().bbox }}));
document.getElementById("newBox").addEventListener("click", newBox);
document.getElementById("saveBox").addEventListener("click", saveCurrent);
document.getElementById("deleteBox").addEventListener("click", deleteBox);
document.getElementById("downloadJson").addEventListener("click", downloadJson);
document.getElementById("downloadTop").addEventListener("click", downloadJson);
fields.selected.addEventListener("change", event => selectAnnotation(event.target.value));
document.getElementById("loadJson").addEventListener("change", event => {{
  const file = event.target.files[0];
  if (file) loadJson(file);
}});
renderPreviews();
newBox();
refresh();
</script>
</body>
</html>
"""


def write_r3d_annotation_app(config: R3DAnnotationAppConfig) -> Path:
    payload = build_r3d_annotation_payload(config)
    output_path = _default_output_path(config, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_r3d_annotation_html(payload), encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> R3DAnnotationAppConfig:
    parser = argparse.ArgumentParser(description="Build a standalone R3D frame annotation HTML app.")
    parser.add_argument("r3d_path", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--confidence-threshold", type=int, default=1)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--max-points-plot", type=int, default=35000)
    parser.add_argument("--point-size", type=float, default=2.5)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    return R3DAnnotationAppConfig(
        r3d_path=args.r3d_path,
        frame_index=args.frame_index,
        output_path=args.output,
        confidence_threshold=args.confidence_threshold,
        z_min=args.z_min,
        z_max=args.z_max,
        sample_step=args.sample_step,
        max_points_plot=args.max_points_plot,
        point_size=args.point_size,
        use_rgb=not args.no_rgb,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> None:
    output_path = write_r3d_annotation_app(parse_args(argv))
    print(output_path)


if __name__ == "__main__":
    main()
