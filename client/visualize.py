from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PALETTE = px.colors.qualitative.Alphabet + px.colors.qualitative.Safe + px.colors.qualitative.Bold

def class_color(class_id: int) -> str:
    if class_id < 0:
        return "lightgray"
    return PALETTE[class_id % len(PALETTE)]

def box_wireframe(bbox: Dict[str, float]):
    x0, y0, z0 = bbox["x_min"], bbox["y_min"], bbox["z_min"]
    x1, y1, z1 = bbox["x_max"], bbox["y_max"], bbox["z_max"]
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]
    return xs, ys, zs

def visualize_scene(
    coord: np.ndarray,
    objects: List[Dict[str, Any]],
    color: Optional[np.ndarray] = None,
    title: str = "Point cloud + predicted boxes",
    max_points: int = 50000,
):
    coord = np.asarray(coord, dtype=np.float32).reshape(-1, 3)
    n = len(coord)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        vis_coord = coord[idx]
        vis_color = None if color is None else np.asarray(color)[idx]
    else:
        vis_coord = coord
        vis_color = None if color is None else np.asarray(color)

    fig = go.Figure()

    if vis_color is not None and np.asarray(vis_color).shape == vis_coord.shape:
        c = np.clip(np.asarray(vis_color, dtype=np.float32), 0, 255)
        fig.add_trace(
            go.Scatter3d(
                x=vis_coord[:, 0],
                y=vis_coord[:, 1],
                z=vis_coord[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=[f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in c], opacity=0.85),
                name="points",
                hoverinfo="skip",
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=vis_coord[:, 0],
                y=vis_coord[:, 1],
                z=vis_coord[:, 2],
                mode="markers",
                marker=dict(size=1.5, color="lightgray", opacity=0.45),
                name="points",
                hoverinfo="skip",
            )
        )

    for obj in objects:
        bbox = obj["bbox"]
        xs, ys, zs = box_wireframe(bbox)
        color = class_color(int(obj["class_id"]))
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=6),
                name=f'{obj["object_name"]} ({obj["score"]:.2f})',
                hovertemplate=(
                    f'object={obj["object_name"]}<br>'
                    f'class={obj["class_name"]}<br>'
                    f'score={obj["score"]:.3f}<br>'
                    f'points={obj["num_points"]}'
                    "<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[(bbox["x_min"] + bbox["x_max"]) / 2.0],
                y=[(bbox["y_min"] + bbox["y_max"]) / 2.0],
                z=[(bbox["z_min"] + bbox["z_max"]) / 2.0],
                mode="text",
                text=[obj["object_name"]],
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        width=1100,
        height=850,
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
