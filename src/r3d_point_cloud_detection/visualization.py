from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go


def sample_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    max_points: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    idx = sample_indices(len(points), max_points=max_points, seed=seed)
    return points[idx], None if colors is None else colors[idx]


def sample_indices(n_points: int, max_points: int, seed: int = 42) -> np.ndarray:
    if n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def bbox_edges(points: np.ndarray) -> tuple[list[float | None], list[float | None], list[float | None]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=float,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for i, j in edges:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])
    return xs, ys, zs


def bbox_edge_points(points: np.ndarray, points_per_edge: int = 32) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=float,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    weights = np.linspace(0.0, 1.0, int(points_per_edge), dtype=float)[:, None]
    segments = [corners[i] * (1.0 - weights) + corners[j] * weights for i, j in edges]
    return np.vstack(segments)


def make_point_cloud_figure(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    boxes: Iterable[dict] | None = None,
    max_points: int = 35000,
    point_size: float = 2.5,
    height: int = 800,
    title: str = "Point cloud",
) -> go.Figure:
    points = np.asarray(points)
    colors = None if colors is None else np.asarray(colors)
    labels = None if labels is None else np.asarray(labels, dtype=object)
    if colors is not None and len(colors) != len(points):
        raise ValueError(f"colors length {len(colors)} does not match point count {len(points)}")
    if labels is not None and len(labels) != len(points):
        raise ValueError(f"labels length {len(labels)} does not match point count {len(points)}")

    idx = sample_indices(len(points), max_points=max_points)
    sampled_points = points[idx]
    sampled_colors = None if colors is None else colors[idx]
    sampled_labels = None if labels is None else labels[idx]
    if sampled_colors is None:
        marker = dict(size=point_size, color=sampled_points[:, 2], colorscale="Viridis", opacity=0.85)
    else:
        marker = dict(
            size=point_size,
            color=[f"rgb({r},{g},{b})" for r, g, b in sampled_colors.tolist()],
            opacity=0.85,
        )
    customdata = sampled_labels[:, None] if sampled_labels is not None else None
    hovertemplate = "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
    if sampled_labels is not None:
        hovertemplate = "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>instance=%{customdata[0]}<extra></extra>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=sampled_points[:, 0],
            y=sampled_points[:, 1],
            z=sampled_points[:, 2],
            mode="markers",
            marker=marker,
            name="points",
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )

    for box in boxes or []:
        box_points = np.asarray(box["points"])
        if len(box_points) == 0:
            continue
        xs, ys, zs = bbox_edges(box_points)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=box.get("color", "red"), width=max(point_size * 2.4, 5.0)),
                name=str(box.get("label", "bbox")),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def save_plotly_html(fig: go.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path
