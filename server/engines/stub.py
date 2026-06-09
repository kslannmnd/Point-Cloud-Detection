from __future__ import annotations

from collections import Counter
from typing import Dict, Any, List

import numpy as np

from .base import InferenceEngine
from ..core import SceneData, quantile_bbox, hash_scene

def _mini_kmeans(x: np.ndarray, k: int, iters: int = 15, seed: int = 42) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)

    # Farthest-point initialization.
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    idx0 = int(rng.integers(0, n))
    centers[0] = x[idx0]
    closest = np.sum((x - centers[0]) ** 2, axis=1)

    for i in range(1, k):
        idx = int(np.argmax(closest))
        centers[i] = x[idx]
        dist = np.sum((x - centers[i]) ** 2, axis=1)
        closest = np.minimum(closest, dist)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            pts = x[labels == j]
            if len(pts) > 0:
                centers[j] = pts.mean(axis=0)
            else:
                centers[j] = x[int(rng.integers(0, n))]
    return labels

class StubEngine(InferenceEngine):
    name = "stub"

    def predict(self, scene: SceneData) -> List[Dict[str, Any]]:
        coord = np.asarray(scene.coord, dtype=np.float32)
        color = np.asarray(scene.color, dtype=np.float32) if scene.color is not None else None
        normal = np.asarray(scene.normal, dtype=np.float32) if scene.normal is not None else None

        if len(coord) == 0:
            return []

        seed = hash_scene(coord)
        n_points = len(coord)
        # Heuristic number of boxes for demo mode.
        k = int(np.clip(round(np.sqrt(n_points / 3000.0)) + 1, 1, 5))
        k = min(k, max(1, n_points // 100))

        # Build feature vector from xyz + optional color + optional normal.
        feat = [coord]
        if color is not None and color.shape == coord.shape:
            feat.append(color / 255.0)
        if normal is not None and normal.shape == coord.shape:
            feat.append(normal)
        x = np.concatenate(feat, axis=1)

        labels = _mini_kmeans(x, k=k, iters=20, seed=seed)
        class_names = self.class_names or ["chair", "table", "sofa"]
        counts = Counter()

        objects: List[Dict[str, Any]] = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            num_points = int(mask.sum())
            if num_points < 60:
                continue

            pts = coord[mask]
            bbox = quantile_bbox(pts)
            class_id = int(cluster_id % max(len(class_names), 1))
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            counts[class_name] += 1
            object_name = f"{class_name}_{counts[class_name]}"
            # Higher score for more compact clusters and more points.
            spread = np.std(pts, axis=0).mean()
            score = float(np.clip(0.95 - 0.15 * spread + 0.02 * np.log1p(num_points), 0.35, 0.93))

            objects.append(
                {
                    "object_id": len(objects),
                    "object_name": object_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "num_points": num_points,
                    "bbox": bbox,
                    "source": "stub",
                    "proposal_id": int(cluster_id),
                }
            )

        objects.sort(key=lambda o: (-o["score"], -o["num_points"]))
        for i, obj in enumerate(objects):
            obj["object_id"] = i
        return objects
