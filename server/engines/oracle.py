from __future__ import annotations

from collections import Counter
from typing import Dict, Any, List

import numpy as np

from .base import InferenceEngine
from ..core import SceneData, quantile_bbox

class OracleEngine(InferenceEngine):
    name = "oracle"

    def predict(self, scene: SceneData) -> List[Dict[str, Any]]:
        if scene.instance is None or scene.segment is None:
            # No labels available: fall back to pseudo detections
            from .stub import StubEngine
            return StubEngine(self.class_names).predict(scene)

        inst = np.asarray(scene.instance, dtype=np.int32).reshape(-1)
        seg = np.asarray(scene.segment, dtype=np.int32).reshape(-1)
        coord = np.asarray(scene.coord, dtype=np.float32)

        objects: List[Dict[str, Any]] = []
        counters = Counter()

        unique_instances = [int(x) for x in np.unique(inst) if int(x) >= 0]
        for instance_id in unique_instances:
            mask = inst == instance_id
            if int(mask.sum()) < 20:
                continue

            labels = seg[mask]
            labels = labels[labels >= 0]
            if len(labels) == 0:
                class_id = 0
            else:
                class_id = int(Counter(labels.tolist()).most_common(1)[0][0])

            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"class_{class_id}"
            counters[class_name] += 1
            object_name = f"{class_name}_{counters[class_name]}"
            bbox = quantile_bbox(coord[mask])

            objects.append(
                {
                    "object_id": len(objects),
                    "object_name": object_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": 0.99,
                    "num_points": int(mask.sum()),
                    "bbox": bbox,
                    "source": "oracle",
                    "proposal_id": instance_id,
                }
            )

        # Stable ordering for UI
        objects.sort(key=lambda o: (o["class_id"], o["object_name"]))
        for i, obj in enumerate(objects):
            obj["object_id"] = i
        return objects
