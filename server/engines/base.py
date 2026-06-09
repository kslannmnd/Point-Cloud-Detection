from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..core import SceneData

class InferenceEngine(ABC):
    name = "base"

    def __init__(self, class_names: List[str] | None = None):
        self.class_names = class_names or ["chair", "table", "sofa"]

    @abstractmethod
    def predict(self, scene: SceneData) -> List[Dict[str, Any]]:
        raise NotImplementedError
