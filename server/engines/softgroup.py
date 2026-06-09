from __future__ import annotations

from typing import List, Dict, Any
import importlib
import os

from .base import InferenceEngine
from ..core import SceneData

class SoftGroupEngine(InferenceEngine):
    """
    Thin adapter for the production SoftGroup/Pointcept pipeline.

    To use it, provide a Python module path in MODEL_ADAPTER_MODULE that exports:
        predict(scene: SceneData, class_names: list[str]) -> list[dict]
    and returns objects in the same schema as the stub/oracle engines.
    """
    name = "model"

    def __init__(self, class_names: List[str] | None = None):
        super().__init__(class_names=class_names)
        self.module_name = os.getenv("MODEL_ADAPTER_MODULE", "").strip()
        self._module = None
        if self.module_name:
            self._module = importlib.import_module(self.module_name)

    def predict(self, scene: SceneData) -> List[Dict[str, Any]]:
        if self._module is None:
            raise RuntimeError(
                "Model mode is enabled but MODEL_ADAPTER_MODULE is not set. "
                "Use ENGINE_MODE=oracle or ENGINE_MODE=stub for the demo."
            )
        if not hasattr(self._module, "predict"):
            raise RuntimeError(f"{self.module_name} must define predict(scene, class_names).")
        return self._module.predict(scene, self.class_names)
