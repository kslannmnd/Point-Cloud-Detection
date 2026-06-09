from __future__ import annotations

from typing import List, Optional, Any
from pydantic import BaseModel, Field

class BBox(BaseModel):
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def cz(self) -> float:
        return (self.z_min + self.z_max) / 2.0

    @property
    def dx(self) -> float:
        return self.x_max - self.x_min

    @property
    def dy(self) -> float:
        return self.y_max - self.y_min

    @property
    def dz(self) -> float:
        return self.z_max - self.z_min

class DetectedObject(BaseModel):
    object_id: int
    object_name: str
    class_id: int
    class_name: str
    score: float
    num_points: int
    bbox: BBox
    source: str = "model"
    proposal_id: Optional[int] = None

class DetectResponse(BaseModel):
    mode: str
    n_points: int
    objects: List[DetectedObject]
    scene_name: Optional[str] = None
    input_format: str = "unknown"
    frame_index: Optional[int] = None
    frame_id: Optional[str] = None
    has_ground_truth: bool = False
    warnings: List[str] = Field(default_factory=list)
