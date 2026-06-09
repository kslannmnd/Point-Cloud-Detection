from __future__ import annotations


S3DIS_CLASSES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
]

TARGET_CLASS_NAMES = ("door", "table", "chair", "sofa", "bookcase", "board")
TARGET_CLASS_NAME_SET = frozenset(TARGET_CLASS_NAMES)
NON_TARGET_CLASS_NAMES = frozenset(name for name in S3DIS_CLASSES if name not in TARGET_CLASS_NAME_SET)
TARGET_CLASS_IDS = frozenset(S3DIS_CLASSES.index(name) for name in TARGET_CLASS_NAMES)


def is_target_class_name(class_name: str) -> bool:
    return str(class_name) in TARGET_CLASS_NAME_SET
