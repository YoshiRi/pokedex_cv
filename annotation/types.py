"""Shared annotation data types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class BBoxAnnotation:
    pokemon_id: int        # national Pokedex number == class_id
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    source: str = ""       # "alpha" | "composite" | "sam" | "feature_match"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_yolo(self, img_width: int, img_height: int) -> tuple[float, float, float, float]:
        """Return (cx, cy, w, h) normalized to [0, 1]."""
        cx = (self.x1 + self.x2) / 2 / img_width
        cy = (self.y1 + self.y2) / 2 / img_height
        w = self.width / img_width
        h = self.height / img_height
        return cx, cy, w, h


@dataclass
class ImageAnnotation:
    image_path: str        # str for JSON serialization
    width: int
    height: int
    bboxes: list[BBoxAnnotation]
    stage: str = ""        # primary stage that generated this record

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ImageAnnotation:
        bboxes = [BBoxAnnotation(**b) for b in d.pop("bboxes", [])]
        return cls(bboxes=bboxes, **d)


class AnnotationStore:
    """Read/write annotations.jsonl (one ImageAnnotation per line)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, ann: ImageAnnotation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ann.to_dict(), ensure_ascii=False) + "\n")

    def append_many(self, anns: list[ImageAnnotation]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for ann in anns:
                f.write(json.dumps(ann.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> list[ImageAnnotation]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(ImageAnnotation.from_dict(json.loads(line)))
        return records
