"""Stage 3: SAM2-based segmentation → bbox (optional dependency).

Requires: pip install sam2 torch torchvision
Model checkpoint: https://github.com/facebookresearch/sam2

Usage:
    from annotation.stages.sam_segment import SAMSegmentStage
    stage = SAMSegmentStage(checkpoint="checkpoints/sam2_hiera_large.pt")
    ann = stage.annotate_single(image_path, pokemon_id=25)
"""

from __future__ import annotations

import logging
from pathlib import Path

from annotation.types import BBoxAnnotation, ImageAnnotation

logger = logging.getLogger(__name__)


class SAMSegmentStage:
    def __init__(
        self,
        checkpoint: str | Path,
        model_cfg: str = "sam2_hiera_large.yaml",
        *,
        device: str = "cuda",
        min_mask_area: int = 500,
        confidence_threshold: float = 0.7,
    ) -> None:
        self._min_area = min_mask_area
        self._conf_thresh = confidence_threshold
        self._predictor = self._load_predictor(checkpoint, model_cfg, device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def annotate_single(
        self,
        img_path: Path,
        pokemon_id: int,
        *,
        hint_bbox: tuple[int, int, int, int] | None = None,
    ) -> ImageAnnotation | None:
        """Segment img_path and return ImageAnnotation.

        hint_bbox: (x1, y1, x2, y2) box prompt; uses automatic mode if None.
        """
        try:
            import numpy as np
            from PIL import Image
        except ImportError as e:
            raise RuntimeError("pillow / numpy are required") from e

        img_np = np.array(Image.open(img_path).convert("RGB"))
        H, W = img_np.shape[:2]

        self._predictor.set_image(img_np)

        if hint_bbox is not None:
            masks, scores, _ = self._predictor.predict(
                box=hint_bbox,
                multimask_output=True,
            )
        else:
            masks, scores, _ = self._predictor.predict(multimask_output=True)

        bboxes = []
        for mask, score in zip(masks, scores):
            if float(score) < self._conf_thresh:
                continue
            bbox = self._mask_to_bbox(mask)
            if bbox is None:
                continue
            bbox.pokemon_id = pokemon_id
            bbox.confidence = float(score)
            bboxes.append(bbox)

        if not bboxes:
            return None

        return ImageAnnotation(
            image_path=str(img_path),
            width=W,
            height=H,
            bboxes=bboxes,
            stage="sam",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_to_bbox(self, mask) -> BBoxAnnotation | None:
        import numpy as np

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None

        y1, y2 = int(np.where(rows)[0][[0, -1]])
        x1, x2 = int(np.where(cols)[0][[0, -1]])
        x2 += 1
        y2 += 1

        if (x2 - x1) * (y2 - y1) < self._min_area:
            return None

        return BBoxAnnotation(
            pokemon_id=-1,
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=0.0,
            source="sam",
        )

    @staticmethod
    def _load_predictor(checkpoint, model_cfg, device):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "SAM2 is not installed. Run: pip install sam2\n"
                "Download checkpoint from: "
                "https://github.com/facebookresearch/sam2#model-checkpoints"
            ) from e

        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        return SAM2ImagePredictor(sam2_model)
