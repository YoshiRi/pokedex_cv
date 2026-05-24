"""Stage 4: template-based feature matching → bbox (planned).

Uses ORB/SIFT keypoint matching or LoFTR dense matching to locate
a known sprite template within a scene image.

Status: PLANNED — interface is defined but not yet fully implemented.
        Use alpha_bbox + composite_gen for training data generation first.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from annotation.schema import BBoxAnnotation, ImageAnnotation

logger = logging.getLogger(__name__)


class FeatureMatchStage:
    """Locate a sprite template in a scene image using keypoint matching.

    Workflow:
      1. Extract keypoints from template (sprite with alpha mask applied)
      2. Extract keypoints from scene
      3. Match descriptors (BF + ratio test)
      4. Estimate homography → project template corners → bbox
    """

    def __init__(
        self,
        *,
        method: str = "orb",        # "orb" | "sift"
        min_matches: int = 8,
        confidence_threshold: float = 0.5,
    ) -> None:
        if method == "sift":
            self._detector = cv2.SIFT_create()
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self._detector = cv2.ORB_create(nfeatures=2000)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._min_matches = min_matches
        self._conf_thresh = confidence_threshold

    def annotate_single(
        self,
        scene_path: Path,
        template_path: Path,
        pokemon_id: int,
    ) -> ImageAnnotation | None:
        scene_bgr = cv2.imread(str(scene_path))
        if scene_bgr is None:
            logger.warning("Cannot load scene: %s", scene_path)
            return None

        template_rgba = np.array(Image.open(template_path).convert("RGBA"))
        template_gray = cv2.cvtColor(template_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
        alpha_mask = template_rgba[:, :, 3]

        kp_t, des_t = self._detector.detectAndCompute(template_gray, alpha_mask.astype(np.uint8))
        kp_s, des_s = self._detector.detectAndCompute(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY), None)

        if des_t is None or des_s is None or len(kp_t) < self._min_matches:
            return None

        matches = self._matcher.knnMatch(des_t, des_s, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < self._min_matches:
            logger.debug(
                "feature_match: not enough matches (%d/%d) for %s",
                len(good), self._min_matches, scene_path.name,
            )
            return None

        src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H_mat is None:
            return None

        h_t, w_t = template_gray.shape
        corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H_mat).reshape(-1, 2)

        x1 = int(projected[:, 0].min())
        y1 = int(projected[:, 1].min())
        x2 = int(projected[:, 0].max())
        y2 = int(projected[:, 1].max())

        inlier_ratio = float(mask.sum()) / len(mask)
        confidence = inlier_ratio * min(1.0, len(good) / 20)

        if confidence < self._conf_thresh:
            return None

        H_scene, W_scene = scene_bgr.shape[:2]
        return ImageAnnotation(
            image_path=str(scene_path),
            width=W_scene,
            height=H_scene,
            bboxes=[
                BBoxAnnotation(
                    pokemon_id=pokemon_id,
                    x1=max(0, x1), y1=max(0, y1),
                    x2=min(W_scene, x2), y2=min(H_scene, y2),
                    confidence=confidence,
                    source="feature_match",
                )
            ],
            stage="feature_match",
        )
