"""
Segmentation scoring engine.

Computes mIoU (Mean Intersection over Union) for image segmentation tasks.

CSV format is dynamically detected from the Ground Truth file.
Ground Truth typically has: [id_col], [class_col], [rle_mask_col], height, width
Prediction must match: [id_col], [class_col], [rle_mask_col]
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from .base import BaseScoringEngine, ScoringResult


def rle_decode(rle_string: str, height: int, width: int) -> np.ndarray:
    """
    Decode Run-Length Encoding (RLE) string to binary mask.
    """
    if pd.isna(rle_string) or rle_string.strip() == "" or rle_string.strip() == "0":
        return np.zeros((height, width), dtype=np.uint8)

    try:
        rle_pairs = list(map(int, rle_string.split()))
    except ValueError:
        return np.zeros((height, width), dtype=np.uint8)

    if len(rle_pairs) % 2 != 0:
        return np.zeros((height, width), dtype=np.uint8)

    mask_flat = np.zeros(height * width, dtype=np.uint8)

    for i in range(0, len(rle_pairs), 2):
        start = rle_pairs[i] - 1
        length = rle_pairs[i + 1]

        if start >= 0 and start + length <= height * width:
            mask_flat[start : start + length] = 1

    return mask_flat.reshape((height, width))


def rle_encode(mask: np.ndarray) -> str:
    """Encode binary mask to Run-Length Encoding (RLE) string."""
    flat = mask.flatten()
    runs = []
    i = 0
    while i < len(flat):
        if flat[i] == 1:
            start = i + 1
            length = 0
            while i < len(flat) and flat[i] == 1:
                length += 1
                i += 1
            runs.extend([start, length])
        else:
            i += 1

    return " ".join(map(str, runs)) if runs else ""


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


class SegmentationScoringEngine(BaseScoringEngine):
    """
    Scoring engine for image segmentation tasks.

    Supports metrics:
    - MIOU: Mean Intersection over Union across all classes

    Column names are auto-detected from the Ground Truth file.
    """

    REQUIRED_COLUMNS = []  # Set dynamically

    def __init__(self, ground_truth_path, metric_type: str = "MIOU"):
        super().__init__(ground_truth_path)
        self.metric_type = metric_type
        self.image_dimensions = {}
        # Column names (auto-detected)
        self.id_col = None
        self.class_col = None
        self.rle_col = None
        self.height_col = None
        self.width_col = None

    def load_ground_truth(self) -> bool:
        """Load ground truth and auto-detect column names."""
        if not super().load_ground_truth():
            return False

        columns = list(self.ground_truth_df.columns)

        if len(columns) < 5:
            self.log(
                "Segmentation ground truth must have at least 5 columns: id, class, rle_mask, height, width",
                "ERROR",
            )
            return False

        # Auto-detect columns
        self.id_col = columns[0]
        self.class_col = columns[1]
        self.rle_col = columns[2]
        self.height_col = columns[3]
        self.width_col = columns[4]

        self.log(
            f"Auto-detected columns: ID='{self.id_col}', Class='{self.class_col}', RLE='{self.rle_col}', H='{self.height_col}', W='{self.width_col}'"
        )

        # Prediction needs id, class, rle
        self.REQUIRED_COLUMNS = [self.id_col, self.class_col, self.rle_col]

        # Cache image dimensions
        for _, row in self.ground_truth_df.iterrows():
            self.image_dimensions[row[self.id_col]] = (
                int(row[self.height_col]),
                int(row[self.width_col]),
            )

        return True

    def validate_prediction_format(
        self, prediction_df: pd.DataFrame
    ) -> tuple[bool, str]:
        """Validate prediction CSV format."""
        missing_columns = set(self.REQUIRED_COLUMNS) - set(prediction_df.columns)
        if missing_columns:
            return (
                False,
                f"Missing required columns: {missing_columns}. Expected: {self.REQUIRED_COLUMNS}",
            )

        return True, ""

    def calculate_score(
        self, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame
    ) -> ScoringResult:
        """Calculate segmentation mIoU score."""
        all_classes = set(ground_truth_df[self.class_col].unique())
        self.log(f"Number of classes: {len(all_classes)}")

        class_ious = {}

        for class_label in all_classes:
            class_gt = ground_truth_df[ground_truth_df[self.class_col] == class_label]
            class_pred = prediction_df[prediction_df[self.class_col] == class_label]

            pred_lookup = {}
            for _, row in class_pred.iterrows():
                key = (row[self.id_col], row[self.class_col])
                pred_lookup[key] = row[self.rle_col]

            ious = []

            for _, gt_row in class_gt.iterrows():
                image_id = gt_row[self.id_col]

                if image_id not in self.image_dimensions:
                    self.log(f"Warning: No dimensions for image {image_id}", "WARNING")
                    continue

                height, width = self.image_dimensions[image_id]
                gt_mask = rle_decode(gt_row[self.rle_col], height, width)

                key = (image_id, class_label)
                if key in pred_lookup:
                    pred_mask = rle_decode(pred_lookup[key], height, width)
                else:
                    pred_mask = np.zeros((height, width), dtype=np.uint8)

                iou = calculate_mask_iou(gt_mask, pred_mask)
                ious.append(iou)

            class_ious[class_label] = np.mean(ious) if ious else 0.0

        mIoU = np.mean(list(class_ious.values())) if class_ious else 0.0
        self.log(f"mIoU: {mIoU:.4f}")

        class_ious_rounded = {k: round(v, 4) for k, v in class_ious.items()}

        return ScoringResult(
            success=True,
            score=round(mIoU, 6),
            metrics={
                "MIOU": round(mIoU, 6),
                "per_class_iou": class_ious_rounded,
                "num_classes": len(all_classes),
                "num_gt_masks": len(ground_truth_df),
                "num_pred_masks": len(prediction_df),
            },
            logs=self.logs,
        )
