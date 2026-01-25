"""
Detection scoring engine.

Computes mAP (Mean Average Precision) for object detection tasks.

CSV format is dynamically detected from the Ground Truth file.
Ground Truth typically has: [id_col], [class_col], [xmin], [ymin], [xmax], [ymax]
Prediction must add: confidence column
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from .base import BaseScoringEngine, ScoringResult


def calculate_iou(box1: tuple, box2: tuple) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: (xmin, ymin, xmax, ymax)
        box2: (xmin, ymin, xmax, ymax)

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_ap(precision: list, recall: list) -> float:
    """
    Calculate Average Precision using the 11-point interpolation method.
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    precision = [0.0] + list(precision) + [0.0]
    recall = [0.0] + list(recall) + [1.0]

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = 0.0
        for r, pr in zip(recall, precision):
            if r >= t:
                p = max(p, pr)
        ap += p / 11

    return ap


class DetectionScoringEngine(BaseScoringEngine):
    """
    Scoring engine for object detection tasks.

    Supports metrics:
    - MAP: mAP@0.5 (IoU threshold = 0.5)
    - MAP_50_95: mAP@[0.5:0.95] (average over IoU thresholds)

    Column names are auto-detected from the Ground Truth file.
    """

    REQUIRED_COLUMNS = []  # Set dynamically

    def __init__(self, ground_truth_path, metric_type: str = "MAP"):
        super().__init__(ground_truth_path)
        self.metric_type = metric_type
        # Column names (auto-detected)
        self.id_col = None
        self.class_col = None
        self.xmin_col = None
        self.ymin_col = None
        self.xmax_col = None
        self.ymax_col = None

    def load_ground_truth(self) -> bool:
        """Load ground truth and auto-detect column names."""
        if not super().load_ground_truth():
            return False

        columns = list(self.ground_truth_df.columns)

        if len(columns) < 6:
            self.log(
                "Detection ground truth must have at least 6 columns: id, class, xmin, ymin, xmax, ymax",
                "ERROR",
            )
            return False

        # Auto-detect: first col = id, second = class, then bbox coords
        self.id_col = columns[0]
        self.class_col = columns[1]
        self.xmin_col = columns[2]
        self.ymin_col = columns[3]
        self.xmax_col = columns[4]
        self.ymax_col = columns[5]

        self.log(
            f"Auto-detected columns: ID='{self.id_col}', Class='{self.class_col}', BBox=[{self.xmin_col}, {self.ymin_col}, {self.xmax_col}, {self.ymax_col}]"
        )

        # Prediction needs same columns + confidence
        self.REQUIRED_COLUMNS = [
            self.id_col,
            self.class_col,
            "confidence",
            self.xmin_col,
            self.ymin_col,
            self.xmax_col,
            self.ymax_col,
        ]

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

        # Check for valid bounding box values
        for col in [
            self.xmin_col,
            self.ymin_col,
            self.xmax_col,
            self.ymax_col,
            "confidence",
        ]:
            if not pd.api.types.is_numeric_dtype(prediction_df[col]):
                return False, f"Column '{col}' must be numeric"

        # Check confidence range
        if (prediction_df["confidence"] < 0).any() or (
            prediction_df["confidence"] > 1
        ).any():
            return False, "Confidence values must be between 0 and 1"

        return True, ""

    def _calculate_ap_per_class(
        self,
        pred_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        class_label: str,
        iou_threshold: float = 0.5,
    ) -> float:
        """Calculate AP for a single class at a given IoU threshold."""
        class_preds = pred_df[pred_df[self.class_col] == class_label].copy()
        class_gts = gt_df[gt_df[self.class_col] == class_label].copy()

        if len(class_gts) == 0 or len(class_preds) == 0:
            return 0.0

        class_preds = class_preds.sort_values("confidence", ascending=False)

        gt_matched = defaultdict(lambda: [False] * 1000)
        gt_per_image = class_gts.groupby(self.id_col)

        gt_boxes = {}
        for image_id, group in gt_per_image:
            gt_boxes[image_id] = [
                (
                    row[self.xmin_col],
                    row[self.ymin_col],
                    row[self.xmax_col],
                    row[self.ymax_col],
                )
                for _, row in group.iterrows()
            ]
            gt_matched[image_id] = [False] * len(gt_boxes[image_id])

        tp = []
        fp = []

        for _, pred in class_preds.iterrows():
            image_id = pred[self.id_col]
            pred_box = (
                pred[self.xmin_col],
                pred[self.ymin_col],
                pred[self.xmax_col],
                pred[self.ymax_col],
            )

            if image_id not in gt_boxes:
                fp.append(1)
                tp.append(0)
                continue

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes[image_id]):
                if gt_matched[image_id][gt_idx]:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                gt_matched[image_id][best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        total_gt = len(class_gts)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt

        return calculate_ap(precision, recall)

    def calculate_score(
        self, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame
    ) -> ScoringResult:
        """Calculate detection mAP score."""
        all_classes = set(ground_truth_df[self.class_col].unique())
        pred_classes = set(prediction_df[self.class_col].unique())

        self.log(f"GT classes: {len(all_classes)}, Pred classes: {len(pred_classes)}")

        if self.metric_type == "MAP_50_95":
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
            all_aps = []

            for iou_thresh in iou_thresholds:
                class_aps = []
                for class_label in all_classes:
                    ap = self._calculate_ap_per_class(
                        prediction_df, ground_truth_df, class_label, iou_thresh
                    )
                    class_aps.append(ap)
                all_aps.append(np.mean(class_aps) if class_aps else 0.0)

            mAP = np.mean(all_aps)
            self.log(f"mAP@[0.5:0.95]: {mAP:.4f}")

            return ScoringResult(
                success=True,
                score=round(mAP, 6),
                metrics={
                    "mAP_50_95": round(mAP, 6),
                    "num_classes": len(all_classes),
                    "num_predictions": len(prediction_df),
                    "num_ground_truth": len(ground_truth_df),
                },
                logs=self.logs,
            )
        else:
            class_aps = {}
            for class_label in all_classes:
                ap = self._calculate_ap_per_class(
                    prediction_df, ground_truth_df, class_label, iou_threshold=0.5
                )
                class_aps[class_label] = round(ap, 4)

            mAP = np.mean(list(class_aps.values())) if class_aps else 0.0
            self.log(f"mAP@0.5: {mAP:.4f}")

            return ScoringResult(
                success=True,
                score=round(mAP, 6),
                metrics={
                    "mAP_50": round(mAP, 6),
                    "per_class_ap": class_aps,
                    "num_classes": len(all_classes),
                    "num_predictions": len(prediction_df),
                    "num_ground_truth": len(ground_truth_df),
                },
                logs=self.logs,
            )
