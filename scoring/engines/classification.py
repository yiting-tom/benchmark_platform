"""
Classification scoring engine.

Computes Accuracy and F1-Score for image classification tasks.

CSV format is dynamically detected from the Ground Truth file.
Typically: [id_column], [label_column]
Example: filename, label OR image_id, label
"""

import pandas as pd
from sklearn.metrics import classification_report

from .base import BaseScoringEngine, ScoringResult


class ClassificationScoringEngine(BaseScoringEngine):
    """
    Scoring engine for image classification tasks.

    Supports metrics:
    - ACCURACY: Overall accuracy
    - F1, PRECISION, RECALL (Macro, Micro, Weighted)
    - CLASS-specific F1, Precision, Recall

    Column names are auto-detected from the Ground Truth file.
    """

    REQUIRED_COLUMNS = []  # Will be set dynamically from ground truth

    def __init__(
        self,
        ground_truth_path,
        metric_type: str = "ACCURACY",
        metric_target_class: str = None,
    ):
        """
        Initialize the classification scorer.

        Args:
            ground_truth_path: Path to ground truth CSV.
            metric_type: One of the MetricType values.
            metric_target_class: Optional class name for Class-Specific metrics.
        """
        super().__init__(ground_truth_path)
        self.metric_type = metric_type
        self.metric_target_class = metric_target_class
        self.id_column = None
        self.label_column = None

    def load_ground_truth(self) -> bool:
        """Load ground truth and auto-detect column names."""
        if not super().load_ground_truth():
            return False

        # Auto-detect columns from ground truth
        columns = list(self.ground_truth_df.columns)

        if len(columns) < 2:
            self.log("Ground truth must have at least 2 columns", "ERROR")
            return False

        # First column is ID, second column is label
        self.id_column = columns[0]
        self.label_column = columns[1]

        self.log(
            f"Auto-detected columns: ID='{self.id_column}', Label='{self.label_column}'"
        )

        # Set required columns for prediction validation
        self.REQUIRED_COLUMNS = [self.id_column, self.label_column]

        return True

    def validate_prediction_format(
        self, prediction_df: pd.DataFrame
    ) -> tuple[bool, str]:
        """Validate prediction CSV format matches ground truth."""
        # Check required columns (auto-detected from ground truth)
        missing_columns = set(self.REQUIRED_COLUMNS) - set(prediction_df.columns)
        if missing_columns:
            gt_cols = ", ".join(self.REQUIRED_COLUMNS)
            return (
                False,
                f"Missing required columns: {missing_columns}. Expected columns: [{gt_cols}]",
            )

        # Check for duplicate IDs
        if prediction_df[self.id_column].duplicated().any():
            return False, f"Prediction contains duplicate {self.id_column} values"

        return True, ""

    def calculate_score(
        self, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame
    ) -> ScoringResult:
        """
        Calculate classification metrics.

        Args:
            prediction_df: DataFrame containing predictions.
            ground_truth_df: DataFrame containing ground truth.

        Returns:
            ScoringResult with granular metrics and per-class reports.
        """
        # Align predictions with ground truth
        merged = pd.merge(
            ground_truth_df,
            prediction_df,
            on=self.id_column,
            how="left",
            suffixes=("_true", "_pred"),
        )

        label_true_col = f"{self.label_column}_true"
        label_pred_col = f"{self.label_column}_pred"

        # Handle missing predictions
        missing_count = merged[label_pred_col].isna().sum()
        if missing_count > 0:
            self.log(f"Warning: {missing_count} items have no prediction", "WARNING")
            merged[label_pred_col] = merged[label_pred_col].fillna("__MISSING__")

        # Generate report
        report = self._generate_report(merged[label_true_col], merged[label_pred_col])
        if not report:
            return ScoringResult(success=False, error_message="Failed to generate report")

        # Extract and organize metrics
        metrics = self._extract_metrics(report, len(merged), int(missing_count))

        # Log summary
        self.log(f"Accuracy: {metrics['ACCURACY']:.4f}")
        self.log(f"F1 (macro): {metrics['F1_MACRO']:.4f}")
        self.log(f"F1 (weighted): {metrics['F1_WEIGHTED']:.4f}")

        if self.metric_target_class:
            self.log(
                f"Target Class '{self.metric_target_class}' F1: {metrics.get('CLASS_F1', 0.0):.4f}"
            )

        # Select primary score
        primary_score = metrics.get(self.metric_type, metrics["ACCURACY"])

        return ScoringResult(
            success=True,
            score=round(primary_score, 6),
            metrics=metrics,
            logs=self.logs,
        )

    def _generate_report(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """Generate classification report using sklearn."""
        try:
            return classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        except Exception as e:
            self.log(f"Failed to generate classification report: {e}", "ERROR")
            return {}

    def _extract_metrics(self, report: dict, total_samples: int, missing_count: int) -> dict:
        """Extract and format metrics from the classification report."""
        macro_avg = report.get("macro avg", {})
        weighted_avg = report.get("weighted avg", {})
        accuracy = report.get("accuracy", 0.0)

        metrics = {
            "ACCURACY": round(accuracy, 6),
            "F1_MACRO": round(macro_avg.get("f1-score", 0.0), 6),
            "F1_MICRO": round(accuracy, 6),
            "F1_WEIGHTED": round(weighted_avg.get("f1-score", 0.0), 6),
            "PRECISION_MACRO": round(macro_avg.get("precision", 0.0), 6),
            "PRECISION_MICRO": round(accuracy, 6),
            "PRECISION_WEIGHTED": round(weighted_avg.get("precision", 0.0), 6),
            "RECALL_MACRO": round(macro_avg.get("recall", 0.0), 6),
            "RECALL_MICRO": round(accuracy, 6),
            "RECALL_WEIGHTED": round(weighted_avg.get("recall", 0.0), 6),
            "total_samples": total_samples,
            "missing_predictions": missing_count,
            "per_class_report": report,
        }

        # Legacy aliases
        metrics["F1"] = metrics["F1_MACRO"]
        metrics["PRECISION"] = metrics["PRECISION_MACRO"]
        metrics["RECALL"] = metrics["RECALL_MACRO"]

        # Per-class metrics
        if self.metric_target_class:
            cls_rpt = report.get(str(self.metric_target_class), {})
            metrics["CLASS_F1"] = round(cls_rpt.get("f1-score", 0.0), 6)
            metrics["CLASS_PRECISION"] = round(cls_rpt.get("precision", 0.0), 6)
            metrics["CLASS_RECALL"] = round(cls_rpt.get("recall", 0.0), 6)
        else:
            metrics["CLASS_F1"] = 0.0
            metrics["CLASS_PRECISION"] = 0.0
            metrics["CLASS_RECALL"] = 0.0

        return metrics
