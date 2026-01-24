"""
Classification scoring engine.

Computes Accuracy and F1-Score for image classification tasks.

CSV format is dynamically detected from the Ground Truth file.
Typically: [id_column], [label_column]
Example: filename, label OR image_id, label
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .base import BaseScoringEngine, ScoringResult


class ClassificationScoringEngine(BaseScoringEngine):
    """
    Scoring engine for image classification tasks.
    
    Supports metrics:
    - ACCURACY: Overall accuracy
    - F1: Macro-averaged F1-score
    
    Column names are auto-detected from the Ground Truth file.
    """
    
    REQUIRED_COLUMNS = []  # Will be set dynamically from ground truth
    
    def __init__(self, ground_truth_path, metric_type: str = "ACCURACY"):
        """
        Initialize the classification scorer.
        
        Args:
            ground_truth_path: Path to ground truth CSV.
            metric_type: Either "ACCURACY" or "F1".
        """
        super().__init__(ground_truth_path)
        self.metric_type = metric_type
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
        
        self.log(f"Auto-detected columns: ID='{self.id_column}', Label='{self.label_column}'")
        
        # Set required columns for prediction validation
        self.REQUIRED_COLUMNS = [self.id_column, self.label_column]
        
        return True
    
    def validate_prediction_format(self, prediction_df: pd.DataFrame) -> tuple[bool, str]:
        """Validate prediction CSV format matches ground truth."""
        # Check required columns (auto-detected from ground truth)
        missing_columns = set(self.REQUIRED_COLUMNS) - set(prediction_df.columns)
        if missing_columns:
            gt_cols = ", ".join(self.REQUIRED_COLUMNS)
            return False, f"Missing required columns: {missing_columns}. Expected columns: [{gt_cols}]"
        
        # Check for duplicate IDs
        if prediction_df[self.id_column].duplicated().any():
            return False, f"Prediction contains duplicate {self.id_column} values"
        
        return True, ""
    
    def calculate_score(
        self, 
        prediction_df: pd.DataFrame, 
        ground_truth_df: pd.DataFrame
    ) -> ScoringResult:
        """
        Calculate classification score.
        
        Args:
            prediction_df: DataFrame with same columns as ground truth
            ground_truth_df: DataFrame with [id_column, label_column]
            
        Returns:
            ScoringResult with accuracy/F1 and per-class metrics.
        """
        # Merge on ID column to align predictions with ground truth
        merged = pd.merge(
            ground_truth_df,
            prediction_df,
            on=self.id_column,
            how="left",
            suffixes=("_true", "_pred")
        )
        
        label_true_col = f"{self.label_column}_true"
        label_pred_col = f"{self.label_column}_pred"
        
        # Check for missing predictions
        missing_count = merged[label_pred_col].isna().sum()
        if missing_count > 0:
            self.log(f"Warning: {missing_count} items have no prediction", "WARNING")
            # Fill missing with a placeholder (will be counted as wrong)
            merged[label_pred_col] = merged[label_pred_col].fillna("__MISSING__")
        
        y_true = merged[label_true_col]
        y_pred = merged[label_pred_col]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        self.log(f"Accuracy: {accuracy:.4f}")
        self.log(f"F1 (macro): {f1_macro:.4f}")
        
        # Determine primary score based on metric_type
        if self.metric_type == "F1":
            primary_score = f1_macro
        else:
            primary_score = accuracy
        
        # Generate per-class report
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception:
            report = {}
        
        return ScoringResult(
            success=True,
            score=round(primary_score, 6),
            metrics={
                "accuracy": round(accuracy, 6),
                "f1_macro": round(f1_macro, 6),
                "total_samples": len(merged),
                "missing_predictions": int(missing_count),
                "per_class_report": report,
            },
            logs=self.logs
        )
