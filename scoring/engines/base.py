"""
Base scoring engine interface.

All task-specific scorers (Classification, Detection, Segmentation)
inherit from this base class and implement the `calculate_score` method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ScoringResult:
    """Result of a scoring operation."""
    success: bool
    score: float | None = None
    metrics: dict[str, Any] | None = None  # Additional metrics (e.g., per-class accuracy)
    error_message: str | None = None
    logs: list[str] | None = None  # Detailed log messages


class BaseScoringEngine(ABC):
    """
    Abstract base class for all scoring engines.
    
    Subclasses must implement:
    - validate_prediction_format(): Check if CSV has required columns
    - calculate_score(): Compute the score given prediction and ground truth
    """
    
    # Required columns for prediction CSV (to be overridden by subclasses)
    REQUIRED_COLUMNS: list[str] = []
    
    def __init__(self, ground_truth_path: Path | str):
        """
        Initialize the scoring engine with ground truth data.
        
        Args:
            ground_truth_path: Path to the ground truth CSV file.
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.ground_truth_df: pd.DataFrame | None = None
        self.logs: list[str] = []
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Add a log message."""
        self.logs.append(f"[{level}] {message}")
    
    def load_ground_truth(self) -> bool:
        """
        Load and validate the ground truth file.
        
        Returns:
            True if successfully loaded, False otherwise.
        """
        try:
            self.ground_truth_df = pd.read_csv(self.ground_truth_path)
            self.log(f"Loaded ground truth with {len(self.ground_truth_df)} rows")
            return True
        except Exception as e:
            self.log(f"Failed to load ground truth: {e}", "ERROR")
            return False
    
    def load_prediction(self, prediction_path: Path | str) -> pd.DataFrame | None:
        """
        Load and validate a prediction file.
        
        Args:
            prediction_path: Path to the prediction CSV file.
            
        Returns:
            DataFrame if valid, None otherwise.
        """
        try:
            df = pd.read_csv(prediction_path)
            self.log(f"Loaded prediction with {len(df)} rows")
            return df
        except Exception as e:
            self.log(f"Failed to load prediction: {e}", "ERROR")
            return None
    
    def validate_prediction_format(self, prediction_df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate that the prediction DataFrame has required columns.
        
        Args:
            prediction_df: The prediction DataFrame to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        missing_columns = set(self.REQUIRED_COLUMNS) - set(prediction_df.columns)
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        return True, ""
    
    @abstractmethod
    def calculate_score(
        self, 
        prediction_df: pd.DataFrame, 
        ground_truth_df: pd.DataFrame
    ) -> ScoringResult:
        """
        Calculate the score for the given prediction.
        
        Args:
            prediction_df: DataFrame containing predictions.
            ground_truth_df: DataFrame containing ground truth.
            
        Returns:
            ScoringResult with score and metrics.
        """
        pass
    
    def score(self, prediction_path: Path | str) -> ScoringResult:
        """
        Main entry point: load, validate, and score a prediction file.
        
        Args:
            prediction_path: Path to the prediction CSV file.
            
        Returns:
            ScoringResult with score, metrics, and logs.
        """
        self.logs = []  # Reset logs
        
        # Load ground truth if not already loaded
        if self.ground_truth_df is None:
            if not self.load_ground_truth():
                return ScoringResult(
                    success=False,
                    error_message="Failed to load ground truth",
                    logs=self.logs
                )
        
        # Load prediction
        prediction_df = self.load_prediction(prediction_path)
        if prediction_df is None:
            return ScoringResult(
                success=False,
                error_message="Failed to load prediction file",
                logs=self.logs
            )
        
        # Validate format
        is_valid, error_msg = self.validate_prediction_format(prediction_df)
        if not is_valid:
            return ScoringResult(
                success=False,
                error_message=error_msg,
                logs=self.logs
            )
        
        # Calculate score
        try:
            result = self.calculate_score(prediction_df, self.ground_truth_df)
            result.logs = self.logs + (result.logs or [])
            return result
        except Exception as e:
            self.log(f"Scoring failed: {e}", "ERROR")
            return ScoringResult(
                success=False,
                error_message=str(e),
                logs=self.logs
            )
