"""
Custom scoring engine.

Loads and executes a Python script provided by the admin.
The script must define either:
1. A class named `CustomScorer` that inherits from `BaseScoringEngine`
2. A function named `calculate_score(prediction_df, ground_truth_df, **kwargs)`
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseScoringEngine, ScoringResult


class CustomScoringEngine(BaseScoringEngine):
    """
    Scoring engine that delegates to a user-provided Python script.
    """

    def __init__(self, ground_truth_path: str | Path, script_path: str | Path):
        """
        Initialize the custom scorer.

        Args:
            ground_truth_path: Path to ground truth CSV.
            script_path: Path to the Python scoring script.
        """
        super().__init__(ground_truth_path)
        self.script_path: Path = Path(script_path)
        self.custom_module: Any | None = None
        self.scorer_instance: BaseScoringEngine | None = None

    def _load_script(self):
        """Dynamic load the scoring script."""
        if self.custom_module:
            return True

        if not self.script_path.exists():
            self.log(f"Scoring script not found: {self.script_path}", "ERROR")
            return False

        try:
            module_name = f"custom_scorer_{os.path.basename(self.script_path).replace('.', '_')}"
            spec = importlib.util.spec_from_file_location(module_name, self.script_path)
            if spec is None or spec.loader is None:
                self.log(f"Failed to create module spec for {self.script_path}", "ERROR")
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self.custom_module = module

            # Check for CustomScorer class
            if hasattr(module, "CustomScorer"):
                scorer_class = getattr(module, "CustomScorer")
                self.scorer_instance = scorer_class(self.ground_truth_path)
                self.log(f"Loaded CustomScorer class from {self.script_path}")
            elif hasattr(module, "calculate_score"):
                self.log(f"Found calculate_score function in {self.script_path}")
            else:
                self.log(
                    f"Script must define 'CustomScorer' class or 'calculate_score' function",
                    "ERROR",
                )
                return False

            return True
        except Exception as e:
            self.log(f"Failed to load scoring script: {e}", "ERROR")
            return False

    def load_ground_truth(self) -> bool:
        """Override to delegate to custom scorer if it's a class."""
        if not self._load_script():
            return False

        if self.scorer_instance:
            # Transfer logs if necessary
            success = self.scorer_instance.load_ground_truth()
            self.ground_truth_df = self.scorer_instance.ground_truth_df
            self.REQUIRED_COLUMNS = self.scorer_instance.REQUIRED_COLUMNS
            return success

        return super().load_ground_truth()

    def validate_prediction_format(
        self, prediction_df: pd.DataFrame
    ) -> tuple[bool, str]:
        """Override to delegate to custom scorer if it's a class."""
        if self.scorer_instance:
            return self.scorer_instance.validate_prediction_format(prediction_df)
        return super().validate_prediction_format(prediction_df)

    def calculate_score(
        self, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame
    ) -> ScoringResult:
        """
        Calculate score using the custom script.
        """
        if not self._load_script():
            return ScoringResult(
                success=False, error_message="Failed to load scoring script"
            )

        try:
            if self.scorer_instance:
                return self.scorer_instance.calculate_score(
                    prediction_df, ground_truth_df
                )

            # Delegate to function
            calc_func = getattr(self.custom_module, "calculate_score")
            result = calc_func(prediction_df, ground_truth_df, logger=self)

            # Handle different return types for convenience
            if isinstance(result, ScoringResult):
                return result
            if isinstance(result, (int, float)):
                return ScoringResult(success=True, score=float(result), logs=self.logs)
            if isinstance(result, dict):
                return ScoringResult(
                    success=result.get("success", True),
                    score=result.get("score"),
                    metrics=result.get("metrics"),
                    error_message=result.get("error_message"),
                    logs=self.logs + (result.get("logs") or []),
                )

            raise ValueError(f"Unsupported return type from custom script: {type(result)}")

        except Exception as e:
            self.log(f"Custom scoring execution failed: {e}", "ERROR")
            return ScoringResult(success=False, error_message=str(e), logs=self.logs)
