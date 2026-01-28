"""
Custom scoring engine that executes an uploaded Python script.
"""

import importlib.util
import sys
from pathlib import Path
import pandas as pd

from .base import BaseScoringEngine, ScoringResult


class CustomScoringEngine(BaseScoringEngine):
    """
    Scoring engine that dynamically loads and executes a Python script.
    
    The script must define:
    - calculate_score(prediction_df, ground_truth_df) -> ScoringResult
    
    Optionally:
    - REQUIRED_COLUMNS: list[str]
    """

    def __init__(self, ground_truth_path: Path | str, script_path: Path | str):
        """
        Initialize with ground truth and the path to the scoring script.
        """
        super().__init__(ground_truth_path)
        self.script_path = Path(script_path)
        self.module = None

    def _load_script(self):
        """Dynamically load the Python script as a module."""
        try:
            module_name = f"custom_scorer_{self.script_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, self.script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {self.script_path}")
            
            self.module = importlib.util.module_from_spec(spec)
            # Add to sys.modules to allow relative imports if needed (though unlikely)
            sys.modules[module_name] = self.module
            spec.loader.exec_module(self.module)
            
            # Set REQUIRED_COLUMNS from script if present
            if hasattr(self.module, "REQUIRED_COLUMNS"):
                self.REQUIRED_COLUMNS = self.module.REQUIRED_COLUMNS
                
            self.log(f"Successfully loaded custom scoring script: {self.script_path.name}")
        except Exception as e:
            self.log(f"Failed to load custom scoring script: {e}", "ERROR")
            raise

    def calculate_score(
        self, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame
    ) -> ScoringResult:
        """
        Execute the calculate_score function from the loaded script.
        """
        if self.module is None:
            try:
                self._load_script()
            except Exception as e:
                return ScoringResult(
                    success=False,
                    error_message=f"Failed to load scoring script: {e}",
                    logs=self.logs
                )

        if not self.module or not hasattr(self.module, "calculate_score"):
            return ScoringResult(
                success=False,
                error_message="Custom script does not implement calculate_score()",
                logs=self.logs,
            )

        try:
            # Pass our logger-augmented logs list to the script if it wants it?
            # Actually, let's keep it simple: the script returns a ScoringResult.
            # We will merge its logs with ours in the base class.
            result = self.module.calculate_score(prediction_df, ground_truth_df)
            
            # Ensure it's a ScoringResult
            if not isinstance(result, ScoringResult):
                # Attempt to wrap if it's just a dict or score
                if isinstance(result, (float, int)):
                    return ScoringResult(success=True, score=float(result), logs=self.logs)
                elif isinstance(result, dict):
                    return ScoringResult(
                        success=result.get("success", True),
                        score=result.get("score"),
                        metrics=result.get("metrics"),
                        error_message=result.get("error_message"),
                        logs=self.logs + (result.get("logs") or []),
                    )
                else:
                    raise TypeError(f"Custom script returned unexpected type: {type(result)}")
            
            return result
        except Exception as e:
            self.log(f"Error executing custom calculate_score: {e}", "ERROR")
            return ScoringResult(
                success=False,
                error_message=f"Custom script execution failed: {e}",
                logs=self.logs,
            )
