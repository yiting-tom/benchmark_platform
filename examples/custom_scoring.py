"""
Example Custom Scoring Script

This script can be uploaded by an admin to handle custom scoring logic.
It can either define a `calculate_score` function or a `CustomScorer` class.
"""

import pandas as pd


def calculate_score(prediction_df, ground_truth_df, **kwargs):
    """
    Simple scoring function that calculates the ratio of exact matches.

    Args:
        prediction_df: DataFrame containing the predictions.
        ground_truth_df: DataFrame containing the ground truth.
        logger: Optional logger instance (provided if the engine is CustomScoringEngine)

    Returns:
        float, dict, or ScoringResult
    """
    logger = kwargs.get("logger")

    if logger:
        logger.log("Starting custom scoring calculation...")

    # Assuming both DFs have 'image_id' and 'label' columns
    # This is just a toy example
    if "image_id" not in prediction_df.columns or "label" not in prediction_df.columns:
        return {"success": False, "error_message": "Missing required columns in prediction"}

    merged = pd.merge(ground_truth_df, prediction_df, on="image_id", suffixes=("_gt", "_pred"))
    matches = (merged["label_gt"] == merged["label_pred"]).sum()
    score = matches / len(ground_truth_df) if len(ground_truth_df) > 0 else 0

    if logger:
        logger.log(f"Custom match count: {matches}")
        logger.log(f"Custom score: {score:.4f}")

    return {
        "success": True,
        "score": float(score),
        "metrics": {"matches": int(matches), "total": len(ground_truth_df)},
    }


# Alternatively, inherit from the base class for more control:
# from scoring.engines.base import BaseScoringEngine, ScoringResult
# class CustomScorer(BaseScoringEngine):
#     REQUIRED_COLUMNS = ['image_id', 'label']
#     def calculate_score(self, prediction_df, ground_truth_df) -> ScoringResult:
#         ...
