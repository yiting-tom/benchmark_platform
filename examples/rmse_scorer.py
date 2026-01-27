import pandas as pd
import numpy as np
from scoring.engines.base import ScoringResult

REQUIRED_COLUMNS = ['filename', 'value']

def calculate_score(prediction_df, ground_truth_df):
    """
    Custom RMSE Scorer.
    """
    merged = pd.merge(ground_truth_df, prediction_df, on='filename', suffixes=('_gt', '_pred'))
    
    if merged.empty:
        return ScoringResult(success=False, error_message="No matching files", logs=["Empty merge"])
        
    rmse = np.sqrt(((merged['value_gt'] - merged['value_pred']) ** 2).mean())
    
    # Leaderboard score: higher is better
    score = 1.0 / (1.0 + rmse)
    
    return ScoringResult(
        success=True,
        score=float(score),
        metrics={'RMSE': float(rmse)},
        logs=[f"RMSE calculated: {rmse:.4f}", f"Matches: {len(merged)}"]
    )
