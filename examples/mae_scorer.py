import pandas as pd
from scoring.engines.base import ScoringResult

# Optional: define required columns
REQUIRED_COLUMNS = ['filename', 'score_val']

def calculate_score(prediction_df, ground_truth_df):
    """
    Custom scoring logic: 
    Calculates the mean absolute error between 'score_val' columns.
    """
    # Merge on filename
    merged = pd.merge(ground_truth_df, prediction_df, on='filename', suffixes=('_gt', '_pred'))
    
    if merged.empty:
        return ScoringResult(
            success=False,
            error_message="No matching filenames found between GT and Prediction",
            logs=["Merge returned empty dataframe"]
        )
    
    # Calculate Mean Absolute Error
    mae = (merged['score_val_gt'] - merged['score_val_pred']).abs().mean()
    
    # We want a "score" where higher is better for the leaderboard, 
    # so let's return 1 / (1 + MAE)
    final_score = 1.0 / (1.0 + mae)
    
    return ScoringResult(
        success=True,
        score=round(float(final_score), 6),
        metrics={
            'MAE': round(float(mae), 4),
            'count': len(merged)
        },
        logs=[
            f"Successfully matched {len(merged)} rows",
            f"Calculated MAE: {mae:.4f}"
        ]
    )
