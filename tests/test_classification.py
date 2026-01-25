"""
Test the classification scoring engine.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.engines.classification import ClassificationScoringEngine


def test_classification_scorer():
    """Test basic classification scoring."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    gt_path = fixtures_dir / "classification_gt.csv"
    pred_path = fixtures_dir / "classification_pred.csv"

    # Test with ACCURACY metric
    engine = ClassificationScoringEngine(gt_path, metric_type="ACCURACY")
    result = engine.score(pred_path)

    print("=== Classification Scoring Test ===")
    print(f"Success: {result.success}")
    print(f"Score (Accuracy): {result.score}")
    print(f"Metrics: {result.metrics}")
    print(f"Logs: {result.logs}")

    # Verify expected results
    # Ground truth: cat, dog, cat, bird, dog
    # Prediction:   cat, dog, dog, bird, cat
    # Correct:      ✓    ✓    ✗    ✓     ✗   = 3/5 = 0.6

    assert result.success, "Scoring should succeed"
    assert result.score == 0.6, f"Expected accuracy 0.6, got {result.score}"
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_classification_scorer()
