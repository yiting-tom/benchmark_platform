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

    # Test with PRECISION metric
    engine = ClassificationScoringEngine(gt_path, metric_type="PRECISION")
    result = engine.score(pred_path)
    print(f"Score (Precision): {result.score}")
    assert result.score > 0, "Precision should be positive"

    # Test with RECALL metric
    engine = ClassificationScoringEngine(gt_path, metric_type="RECALL")
    result = engine.score(pred_path)
    print(f"Score (Recall): {result.score}")
    assert result.score > 0, "Recall should be positive"

    # Verify all metrics are present in results
    assert "PRECISION_MACRO" in result.metrics
    assert "F1_WEIGHTED" in result.metrics
    assert "ACCURACY" in result.metrics
    assert "F1" in result.metrics

    # Test with WEIGHTED metric
    engine = ClassificationScoringEngine(gt_path, metric_type="F1_WEIGHTED")
    result = engine.score(pred_path)
    print(f"Score (F1 Weighted): {result.score}")
    assert result.score == result.metrics["F1_WEIGHTED"]

    # Test with CLASS-specific metric (Target: cat)
    engine = ClassificationScoringEngine(
        gt_path, metric_type="CLASS_F1", metric_target_class="cat"
    )
    result = engine.score(pred_path)
    print(f"Score (Class F1 - cat): {result.score}")
    
    # Ground truth: cat, dog, cat, bird, dog
    # Prediction:   cat, dog, dog, bird, cat
    # For 'cat': TP=1 (image 1), FP=1 (image 5), FN=1 (image 3)
    # Precision = 1 / (1+1) = 0.5
    # Recall = 1 / (1+1) = 0.5
    # F1 = 2 * (0.5*0.5)/(0.5+0.5) = 0.5
    assert result.score == 0.5, f"Expected 0.5 for cat F1, got {result.score}"
    assert result.metrics["CLASS_F1"] == 0.5
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_classification_scorer()
