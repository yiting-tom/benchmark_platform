"""
Test the classification scoring engine.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.engines.classification import ClassificationScoringEngine


def run_test(name, engine, pred_path):
    """Helper to run a test and print results."""
    print(f"\n--- {name} ---")
    result = engine.score(pred_path)
    print(f"Success: {result.success}")
    print(f"Score: {result.score}")
    return result


def test_basic_accuracy(fixtures_dir):
    gt_path = fixtures_dir / "classification_gt.csv"
    pred_path = fixtures_dir / "classification_pred.csv"
    engine = ClassificationScoringEngine(gt_path, metric_type="ACCURACY")
    result = run_test("Basic Accuracy", engine, pred_path)
    assert result.success
    assert result.score == 0.6


def test_granular_metrics(fixtures_dir):
    gt_path = fixtures_dir / "classification_gt.csv"
    pred_path = fixtures_dir / "classification_pred.csv"
    
    # Test Precision (Macro)
    engine = ClassificationScoringEngine(gt_path, metric_type="PRECISION")
    result = run_test("Precision (Macro)", engine, pred_path)
    assert result.score > 0
    
    # Test Recall (Macro)
    engine = ClassificationScoringEngine(gt_path, metric_type="RECALL")
    result = run_test("Recall (Macro)", engine, pred_path)
    assert result.score > 0
    
    # Test F1 (Weighted)
    engine = ClassificationScoringEngine(gt_path, metric_type="F1_WEIGHTED")
    result = run_test("F1 (Weighted)", engine, pred_path)
    assert result.score == result.metrics["F1_WEIGHTED"]


def test_class_specific_metrics(fixtures_dir):
    gt_path = fixtures_dir / "classification_gt.csv"
    pred_path = fixtures_dir / "classification_pred.csv"
    engine = ClassificationScoringEngine(
        gt_path, metric_type="CLASS_F1", metric_target_class="cat"
    )
    result = run_test("Class-Specific F1 (cat)", engine, pred_path)
    assert result.score == 0.5


def test_edge_cases(fixtures_dir):
    gt_path = fixtures_dir / "classification_gt.csv"
    pred_path = fixtures_dir / "classification_pred.csv"
    
    # Partial predictions
    partial_pred_path = fixtures_dir / "classification_partial_pred.csv"
    engine = ClassificationScoringEngine(gt_path, metric_type="ACCURACY")
    result = run_test("Partial Predictions", engine, partial_pred_path)
    assert result.score == 0.4
    assert result.metrics["missing_predictions"] == 3

    # Invalid target class
    engine = ClassificationScoringEngine(
        gt_path, metric_type="CLASS_F1", metric_target_class="not_a_class"
    )
    result = run_test("Invalid Target Class", engine, pred_path)
    assert result.score == 0.0


def main():
    print("=== Classification Scoring Tests ===")
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    test_basic_accuracy(fixtures_dir)
    test_granular_metrics(fixtures_dir)
    test_class_specific_metrics(fixtures_dir)
    test_edge_cases(fixtures_dir)
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
