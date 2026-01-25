"""
Test all scoring engines: Classification, Detection, Segmentation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.engines.classification import ClassificationScoringEngine
from scoring.engines.detection import DetectionScoringEngine
from scoring.engines.segmentation import (
    SegmentationScoringEngine,
    rle_decode,
    rle_encode,
)


def test_classification():
    """Test Classification scoring."""
    print("\n=== Classification Scoring Test ===")
    fixtures_dir = Path(__file__).parent / "fixtures"

    engine = ClassificationScoringEngine(
        fixtures_dir / "classification_gt.csv", metric_type="ACCURACY"
    )
    result = engine.score(fixtures_dir / "classification_pred.csv")

    print(f"Success: {result.success}")
    print(f"Score (Accuracy): {result.score}")

    assert result.success, "Classification scoring should succeed"
    assert result.score == 0.6, f"Expected 0.6, got {result.score}"
    print("✅ Classification test passed!")


def test_detection():
    """Test Detection scoring."""
    print("\n=== Detection Scoring Test ===")
    fixtures_dir = Path(__file__).parent / "fixtures"

    engine = DetectionScoringEngine(
        fixtures_dir / "detection_gt.csv", metric_type="MAP"
    )
    result = engine.score(fixtures_dir / "detection_pred.csv")

    print(f"Success: {result.success}")
    print(f"Score (mAP@0.5): {result.score}")
    print(f"Metrics: {result.metrics}")

    assert result.success, "Detection scoring should succeed"
    assert result.score is not None and result.score > 0, "mAP should be positive"
    print("✅ Detection test passed!")


def test_rle_roundtrip():
    """Test RLE encode/decode roundtrip."""
    print("\n=== RLE Encode/Decode Test ===")
    import numpy as np

    # Create a simple mask
    original_mask = np.zeros((10, 10), dtype=np.uint8)
    original_mask[0:2, 0:5] = 1  # Top rows
    original_mask[5:7, 3:8] = 1  # Middle rows

    # Encode
    rle = rle_encode(original_mask)
    print(f"RLE: {rle}")

    # Decode
    decoded_mask = rle_decode(rle, 10, 10)

    # Compare
    assert np.array_equal(original_mask, decoded_mask), "RLE roundtrip failed"
    print("✅ RLE roundtrip test passed!")


def test_segmentation():
    """Test Segmentation scoring."""
    print("\n=== Segmentation Scoring Test ===")
    fixtures_dir = Path(__file__).parent / "fixtures"

    engine = SegmentationScoringEngine(
        fixtures_dir / "segmentation_gt.csv", metric_type="MIOU"
    )
    result = engine.score(fixtures_dir / "segmentation_pred.csv")

    print(f"Success: {result.success}")
    print(f"Score (mIoU): {result.score}")
    if result.metrics:
        print(f"Per-class IoU: {result.metrics.get('per_class_iou', {})}")

    assert result.success, "Segmentation scoring should succeed"
    assert result.score is not None and result.score > 0, "mIoU should be positive"
    print("✅ Segmentation test passed!")


if __name__ == "__main__":
    test_classification()
    test_detection()
    test_rle_roundtrip()
    test_segmentation()
    print("\n" + "=" * 50)
    print("🎉 All scoring engine tests passed!")
    print("=" * 50)
