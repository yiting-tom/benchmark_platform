import os
import tempfile
from pathlib import Path
from django.test import TestCase
from django.utils import timezone
from django.core.files.uploadedfile import SimpleUploadedFile
from competitions.models import Competition, TaskType, MetricType, CompetitionStatus
from scoring.engines.custom import CustomScoringEngine
from scoring.tasks import get_scoring_engine

class CustomScoringTest(TestCase):
    def setUp(self):
        self.fixtures_dir = Path(__file__).parent / "fixtures"
        self.gt_path = self.fixtures_dir / "classification_gt.csv"
        self.pred_path = self.fixtures_dir / "classification_pred.csv"
        
        # Create a temp scoring script
        self.script_content = b"""
import pandas as pd
def calculate_score(prediction_df, ground_truth_df, **kwargs):
    # Intentional specific logic for testing:
    # return the number of rows in prediction as the score
    return float(len(prediction_df))
"""
        self.script_file = SimpleUploadedFile("test_scorer.py", self.script_content)

    def test_custom_scoring_engine_directly(self):
        """Test the CustomScoringEngine class directly with a script file."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            tmp.write(self.script_content)
            tmp_path = tmp.name
        
        try:
            engine = CustomScoringEngine(self.gt_path, tmp_path)
            result = engine.score(self.pred_path)
            
            self.assertTrue(result.success)
            # classification_pred.csv has 5 rows
            self.assertEqual(result.score, 5.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_get_scoring_engine_for_custom(self):
        """Test that get_scoring_engine factory returns CustomScoringEngine."""
        competition = Competition.objects.create(
            name="Custom Competition",
            task_type=TaskType.CLASSIFICATION,
            metric_type=MetricType.CUSTOM,
            status=CompetitionStatus.ACTIVE,
            public_ground_truth=SimpleUploadedFile("gt.csv", b"id,label\n1,cat"),
            scoring_script=self.script_file
        )
        
        engine = get_scoring_engine(competition)
        self.assertIsInstance(engine, CustomScoringEngine)
        self.assertEqual(Path(engine.script_path), Path(competition.scoring_script.path))
