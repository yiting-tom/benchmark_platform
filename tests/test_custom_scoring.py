from django.test import TestCase
from django.core.files.base import ContentFile
from django.contrib.auth import get_user_model
from competitions.models import Competition, TaskType, Submission, SubmissionStatus, SubmissionLog
from scoring.tasks import score_submission

class CustomScoringTest(TestCase):
    def setUp(self):
        self.User = get_user_model()
        self.user = self.User.objects.create_user(username="testuser")
        
        # Create a dummy ground truth
        self.gt_content = "filename,label\ntest.jpg,1"
        
    def test_custom_scoring_script_execution(self):
        # 1. Create a dummy scoring script content
        script_content = """
from scoring.engines.base import ScoringResult
import pandas as pd

REQUIRED_COLUMNS = ['filename', 'label']

def calculate_score(prediction_df, ground_truth_df):
    # Just return a hardcoded score and a log
    return ScoringResult(
        success=True,
        score=0.95,
        metrics={'custom_metric': 123},
        logs=['Log from custom script']
    )
"""
        
        # 2. Setup Competition with the script
        competition = Competition.objects.create(
            name="Custom Script Test",
            task_type=TaskType.CUSTOM,
            public_ground_truth=ContentFile(self.gt_content, name="gt.csv"),
        )
        competition.scoring_script.save("test_scorer.py", ContentFile(script_content))
        
        # 3. Create a submission
        submission = Submission.objects.create(
            competition=competition,
            user=self.user,
            prediction_file=ContentFile("filename,label\ntest.jpg,1", name="pred.csv")
        )
        
        # 4. Run the scoring task
        score_submission(submission.id)
        
        # 5. Verify results
        submission.refresh_from_db()
        self.assertEqual(submission.status, SubmissionStatus.SUCCESS)
        self.assertEqual(submission.public_score, 0.95)
        self.assertEqual(submission.all_scores, {'custom_metric': 123})
        
        # Check logs
        logs = list(SubmissionLog.objects.filter(submission=submission).values_list('message', flat=True))
        self.assertIn('Log from custom script', logs)
        self.assertTrue(any('Successfully loaded custom scoring script: test_scorer' in log for log in logs))

    def test_custom_scoring_script_with_error(self):
        # Script with syntax error or logical error
        script_content = """
def calculate_score(prediction_df, ground_truth_df):
    raise ValueError("Something went wrong in custom script")
"""
        
        competition = Competition.objects.create(
            name="Custom Script Error Test",
            task_type=TaskType.CUSTOM,
            public_ground_truth=ContentFile(self.gt_content, name="gt.csv"),
        )
        competition.scoring_script.save("error_scorer.py", ContentFile(script_content))
        
        submission = Submission.objects.create(
            competition=competition,
            user=self.user,
            prediction_file=ContentFile("filename,label\ntest.jpg,1", name="pred.csv")
        )
        
        score_submission(submission.id)
        
        submission.refresh_from_db()
        self.assertEqual(submission.status, SubmissionStatus.FAILED)
        self.assertIn("Something went wrong in custom script", submission.error_message)
