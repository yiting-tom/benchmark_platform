from django.test import TestCase
from django.contrib.auth.models import User
from competitions.models import Competition, Submission, SubmissionStatus, TaskType, MetricType
from scoring.tasks import score_submission
from django.core.files.uploadedfile import SimpleUploadedFile

class MultiMetricTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="password123")
        self.competition = Competition.objects.create(
            name="Multi Metric Competition",
            task_type=TaskType.CLASSIFICATION,
            metric_type=MetricType.ACCURACY,
            available_metrics=[MetricType.ACCURACY, MetricType.F1],
            public_ground_truth=SimpleUploadedFile("gt.csv", b"id,label\n1,cat\n2,dog\n")
        )

    def test_multi_metric_saved(self):
        """Test that multiple metrics are saved in Submission.all_scores."""
        # Create a mock prediction file
        prediction_content = b"id,label\n1,cat\n2,cat\n"
        prediction_file = SimpleUploadedFile("pred.csv", prediction_content)
        
        submission = Submission.objects.create(
            competition=self.competition,
            user=self.user,
            prediction_file=prediction_file,
            status=SubmissionStatus.PENDING
        )
        
        # Run scoring (synchronously for testing)
        score_submission(submission.id)
        
        # Refresh from DB
        submission.refresh_from_db()
        
        self.assertEqual(submission.status, SubmissionStatus.SUCCESS)
        self.assertIsNotNone(submission.all_scores)
        self.assertIn(MetricType.ACCURACY, submission.all_scores)
        self.assertIn(MetricType.F1, submission.all_scores)
        
        # Accuracy should be 0.5 (1 correct out of 2)
        self.assertEqual(submission.all_scores[MetricType.ACCURACY], 0.5)
        # Primary score should also be 0.5
        self.assertEqual(submission.public_score, 0.5)
