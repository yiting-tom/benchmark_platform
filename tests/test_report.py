from django.test import TestCase, Client
from django.contrib.auth.models import User
from competitions.models import Competition, Submission, SubmissionStatus
from django.utils import timezone

class ReportTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username="testuser", password="ComplexPass123!")
        self.client.force_login(self.user)
        self.competition = Competition.objects.create(
            name="Test Competition",
            metric_type="ACCURACY",
            daily_upload_limit=5,
            total_upload_limit=10,
        )
        self.submission = Submission.objects.create(
            competition=self.competition,
            user=self.user,
            status=SubmissionStatus.SUCCESS,
            scores={
                "per_class_report": {
                    "cat": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 10},
                    "dog": {"precision": 0.7, "recall": 0.75, "f1-score": 0.72, "support": 10},
                    "accuracy": 0.785,
                    "macro avg": {"precision": 0.8, "recall": 0.775, "f1-score": 0.785, "support": 20},
                }
            }
        )
        self.report_url = f"/submission/{self.submission.id}/report/"

    def test_report_content(self):
        """Test that the report modal returns correct per-class data."""
        response = self.client.get(self.report_url)
        self.assertEqual(response.status_code, 200)
        # Should contain class names
        self.assertContains(response, "cat")
        self.assertContains(response, "dog")
        # Should contain renamed f1 score (0.85)
        self.assertContains(response, "0.8500")
        # Should contain macro f1 score (0.7850)
        self.assertContains(response, "0.7850")

    def test_report_unauthorized(self):
        """Test that users cannot view reports for other users' submissions."""
        other_user = User.objects.create_user(username="other", password="ComplexPass123!")
        self.client.force_login(other_user)
        response = self.client.get(self.report_url)
        self.assertEqual(response.status_code, 404)
