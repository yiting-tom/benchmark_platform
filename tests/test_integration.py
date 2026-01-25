from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from competitions.models import (
    Competition,
    CompetitionParticipant,
    Submission,
    SubmissionStatus,
    TaskType,
    MetricType,
    CompetitionStatus,
)
from scoring.tasks import score_private_submissions
import os
import pandas as pd
from pathlib import Path
from datetime import timedelta


class IntegrationTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="password")
        self.client = Client()
        self.client.login(username="testuser", password="password")

        # Create fixtures dir if not exists
        self.fixtures_dir = Path(__file__).parent / "fixtures"
        self.gt_path = self.fixtures_dir / "classification_gt.csv"

        # Create a test competition
        self.competition = Competition.objects.create(
            name="Test Competition",
            task_type=TaskType.CLASSIFICATION,
            metric_type=MetricType.ACCURACY,
            status=CompetitionStatus.ACTIVE,
            public_ground_truth=SimpleUploadedFile(
                "gt.csv", open(self.gt_path, "rb").read()
            ),
            private_ground_truth=SimpleUploadedFile(
                "private_gt.csv", open(self.gt_path, "rb").read()
            ),
            daily_upload_limit=5,
            total_upload_limit=10,
        )

        # Add participant
        self.participant = CompetitionParticipant.objects.create(
            competition=self.competition,
            user=self.user,
            start_time=timezone.now() - timedelta(days=1),
            end_time=timezone.now() + timedelta(days=1),
        )

    def test_csv_validation_failed(self):
        """Test that invalid CSV headers are caught immediately."""
        invalid_csv = SimpleUploadedFile("invalid.csv", b"wrong_id,wrong_label\n1,cat")

        url = f"/{self.competition.id}/upload/"
        response = self.client.post(url, {"prediction_file": invalid_csv})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Missing columns", response.content.decode())

        # Verify submission status
        submission = Submission.objects.filter(
            user=self.user, competition=self.competition
        ).last()
        assert submission is not None
        self.assertEqual(submission.status, SubmissionStatus.FAILED)
        self.assertIn("Missing columns", submission.error_message)

    def test_private_scoring_task(self):
        """Test the automated private scoring task."""
        # Create a successful submission
        pred_path = self.fixtures_dir / "classification_pred.csv"
        submission = Submission.objects.create(
            competition=self.competition,
            user=self.user,
            prediction_file=SimpleUploadedFile(
                "pred.csv", open(pred_path, "rb").read()
            ),
            status=SubmissionStatus.SUCCESS,
            public_score=0.6,
            is_final_selection=True,
        )

        # Run private scoring
        result = score_private_submissions(self.competition.id)

        self.assertTrue(result["success"])
        self.assertEqual(result["processed_count"], 1)

        # Verify private score
        submission.refresh_from_db()
        self.assertEqual(submission.private_score, 0.6)

        # Verify competition status
        self.competition.refresh_from_db()
        self.assertTrue(self.competition.private_scoring_completed)
