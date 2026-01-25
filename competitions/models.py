"""
Competition models for the CV Benchmark Platform.

This module defines the core data models:
- Competition: The main competition entity
- CompetitionParticipant: Whitelist with per-user time windows
- Submission: Prediction file uploads and scores
- SubmissionLog: Detailed scoring logs for debugging
"""

from django.conf import settings
from django.db import models
from django.utils import timezone
from typing import TYPE_CHECKING, List, Tuple, Optional

if TYPE_CHECKING:
    from django.contrib.auth.models import User


class TaskType(models.TextChoices):
    """Supported CV task types."""

    CLASSIFICATION = "CLASSIFICATION", "Image Classification"
    DETECTION = "DETECTION", "Object Detection"
    SEGMENTATION = "SEGMENTATION", "Image Segmentation"


class MetricType(models.TextChoices):
    """Supported evaluation metrics."""

    ACCURACY = "ACCURACY", "Accuracy"
    F1 = "F1", "F1-Score"
    MAP = "MAP", "mAP@0.5"
    MAP_50_95 = "MAP_50_95", "mAP@[0.5:0.95]"
    MIOU = "MIOU", "mIoU"
    PRECISION = "PRECISION", "Precision"
    RECALL = "RECALL", "Recall"
    AP75 = "AP75", "mAP@0.75"
    CUSTOM = "CUSTOM", "Custom Script"


class Metric(models.Model):
    """Evaluation metric lookup."""

    name = models.CharField(
        max_length=20, choices=MetricType.choices, unique=True, verbose_name="Metric Name"
    )

    def __str__(self):
        return str(self.get_name_display())  # type: ignore

    class Meta:
        verbose_name = "Metric"
        verbose_name_plural = "Metrics"


class CompetitionStatus(models.TextChoices):
    """Competition lifecycle status."""

    DRAFT = "DRAFT", "Draft"
    ACTIVE = "ACTIVE", "Active"
    ENDED = "ENDED", "Ended"


class SubmissionStatus(models.TextChoices):
    """Submission processing status."""

    PENDING = "PENDING", "Pending"
    PROCESSING = "PROCESSING", "Processing"
    SUCCESS = "SUCCESS", "Success"
    FAILED = "FAILED", "Failed"


class LogLevel(models.TextChoices):
    """Log severity levels."""

    INFO = "INFO", "Info"
    WARNING = "WARNING", "Warning"
    ERROR = "ERROR", "Error"


def competition_ground_truth_path(instance, filename):
    """Generate upload path for ground truth files."""
    return f"competitions/{instance.id}/ground_truth/{filename}"


def submission_prediction_path(instance, filename):
    """Generate upload path for prediction files."""
    return f"submissions/{instance.competition_id}/{instance.user_id}/{filename}"


class Competition(models.Model):
    """
    A CV competition created by an admin.

    Stores competition metadata, ground truth files, and upload limits.
    """

    name = models.CharField(
        max_length=100,
        verbose_name="Competition Name",
        help_text="e.g., Defect Detection Challenge",
    )
    description = models.TextField(
        verbose_name="Description", help_text="Markdown supported", blank=True
    )
    task_type = models.CharField(
        max_length=20, choices=TaskType.choices, verbose_name="Task Type"
    )
    metric_type = models.CharField(
        max_length=20,
        choices=MetricType.choices,
        verbose_name="Primary Metric",
        help_text="Metric used for main ranking",
    )
    additional_metrics = models.ManyToManyField(
        Metric, blank=True, verbose_name="Additional Metrics"
    )

    # Ground truth files
    public_ground_truth = models.FileField(
        upload_to=competition_ground_truth_path,
        verbose_name="Public Ground Truth",
        help_text="CSV format",
    )
    private_ground_truth = models.FileField(
        upload_to=competition_ground_truth_path,
        verbose_name="Private Ground Truth",
        help_text="CSV format, only visible to Validators",
        blank=True,
        null=True,
    )
    scoring_script = models.FileField(
        upload_to="competitions/scoring_scripts/",
        verbose_name="Scoring Script",
        help_text="Custom Python script for scoring. Must implement calculate_score function.",
        blank=True,
        null=True,
    )

    # Dataset access
    dataset_url = models.URLField(verbose_name="Dataset Download URL", blank=True)

    # Upload limits
    daily_upload_limit = models.PositiveIntegerField(
        default=5, verbose_name="Daily Upload Limit"
    )
    total_upload_limit = models.PositiveIntegerField(
        default=100, verbose_name="Total Upload Limit"
    )

    # Status
    status = models.CharField(
        max_length=20,
        choices=CompetitionStatus.choices,
        default=CompetitionStatus.DRAFT,
        verbose_name="Status",
    )
    private_scoring_completed = models.BooleanField(
        default=False,
        verbose_name="Private Scoring Completed",
        help_text="Whether private scores have been automatically calculated",
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")

    class Meta:
        verbose_name = "Competition"
        verbose_name_plural = "Competitions"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class CompetitionParticipant(models.Model):
    """
    Whitelist entry linking a user to a competition with a time window.

    Each participant has their own start/end time, allowing staggered
    competition access for different team members.
    """

    competition = models.ForeignKey(
        Competition,
        on_delete=models.CASCADE,
        related_name="participants",
        verbose_name="Competition",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="competition_participations",
        verbose_name="Participant",
    )

    # Per-user time window
    start_time = models.DateTimeField(verbose_name="Start Time")
    end_time = models.DateTimeField(verbose_name="End Time")

    # Manual control
    is_active = models.BooleanField(
        default=True,
        verbose_name="Active",
        help_text="Uncheck to suspend participation",
    )

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")

    class Meta:
        verbose_name = "Competition Participant"
        verbose_name_plural = "Competition Participants"
        unique_together = ["competition", "user"]
        indexes = [
            models.Index(fields=["competition", "user"]),
            models.Index(fields=["start_time", "end_time"]),
        ]

    def __str__(self):
        return f"{self.user} @ {self.competition}"

    def is_within_time_window(self) -> bool:
        """Check if current time is within the participation window."""
        now = timezone.now()
        return self.start_time <= now <= self.end_time

    def can_participate(self) -> bool:
        """Check if user is active and within time window."""
        return self.is_active and self.is_within_time_window()


class Submission(models.Model):
    """
    A prediction file submission from a participant.

    Tracks the upload, scoring status, and both public/private scores.
    """

    id: int  # Explicitly typed for static analysis
    competition = models.ForeignKey(
        Competition,
        on_delete=models.CASCADE,
        related_name="submissions",
        verbose_name="Competition",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="submissions",
        verbose_name="Submitter",
    )

    # Uploaded file
    prediction_file = models.FileField(
        upload_to=submission_prediction_path, verbose_name="Prediction File"
    )

    # Processing status
    status = models.CharField(
        max_length=20,
        choices=SubmissionStatus.choices,
        default=SubmissionStatus.PENDING,
        verbose_name="Status",
    )

    # Scores
    public_score = models.FloatField(null=True, blank=True, verbose_name="Public Score")
    private_score = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Private Score",
        help_text="Filled by Validator",
    )
    scores = models.JSONField(
        null=True,
        blank=True,
        verbose_name="All Scores",
        help_text="Stores scores for all selected metrics",
    )
    private_scores = models.JSONField(
        null=True,
        blank=True,
        verbose_name="All Private Scores",
        help_text="Stores private scores for all selected metrics",
    )

    # Final selection flag
    is_final_selection = models.BooleanField(
        default=False,
        verbose_name="Final Selection",
        help_text="Mark this as the final version for scoring",
    )

    # Error handling
    error_message = models.TextField(blank=True, verbose_name="Error Message")

    # Timestamps
    submitted_at = models.DateTimeField(auto_now_add=True, verbose_name="Submitted At")
    scored_at = models.DateTimeField(null=True, blank=True, verbose_name="Scored At")

    class Meta:
        verbose_name = "Submission"
        verbose_name_plural = "Submissions"
        ordering = ["-submitted_at"]
        indexes = [
            models.Index(fields=["competition", "user", "-submitted_at"]),
            models.Index(fields=["competition", "is_final_selection"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return f"Submission #{self.id} by {self.user}"

    @classmethod
    def get_today_count(cls, competition: "Competition", user: "User") -> int:
        """Get number of submissions by this user today."""
        today = timezone.now().date()
        return cls.objects.filter(
            competition=competition, user=user, submitted_at__date=today
        ).count()

    @classmethod
    def get_total_count(cls, competition: "Competition", user: "User") -> int:
        """Get total number of submissions by this user."""
        return cls.objects.filter(competition=competition, user=user).count()

    def can_submit_more_today(self) -> bool:
        """Check if user hasn't exceeded daily limit."""
        today_count = self.get_today_count(self.competition, self.user)
        return today_count < self.competition.daily_upload_limit

    def can_submit_more_total(self):
        """Check if user hasn't exceeded total limit."""
        total_count = self.get_total_count(self.competition, self.user)
        return total_count < self.competition.total_upload_limit


class SubmissionLog(models.Model):
    """
    Detailed log entries for a submission's scoring process.

    Useful for debugging scoring failures and tracking progress.
    """

    submission = models.ForeignKey(
        Submission,
        on_delete=models.CASCADE,
        related_name="logs",
        verbose_name="Submission",
    )
    level = models.CharField(
        max_length=10,
        choices=LogLevel.choices,
        default=LogLevel.INFO,
        verbose_name="Level",
    )
    message = models.TextField(verbose_name="Message")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")

    class Meta:
        verbose_name = "Scoring Log"
        verbose_name_plural = "Scoring Logs"
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.level}] {self.message[:50]}"
