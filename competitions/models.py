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


class TaskType(models.TextChoices):
    """Supported CV task types."""
    CLASSIFICATION = 'CLASSIFICATION', '圖像分類'
    DETECTION = 'DETECTION', '物件偵測'
    SEGMENTATION = 'SEGMENTATION', '影像分割'


class MetricType(models.TextChoices):
    """Supported evaluation metrics."""
    ACCURACY = 'ACCURACY', 'Accuracy'
    F1 = 'F1', 'F1-Score'
    MAP = 'MAP', 'mAP@0.5'
    MAP_50_95 = 'MAP_50_95', 'mAP@[0.5:0.95]'
    MIOU = 'MIOU', 'mIoU'


class CompetitionStatus(models.TextChoices):
    """Competition lifecycle status."""
    DRAFT = 'DRAFT', '草稿'
    ACTIVE = 'ACTIVE', '進行中'
    ENDED = 'ENDED', '已結束'


class SubmissionStatus(models.TextChoices):
    """Submission processing status."""
    PENDING = 'PENDING', '等待中'
    PROCESSING = 'PROCESSING', '處理中'
    SUCCESS = 'SUCCESS', '成功'
    FAILED = 'FAILED', '失敗'


class LogLevel(models.TextChoices):
    """Log severity levels."""
    INFO = 'INFO', 'Info'
    WARNING = 'WARNING', 'Warning'
    ERROR = 'ERROR', 'Error'


def competition_ground_truth_path(instance, filename):
    """Generate upload path for ground truth files."""
    return f'competitions/{instance.id}/ground_truth/{filename}'


def submission_prediction_path(instance, filename):
    """Generate upload path for prediction files."""
    return f'submissions/{instance.competition_id}/{instance.user_id}/{filename}'


class Competition(models.Model):
    """
    A CV competition created by an admin.
    
    Stores competition metadata, ground truth files, and upload limits.
    """
    name = models.CharField(
        max_length=100,
        verbose_name='競賽名稱',
        help_text='例如：瑕疵偵測挑戰賽'
    )
    description = models.TextField(
        verbose_name='競賽說明',
        help_text='支援 Markdown 格式',
        blank=True
    )
    task_type = models.CharField(
        max_length=20,
        choices=TaskType.choices,
        verbose_name='任務類型'
    )
    metric_type = models.CharField(
        max_length=20,
        choices=MetricType.choices,
        verbose_name='評分指標'
    )
    
    # Ground truth files
    public_ground_truth = models.FileField(
        upload_to=competition_ground_truth_path,
        verbose_name='Public Set 標準答案',
        help_text='CSV 格式'
    )
    private_ground_truth = models.FileField(
        upload_to=competition_ground_truth_path,
        verbose_name='Private Set 標準答案',
        help_text='CSV 格式，僅 Validator 可見',
        blank=True,
        null=True
    )
    
    # Dataset access
    dataset_url = models.URLField(
        verbose_name='資料集下載連結',
        blank=True
    )
    
    # Upload limits
    daily_upload_limit = models.PositiveIntegerField(
        default=5,
        verbose_name='每日上傳上限'
    )
    total_upload_limit = models.PositiveIntegerField(
        default=100,
        verbose_name='總上傳上限'
    )
    
    # Status
    status = models.CharField(
        max_length=20,
        choices=CompetitionStatus.choices,
        default=CompetitionStatus.DRAFT,
        verbose_name='狀態'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='建立時間')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新時間')

    class Meta:
        verbose_name = '競賽'
        verbose_name_plural = '競賽'
        ordering = ['-created_at']

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
        related_name='participants',
        verbose_name='競賽'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='competition_participations',
        verbose_name='參賽者'
    )
    
    # Per-user time window
    start_time = models.DateTimeField(verbose_name='開始時間')
    end_time = models.DateTimeField(verbose_name='結束時間')
    
    # Manual control
    is_active = models.BooleanField(
        default=True,
        verbose_name='啟用',
        help_text='取消勾選可暫停該使用者的參賽權限'
    )
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='建立時間')

    class Meta:
        verbose_name = '參賽白名單'
        verbose_name_plural = '參賽白名單'
        unique_together = ['competition', 'user']
        indexes = [
            models.Index(fields=['competition', 'user']),
            models.Index(fields=['start_time', 'end_time']),
        ]

    def __str__(self):
        return f'{self.user} @ {self.competition}'

    def is_within_time_window(self):
        """Check if current time is within the participation window."""
        now = timezone.now()
        return self.start_time <= now <= self.end_time

    def can_participate(self):
        """Check if user can currently participate (active + within time)."""
        return self.is_active and self.is_within_time_window()


class Submission(models.Model):
    """
    A prediction file submission from a participant.
    
    Tracks the upload, scoring status, and both public/private scores.
    """
    competition = models.ForeignKey(
        Competition,
        on_delete=models.CASCADE,
        related_name='submissions',
        verbose_name='競賽'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='submissions',
        verbose_name='提交者'
    )
    
    # Uploaded file
    prediction_file = models.FileField(
        upload_to=submission_prediction_path,
        verbose_name='預測檔案'
    )
    
    # Processing status
    status = models.CharField(
        max_length=20,
        choices=SubmissionStatus.choices,
        default=SubmissionStatus.PENDING,
        verbose_name='狀態'
    )
    
    # Scores
    public_score = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Public Score'
    )
    private_score = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Private Score',
        help_text='由 Validator 填入'
    )
    
    # Final selection flag
    is_final_selection = models.BooleanField(
        default=False,
        verbose_name='最終決定版',
        help_text='勾選此項作為最終評分版本'
    )
    
    # Error handling
    error_message = models.TextField(
        blank=True,
        verbose_name='錯誤訊息'
    )
    
    # Timestamps
    submitted_at = models.DateTimeField(auto_now_add=True, verbose_name='提交時間')
    scored_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name='算分完成時間'
    )

    class Meta:
        verbose_name = '提交紀錄'
        verbose_name_plural = '提交紀錄'
        ordering = ['-submitted_at']
        indexes = [
            models.Index(fields=['competition', 'user', '-submitted_at']),
            models.Index(fields=['competition', 'is_final_selection']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f'Submission #{self.id} by {self.user}'

    @classmethod
    def get_today_count(cls, competition, user):
        """Get number of submissions by this user today."""
        today = timezone.now().date()
        return cls.objects.filter(
            competition=competition,
            user=user,
            submitted_at__date=today
        ).count()

    @classmethod
    def get_total_count(cls, competition, user):
        """Get total number of submissions by this user."""
        return cls.objects.filter(
            competition=competition,
            user=user
        ).count()

    def can_submit_more_today(self):
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
        related_name='logs',
        verbose_name='提交紀錄'
    )
    level = models.CharField(
        max_length=10,
        choices=LogLevel.choices,
        default=LogLevel.INFO,
        verbose_name='等級'
    )
    message = models.TextField(verbose_name='訊息')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='時間')

    class Meta:
        verbose_name = '算分日誌'
        verbose_name_plural = '算分日誌'
        ordering = ['created_at']

    def __str__(self):
        return f'[{self.level}] {self.message[:50]}'
