from django.utils import timezone

from competitions.models import (
    Competition,
    Submission,
    SubmissionLog,
    SubmissionStatus,
    LogLevel,
    TaskType,
    MetricType,
)
from .engines.classification import ClassificationScoringEngine
from .engines.detection import DetectionScoringEngine
from .engines.segmentation import SegmentationScoringEngine


def get_scoring_engine(competition: Competition):
    """
    Factory function to get the appropriate scoring engine for a competition.
    
    Args:
        competition: The Competition instance.
        
    Returns:
        An initialized scoring engine.
        
    Raises:
        ValueError: If task type is not supported.
    """
    ground_truth_path = competition.public_ground_truth.path
    metric_type = competition.metric_type
    
    if competition.task_type == TaskType.CLASSIFICATION:
        return ClassificationScoringEngine(ground_truth_path, metric_type)
    elif competition.task_type == TaskType.DETECTION:
        return DetectionScoringEngine(ground_truth_path, metric_type)
    elif competition.task_type == TaskType.SEGMENTATION:
        return SegmentationScoringEngine(ground_truth_path, metric_type)
    else:
        raise ValueError(f"Unknown task type: {competition.task_type}")


def add_submission_log(submission: Submission, message: str, level: str = LogLevel.INFO):
    """Helper to add a log entry to a submission."""
    SubmissionLog.objects.create(
        submission=submission,
        level=level,
        message=message
    )


def score_submission(submission_id: int) -> dict:
    """
    Score a single submission.
    
    This is the main task that gets queued by Django-Q2.
    
    Args:
        submission_id: ID of the Submission to score.
        
    Returns:
        Dict with scoring result summary.
    """
    try:
        submission = Submission.objects.select_related("competition").get(id=submission_id)
    except Submission.DoesNotExist:
        return {"success": False, "error": f"Submission {submission_id} not found"}
    
    # Update status to PROCESSING
    submission.status = SubmissionStatus.PROCESSING
    submission.save(update_fields=["status"])
    add_submission_log(submission, "Started scoring", LogLevel.INFO)
    
    try:
        # Get the appropriate scoring engine
        competition = submission.competition
        engine = get_scoring_engine(competition)
        
        # Run scoring
        prediction_path = submission.prediction_file.path
        result = engine.score(prediction_path)
        
        # Save logs from engine
        for log_msg in (result.logs or []):
            # Parse log level from message format "[LEVEL] message"
            if log_msg.startswith("[ERROR]"):
                level = LogLevel.ERROR
            elif log_msg.startswith("[WARNING]"):
                level = LogLevel.WARNING
            else:
                level = LogLevel.INFO
            add_submission_log(submission, log_msg, level)
        
        if result.success:
            submission.status = SubmissionStatus.SUCCESS
            submission.public_score = result.score
            submission.scored_at = timezone.now()
            submission.save(update_fields=["status", "public_score", "scored_at"])
            
            add_submission_log(
                submission, 
                f"Scoring completed. Score: {result.score}", 
                LogLevel.INFO
            )
            
            return {
                "success": True,
                "submission_id": submission_id,
                "score": result.score,
                "metrics": result.metrics,
            }
        else:
            submission.status = SubmissionStatus.FAILED
            submission.error_message = result.error_message or "Unknown error"
            submission.save(update_fields=["status", "error_message"])
            
            add_submission_log(
                submission,
                f"Scoring failed: {result.error_message}",
                LogLevel.ERROR
            )
            
            return {
                "success": False,
                "submission_id": submission_id,
                "error": result.error_message,
            }
            
    except NotImplementedError as e:
        submission.status = SubmissionStatus.FAILED
        submission.error_message = str(e)
        submission.save(update_fields=["status", "error_message"])
        add_submission_log(submission, str(e), LogLevel.ERROR)
        
        return {"success": False, "submission_id": submission_id, "error": str(e)}
        
    except Exception as e:
        submission.status = SubmissionStatus.FAILED
        submission.error_message = f"Unexpected error: {e}"
        submission.save(update_fields=["status", "error_message"])
        add_submission_log(submission, f"Unexpected error: {e}", LogLevel.ERROR)
        
        return {"success": False, "submission_id": submission_id, "error": str(e)}
