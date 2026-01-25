from django.utils import timezone
from django_q.tasks import async_task
from typing import Any, Optional

from competitions.models import (
    Competition,
    CompetitionStatus,
    Submission,
    SubmissionLog,
    SubmissionStatus,
    LogLevel,
    TaskType,
    MetricType,
)
from .engines.base import BaseScoringEngine, ScoringResult
from .engines.classification import ClassificationScoringEngine
from .engines.detection import DetectionScoringEngine
from .engines.segmentation import SegmentationScoringEngine
from .engines.custom import CustomScoringEngine


def get_scoring_engine(
    competition: Competition, ground_truth_path: Optional[str] = None
) -> BaseScoringEngine:
    """
    Factory function to get the appropriate scoring engine for a competition.

    Args:
        competition: The Competition instance.
        ground_truth_path: Optional override for ground truth file path.

    Returns:
        An initialized scoring engine.

    Raises:
        ValueError: If task type is not supported.
    """
    if not ground_truth_path:
        ground_truth_path = competition.public_ground_truth.path

    metric_type = competition.metric_type

    if competition.metric_type == MetricType.CUSTOM:
        if not competition.scoring_script:
            raise ValueError("Competition set to CUSTOM metric but no scoring script uploaded.")
        return CustomScoringEngine(ground_truth_path, competition.scoring_script.path)

    if competition.task_type == TaskType.CLASSIFICATION:
        return ClassificationScoringEngine(ground_truth_path, metric_type)
    elif competition.task_type == TaskType.DETECTION:
        return DetectionScoringEngine(ground_truth_path, metric_type)
    elif competition.task_type == TaskType.SEGMENTATION:
        return SegmentationScoringEngine(ground_truth_path, metric_type)
    else:
        raise ValueError(f"Unknown task type: {competition.task_type}")


def add_submission_log(
    submission: Submission, message: str, level: str = LogLevel.INFO
) -> None:
    """Helper to add a log entry to a submission."""
    SubmissionLog.objects.create(submission=submission, level=level, message=message)


def score_submission(submission_id: int) -> dict[str, Any]:
    """
    Score a single submission.

    This is the main task that gets queued by Django-Q2.

    Args:
        submission_id: ID of the Submission to score.

    Returns:
        Dict with scoring result summary.
    """
    try:
        submission = Submission.objects.select_related("competition").get(
            id=submission_id
        )
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
        for log_msg in result.logs or []:
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
                submission, f"Scoring completed. Score: {result.score}", LogLevel.INFO
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
                submission, f"Scoring failed: {result.error_message}", LogLevel.ERROR
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


def score_private_submissions(competition_id: int) -> dict[str, Any]:
    """
    Score all final submissions for a competition using private ground truth.

    This is intended to be run manually or automatically when a competition ends.
    """
    try:
        competition = Competition.objects.get(id=competition_id)
    except Competition.DoesNotExist:
        return {"success": False, "error": f"Competition {competition_id} not found"}

    if not competition.private_ground_truth:
        return {
            "success": False,
            "error": "No private ground truth uploaded for this competition",
        }

    # Get all final selections
    final_submissions = Submission.objects.filter(
        competition=competition,
        is_final_selection=True,
        status=SubmissionStatus.SUCCESS,
    )

    # Initialize engine with private GT
    engine = get_scoring_engine(competition, competition.private_ground_truth.path)

    count = 0
    errors = []

    for submission in final_submissions:
        try:
            prediction_path = submission.prediction_file.path
            result = engine.score(prediction_path)

            if result.success:
                submission.private_score = result.score
                submission.save(update_fields=["private_score"])
                count += 1
                add_submission_log(
                    submission,
                    f"Private scoring completed. Score: {result.score}",
                    LogLevel.INFO,
                )
            else:
                errors.append(
                    f"Submission {submission.id} failed: {result.error_message}"
                )
                add_submission_log(
                    submission,
                    f"Private scoring failed: {result.error_message}",
                    LogLevel.ERROR,
                )
        except Exception as e:
            errors.append(f"Submission {submission.id} error: {e}")
            add_submission_log(
                submission, f"Private scoring unexpected error: {e}", LogLevel.ERROR
            )

    competition.private_scoring_completed = True
    competition.save(update_fields=["private_scoring_completed"])

    return {
        "success": len(errors) == 0,
        "processed_count": count,
        "error_count": len(errors),
        "errors": errors,
    }
