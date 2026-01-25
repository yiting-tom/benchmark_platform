"""
Views for the competitions app.

Handles participant-facing pages:
- Competition list (dashboard)
- Competition detail with upload
- HTMX partials for history, leaderboard
"""

from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count, Max
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views import View
from django.views.decorators.http import require_POST
from django_q.tasks import async_task
from typing import Any, Dict, List, Optional, Union
from django.http import HttpRequest

from .models import (
    Competition,
    CompetitionParticipant,
    Submission,
    SubmissionStatus,
    CompetitionStatus,
)


class CompetitionListView(LoginRequiredMixin, View):
    """List competitions the current user can participate in."""

    def get(self, request: HttpRequest) -> HttpResponse:
        now = timezone.now()

        # Get all participations for this user
        participations = CompetitionParticipant.objects.filter(
            user=request.user,
            is_active=True,
        ).select_related("competition")

        competitions: List[Dict[str, Any]] = []
        for p in participations:
            # Check if within time window
            is_active = (
                p.start_time <= now <= p.end_time
                and p.competition.status == CompetitionStatus.ACTIVE
            )

            # Get upload counts
            user = request.user
            if user.is_anonymous:
                continue

            today_count = Submission.get_today_count(p.competition, user)  # type: ignore
            total_count = Submission.get_total_count(p.competition, user)  # type: ignore

            competitions.append(
                {
                    "competition": p.competition,
                    "participant": p,
                    "is_active": is_active,
                    "today_count": today_count,
                    "total_count": total_count,
                }
            )

        return render(request, "competitions/list.html", {"competitions": competitions})


class CompetitionDetailView(LoginRequiredMixin, View):
    """Competition detail page with upload form."""

    def get(self, request: HttpRequest, competition_id: int) -> HttpResponse:
        competition = get_object_or_404(Competition, id=competition_id)

        # Check if user is a participant
        participant = get_object_or_404(
            CompetitionParticipant,
            competition=competition,
            user=request.user,
            is_active=True,
        )

        now = timezone.now()
        can_upload = True
        upload_error = None

        # Check time window
        if now < participant.start_time:
            can_upload = False
            upload_error = f"Competition has not started yet (Start Time: {participant.start_time.strftime('%Y-%m-%d %H:%M')})"
        elif now > participant.end_time:
            can_upload = False
            upload_error = "Your participation time has ended"

        # Check competition status
        if competition.status != CompetitionStatus.ACTIVE:
            can_upload = False
            upload_error = "Competition has ended"

        # Check upload limits
        user = request.user
        if user.is_anonymous:
            return HttpResponse("Unauthorized", status=401)

        today_count = Submission.get_today_count(competition, user)  # type: ignore
        total_count = Submission.get_total_count(competition, user)  # type: ignore

        if today_count >= competition.daily_upload_limit:
            can_upload = False
            upload_error = "Daily upload limit reached"

        if total_count >= competition.total_upload_limit:
            can_upload = False
            upload_error = "Total upload limit reached"

        # Detect expected columns from ground truth file
        expected_format = self._get_expected_format(competition)

        return render(
            request,
            "competitions/detail.html",
            {
                "competition": competition,
                "participant": participant,
                "can_upload": can_upload,
                "upload_error": upload_error,
                "today_count": today_count,
                "total_count": total_count,
                "expected_format": expected_format,
            },
        )

    def _get_expected_format(self, competition: Competition) -> str:
        """Detect expected CSV format from ground truth file."""
        import pandas as pd
        from .models import TaskType

        try:
            if competition.public_ground_truth:
                df = pd.read_csv(competition.public_ground_truth.path, nrows=0)
                columns = list(df.columns)

                if competition.task_type == TaskType.DETECTION:
                    # Detection needs additional 'confidence' column
                    return (
                        ", ".join(columns[:2])
                        + ", confidence, "
                        + ", ".join(columns[2:])
                    )
                elif competition.task_type == TaskType.SEGMENTATION:
                    # Segmentation prediction doesn't need height/width
                    return ", ".join(columns[:3])
                else:
                    return ", ".join(columns)
        except Exception:
            pass

        # Fallback to default hints
        if competition.task_type == TaskType.CLASSIFICATION:
            return "id_column, label_column"
        elif competition.task_type == TaskType.DETECTION:
            return "id, class, confidence, xmin, ymin, xmax, ymax"
        else:
            return "id, class, rle_mask"


@login_required  # type: ignore
@require_POST  # type: ignore
def upload_prediction(request: HttpRequest, competition_id: int) -> HttpResponse:
    """Handle prediction file upload (HTMX endpoint)."""
    competition = get_object_or_404(Competition, id=competition_id)

    # Verify participant
    participant = get_object_or_404(
        CompetitionParticipant,
        competition=competition,
        user=request.user,
        is_active=True,
    )

    # Check limits
    # Cast request.user or ensure it's a User object for the method call
    user = request.user
    if user.is_anonymous:
        return HttpResponse("Unauthorized", status=401)

    today_count = Submission.get_today_count(competition, user)  # type: ignore
    total_count = Submission.get_total_count(competition, user)  # type: ignore

    if today_count >= competition.daily_upload_limit:
        return render(
            request,
            "competitions/partials/upload_result.html",
            {"success": False, "error": "Daily upload limit reached"},
        )

    if total_count >= competition.total_upload_limit:
        return render(
            request,
            "competitions/partials/upload_result.html",
            {"success": False, "error": "Total upload limit reached"},
        )

    # Check time window
    now = timezone.now()
    if not (participant.start_time <= now <= participant.end_time):
        return render(
            request,
            "competitions/partials/upload_result.html",
            {"success": False, "error": "Not within participation time window"},
        )

    # Get uploaded file
    prediction_file = request.FILES.get("prediction_file")
    if not prediction_file:
        return render(
            request,
            "competitions/partials/upload_result.html",
            {"success": False, "error": "Please select a file"},
        )

    prediction_file_name: str = prediction_file.name or "unknown.csv"
    if not prediction_file_name.endswith(".csv"):
        return render(
            request,
            "competitions/partials/upload_result.html",
            {"success": False, "error": "Please upload a CSV file"},
        )

    # Create submission
    submission = Submission.objects.create(
        competition=competition,
        user=request.user,
        prediction_file=prediction_file,
        status=SubmissionStatus.PENDING,
    )

    # Pre-validation (Quick header check)
    import pandas as pd

    try:
        df = pd.read_csv(submission.prediction_file.path, nrows=0)
        expected_cols = _get_expected_columns(competition)
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            submission.status = SubmissionStatus.FAILED
            submission.error_message = f"Missing columns: {', '.join(missing_cols)}"
            submission.save(update_fields=["status", "error_message"])
            from .models import LogLevel, SubmissionLog

            SubmissionLog.objects.create(
                submission=submission,
                level=LogLevel.ERROR,
                message=f"CSV Validation failed: Missing columns {missing_cols}",
            )
            return render(
                request,
                "competitions/partials/upload_result.html",
                {
                    "success": False,
                    "error": f"Invalid CSV format: Missing columns {missing_cols}",
                },
            )
    except Exception as e:
        submission.status = SubmissionStatus.FAILED
        submission.error_message = f"Invalid CSV file: {e}"
        submission.save(update_fields=["status", "error_message"])
        return render(
            request,
            "competitions/partials/upload_result.html",
            {"success": False, "error": f"Invalid CSV file: {e}"},
        )

    # Queue scoring task
    async_task("scoring.tasks.score_submission", submission.id)

    # Return success response
    return render(
        request,
        "competitions/partials/upload_result.html",
        {
            "success": True,
            "submission_id": submission.id,
            "today_count": today_count + 1,
            "total_count": total_count + 1,
        },
    )


def _get_expected_columns(competition: Competition) -> List[str]:
    """Helper to get expected columns for a task type."""
    from .models import TaskType

    if competition.task_type == TaskType.CLASSIFICATION:
        return ["image_id", "label"]
    elif competition.task_type == TaskType.DETECTION:
        return ["image_id", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"]
    elif competition.task_type == TaskType.SEGMENTATION:
        return ["image_id", "class_id", "rle"]
    return []


@login_required
def submission_logs(request: HttpRequest, submission_id: int) -> HttpResponse:
    """Get submission logs (HTMX endpoint)."""
    submission = get_object_or_404(Submission, id=submission_id, user=request.user)

    # submission.logs is a related manager. Casting to Any to satisfy the type checker
    # if it doesn't recognize the reverse relation.
    logs = getattr(submission, "logs").all().order_by("created_at")

    return render(
        request,
        "competitions/partials/logs_modal.html",
        {"submission": submission, "logs": logs},
    )


@login_required
def submission_history(request: HttpRequest, competition_id: int) -> HttpResponse:
    """Get submission history for current user (HTMX endpoint)."""
    competition = get_object_or_404(Competition, id=competition_id)

    submissions = Submission.objects.filter(
        competition=competition, user=request.user
    ).order_by("-submitted_at")[:50]

    return render(
        request, "competitions/partials/history.html", {"submissions": submissions}
    )


@login_required  # type: ignore
@require_POST  # type: ignore
def set_final_selection(request: HttpRequest, submission_id: int) -> HttpResponse:
    """Set a submission as the final selection (HTMX endpoint)."""
    submission = get_object_or_404(
        Submission, id=submission_id, user=request.user, status=SubmissionStatus.SUCCESS
    )

    # Clear previous final selection for this competition
    Submission.objects.filter(
        competition=submission.competition, user=request.user, is_final_selection=True
    ).update(is_final_selection=False)

    # Set new final selection
    submission.is_final_selection = True
    submission.save(update_fields=["is_final_selection"])

    # Return updated row
    return render(
        request, "competitions/partials/history_row.html", {"submission": submission}
    )


@login_required
def leaderboard(request: HttpRequest, competition_id: int) -> HttpResponse:
    """Get competition leaderboard (HTMX endpoint)."""
    competition = get_object_or_404(Competition, id=competition_id)

    # Determine which score to use
    show_private = competition.status == CompetitionStatus.ENDED
    score_field = "private_score" if show_private else "public_score"

    # Determine which metrics to show
    from .models import MetricType

    metric_choices = dict(MetricType.choices)
    metrics_info = [
        (competition.metric_type, metric_choices.get(competition.metric_type, competition.metric_type))
    ]
    for m in competition.additional_metrics.all():
        if m.name != competition.metric_type:
            metrics_info.append((m.name, metric_choices.get(m.name, m.name)))

    # Get best score per user
    # For public: use user's best public score
    # For private: use their final selection's private score
    leaderboard_data: List[Dict[str, Any]] = []

    # Get all users who have submitted
    from django.db.models import Max, Q

    # Get users with any successful submission
    users_with_submissions = (
        Submission.objects.filter(competition=competition, status=SubmissionStatus.SUCCESS)
        .values_list("user_id", flat=True)
        .distinct()
    )

    leaderboard_data = []
    for user_id in users_with_submissions:
        # Get final selection or best public submission
        final_s = Submission.objects.filter(
            competition=competition, user_id=user_id, is_final_selection=True
        ).first()

        best_public_s = (
            Submission.objects.filter(
                competition=competition,
                user_id=user_id,
                status=SubmissionStatus.SUCCESS,
                public_score__isnull=False,
            )
            .order_by("-public_score", "-submitted_at")
            .first()
        )

        # Decide which submission to use for which score type
        # For public scores: always use best_public_s
        # For private scores: use final_s if it exists and has private_score
        
        public_display_scores = []
        if best_public_s:
            username = best_public_s.user.username
            for m_code, _ in metrics_info:
                if m_code == competition.metric_type:
                    public_display_scores.append(best_public_s.public_score)
                else:
                    public_display_scores.append((best_public_s.scores or {}).get(m_code))
        else:
            continue # Should not happen given users_with_submissions filter

        private_display_scores = []
        private_score = None
        if final_s and final_s.private_score is not None:
            private_score = final_s.private_score
            for m_code, _ in metrics_info:
                if m_code == competition.metric_type:
                    private_display_scores.append(final_s.private_score)
                else:
                    private_display_scores.append((final_s.private_scores or {}).get(m_code))
        else:
            # Fill with None if no private score available
            private_display_scores = [None] * len(metrics_info)

        leaderboard_data.append(
            {
                "user_id": user_id,
                "username": username,
                "public_score": best_public_s.public_score,
                "private_score": private_score,
                "public_display_scores": public_display_scores,
                "private_display_scores": private_display_scores,
                "submission_count": Submission.get_total_count(competition, best_public_s.user),
                "last_submission": best_public_s.submitted_at,
            }
        )

    # Sort leaderboard
    # If competition ended, sort by private score (fall back to public if private is None)
    if show_private:
        leaderboard_data.sort(
            key=lambda x: (x["private_score"] is not None, x["private_score"] or -1, x["public_score"]),
            reverse=True,
        )
    else:
        leaderboard_data.sort(key=lambda x: x["public_score"], reverse=True)

    # Sort already performed above based on show_private status

    for i, entry in enumerate(leaderboard_data, 1):
        entry["rank"] = i
        # Use getattr to satisfy type checker for AnonymousUser/User id
        current_user_id = getattr(request.user, "id", None)
        entry["is_current_user"] = entry["user_id"] == current_user_id

    return render(
        request,
        "competitions/partials/leaderboard.html",
        {
            "competition": competition,
            "leaderboard": leaderboard_data,
            "show_private": show_private,
            "metrics_info": metrics_info,
        },
    )


@login_required
def leaderboard_chart_data(request: HttpRequest, competition_id: int) -> JsonResponse:
    """Return JSON data for leaderboard charts."""
    from collections import defaultdict

    competition = get_object_or_404(Competition, id=competition_id)
    show_private = competition.status == CompetitionStatus.ENDED
    score_field = "private_score" if show_private else "public_score"

    # Get all successful submissions with scores
    submissions = (
        Submission.objects.filter(
            competition=competition,
            status=SubmissionStatus.SUCCESS,
            **{f"{score_field}__isnull": False},
        )
        .select_related("user")
        .order_by("submitted_at")
    )

    # --- Score Trend Data ---
    # Track best score per user over time
    user_best_scores = defaultdict(lambda: {"scores": [], "timestamps": []})
    user_running_best = {}

    for s in submissions:
        username = s.user.username
        score = getattr(s, score_field)
        timestamp = s.submitted_at.isoformat()

        # Update running best
        if username not in user_running_best or score > user_running_best[username]:
            user_running_best[username] = score
            user_best_scores[username]["scores"].append(score)
            user_best_scores[username]["timestamps"].append(timestamp)

    # Build datasets for Chart.js
    colors = [
        "rgb(59, 130, 246)",  # blue
        "rgb(239, 68, 68)",  # red
        "rgb(34, 197, 94)",  # green
        "rgb(168, 85, 247)",  # purple
        "rgb(249, 115, 22)",  # orange
        "rgb(236, 72, 153)",  # pink
        "rgb(20, 184, 166)",  # teal
        "rgb(245, 158, 11)",  # amber
    ]

    trend_datasets = []
    for i, (username, data) in enumerate(user_best_scores.items()):
        color = colors[i % len(colors)]
        trend_datasets.append(
            {
                "label": username,
                "data": [
                    {"x": t, "y": s} for t, s in zip(data["timestamps"], data["scores"])
                ],
                "borderColor": color,
                "backgroundColor": color,
                "tension": 0.3,
                "fill": False,
            }
        )

    # --- Score Distribution Data ---
    # Get final best scores per user
    final_scores = list(user_running_best.values())

    if final_scores:
        min_score = min(final_scores)
        max_score = max(final_scores)
        range_size = (max_score - min_score) / 5 if max_score > min_score else 0.2

        # Create 5 bins
        bins = []
        counts = []
        for i in range(5):
            bin_start = min_score + i * range_size
            bin_end = min_score + (i + 1) * range_size
            bins.append(f"{bin_start:.2f}-{bin_end:.2f}")
            count = sum(
                1
                for s in final_scores
                if bin_start <= s < bin_end or (i == 4 and s == bin_end)
            )
            counts.append(count)
    else:
        bins = []
        counts = []

    return JsonResponse(
        {
            "trend": {
                "datasets": trend_datasets,
            },
            "distribution": {
                "labels": bins,
                "data": counts,
            },
        }
    )
