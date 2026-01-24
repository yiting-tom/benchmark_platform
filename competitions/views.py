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

from .models import (
    Competition,
    CompetitionParticipant,
    Submission,
    SubmissionStatus,
    CompetitionStatus,
)


class CompetitionListView(LoginRequiredMixin, View):
    """List competitions the current user can participate in."""
    
    def get(self, request):
        now = timezone.now()
        
        # Get all participations for this user
        participations = CompetitionParticipant.objects.filter(
            user=request.user,
            is_active=True,
        ).select_related('competition')
        
        competitions = []
        for p in participations:
            # Check if within time window
            is_active = p.start_time <= now <= p.end_time and p.competition.status == CompetitionStatus.ACTIVE
            
            # Get upload counts
            today_count = Submission.get_today_count(p.competition, request.user)
            total_count = Submission.get_total_count(p.competition, request.user)
            
            competitions.append({
                'competition': p.competition,
                'participant': p,
                'is_active': is_active,
                'today_count': today_count,
                'total_count': total_count,
            })
        
        return render(request, 'competitions/list.html', {
            'competitions': competitions
        })


class CompetitionDetailView(LoginRequiredMixin, View):
    """Competition detail page with upload form."""
    
    def get(self, request, competition_id):
        competition = get_object_or_404(Competition, id=competition_id)
        
        # Check if user is a participant
        participant = get_object_or_404(
            CompetitionParticipant,
            competition=competition,
            user=request.user,
            is_active=True
        )
        
        now = timezone.now()
        can_upload = True
        upload_error = None
        
        # Check time window
        if now < participant.start_time:
            can_upload = False
            upload_error = f"競賽尚未開始（開始時間：{participant.start_time.strftime('%Y-%m-%d %H:%M')}）"
        elif now > participant.end_time:
            can_upload = False
            upload_error = "您的參賽時間已結束"
        
        # Check competition status
        if competition.status != CompetitionStatus.ACTIVE:
            can_upload = False
            upload_error = "競賽已結束"
        
        # Check upload limits
        today_count = Submission.get_today_count(competition, request.user)
        total_count = Submission.get_total_count(competition, request.user)
        
        if today_count >= competition.daily_upload_limit:
            can_upload = False
            upload_error = "已達今日上傳上限"
        
        if total_count >= competition.total_upload_limit:
            can_upload = False
            upload_error = "已達總上傳上限"
        
        # Detect expected columns from ground truth file
        expected_format = self._get_expected_format(competition)
        
        return render(request, 'competitions/detail.html', {
            'competition': competition,
            'participant': participant,
            'can_upload': can_upload,
            'upload_error': upload_error,
            'today_count': today_count,
            'total_count': total_count,
            'expected_format': expected_format,
        })
    
    def _get_expected_format(self, competition):
        """Detect expected CSV format from ground truth file."""
        import pandas as pd
        from .models import TaskType
        
        try:
            if competition.public_ground_truth:
                df = pd.read_csv(competition.public_ground_truth.path, nrows=0)
                columns = list(df.columns)
                
                if competition.task_type == TaskType.DETECTION:
                    # Detection needs additional 'confidence' column
                    return ", ".join(columns[:2]) + ", confidence, " + ", ".join(columns[2:])
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


@login_required
@require_POST
def upload_prediction(request, competition_id):
    """Handle prediction file upload (HTMX endpoint)."""
    competition = get_object_or_404(Competition, id=competition_id)
    
    # Verify participant
    participant = get_object_or_404(
        CompetitionParticipant,
        competition=competition,
        user=request.user,
        is_active=True
    )
    
    # Check limits
    today_count = Submission.get_today_count(competition, request.user)
    total_count = Submission.get_total_count(competition, request.user)
    
    if today_count >= competition.daily_upload_limit:
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': '已達今日上傳上限'
        })
    
    if total_count >= competition.total_upload_limit:
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': '已達總上傳上限'
        })
    
    # Check time window
    now = timezone.now()
    if not (participant.start_time <= now <= participant.end_time):
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': '不在參賽時間內'
        })
    
    # Get uploaded file
    prediction_file = request.FILES.get('prediction_file')
    if not prediction_file:
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': '請選擇檔案'
        })
    
    if not prediction_file.name.endswith('.csv'):
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': '請上傳 CSV 格式檔案'
        })
    
    # Create submission
    submission = Submission.objects.create(
        competition=competition,
        user=request.user,
        prediction_file=prediction_file,
        status=SubmissionStatus.PENDING
    )
    
    # Queue scoring task
    async_task('scoring.tasks.score_submission', submission.id)
    
    # Return success response
    return render(request, 'competitions/partials/upload_result.html', {
        'success': True,
        'submission_id': submission.id,
        'today_count': today_count + 1,
        'total_count': total_count + 1,
    })


@login_required
def submission_history(request, competition_id):
    """Get submission history for current user (HTMX endpoint)."""
    competition = get_object_or_404(Competition, id=competition_id)
    
    submissions = Submission.objects.filter(
        competition=competition,
        user=request.user
    ).order_by('-submitted_at')[:50]
    
    return render(request, 'competitions/partials/history.html', {
        'submissions': submissions
    })


@login_required
@require_POST
def set_final_selection(request, submission_id):
    """Set a submission as the final selection (HTMX endpoint)."""
    submission = get_object_or_404(
        Submission,
        id=submission_id,
        user=request.user,
        status=SubmissionStatus.SUCCESS
    )
    
    # Clear previous final selection for this competition
    Submission.objects.filter(
        competition=submission.competition,
        user=request.user,
        is_final_selection=True
    ).update(is_final_selection=False)
    
    # Set new final selection
    submission.is_final_selection = True
    submission.save(update_fields=['is_final_selection'])
    
    # Return updated row
    return render(request, 'competitions/partials/history_row.html', {
        'submission': submission
    })


@login_required
def leaderboard(request, competition_id):
    """Get competition leaderboard (HTMX endpoint)."""
    competition = get_object_or_404(Competition, id=competition_id)
    
    # Determine which score to use
    show_private = competition.status == CompetitionStatus.ENDED
    score_field = 'private_score' if show_private else 'public_score'
    
    # Get best score per user
    # For public: use user's best public score
    # For private: use their final selection's private score
    
    if show_private:
        # Use final selections only
        submissions = Submission.objects.filter(
            competition=competition,
            is_final_selection=True,
            private_score__isnull=False
        ).select_related('user')
        
        leaderboard_data = []
        for s in submissions:
            leaderboard_data.append({
                'user_id': s.user_id,
                'username': s.user.username,
                'score': s.private_score,
                'submission_count': Submission.get_total_count(competition, s.user),
                'last_submission': s.submitted_at,
            })
    else:
        # Use best public score per user
        from django.db.models import Max
        
        user_best = Submission.objects.filter(
            competition=competition,
            status=SubmissionStatus.SUCCESS,
            public_score__isnull=False
        ).values('user_id', 'user__username').annotate(
            score=Max('public_score'),
            submission_count=Count('id'),
            last_submission=Max('submitted_at')
        ).order_by('-score')
        
        leaderboard_data = [
            {
                'user_id': entry['user_id'],
                'username': entry['user__username'],
                'score': entry['score'],
                'submission_count': entry['submission_count'],
                'last_submission': entry['last_submission'],
            }
            for entry in user_best
        ]
    
    # Sort by score (descending) and add ranks
    leaderboard_data.sort(key=lambda x: x['score'] or 0, reverse=True)
    
    for i, entry in enumerate(leaderboard_data, 1):
        entry['rank'] = i
        entry['is_current_user'] = entry['user_id'] == request.user.id
    
    return render(request, 'competitions/partials/leaderboard.html', {
        'leaderboard': leaderboard_data,
        'show_private': show_private,
    })


@login_required
def leaderboard_chart_data(request, competition_id):
    """Return JSON data for leaderboard charts."""
    from collections import defaultdict
    
    competition = get_object_or_404(Competition, id=competition_id)
    show_private = competition.status == CompetitionStatus.ENDED
    score_field = 'private_score' if show_private else 'public_score'
    
    # Get all successful submissions with scores
    submissions = Submission.objects.filter(
        competition=competition,
        status=SubmissionStatus.SUCCESS,
        **{f'{score_field}__isnull': False}
    ).select_related('user').order_by('submitted_at')
    
    # --- Score Trend Data ---
    # Track best score per user over time
    user_best_scores = defaultdict(lambda: {'scores': [], 'timestamps': []})
    user_running_best = {}
    
    for s in submissions:
        username = s.user.username
        score = getattr(s, score_field)
        timestamp = s.submitted_at.isoformat()
        
        # Update running best
        if username not in user_running_best or score > user_running_best[username]:
            user_running_best[username] = score
            user_best_scores[username]['scores'].append(score)
            user_best_scores[username]['timestamps'].append(timestamp)
    
    # Build datasets for Chart.js
    colors = [
        'rgb(59, 130, 246)',   # blue
        'rgb(239, 68, 68)',    # red
        'rgb(34, 197, 94)',    # green
        'rgb(168, 85, 247)',   # purple
        'rgb(249, 115, 22)',   # orange
        'rgb(236, 72, 153)',   # pink
        'rgb(20, 184, 166)',   # teal
        'rgb(245, 158, 11)',   # amber
    ]
    
    trend_datasets = []
    for i, (username, data) in enumerate(user_best_scores.items()):
        color = colors[i % len(colors)]
        trend_datasets.append({
            'label': username,
            'data': [{'x': t, 'y': s} for t, s in zip(data['timestamps'], data['scores'])],
            'borderColor': color,
            'backgroundColor': color,
            'tension': 0.3,
            'fill': False,
        })
    
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
            count = sum(1 for s in final_scores if bin_start <= s < bin_end or (i == 4 and s == bin_end))
            counts.append(count)
    else:
        bins = []
        counts = []
    
    return JsonResponse({
        'trend': {
            'datasets': trend_datasets,
        },
        'distribution': {
            'labels': bins,
            'data': counts,
        }
    })

