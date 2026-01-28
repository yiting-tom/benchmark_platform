"""
Views for the competitions app.

Handles participant-facing pages:
- Competition list (dashboard)
- Competition detail with upload
- HTMX partials for history, leaderboard
"""

from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render, redirect
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
    MetricType,
    RegistrationWhitelist,
)
from .utils import (
    get_leaderboard_data,
    get_score_trend_data,
    get_score_distribution_data,
    get_expected_format_hint,
)


class CompetitionListView(LoginRequiredMixin, View):
    """List competitions the current user can participate in."""
    
    def get(self, request: HttpRequest) -> HttpResponse:
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
    
    def get(self, request: HttpRequest, competition_id: int) -> HttpResponse:
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
            upload_error = f"Competition has not started yet (Start Time: {participant.start_time.strftime('%Y-%m-%d %H:%M')})"
        elif now > participant.end_time:
            can_upload = False
            upload_error = "Your participation time has ended"
        
        # Check competition status
        if competition.status != CompetitionStatus.ACTIVE:
            can_upload = False
            upload_error = "Competition has ended"
        
        # Check upload limits
        today_count = Submission.get_today_count(competition, request.user)
        total_count = Submission.get_total_count(competition, request.user)
        
        if today_count >= competition.daily_upload_limit:
            can_upload = False
            upload_error = "Daily upload limit reached"
        
        if total_count >= competition.total_upload_limit:
            can_upload = False
            upload_error = "Total upload limit reached"
        
        # Detect expected columns from ground truth file
        expected_format = get_expected_format_hint(competition)
        
        return render(request, 'competitions/detail.html', {
            'competition': competition,
            'participant': participant,
            'can_upload': can_upload,
            'upload_error': upload_error,
            'today_count': today_count,
            'total_count': total_count,
            'expected_format': expected_format,
        })
    



@login_required
@require_POST
def upload_prediction(request: HttpRequest, competition_id: int) -> HttpResponse:
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
    if not Submission(competition=competition, user=request.user).can_submit_more_today():
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': 'Daily upload limit reached'
        })
    
    if not Submission(competition=competition, user=request.user).can_submit_more_total():
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': 'Total upload limit reached'
        })
    
    # Check time window
    now = timezone.now()
    if not (participant.start_time <= now <= participant.end_time):
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': 'Not within participation time window'
        })
    
    # Get uploaded file
    prediction_file = request.FILES.get('prediction_file')
    if not prediction_file:
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': 'Please select a file'
        })
    
    if not prediction_file.name.endswith('.csv'):
        return render(request, 'competitions/partials/upload_result.html', {
            'success': False,
            'error': 'Please upload a CSV file'
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
        'today_count': Submission.get_today_count(competition, request.user),
        'total_count': Submission.get_total_count(competition, request.user),
    })


@login_required
def submission_history(request: HttpRequest, competition_id: int) -> HttpResponse:
    """Get submission history for current user (HTMX endpoint)."""
    competition = get_object_or_404(Competition, id=competition_id)
    
    submissions = Submission.objects.filter(
        competition=competition,
        user=request.user
    ).prefetch_related('logs').order_by('-submitted_at')[:50]
    
    # Get labels for metrics
    metric_labels = {m.value: m.label for m in MetricType}
    available_metrics = [
        {'key': m, 'label': metric_labels.get(m, m)}
        for m in (competition.available_metrics or [])
    ]
    
    return render(request, 'competitions/partials/history.html', {
        'competition': competition,
        'submissions': submissions,
        'available_metrics': available_metrics,
    })


@login_required
@require_POST
def set_final_selection(request: HttpRequest, submission_id: int) -> HttpResponse:
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
def leaderboard(request: HttpRequest, competition_id: int) -> HttpResponse:
    """Get competition leaderboard (HTMX endpoint)."""
    competition = get_object_or_404(Competition, id=competition_id)
    
    # Determine which score to use
    show_private = competition.status == CompetitionStatus.ENDED
    
    # Get leaderboard data using utility
    leaderboard_data = get_leaderboard_data(competition, show_private)
    
    # Get labels for available metrics
    metric_labels = {m.value: m.label for m in MetricType}
    available_metrics = [
        {'key': m, 'label': metric_labels.get(m, m)}
        for m in (competition.available_metrics or [])
    ]
    
    for entry in leaderboard_data:
        entry['is_current_user'] = entry['user_id'] == request.user.id
        
        # Format additional scores for display
        entry['display_scores'] = []
        for m in (competition.available_metrics or []):
            val = entry.get('all_scores', {}).get(m)
            entry['display_scores'].append(val)
    
    return render(request, 'competitions/partials/leaderboard.html', {
        'competition': competition,
        'leaderboard': leaderboard_data,
        'show_private': show_private,
        'available_metrics': available_metrics,
    })


@login_required
def leaderboard_chart_data(request: HttpRequest, competition_id: int) -> JsonResponse:
    """Return JSON data for leaderboard charts."""
    competition = get_object_or_404(Competition, id=competition_id)
    
    # Determine which score to use
    show_private = competition.status == CompetitionStatus.ENDED
    score_field = 'private_score' if show_private else 'public_score'
    
    # --- Score Trend Data ---
    trend_datasets = get_score_trend_data(competition, score_field)
    
    # --- Score Distribution Data ---
    # Extract just the best scores for distribution
    leaderboard_data = get_leaderboard_data(competition, show_private)
    best_scores = [entry['score'] for entry in leaderboard_data if entry['score'] is not None]
    distribution_data = get_score_distribution_data(best_scores)
    
    return JsonResponse({
        'trend': {
            'datasets': trend_datasets,
        },
        'distribution': distribution_data
    })


class RegisterView(View):
    """User registration view with whitelist check."""
    
    def get(self, request: HttpRequest) -> HttpResponse:
        if request.user.is_authenticated:
            return redirect('competition_list')
        return render(request, 'registration/register.html')
        
    def post(self, request: HttpRequest) -> HttpResponse:
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        # Check whitelist
        if not RegistrationWhitelist.objects.filter(username=username).exists():
            return render(request, 'registration/register.html', {
                'error': 'This username is not in the registration whitelist.',
                'username': username,
                'email': email
            })
            
        # Check if user exists
        if User.objects.filter(username=username).exists():
            return render(request, 'registration/register.html', {
                'error': 'A user with that username already exists.',
                'username': username,
                'email': email
            })
            
        # Basic validation
        if not password1 or password1 != password2:
            return render(request, 'registration/register.html', {
                'error': 'Passwords do not match or are empty.',
                'username': username,
                'email': email
            })
            
        # Create user
        User.objects.create_user(username=username, email=email, password=password1)
        messages.success(request, 'Registration successful! Please log in.')
        return redirect('login')


@login_required
def submission_report(request: HttpRequest, submission_id: int) -> HttpResponse:
    """Get detailed scoring report for a submission (HTMX endpoint)."""
    submission = get_object_or_404(
        Submission, 
        id=submission_id, 
        user=request.user
    )
    
    # Extract per-class report if available
    report_data = submission.all_scores.get('per_class_report', {}) if submission.all_scores else {}
    
    return render(request, 'competitions/partials/report.html', {
        'submission': submission,
        'report_data': report_data,
    })


