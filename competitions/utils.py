from django.db.models import Count, Max
from .models import Competition, Submission, SubmissionStatus
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict

def get_leaderboard_data(competition: Competition, show_private: bool) -> List[Dict[str, Any]]:
    """Aggregate leaderboard data for a competition."""
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
                'all_scores': s.all_scores or {},
                'submission_count': Submission.get_total_count(competition, s.user),
                'last_submission': s.submitted_at,
            })
    else:
        # Use best public score per user
        user_best = Submission.objects.filter(
            competition=competition,
            status=SubmissionStatus.SUCCESS,
            public_score__isnull=False
        ).values('user_id', 'user__username').annotate(
            best_score=Max('public_score'),
            submission_count=Count('id'),
            last_submission=Max('submitted_at')
        ).order_by('-best_score')
        
        leaderboard_data = []
        for entry in user_best:
            # Find the actual submission object that has this best score to get all_scores
            best_submission = Submission.objects.filter(
                competition=competition,
                user_id=entry['user_id'],
                status=SubmissionStatus.SUCCESS,
                public_score=entry['best_score']
            ).order_by('-submitted_at').first()
            
            leaderboard_data.append({
                'user_id': entry['user_id'],
                'username': entry['user__username'],
                'score': entry['best_score'],
                'all_scores': best_submission.all_scores if best_submission else {},
                'submission_count': entry['submission_count'],
                'last_submission': entry['last_submission'],
            })
    
    # Sort by score (descending)
    leaderboard_data.sort(key=lambda x: x['score'] or 0, reverse=True)
    
    # Add ranks
    for i, entry in enumerate(leaderboard_data, 1):
        entry['rank'] = i
        
    return leaderboard_data

def get_score_trend_data(competition: Competition, score_field: str) -> List[Dict[str, Any]]:
    """Generate score trend data for Chart.js."""
    submissions = Submission.objects.filter(
        competition=competition,
        status=SubmissionStatus.SUCCESS,
        **{f'{score_field}__isnull': False}
    ).select_related('user').order_by('submitted_at')
    
    user_best_scores = defaultdict(lambda: {'scores': [], 'timestamps': []})
    user_running_best = {}
    
    for s in submissions:
        username = s.user.username
        score = getattr(s, score_field)
        timestamp = s.submitted_at.isoformat()
        
        if username not in user_running_best or score > user_running_best[username]:
            user_running_best[username] = score
            user_best_scores[username]['scores'].append(score)
            user_best_scores[username]['timestamps'].append(timestamp)
            
    colors = [
        'rgb(59, 130, 246)', 'rgb(239, 68, 68)', 'rgb(34, 197, 94)',
        'rgb(168, 85, 247)', 'rgb(249, 115, 22)', 'rgb(236, 72, 153)',
        'rgb(20, 184, 166)', 'rgb(245, 158, 11)',
    ]
    
    datasets = []
    for i, (username, data) in enumerate(user_best_scores.items()):
        color = colors[i % len(colors)]
        datasets.append({
            'label': username,
            'data': [{'x': t, 'y': s} for t, s in zip(data['timestamps'], data['scores'])],
            'borderColor': color,
            'backgroundColor': color,
            'tension': 0.3,
            'fill': False,
        })
    return datasets

def get_score_distribution_data(scores: List[float]) -> Dict[str, Any]:
    """Generate score distribution data for Chart.js."""
    if not scores:
        return {'labels': [], 'data': []}
        
    min_score = min(scores)
    max_score = max(scores)
    range_size = (max_score - min_score) / 5 if max_score > min_score else 0.2
    
    bins = []
    counts = []
    for i in range(5):
        bin_start = min_score + i * range_size
        bin_end = min_score + (i + 1) * range_size
        bins.append(f"{bin_start:.2f}-{bin_end:.2f}")
        count = sum(1 for s in scores if bin_start <= s < bin_end or (i == 4 and s == bin_end))
        counts.append(count)
        
    return {'labels': bins, 'data': counts}

def get_expected_format_hint(competition: Competition) -> str:
    """Detect expected CSV format hint from ground truth file."""
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
