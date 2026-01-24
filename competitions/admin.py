"""
Django Admin configuration for competitions.

Provides admin interfaces for:
- Competition management (for admins/organizers)
- Participant whitelist management
- Submission review (for validators)
"""

from django.contrib import admin
from django.utils.html import format_html

from .models import (
    Competition,
    CompetitionParticipant,
    Submission,
    SubmissionLog,
)


class CompetitionParticipantInline(admin.TabularInline):
    """Inline editor for adding participants to a competition."""
    model = CompetitionParticipant
    extra = 1
    autocomplete_fields = ['user']
    fields = ['user', 'start_time', 'end_time', 'is_active']


@admin.register(Competition)
class CompetitionAdmin(admin.ModelAdmin):
    """Admin interface for competition management."""
    list_display = [
        'name', 
        'task_type', 
        'metric_type', 
        'status', 
        'participant_count',
        'submission_count',
        'created_at'
    ]
    list_filter = ['status', 'task_type', 'metric_type']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    inlines = [CompetitionParticipantInline]
    
    fieldsets = (
        ('基本資訊', {
            'fields': ('name', 'description', 'task_type', 'metric_type')
        }),
        ('資料設定', {
            'fields': ('public_ground_truth', 'private_ground_truth', 'dataset_url')
        }),
        ('上傳限制', {
            'fields': ('daily_upload_limit', 'total_upload_limit')
        }),
        ('狀態', {
            'fields': ('status',)
        }),
        ('時間戳記', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    @admin.display(description='參賽人數')
    def participant_count(self, obj):
        return obj.participants.count()

    @admin.display(description='提交數')
    def submission_count(self, obj):
        return obj.submissions.count()


@admin.register(CompetitionParticipant)
class CompetitionParticipantAdmin(admin.ModelAdmin):
    """Admin interface for participant whitelist management."""
    list_display = [
        'user', 
        'competition', 
        'start_time', 
        'end_time', 
        'is_active',
        'participation_status'
    ]
    list_filter = ['competition', 'is_active']
    search_fields = ['user__username', 'user__email']
    autocomplete_fields = ['user', 'competition']
    list_editable = ['is_active']

    @admin.display(description='狀態')
    def participation_status(self, obj):
        if not obj.is_active:
            return format_html('<span style="color: gray;">⏸️ 已停權</span>')
        if obj.can_participate():
            return format_html('<span style="color: green;">✅ 可參賽</span>')
        return format_html('<span style="color: orange;">⏰ 非活動時間</span>')


class SubmissionLogInline(admin.TabularInline):
    """Inline viewer for submission logs."""
    model = SubmissionLog
    extra = 0
    readonly_fields = ['level', 'message', 'created_at']
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    """
    Admin interface for submission review.
    
    Validators can:
    - Filter to see only final selections
    - Fill in private scores directly in the list view
    """
    list_display = [
        'id',
        'competition',
        'user',
        'status_badge',
        'public_score',
        'private_score',
        'is_final_selection',
        'submitted_at'
    ]
    list_filter = ['competition', 'status', 'is_final_selection']
    list_editable = ['private_score']  # Allow validators to fill in scores
    search_fields = ['user__username', 'user__email']
    readonly_fields = [
        'competition',
        'user', 
        'prediction_file',
        'status',
        'public_score', 
        'error_message',
        'submitted_at', 
        'scored_at'
    ]
    inlines = [SubmissionLogInline]
    
    fieldsets = (
        ('提交資訊', {
            'fields': ('competition', 'user', 'prediction_file', 'submitted_at')
        }),
        ('評分結果', {
            'fields': ('status', 'public_score', 'private_score', 'scored_at')
        }),
        ('最終選擇', {
            'fields': ('is_final_selection',)
        }),
        ('錯誤訊息', {
            'fields': ('error_message',),
            'classes': ('collapse',)
        }),
    )

    @admin.display(description='狀態')
    def status_badge(self, obj):
        colors = {
            'PENDING': 'gray',
            'PROCESSING': 'blue',
            'SUCCESS': 'green',
            'FAILED': 'red',
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.get_status_display()
        )


@admin.register(SubmissionLog)
class SubmissionLogAdmin(admin.ModelAdmin):
    """Admin interface for viewing submission logs."""
    list_display = ['submission', 'level', 'short_message', 'created_at']
    list_filter = ['level', 'submission__competition']
    search_fields = ['message', 'submission__user__username']
    readonly_fields = ['submission', 'level', 'message', 'created_at']

    @admin.display(description='訊息')
    def short_message(self, obj):
        return obj.message[:80] + '...' if len(obj.message) > 80 else obj.message

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
