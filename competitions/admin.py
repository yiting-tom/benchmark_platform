"""
Django Admin configuration for competitions.

Provides admin interfaces for:
- Competition management (for admins/organizers)
- Participant whitelist management
- Submission review (for validators)
"""

from django import forms
from django.contrib import admin
from django.utils.html import format_html

from .models import (
    Competition,
    CompetitionParticipant,
    Submission,
    SubmissionLog,
    MetricType,
)


class CompetitionParticipantInline(admin.TabularInline):
    """Inline editor for adding participants to a competition."""

    model = CompetitionParticipant
    extra = 1
    autocomplete_fields = ["user"]
    fields = ["user", "start_time", "end_time", "is_active"]


class CompetitionAdminForm(forms.ModelForm):
    """Custom form for CompetitionAdmin to handle available_metrics as a list of choices."""

    available_metrics = forms.MultipleChoiceField(
        choices=MetricType.choices,
        widget=forms.CheckboxSelectMultiple,
        required=False,
        help_text="Select all metrics to be displayed on the leaderboard",
    )

    class Meta:
        model = Competition
        fields = "__all__"


@admin.register(Competition)
class CompetitionAdmin(admin.ModelAdmin):
    """Admin interface for competition management."""

    form = CompetitionAdminForm

    list_display = [
        "name",
        "task_type",
        "metric_type",
        "status",
        "participant_count",
        "submission_count",
        "created_at",
    ]
    list_filter = ["status", "task_type", "metric_type"]
    search_fields = ["name", "description"]
    readonly_fields = ["created_at", "updated_at"]
    inlines = [CompetitionParticipantInline]

    fieldsets = (
        (
            "General Information",
            {"fields": ("name", "description", "task_type", "metric_type", "available_metrics")},
        ),
        (
            "Data Settings",
            {"fields": ("public_ground_truth", "private_ground_truth", "scoring_script", "dataset_url")},
        ),
        ("Upload Limits", {"fields": ("daily_upload_limit", "total_upload_limit")}),
        ("Status", {"fields": ("status",)}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    @admin.display(description="Participants")
    def participant_count(self, obj):
        return obj.participants.count()

    @admin.display(description="Submissions")
    def submission_count(self, obj):
        return obj.submissions.count()


@admin.register(CompetitionParticipant)
class CompetitionParticipantAdmin(admin.ModelAdmin):
    """Admin interface for participant whitelist management."""

    list_display = [
        "user",
        "competition",
        "start_time",
        "end_time",
        "is_active",
        "participation_status",
    ]
    list_filter = ["competition", "is_active"]
    search_fields = ["user__username", "user__email"]
    autocomplete_fields = ["user", "competition"]
    list_editable = ["is_active"]

    @admin.display(description="Status")
    def participation_status(self, obj):
        if not obj.is_active:
            return format_html(
                '<span style="color: {};">{}</span>', "gray", "[Suspended]"
            )
        if obj.can_participate():
            return format_html(
                '<span style="color: {};">{}</span>', "green", "[Active]"
            )
        return format_html(
            '<span style="color: {};">{}</span>', "orange", "[Out of Window]"
        )


class SubmissionLogInline(admin.TabularInline):
    """Inline viewer for submission logs."""

    model = SubmissionLog
    extra = 0
    readonly_fields = ["level", "message", "created_at"]
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
        "id",
        "competition",
        "user",
        "status_badge",
        "public_score",
        "private_score",
        "is_final_selection",
        "submitted_at",
    ]
    list_filter = ["competition", "status", "is_final_selection"]
    list_editable = ["private_score"]  # Allow validators to fill in scores
    search_fields = ["user__username", "user__email"]
    readonly_fields = [
        "competition",
        "user",
        "prediction_file",
        "status",
        "public_score",
        "error_message",
        "all_scores",
        "submitted_at",
        "scored_at",
    ]
    inlines = [SubmissionLogInline]

    fieldsets = (
        (
            "Submission Info",
            {"fields": ("competition", "user", "prediction_file", "submitted_at")},
        ),
        (
            "Scoring Results",
            {"fields": ("status", "public_score", "private_score", "all_scores", "scored_at")},
        ),
        ("Final Selection", {"fields": ("is_final_selection",)}),
        ("Error Message", {"fields": ("error_message",), "classes": ("collapse",)}),
    )

    @admin.display(description="Status")
    def status_badge(self, obj):
        colors = {
            "PENDING": "gray",
            "PROCESSING": "blue",
            "SUCCESS": "green",
            "FAILED": "red",
        }
        color = colors.get(obj.status, "gray")
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.get_status_display(),
        )


@admin.register(SubmissionLog)
class SubmissionLogAdmin(admin.ModelAdmin):
    """Admin interface for viewing submission logs."""

    list_display = ["submission", "level", "short_message", "created_at"]
    list_filter = ["level", "submission__competition"]
    search_fields = ["message", "submission__user__username"]
    readonly_fields = ["submission", "level", "message", "created_at"]

    @admin.display(description="Message")
    def short_message(self, obj):
        return obj.message[:80] + "..." if len(obj.message) > 80 else obj.message

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
