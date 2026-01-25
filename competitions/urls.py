"""URL configuration for competitions app."""

from django.urls import path

from . import views

urlpatterns = [
    # Authentication & Profile
    path("register/", views.register, name="register"),
    # Main pages
    path("", views.CompetitionListView.as_view(), name="dashboard"),
    path(
        "<int:competition_id>/",
        views.CompetitionDetailView.as_view(),
        name="competition_detail",
    ),
    # HTMX endpoints
    path(
        "<int:competition_id>/upload/",
        views.upload_prediction,
        name="upload_prediction",
    ),
    path(
        "<int:competition_id>/history/",
        views.submission_history,
        name="submission_history",
    ),
    path("<int:competition_id>/leaderboard/", views.leaderboard, name="leaderboard"),
    path(
        "<int:competition_id>/leaderboard/chart-data/",
        views.leaderboard_chart_data,
        name="leaderboard_chart_data",
    ),
    path(
        "submission/<int:submission_id>/final/",
        views.set_final_selection,
        name="set_final_selection",
    ),
    path(
        "submission/<int:submission_id>/logs/",
        views.submission_logs,
        name="submission_logs",
    ),
    path(
        "submission/<int:submission_id>/report/",
        views.submission_report,
        name="submission_report",
    ),
]
