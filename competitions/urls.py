"""URL configuration for competitions app."""

from django.urls import path

from . import views

urlpatterns = [
    # Main pages
    path('', views.CompetitionListView.as_view(), name='competition_list'),
    path('<int:competition_id>/', views.CompetitionDetailView.as_view(), name='competition_detail'),
    
    # HTMX endpoints
    path('<int:competition_id>/upload/', views.upload_prediction, name='upload_prediction'),
    path('<int:competition_id>/history/', views.submission_history, name='submission_history'),
    path('<int:competition_id>/leaderboard/', views.leaderboard, name='leaderboard'),
    path('<int:competition_id>/leaderboard/chart-data/', views.leaderboard_chart_data, name='leaderboard_chart_data'),
    path('submission/<int:submission_id>/final/', views.set_final_selection, name='set_final_selection'),
]
