from django.urls import path
from .views import (
    IncidentLogListCreateView,
    IncidentLogDetailView,
    LabelLogView,
    LogStatsView,
)

urlpatterns = [
    path('', IncidentLogListCreateView.as_view(), name='log_list_create'),
    path('stats/', LogStatsView.as_view(), name='log_stats'),
    path('<int:pk>/', IncidentLogDetailView.as_view(), name='log_detail'),
    path('<int:pk>/label/', LabelLogView.as_view(), name='log_label'),
]
