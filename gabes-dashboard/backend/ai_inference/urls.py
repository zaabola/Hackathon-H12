from django.urls import path
from .views import PPEAnalyzeView, MarineAnalyzeView, LandAnalyzeView

urlpatterns = [
    path('ppe/analyze/', PPEAnalyzeView.as_view(), name='ppe_analyze'),
    path('marine/analyze/', MarineAnalyzeView.as_view(), name='marine_analyze'),
    path('land/analyze/', LandAnalyzeView.as_view(), name='land_analyze'),
]
