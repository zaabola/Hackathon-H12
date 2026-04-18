from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views import (
    CustomTokenObtainPairView, MeView,
    UserListCreateView, UserRetrieveUpdateDestroyView,
    UserStatsView,
)

urlpatterns = [
    path('login/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('me/', MeView.as_view(), name='me'),
    path('users/', UserListCreateView.as_view(), name='user_list_create'),
    path('users/stats/', UserStatsView.as_view(), name='user_stats'),
    path('users/<int:pk>/', UserRetrieveUpdateDestroyView.as_view(), name='user_detail'),
]
