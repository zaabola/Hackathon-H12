from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views import (
    CustomTokenObtainPairView, MeView,
    FarmerRegisterView,
    PendingFarmerListView, FarmerApproveView,
    UserListCreateView, UserRetrieveUpdateDestroyView,
    UserStatsView, UserBanToggleView, UserResetPasswordView,
)

urlpatterns = [
    # ── Auth ──────────────────────────────────────────────────────────────────
    path('login/',   CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('refresh/', TokenRefreshView.as_view(),          name='token_refresh'),
    path('me/',      MeView.as_view(),                    name='me'),

    # ── Farmer self-registration (public) ─────────────────────────────────────
    path('register/farmer/', FarmerRegisterView.as_view(), name='farmer_register'),

    # ── Admin: farmer approval queue ──────────────────────────────────────────
    path('farmers/pending/',          PendingFarmerListView.as_view(), name='farmers_pending'),
    path('farmers/<int:pk>/approve/', FarmerApproveView.as_view(),     name='farmer_approve'),

    # ── Admin: full user management ───────────────────────────────────────────
    path('users/',          UserListCreateView.as_view(),            name='user_list_create'),
    path('users/stats/',    UserStatsView.as_view(),                 name='user_stats'),
    path('users/<int:pk>/', UserRetrieveUpdateDestroyView.as_view(), name='user_detail'),
    path('users/<int:pk>/ban/',            UserBanToggleView.as_view(),      name='user_ban'),
    path('users/<int:pk>/reset-password/', UserResetPasswordView.as_view(), name='user_reset_password'),
]
