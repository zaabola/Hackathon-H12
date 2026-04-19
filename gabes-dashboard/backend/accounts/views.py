from django.contrib.auth import get_user_model
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .serializers import (
    CustomTokenObtainPairSerializer,
    UserSerializer,
    FarmerRegisterSerializer,
)
from .permissions import IsAdminRole

User = get_user_model()


class CustomTokenObtainPairView(TokenObtainPairView):
    """Login endpoint — returns JWT with role + approval info."""
    serializer_class = CustomTokenObtainPairSerializer
    permission_classes = [AllowAny]


class MeView(APIView):
    """Get currently authenticated user info."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user, context={'request': request})
        return Response(serializer.data)


# ── Farmer Self-Registration (public) ─────────────────────────────────────────
class FarmerRegisterView(APIView):
    """
    Public endpoint — farmers submit their details + ID document.
    Account is created with is_active=False, is_approved=False.
    Login is blocked until an admin approves.
    """
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = FarmerRegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({
                'message': (
                    'Your registration has been submitted. '
                    'An administrator will review your ID and approve your account. '
                    'You will be able to log in once approved.'
                ),
                'username': user.username,
                'status': 'pending_approval',
            }, status=201)
        return Response(serializer.errors, status=400)


# ── Admin: Pending Farmer Approvals ───────────────────────────────────────────
class PendingFarmerListView(generics.ListAPIView):
    """Admin: list all farmers awaiting approval."""
    serializer_class = UserSerializer
    permission_classes = [IsAdminRole]

    def get_queryset(self):
        return User.objects.filter(role='farmer', is_approved=False).order_by('date_joined')

    def get_serializer_context(self):
        return {'request': self.request}


class FarmerApproveView(APIView):
    """Admin: approve or reject a pending farmer registration."""
    permission_classes = [IsAdminRole]

    def post(self, request, pk):
        try:
            farmer = User.objects.get(pk=pk, role='farmer')
        except User.DoesNotExist:
            return Response({'error': 'Farmer not found.'}, status=404)

        action = request.data.get('action')  # 'approve' or 'reject'

        if action == 'approve':
            farmer.is_approved = True
            farmer.is_active = True
            farmer.rejection_note = ''
            farmer.save()
            return Response({
                'id': farmer.id,
                'username': farmer.username,
                'message': f'{farmer.username} has been approved and can now log in.',
                'status': 'approved',
            })
        elif action == 'reject':
            note = request.data.get('rejection_note', '')
            farmer.is_approved = False
            farmer.is_active = False
            farmer.rejection_note = note
            farmer.save()
            return Response({
                'id': farmer.id,
                'username': farmer.username,
                'message': f'{farmer.username} has been rejected.',
                'rejection_note': note,
                'status': 'rejected',
            })
        else:
            return Response({'error': "action must be 'approve' or 'reject'."}, status=400)


# ── Admin: Full User CRUD ──────────────────────────────────────────────────────
class UserListCreateView(generics.ListCreateAPIView):
    """Admin: list all users or create a new one (workers/technicians/admins)."""
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [IsAdminRole]

    def get_serializer_context(self):
        return {'request': self.request}


class UserRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    """Admin: get, update or delete a user."""
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdminRole]

    def get_serializer_context(self):
        return {'request': self.request}


class UserBanToggleView(APIView):
    """Admin: ban (deactivate) or unban (reactivate) a user."""
    permission_classes = [IsAdminRole]

    def post(self, request, pk):
        try:
            target = User.objects.get(pk=pk)
        except User.DoesNotExist:
            return Response({'error': 'User not found.'}, status=404)
        if target == request.user:
            return Response({'error': 'You cannot ban your own account.'}, status=400)
        target.is_active = not target.is_active
        target.save()
        action = 'banned' if not target.is_active else 'unbanned'
        return Response({
            'id': target.id,
            'username': target.username,
            'is_active': target.is_active,
            'message': f'User {target.username} has been {action}.',
        })


class UserResetPasswordView(APIView):
    """Admin: force-reset a user password."""
    permission_classes = [IsAdminRole]

    def post(self, request, pk):
        try:
            target = User.objects.get(pk=pk)
        except User.DoesNotExist:
            return Response({'error': 'User not found.'}, status=404)
        new_password = request.data.get('password')
        if not new_password or len(new_password) < 6:
            return Response({'error': 'Password must be at least 6 characters.'}, status=400)
        target.set_password(new_password)
        target.save()
        return Response({'message': f'Password for {target.username} has been reset.'})


class UserStatsView(APIView):
    """Admin: user counts by role and status."""
    permission_classes = [IsAdminRole]

    def get(self, request):
        return Response({
            'total':            User.objects.count(),
            'workers':          User.objects.filter(role='worker').count(),
            'technicians':      User.objects.filter(role='technician').count(),
            'admins':           User.objects.filter(role='admin').count(),
            'farmers':          User.objects.filter(role='farmer').count(),
            'farmers_pending':  User.objects.filter(role='farmer', is_approved=False).count(),
            'active':           User.objects.filter(is_active=True).count(),
            'banned':           User.objects.filter(is_active=False).count(),
        })
