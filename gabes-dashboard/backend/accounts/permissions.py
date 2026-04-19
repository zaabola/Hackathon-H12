from rest_framework.permissions import BasePermission


class IsWorker(BasePermission):
    """Allow access to workers only."""
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and
                    request.user.role == 'worker')


class IsTechnician(BasePermission):
    """Allow access to technicians only."""
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and
                    request.user.role == 'technician')


class IsAdminRole(BasePermission):
    """Allow access to admin role users only."""
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and
                    request.user.role == 'admin')


class IsWorkerOrAdmin(BasePermission):
    """Allow workers and admins (NOT farmers — they have a separate permission)."""
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and
                    request.user.role in ('worker', 'admin'))


class IsTechnicianOrAdmin(BasePermission):
    """Allow technicians and admins."""
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and
                    request.user.role in ('technician', 'admin'))


class IsFarmerOrAdmin(BasePermission):
    """
    Allow approved farmers and admins.
    Farmers must have is_approved=True AND is_active=True (set by admin upon approval).
    """
    def has_permission(self, request, view):
        if not (request.user and request.user.is_authenticated):
            return False
        if request.user.role == 'admin':
            return True
        if request.user.role == 'farmer':
            return request.user.is_approved and request.user.is_active
        return False
