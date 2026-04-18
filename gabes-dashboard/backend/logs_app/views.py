from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.db.models import Count
from .models import IncidentLog
from .serializers import IncidentLogSerializer, LabelLogSerializer
from accounts.permissions import IsWorkerOrAdmin, IsTechnicianOrAdmin, IsAdminRole


class IncidentLogListCreateView(generics.ListCreateAPIView):
    """
    GET  /api/logs/         → Technicians & Admins see all logs
                              Workers see only their own logs
    POST /api/logs/         → Workers & Admins create a new log
    """
    serializer_class = IncidentLogSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        qs = IncidentLog.objects.select_related('created_by', 'labeled_by')

        # Workers only see their own logs
        if user.role == 'worker':
            qs = qs.filter(created_by=user)

        # Filters
        module = self.request.query_params.get('module')
        status_filter = self.request.query_params.get('status')
        source = self.request.query_params.get('source')

        if module:
            qs = qs.filter(module=module)
        if status_filter:
            qs = qs.filter(status=status_filter)
        if source:
            qs = qs.filter(source=source)

        return qs

    def get_serializer_context(self):
        return {'request': self.request}

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class IncidentLogDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    GET    /api/logs/<id>/  → Technicians & Admins
    PATCH  /api/logs/<id>/  → Admin can update anything
    DELETE /api/logs/<id>/  → Admin only
    """
    serializer_class = IncidentLogSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.role == 'worker':
            return IncidentLog.objects.filter(created_by=user)
        return IncidentLog.objects.all()

    def get_serializer_context(self):
        return {'request': self.request}

    def destroy(self, request, *args, **kwargs):
        if request.user.role != 'admin':
            return Response(
                {'error': 'Only admins can delete logs'},
                status=status.HTTP_403_FORBIDDEN
            )
        return super().destroy(request, *args, **kwargs)


class LabelLogView(APIView):
    """
    PATCH /api/logs/<id>/label/  → Technician or Admin sets status
    """
    permission_classes = [IsTechnicianOrAdmin]

    def patch(self, request, pk):
        try:
            log = IncidentLog.objects.get(pk=pk)
        except IncidentLog.DoesNotExist:
            return Response({'error': 'Log not found'}, status=status.HTTP_404_NOT_FOUND)

        serializer = LabelLogSerializer(log, data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            full = IncidentLogSerializer(log, context={'request': request})
            return Response(full.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogStatsView(APIView):
    """
    GET /api/logs/stats/  → Dashboard KPI counts
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        qs = IncidentLog.objects.all()
        if user.role == 'worker':
            qs = qs.filter(created_by=user)

        by_module = {
            item['module']: item['count']
            for item in qs.values('module').annotate(count=Count('id'))
        }

        return Response({
            'total': qs.count(),
            'in_progress': qs.filter(status='in_progress').count(),
            'resolved': qs.filter(status='resolved').count(),
            'non_resolved': qs.filter(status='non_resolved').count(),
            'by_module': by_module,
        })
