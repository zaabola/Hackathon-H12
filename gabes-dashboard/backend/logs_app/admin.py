from django.contrib import admin
from .models import IncidentLog


@admin.register(IncidentLog)
class IncidentLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'module', 'source', 'title', 'status', 'created_by', 'zone_name', 'created_at']
    list_filter = ['module', 'status', 'source', 'created_at']
    search_fields = ['title', 'description', 'created_by__username', 'zone_name']
    readonly_fields = ['created_at', 'updated_at', 'labeled_at', 'result_data']
    ordering = ['-created_at']

    fieldsets = (
        ('Incident Info', {
            'fields': ('module', 'source', 'title', 'description', 'zone_name', 'latitude', 'longitude')
        }),
        ('Media', {
            'fields': ('photo', 'video')
        }),
        ('AI Result', {
            'fields': ('result_data',),
            'classes': ('collapse',),
        }),
        ('Status', {
            'fields': ('status', 'labeled_by', 'label_note', 'labeled_at')
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )
