from django.db import models
from django.conf import settings


class IncidentLog(models.Model):
    STATUS_IN_PROGRESS = 'in_progress'
    STATUS_RESOLVED = 'resolved'
    STATUS_NON_RESOLVED = 'non_resolved'

    STATUS_CHOICES = [
        (STATUS_IN_PROGRESS, 'In Progress'),
        (STATUS_RESOLVED, 'Resolved'),
        (STATUS_NON_RESOLVED, 'Non Resolved'),
    ]

    MODULE_PPE = 'ppe'
    MODULE_MARINE = 'marine'
    MODULE_LAND = 'land'

    MODULE_CHOICES = [
        (MODULE_PPE, 'Industrial Safety (PPE)'),
        (MODULE_MARINE, 'Marine Contamination'),
        (MODULE_LAND, 'Land Regeneration'),
    ]

    SOURCE_UPLOAD = 'upload'
    SOURCE_CAMERA = 'camera'

    SOURCE_CHOICES = [
        (SOURCE_UPLOAD, 'File Upload'),
        (SOURCE_CAMERA, 'Live Camera'),
    ]

    # Who created this log
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='created_logs',
    )

    # Core fields
    module = models.CharField(max_length=20, choices=MODULE_CHOICES)
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default=SOURCE_UPLOAD)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    # Media attachments
    photo = models.ImageField(upload_to='logs/photos/%Y/%m/', blank=True, null=True)
    video = models.FileField(upload_to='logs/videos/%Y/%m/', blank=True, null=True)

    # AI result data (raw JSON from the model)
    result_data = models.JSONField(default=dict)

    # Location context
    zone_name = models.CharField(max_length=100, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    # Status labeling (done by technician)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_IN_PROGRESS)
    labeled_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='labeled_logs',
    )
    label_note = models.TextField(blank=True)
    labeled_at = models.DateTimeField(null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Incident Log'
        verbose_name_plural = 'Incident Logs'

    def __str__(self):
        return f'[{self.get_module_display()}] {self.title} — {self.get_status_display()}'
