from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    ROLE_WORKER     = 'worker'
    ROLE_TECHNICIAN = 'technician'
    ROLE_ADMIN      = 'admin'
    ROLE_FARMER     = 'farmer'

    ROLE_CHOICES = [
        (ROLE_WORKER,     'Worker'),
        (ROLE_TECHNICIAN, 'Technician'),
        (ROLE_ADMIN,      'Admin'),
        (ROLE_FARMER,     'Farmer'),
    ]

    role       = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_WORKER)
    department = models.CharField(max_length=100, blank=True)
    phone      = models.CharField(max_length=20, blank=True)

    # Farmer-specific fields
    # is_active (inherited from AbstractUser) acts as the "approved" flag for farmers.
    # While pending approval, is_active=False prevents login.
    is_approved    = models.BooleanField(default=True)   # False = pending admin review
    id_document    = models.ImageField(upload_to='id_documents/', null=True, blank=True)
    rejection_note = models.TextField(blank=True)        # optional note when rejected

    def __str__(self):
        return f'{self.username} ({self.get_role_display()})'

    @property
    def is_worker(self):
        return self.role == self.ROLE_WORKER

    @property
    def is_technician(self):
        return self.role == self.ROLE_TECHNICIAN

    @property
    def is_admin_role(self):
        return self.role == self.ROLE_ADMIN

    @property
    def is_farmer(self):
        return self.role == self.ROLE_FARMER
