from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    ROLE_WORKER = 'worker'
    ROLE_TECHNICIAN = 'technician'
    ROLE_ADMIN = 'admin'

    ROLE_CHOICES = [
        (ROLE_WORKER, 'Worker'),
        (ROLE_TECHNICIAN, 'Technician'),
        (ROLE_ADMIN, 'Admin'),
    ]

    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_WORKER)
    department = models.CharField(max_length=100, blank=True)
    phone = models.CharField(max_length=20, blank=True)

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
