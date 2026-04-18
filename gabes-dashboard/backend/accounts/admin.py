from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'role', 'department', 'is_active']
    list_filter = ['role', 'is_active', 'is_staff']
    search_fields = ['username', 'email', 'first_name', 'last_name']
    ordering = ['-date_joined']

    fieldsets = UserAdmin.fieldsets + (
        ('Gabes AI Role', {
            'fields': ('role', 'department', 'phone'),
        }),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Gabes AI Role', {
            'fields': ('role', 'department', 'phone'),
        }),
    )
