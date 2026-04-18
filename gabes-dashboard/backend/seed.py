"""
Seed script — creates demo users for all 3 roles.
Run with: py -3 manage.py shell < seed.py
"""
import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gabes_project.settings')
django.setup()

from accounts.models import CustomUser
from logs_app.models import IncidentLog

print("Seeding database...")

# Admin
admin, _ = CustomUser.objects.get_or_create(username='admin')
admin.set_password('admin123')
admin.role = 'admin'
admin.email = 'admin@gabes-ai.tn'
admin.first_name = 'Admin'
admin.last_name = 'Gabes'
admin.is_staff = True
admin.is_superuser = True
admin.save()
print(f"  ✅ Admin:      admin / admin123")

# Technician
tech, _ = CustomUser.objects.get_or_create(username='tech1')
tech.set_password('tech123')
tech.role = 'technician'
tech.email = 'tech@gabes-ai.tn'
tech.first_name = 'Ahmed'
tech.last_name = 'Technician'
tech.department = 'Environmental Monitoring'
tech.save()
print(f"  ✅ Technician: tech1 / tech123")

# Worker
worker, _ = CustomUser.objects.get_or_create(username='worker1')
worker.set_password('worker123')
worker.role = 'worker'
worker.email = 'worker@gabes-ai.tn'
worker.first_name = 'Nour'
worker.last_name = 'Worker'
worker.department = 'Chemical Plant A'
worker.save()
print(f"  ✅ Worker:     worker1 / worker123")

# Sample logs
sample_logs = [
    {
        'module': 'ppe',
        'source': 'upload',
        'title': 'PPE Violation — Sector B',
        'description': 'Worker #3 detected without gas mask near reactor zone.',
        'status': 'in_progress',
        'zone_name': 'Chemical Plant Sector B',
        'result_data': {'violations': 1, 'workers_detected': 4},
        'created_by': worker,
    },
    {
        'module': 'marine',
        'source': 'camera',
        'title': 'Fish Erratic Behavior — Tank 2',
        'description': 'Multiple fish showing erratic swimming patterns. Contamination suspected.',
        'status': 'in_progress',
        'zone_name': 'Marine Monitoring Tank 2',
        'result_data': {'contamination_probability': 0.87, 'fish_count': 12, 'erratic_count': 4},
        'created_by': worker,
    },
    {
        'module': 'land',
        'source': 'upload',
        'title': 'Satellite Scan — Oasis North',
        'description': 'High saline zone detected in northern oasis. Immediate remediation needed.',
        'status': 'resolved',
        'zone_name': 'Oasis Zone North',
        'result_data': {'healthy_pct': 22.5, 'dead_pct': 51.3, 'regeneration_score': 38},
        'created_by': worker,
        'labeled_by': tech,
    },
]

for log_data in sample_logs:
    if not IncidentLog.objects.filter(title=log_data['title']).exists():
        IncidentLog.objects.create(**log_data)
        print(f"  📋 Log created: {log_data['title']}")

print("\n✅ Seed complete! Visit http://localhost:8000/django-admin/ with admin/admin123")
