from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('django-admin/', admin.site.urls),
    path('api/auth/', include('accounts.urls')),
    path('api/logs/', include('logs_app.urls')),
    path('api/', include('ai_inference.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
