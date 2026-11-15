from django.contrib import admin
from django.urls import path

from direct_evolution_together import views

urlpatterns = [
    path("", views.index, name="index"),
    path("p/<slug:room>/", views.index, name="private_room"),
    path("health/", views.health, name="health"),
    path("api/rooms/", views.api_create_room, name="api_create_room"),
    path("api/weights/validate/", views.api_validate_weights, name="api_validate_weights"),
    path("export/bundle/", views.export_bundle, name="export_bundle"),
    path("export/weights/", views.export_weights, name="export_weights"),
    path("export/script/", views.export_script, name="export_script"),
    path("reset/", views.admin_reset_page, name="admin_reset"),
    path("api/reset/", views.api_reset, name="api_reset"),
    path("api/clear/", views.api_clear, name="api_clear"),
    path("admin/", admin.site.urls),
]
