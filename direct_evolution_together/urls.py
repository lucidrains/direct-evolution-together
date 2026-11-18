from django.contrib import admin
from django.urls import path

from direct_evolution_together import views

urlpatterns = [
    path("", views.index, name="index"),
    path("p/<slug:room>/", views.index, name="private_room"),
    path("health/", views.health, name="health"),
    path("assets/<path:asset_name>", views.serve_asset, name="serve_asset"),
    path("api/config/", views.api_config, name="api_config"),
    path("api/rooms/", views.api_create_room, name="api_create_room"),
    path("api/rooms/<slug:room>/worker/jobs/", views.api_worker_jobs, name="api_worker_jobs"),
    path("api/rooms/<slug:room>/worker/results/", views.api_worker_results, name="api_worker_results"),
    path("api/rooms/<slug:room>/worker/evals/", views.api_worker_evals, name="api_worker_evals"),
    path("api/rooms/<slug:room>/crowd/results/", views.api_crowd_results, name="api_crowd_results"),
    path("api/rooms/<slug:room>/crowd/evals/", views.api_crowd_evals, name="api_crowd_evals"),
    path("api/sim/eval-batch/", views.api_sim_eval_batch, name="api_sim_eval_batch"),
    path("api/sim/rollout/", views.api_sim_rollout, name="api_sim_rollout"),
    path("api/sim/base-model/", views.api_sim_base_model, name="api_sim_base_model"),
    path("api/sim/scene/<slug:scene_id>/", views.api_sim_scene_manifest, name="api_sim_scene_manifest"),
    path("api/weights/validate/", views.api_validate_weights, name="api_validate_weights"),
    path("api/render/video/", views.api_render_video, name="api_render_video"),
    path("media/<path:rel_path>", views.serve_media, name="serve_media"),
    path("export/bundle/", views.export_bundle, name="export_bundle"),
    path("export/weights/", views.export_weights, name="export_weights"),
    path("export/script/", views.export_script, name="export_script"),
    path("export/worker-script/", views.export_worker_script, name="export_worker_script"),
    path("reset/", views.admin_reset_page, name="admin_reset"),
    path("api/reset/", views.api_reset, name="api_reset"),
    path("api/clear/", views.api_clear, name="api_clear"),
    path("admin/", admin.site.urls),
]
