from django.contrib import admin

from direct_evolution_together.models import ChatMessage, Room


@admin.register(Room)
class RoomAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "description", "created_at")
    search_fields = ("name", "slug", "description")
    ordering = ("-created_at", "-id")


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("room", "username", "message", "created_at")
    list_filter = ("room",)
    search_fields = ("username", "message")
    ordering = ("-created_at", "-id")
