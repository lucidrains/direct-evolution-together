from django.contrib import admin

from direct_evolution_together.models import ChatMessage


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("room", "username", "message", "created_at")
    list_filter = ("room",)
    search_fields = ("username", "message")
    ordering = ("-created_at", "-id")
