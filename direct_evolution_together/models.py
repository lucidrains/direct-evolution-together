from datetime import timezone

from django.db import models


class Room(models.Model):
    """Metadata for a named private room."""

    slug = models.SlugField(max_length=64, unique=True)
    name = models.CharField(max_length=64)
    description = models.CharField(max_length=280, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self) -> str:
        return f"{self.name} (/p/{self.slug}/)"


class ChatMessage(models.Model):
    """Persisted room chat message."""

    room = models.CharField(max_length=64, db_index=True)
    username = models.CharField(max_length=24)
    message = models.TextField(max_length=2000)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["created_at", "id"]

    def to_payload(self) -> dict[str, str]:
        return {
            "username": self.username,
            "message": self.message,
            "timestamp": self.created_at.astimezone(timezone.utc).strftime("%H:%M:%S"),
        }

    def __str__(self) -> str:
        return f"[{self.room}] {self.username}: {self.message[:40]}"


class RoomEvolutionState(models.Model):
    """Persisted room evolution state so server restarts/deploys resume cleanly."""

    room = models.CharField(max_length=64, unique=True)
    state_json = models.TextField()
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"EvolutionState({self.room})"
