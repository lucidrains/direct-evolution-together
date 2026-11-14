from datetime import timezone

from django.db import models


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
