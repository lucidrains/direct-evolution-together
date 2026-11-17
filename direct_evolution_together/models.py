from datetime import timezone

from django.db import models


class Room(models.Model):
    """Metadata for a named private room."""

    slug = models.SlugField(max_length=64, unique=True)
    name = models.CharField(max_length=64)
    description = models.CharField(max_length=280, blank=True, default="")
    worker_token_hash = models.CharField(max_length=64, blank=True, default="")
    external_spec_json = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self) -> str:
        return f"{self.name} (/p/{self.slug}/)"


class ExternalEvalResult(models.Model):
    """Unified simulation/evaluation consensus record (crowd WASM/WebGPU peers or researcher workers)."""

    room = models.CharField(max_length=64, db_index=True)
    goal = models.CharField(max_length=64, db_index=True)
    generation = models.IntegerField(default=0, db_index=True)
    genome_hash = models.CharField(max_length=32, db_index=True)
    status = models.CharField(max_length=16, default="pending", db_index=True)  # pending | verified | disputed
    attestations_json = models.TextField(default="[]")
    payload_json = models.TextField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["room", "goal", "generation", "genome_hash"],
                name="uniq_external_eval",
            )
        ]

    def __str__(self) -> str:
        return f"EvalRecord({self.room}/{self.goal} g{self.generation} {self.genome_hash} [{self.status}])"


class ContributorTrust(models.Model):
    """Per-room reputation record for crowd WASM/WebGPU simulation contributors."""

    room = models.CharField(max_length=64, db_index=True)
    contributor_id = models.CharField(max_length=64, db_index=True)
    reputation = models.FloatField(default=0.80)
    submissions = models.IntegerField(default=0)
    agreements = models.IntegerField(default=0)
    mismatches = models.IntegerField(default=0)
    quarantined = models.BooleanField(default=False, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["room", "contributor_id"],
                name="uniq_room_contributor_trust",
            )
        ]

    def __str__(self) -> str:
        flag = " [QUARANTINED]" if self.quarantined else ""
        return f"ContributorTrust({self.room}/{self.contributor_id}: rep={self.reputation:.2f}{flag})"


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
