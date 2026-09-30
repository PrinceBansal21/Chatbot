"""Deployment dependency models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DeploymentDependencies:
    """Represents non-code cloud dependencies declared in configuration."""

    container_repository: Optional[str] = None
    queues: Dict[str, str] = field(default_factory=dict)
    config_parameters: Dict[str, str] = field(default_factory=dict)
    vectordb_index: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> "DeploymentDependencies":
        data = raw or {}
        return cls(
            container_repository=data.get("container_repository"),
            queues=data.get("queues", {}) or {},
            config_parameters=data.get("config_parameters", {}) or {},
            vectordb_index=data.get("vectordb_index"),
        )

    def resolved_queue(self, queue_type: str, fallback: Optional[str] = None) -> Optional[str]:
        return self.queues.get(queue_type) or fallback
