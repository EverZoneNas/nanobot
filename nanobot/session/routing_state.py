"""Session metadata helpers for cache-aware model routing."""

from __future__ import annotations

from typing import Any

MODEL_ROUTING_AFFINITY_KEY = "_model_routing_affinity"


def clear_model_routing_affinity(metadata: dict[str, Any] | None) -> None:
    if isinstance(metadata, dict):
        metadata.pop(MODEL_ROUTING_AFFINITY_KEY, None)
