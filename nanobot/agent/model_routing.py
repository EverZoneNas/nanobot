"""Cache-aware task-based model routing."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import json_repair
from loguru import logger

from nanobot.agent.model_presets import PresetSnapshotLoader
from nanobot.config.schema import (
    DreamConfig,
    ModelPresetConfig,
    ModelRouteRule,
    ModelRoutingConfig,
    TaskComplexity,
    TaskKind,
    TaskType,
)
from nanobot.providers.factory import (
    ProviderSnapshot,
    runtime_provider_cache_identity,
)
from nanobot.session.goal_state import sustained_goal_turn
from nanobot.session.routing_state import (
    MODEL_ROUTING_AFFINITY_KEY,
    clear_model_routing_affinity,
)

_CLASSIFIER_MAX_TOKENS = 128
_USER_TEXT_MAX_CHARS = 2000
_COMPLEXITY_BENEFIT: dict[TaskComplexity, float] = {
    "low": 0.25,
    "medium": 0.60,
    "high": 1.0,
}

_CLASSIFIER_SYSTEM = """You classify user requests for model routing.
Output JSON only with this shape:
{"task_type":"coding|research|admin|chat|other","complexity":"low|medium|high","confidence":0.0,"reason":"brief"}

Guidelines:
- coding: implementation, debugging, refactors, shell automation, multi-file changes
- research: exploration, comparisons, reading docs or URLs, analysis
- admin: scheduling, configuration, reminders, lightweight operational tasks
- chat: simple Q&A, greetings, short explanations
- other: anything that does not fit above
- low: quick, single-step, or conversational
- medium: moderate scope, a few steps or files
- high: large, ambiguous, or multi-step work
- confidence: number from 0.0 to 1.0 indicating classification certainty
"""

BuildInlineSnapshot = Callable[[ModelPresetConfig], ProviderSnapshot]
RouteDecisionReason = Literal[
    "initial_candidate",
    "initial_baseline",
    "candidate_unchanged",
    "same_cache_identity",
    "switched_score",
    "kept_for_cache",
    "classifier_fallback",
    "no_candidate_affinity",
    "no_candidate_baseline",
    "deterministic_rule",
    "dream_override",
]


@dataclass(slots=True)
class RoutingContext:
    """Inputs used to resolve a model route."""

    user_text: str
    task_kind: TaskKind
    task_type: TaskType | None = None
    complexity: TaskComplexity | None = None
    confidence: float | None = None
    prompt_tokens_estimate: int = 0
    session_metadata: dict[str, Any] | None = None
    message_metadata: dict[str, Any] | None = None
    session_key: str | None = None


@dataclass(slots=True)
class TurnRoute:
    """Resolved model and generation settings for one agent run."""

    snapshot: ProviderSnapshot
    preset_name: str
    preset: ModelPresetConfig
    task_kind: TaskKind
    task_type: TaskType | None = None
    complexity: TaskComplexity | None = None
    confidence: float | None = None

    def to_run_spec_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.snapshot.model,
            "route_provider": self.snapshot.provider,
            "routed_preset": self.preset_name,
            "temperature": self.preset.temperature,
            "max_tokens": self.preset.max_tokens,
            "reasoning_effort": self.preset.reasoning_effort,
            "context_window_tokens": self.snapshot.context_window_tokens,
        }


@dataclass(slots=True)
class RoutingDecision:
    """Selected route plus cache-aware scoring diagnostics."""

    selected: TurnRoute
    candidate: TurnRoute | None
    reason: RouteDecisionReason
    prompt_tokens_estimate: int = 0
    quality_benefit: float | None = None
    cache_penalty: float | None = None
    switch_score: float | None = None
    estimated_reusable_tokens: int = 0


def infer_task_kind(
    *,
    session_key: str | None,
    session_metadata: dict[str, Any] | None,
    message_metadata: dict[str, Any] | None,
    explicit_task_kind: TaskKind | None = None,
) -> TaskKind:
    if explicit_task_kind is not None:
        return explicit_task_kind
    key = (session_key or "").strip()
    if key.startswith("dream:"):
        return "dream"
    if key == "heartbeat" or key.startswith("cron:"):
        return "cron"
    if sustained_goal_turn(session_metadata, message_metadata=message_metadata):
        return "sustained_goal"
    return "chat"


def extract_user_text(initial_messages: list[dict[str, Any]]) -> str:
    for message in reversed(initial_messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            if parts:
                return "\n".join(parts)
    return ""


def _truncate_user_text(text: str) -> str:
    text = text.strip()
    if len(text) <= _USER_TEXT_MAX_CHARS:
        return text
    return text[:_USER_TEXT_MAX_CHARS] + "…"


def _parse_classifier_response(
    content: str | None,
) -> tuple[TaskType | None, TaskComplexity | None, float | None]:
    if not content:
        return None, None, None
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            parsed = json_repair.loads(stripped)
        except Exception:
            return None, None, None
    if not isinstance(parsed, dict):
        return None, None, None

    task_type = parsed.get("task_type")
    complexity = parsed.get("complexity")
    valid_types = {"coding", "research", "admin", "chat", "other"}
    valid_complexity = {"low", "medium", "high"}
    resolved_type = task_type if task_type in valid_types else None
    resolved_complexity = complexity if complexity in valid_complexity else None
    if resolved_type is None or resolved_complexity is None:
        return resolved_type, resolved_complexity, None

    confidence = parsed.get("confidence", 0.5)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = 0.5
    return resolved_type, resolved_complexity, max(0.0, min(1.0, float(confidence)))


def _rule_matches(ctx: RoutingContext, rule: ModelRouteRule) -> bool:
    match = rule.match
    if match.task_kind is not None and ctx.task_kind != match.task_kind:
        return False
    if match.task_type is not None and ctx.task_type != match.task_type:
        return False
    if match.complexity is not None and ctx.complexity != match.complexity:
        return False
    return True


class ModelRouter:
    """Resolve model presets using task classification and cache affinity."""

    def __init__(
        self,
        *,
        routing: ModelRoutingConfig,
        dream: DreamConfig,
        load_preset: PresetSnapshotLoader,
        build_inline_snapshot: BuildInlineSnapshot,
        resolve_preset: Callable[[str], ModelPresetConfig],
    ) -> None:
        self._routing = routing
        self._dream = dream
        self._load_preset = load_preset
        self._build_inline_snapshot = build_inline_snapshot
        self._resolve_preset = resolve_preset
        self._classifier_snapshot: ProviderSnapshot | None = None
        self._classifier_signature: tuple[object, ...] | None = None

    @property
    def enabled(self) -> bool:
        return self._routing.enabled

    @staticmethod
    def _cache_identity(snapshot: ProviderSnapshot) -> str:
        return snapshot.cache_identity or runtime_provider_cache_identity(
            snapshot.provider,
            snapshot.model,
        )

    def _refresh_classifier_snapshot(self) -> ProviderSnapshot:
        signature = ("classifier", self._routing.classifier_preset)
        if self._classifier_snapshot is not None and self._classifier_signature == signature:
            return self._classifier_snapshot
        snapshot = self._load_preset(self._routing.classifier_preset)
        self._classifier_snapshot = snapshot
        self._classifier_signature = signature
        return snapshot

    def _snapshot_for_preset(self, preset_name: str) -> tuple[ProviderSnapshot, ModelPresetConfig]:
        preset = self._resolve_preset(preset_name)
        snapshot = self._load_preset(preset_name)
        snapshot.provider.generation = preset.to_generation_settings()
        return snapshot, preset

    def _route_for_preset(self, name: str, ctx: RoutingContext) -> TurnRoute:
        snapshot, preset = self._snapshot_for_preset(name)
        return TurnRoute(
            snapshot=snapshot,
            preset_name=name,
            preset=preset,
            task_kind=ctx.task_kind,
            task_type=ctx.task_type,
            complexity=ctx.complexity,
            confidence=ctx.confidence,
        )

    def _baseline_route(
        self,
        ctx: RoutingContext,
        baseline_snapshot: ProviderSnapshot,
        baseline_preset: str | None,
    ) -> TurnRoute:
        name = baseline_preset or "default"
        return TurnRoute(
            snapshot=baseline_snapshot,
            preset_name=name,
            preset=self._resolve_preset(name),
            task_kind=ctx.task_kind,
            task_type=ctx.task_type,
            complexity=ctx.complexity,
            confidence=ctx.confidence,
        )

    def _dream_override_snapshot(self) -> ProviderSnapshot | None:
        override = (self._dream.model_override or "").strip()
        if not override:
            return None
        return self._build_inline_snapshot(ModelPresetConfig(model=override, provider="auto"))

    def _match_rule(self, ctx: RoutingContext) -> str | None:
        for rule in self._routing.rules:
            if _rule_matches(ctx, rule):
                return rule.preset
        return self._routing.default_preset

    async def _classify(
        self,
        user_text: str,
    ) -> tuple[TaskType | None, TaskComplexity | None, float | None]:
        snapshot = self._refresh_classifier_snapshot()
        provider = snapshot.provider
        preset = self._resolve_preset(self._routing.classifier_preset)
        try:
            response = await provider.chat_with_retry(
                model=snapshot.model,
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM},
                    {"role": "user", "content": _truncate_user_text(user_text)},
                ],
                tools=None,
                tool_choice=None,
                max_tokens=_CLASSIFIER_MAX_TOKENS,
                temperature=0.0,
            )
        except Exception:
            logger.warning("Model routing classifier call failed")
            return None, None, None
        if response.finish_reason == "error":
            logger.warning(
                "Model routing classifier returned error: {}",
                (response.content or "")[:200],
            )
            return None, None, None
        task_type, complexity, confidence = _parse_classifier_response(response.content)
        logger.debug(
            "Model routing classifier: task_type={} complexity={} confidence={} model={}",
            task_type,
            complexity,
            confidence,
            preset.model,
        )
        return task_type, complexity, confidence

    def _load_affinity(self, ctx: RoutingContext) -> tuple[TurnRoute, dict[str, Any]] | None:
        metadata = ctx.session_metadata
        if not isinstance(metadata, dict):
            return None
        raw = metadata.get(MODEL_ROUTING_AFFINITY_KEY)
        if not isinstance(raw, dict):
            return None
        updated_at = raw.get("updated_at")
        if not isinstance(updated_at, (int, float)) or (
            time.time() - float(updated_at) > self._routing.affinity_ttl_seconds
        ):
            clear_model_routing_affinity(metadata)
            return None
        name = raw.get("preset")
        if not isinstance(name, str) or not name:
            clear_model_routing_affinity(metadata)
            return None
        try:
            route = self._route_for_preset(name, ctx)
        except (KeyError, ValueError):
            clear_model_routing_affinity(metadata)
            return None
        if raw.get("cache_identity") != self._cache_identity(route.snapshot):
            clear_model_routing_affinity(metadata)
            return None
        return route, raw

    async def resolve_turn_route(
        self,
        ctx: RoutingContext,
        *,
        baseline_snapshot: ProviderSnapshot,
        baseline_preset: str | None,
    ) -> RoutingDecision:
        baseline = self._baseline_route(ctx, baseline_snapshot, baseline_preset)

        if ctx.task_kind == "dream":
            override_snapshot = self._dream_override_snapshot()
            if override_snapshot is not None:
                route = TurnRoute(
                    snapshot=override_snapshot,
                    preset_name="dream:override",
                    preset=ModelPresetConfig(model=override_snapshot.model, provider="auto"),
                    task_kind=ctx.task_kind,
                )
                return RoutingDecision(route, route, "dream_override")

        if ctx.task_kind != "chat":
            preset_name = self._match_rule(ctx)
            if preset_name is None:
                return RoutingDecision(baseline, None, "no_candidate_baseline")
            candidate = self._route_for_preset(preset_name, ctx)
            return RoutingDecision(candidate, candidate, "deterministic_rule")

        task_type, complexity, confidence = await self._classify(ctx.user_text)
        ctx.task_type = task_type
        ctx.complexity = complexity
        ctx.confidence = confidence
        affinity = self._load_affinity(ctx)

        if task_type is None or complexity is None:
            if affinity is not None:
                return RoutingDecision(
                    affinity[0],
                    None,
                    "classifier_fallback",
                    prompt_tokens_estimate=ctx.prompt_tokens_estimate,
                )
            fallback_name = self._routing.default_preset
            if fallback_name is None:
                return RoutingDecision(
                    baseline,
                    None,
                    "initial_baseline",
                    prompt_tokens_estimate=ctx.prompt_tokens_estimate,
                )
            candidate = self._route_for_preset(fallback_name, ctx)
            return RoutingDecision(
                candidate,
                candidate,
                "initial_candidate",
                prompt_tokens_estimate=ctx.prompt_tokens_estimate,
            )

        preset_name = self._match_rule(ctx)
        if preset_name is None:
            if affinity is not None:
                return RoutingDecision(
                    affinity[0],
                    None,
                    "no_candidate_affinity",
                    prompt_tokens_estimate=ctx.prompt_tokens_estimate,
                )
            return RoutingDecision(
                baseline,
                None,
                "no_candidate_baseline",
                prompt_tokens_estimate=ctx.prompt_tokens_estimate,
            )

        candidate = self._route_for_preset(preset_name, ctx)
        if affinity is None:
            return RoutingDecision(
                candidate,
                candidate,
                "initial_candidate",
                prompt_tokens_estimate=ctx.prompt_tokens_estimate,
            )

        current, state = affinity
        if candidate.preset_name == current.preset_name:
            return RoutingDecision(
                candidate,
                candidate,
                "candidate_unchanged",
                prompt_tokens_estimate=ctx.prompt_tokens_estimate,
            )
        if self._cache_identity(candidate.snapshot) == self._cache_identity(current.snapshot):
            return RoutingDecision(
                candidate,
                candidate,
                "same_cache_identity",
                prompt_tokens_estimate=ctx.prompt_tokens_estimate,
            )

        previous_prompt = max(0, int(state.get("prompt_tokens_estimate") or 0))
        cached_tokens = max(0, int(state.get("cached_tokens") or 0))
        reusable_tokens = min(
            max(0, ctx.prompt_tokens_estimate),
            max(previous_prompt, cached_tokens),
        )
        resolved_confidence = confidence if confidence is not None else 0.5
        quality_benefit = resolved_confidence * _COMPLEXITY_BENEFIT[complexity]
        cache_penalty = self._routing.cache_weight * min(
            reusable_tokens / self._routing.warm_prefix_tokens,
            1.0,
        )
        switch_score = quality_benefit - cache_penalty
        selected = candidate if switch_score >= self._routing.switch_threshold else current
        reason: RouteDecisionReason = (
            "switched_score" if selected is candidate else "kept_for_cache"
        )
        return RoutingDecision(
            selected,
            candidate,
            reason,
            prompt_tokens_estimate=ctx.prompt_tokens_estimate,
            quality_benefit=quality_benefit,
            cache_penalty=cache_penalty,
            switch_score=switch_score,
            estimated_reusable_tokens=reusable_tokens,
        )

    def record_outcome(
        self,
        decision: RoutingDecision,
        *,
        session_metadata: dict[str, Any] | None,
        usage: dict[str, int],
        stop_reason: str,
    ) -> None:
        """Persist cache affinity after a successful normal chat run."""
        route = decision.selected
        if (
            route.task_kind != "chat"
            or not isinstance(session_metadata, dict)
            or stop_reason in {"error", "tool_error", "empty_final_response"}
        ):
            return
        session_metadata[MODEL_ROUTING_AFFINITY_KEY] = {
            "version": 1,
            "preset": route.preset_name,
            "model": route.snapshot.model,
            "cache_identity": self._cache_identity(route.snapshot),
            "task_type": route.task_type,
            "complexity": route.complexity,
            "confidence": route.confidence,
            "prompt_tokens_estimate": max(0, decision.prompt_tokens_estimate),
            "cached_tokens": max(0, int(usage.get("cached_tokens") or 0)),
            "updated_at": time.time(),
        }
