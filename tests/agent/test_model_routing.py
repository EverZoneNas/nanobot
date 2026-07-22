"""Tests for run-based per-turn model routing."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.model_routing import (
    ModelRouter,
    RoutingContext,
    RoutingDecision,
    TurnRoute,
    _parse_classifier_response,
    _rule_matches,
    infer_run_kind,
    infer_task_kind,
)
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import (
    Config,
    DreamConfig,
    ModelPresetConfig,
    ModelRouteMatch,
    ModelRouteRule,
    ModelRoutingConfig,
)
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.factory import ProviderSnapshot, provider_cache_identity
from nanobot.session.routing_state import MODEL_ROUTING_AFFINITY_KEY


def _provider(model: str = "test-model") -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = model
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok"))
    provider.chat = AsyncMock(
        return_value=LLMResponse(
            content='{"task_type":"coding","complexity":"high","reason":"debug"}',
        ),
    )
    return provider


def _snapshot(
    model: str,
    provider: MagicMock | None = None,
    *,
    cache_identity: str | None = None,
) -> ProviderSnapshot:
    return ProviderSnapshot(
        provider=provider or _provider(model),
        model=model,
        context_window_tokens=128_000,
        signature=("test", model),
        cache_identity=cache_identity or model,
    )


def _router(
    *,
    routing: ModelRoutingConfig,
    presets: dict[str, ModelPresetConfig] | None = None,
    classifier_response: str | None = None,
    dream: DreamConfig | None = None,
) -> ModelRouter:
    presets = presets or {
        "fast": ModelPresetConfig(model="fast-model", provider="auto"),
        "deep": ModelPresetConfig(model="deep-model", provider="auto"),
    }
    classifier_provider = _provider("classifier-model")
    classifier_provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=classifier_response
            or '{"task_type":"coding","complexity":"high","reason":"debug"}',
        ),
    )

    def load_preset(name: str) -> ProviderSnapshot:
        preset = presets[name if name != "default" else "fast"]
        if name == routing.classifier_preset:
            return _snapshot(preset.model, classifier_provider)
        return _snapshot(preset.model, _provider(preset.model))

    def resolve_preset(name: str) -> ModelPresetConfig:
        if name == "default":
            return presets["fast"]
        return presets[name]

    return ModelRouter(
        routing=routing,
        dream=dream or Config().agents.defaults.dream,
        load_preset=load_preset,
        build_inline_snapshot=lambda preset: _snapshot(preset.model),
        resolve_preset=resolve_preset,
    )


def test_config_rejects_unknown_routing_preset() -> None:
    with pytest.raises(ValueError, match="smart_model_routing"):
        Config.model_validate({
            "modelPresets": {
                "fast": {"model": "fast-model", "provider": "auto"},
            },
            "agents": {
                "defaults": {
                    "smartModelRouting": {
                        "enabled": True,
                        "classifierPreset": "fast",
                        "rules": [{"match": {"complexity": "low"}, "preset": "missing"}],
                    }
                }
            },
        })


def test_infer_run_kind_for_subagent_and_dream() -> None:
    assert infer_run_kind(
        session_key="cli:direct",
        session_metadata=None,
        message_metadata=None,
        explicit_run_kind="subagent",
    ) == "subagent"
    assert infer_run_kind(
        session_key="dream:20260101-120000",
        session_metadata=None,
        message_metadata=None,
    ) == "dream"
    assert infer_run_kind(
        session_key="cron:job-1",
        session_metadata=None,
        message_metadata=None,
    ) == "cron"
    assert infer_task_kind(
        session_key=None,
        session_metadata=None,
        message_metadata=None,
        explicit_task_kind="subagent",
    ) == "subagent"


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("_cron_trigger", "cron"),
        ("_local_trigger", "local_trigger"),
    ],
)
def test_infer_run_kind_for_bound_automation_precedes_active_goal(
    marker: str,
    expected: str,
) -> None:
    assert infer_run_kind(
        session_key="websocket:existing-session",
        session_metadata={"goal_state": {"status": "active", "objective": "ship release"}},
        message_metadata={marker: {"id": "automation-1"}},
    ) == expected


def test_infer_run_kind_authoritative_precedence_and_marker_shape() -> None:
    local_marker = {"_local_trigger": {"trigger_id": "trigger-1"}}
    assert infer_run_kind(
        session_key="dream:20260722-120000",
        session_metadata=None,
        message_metadata=local_marker,
    ) == "dream"
    assert infer_run_kind(
        session_key="heartbeat",
        session_metadata=None,
        message_metadata=local_marker,
    ) == "cron"
    assert infer_run_kind(
        session_key="websocket:session",
        session_metadata={"goal_state": {"status": "active"}},
        message_metadata=local_marker,
        explicit_run_kind="subagent",
    ) == "subagent"
    assert infer_run_kind(
        session_key="websocket:session",
        session_metadata=None,
        message_metadata={"_local_trigger": True},
    ) == "chat"


@pytest.mark.parametrize("alias", ["runKind", "run_kind", "taskKind", "task_kind"])
def test_model_route_match_accepts_run_kind_aliases(alias: str) -> None:
    match = ModelRouteMatch.model_validate({alias: "cron"})

    assert match.run_kind == "cron"
    assert match.task_kind == "cron"
    dumped = match.model_dump(by_alias=True)
    assert dumped["runKind"] == "cron"
    assert "taskKind" not in dumped


def test_rule_matching_precedence() -> None:
    ctx = RoutingContext(user_text="fix bug", run_kind="chat", task_type="coding", complexity="high")
    rules = [
        ModelRouteRule(match=ModelRouteMatch(complexity="low"), preset="fast"),
        ModelRouteRule(match=ModelRouteMatch(task_type="coding", complexity="high"), preset="deep"),
    ]
    assert _rule_matches(ctx, rules[0]) is False
    assert _rule_matches(ctx, rules[1]) is True


def test_parse_classifier_response_accepts_json_and_fenced_json() -> None:
    task_type, complexity, confidence = _parse_classifier_response(
        '```json\n{"task_type":"research","complexity":"medium","reason":"docs"}\n```'
    )
    assert task_type == "research"
    assert complexity == "medium"
    assert confidence == 0.5
    assert _parse_classifier_response("not json") == (None, None, None)


def test_config_parses_cache_routing_defaults_and_aliases() -> None:
    defaults = ModelRoutingConfig()
    assert defaults.affinity_ttl_seconds == 300
    assert defaults.cache_weight == 0.65
    assert defaults.switch_threshold == 0.15
    assert defaults.warm_prefix_tokens == 16_000

    routing = ModelRoutingConfig.model_validate({
        "affinityTtlSeconds": 120,
        "cacheWeight": 0.4,
        "switchThreshold": 0.2,
        "warmPrefixTokens": 8000,
    })
    assert routing.affinity_ttl_seconds == 120
    assert routing.cache_weight == 0.4
    assert routing.switch_threshold == 0.2
    assert routing.warm_prefix_tokens == 8000
    assert routing.model_dump(by_alias=True) == {
        "enabled": False,
        "classifierPreset": "fast",
        "rules": [],
        "defaultPreset": None,
        "affinityTtlSeconds": 120,
        "cacheWeight": 0.4,
        "switchThreshold": 0.2,
        "warmPrefixTokens": 8000,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("affinityTtlSeconds", 0),
        ("cacheWeight", -0.1),
        ("cacheWeight", 1.1),
        ("switchThreshold", -1.1),
        ("switchThreshold", 1.1),
        ("warmPrefixTokens", 0),
    ],
)
def test_config_rejects_invalid_cache_routing_bounds(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        ModelRoutingConfig.model_validate({field: value})


def test_cache_identity_excludes_generation_settings_and_hides_credentials() -> None:
    config = Config.model_validate({
        "providers": {"openai": {"apiKey": "sk-secret"}},
        "modelPresets": {
            "fast": {
                "model": "openai/gpt-4.1",
                "provider": "openai",
                "temperature": 0.0,
            },
            "deep": {
                "model": "openai/gpt-4.1",
                "provider": "auto",
                "temperature": 0.8,
                "reasoningEffort": "high",
            },
            "other": {
                "model": "openai/gpt-4.1-mini",
                "provider": "openai",
            },
        },
    })
    fast = provider_cache_identity(config, preset_name="fast")
    deep = provider_cache_identity(config, preset_name="deep")
    other = provider_cache_identity(config, preset_name="other")

    assert fast == deep
    assert fast != other
    assert "sk-secret" not in fast


@pytest.mark.asyncio
async def test_zero_classifier_confidence_has_no_quality_benefit() -> None:
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            cache_weight=0.0,
            rules=[ModelRouteRule(match=ModelRouteMatch(complexity="high"), preset="deep")],
        ),
        classifier_response=(
            '{"task_type":"coding","complexity":"high","confidence":0.0,"reason":"uncertain"}'
        ),
    )
    metadata = {
        MODEL_ROUTING_AFFINITY_KEY: {
            "preset": "fast",
            "cache_identity": "fast-model",
            "prompt_tokens_estimate": 1,
            "updated_at": time.time(),
        }
    }
    decision = await router.resolve_turn_route(
        RoutingContext(
            user_text="maybe refactor",
            run_kind="chat",
            prompt_tokens_estimate=1,
            session_metadata=metadata,
        ),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )
    assert decision.quality_benefit == 0.0
    assert decision.selected.preset_name == "fast"
    assert decision.reason == "kept_for_cache"


@pytest.mark.asyncio
async def test_resolve_turn_route_uses_classifier_for_chat() -> None:
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[
                ModelRouteRule(
                    match=ModelRouteMatch(task_type="coding", complexity="high"),
                    preset="deep",
                ),
            ],
        ),
    )
    decision = await router.resolve_turn_route(
        RoutingContext(
            user_text="refactor the auth module",
            run_kind="chat",
            session_metadata={},
        ),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )
    assert isinstance(decision, RoutingDecision)
    route = decision.selected
    assert isinstance(route, TurnRoute)
    assert route.preset_name == "deep"
    assert route.snapshot.model == "deep-model"
    assert route.task_type == "coding"
    assert route.complexity == "high"
    assert decision.reason == "initial_candidate"


@pytest.mark.parametrize(
    "run_kind",
    ["subagent", "cron", "local_trigger", "dream", "sustained_goal"],
)
@pytest.mark.asyncio
async def test_resolve_turn_route_uses_semantic_rules_for_every_run_kind(
    run_kind: str,
) -> None:
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[
                ModelRouteRule(
                    match=ModelRouteMatch(
                        run_kind=run_kind,
                        task_type="coding",
                        complexity="high",
                    ),
                    preset="deep",
                ),
            ],
        ),
    )
    metadata: dict = {}

    decision = await router.resolve_turn_route(
        RoutingContext(
            user_text="refactor the auth module",
            run_kind=run_kind,
            session_metadata=metadata,
        ),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )

    assert decision.selected.preset_name == "deep"
    assert decision.selected.run_kind == run_kind
    assert decision.selected.task_type == "coding"
    assert decision.selected.complexity == "high"
    router.record_outcome(
        decision,
        session_metadata=metadata,
        usage={"cached_tokens": 100},
        stop_reason="completed",
    )
    assert MODEL_ROUTING_AFFINITY_KEY not in metadata


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("_cron_trigger", "cron"),
        ("_local_trigger", "local_trigger"),
    ],
)
@pytest.mark.asyncio
async def test_bound_automation_inference_routes_with_semantic_rules(
    marker: str,
    expected: str,
) -> None:
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[
                ModelRouteRule(
                    match=ModelRouteMatch(
                        run_kind=expected,
                        task_type="coding",
                        complexity="high",
                    ),
                    preset="deep",
                ),
            ],
        ),
    )
    session_metadata = {
        "goal_state": {"status": "active", "objective": "keep the release moving"}
    }
    run_kind = infer_run_kind(
        session_key="websocket:existing-session",
        session_metadata=session_metadata,
        message_metadata={marker: {"id": "automation-1"}},
    )

    decision = await router.resolve_turn_route(
        RoutingContext(
            user_text="refactor the release automation",
            run_kind=run_kind,
            session_metadata=session_metadata,
        ),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )

    assert decision.selected.run_kind == expected
    assert decision.selected.preset_name == "deep"
    assert decision.selected.task_type == "coding"
    assert decision.selected.complexity == "high"


@pytest.mark.asyncio
async def test_resolve_turn_route_selects_same_baseline_candidate() -> None:
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[
                ModelRouteRule(match=ModelRouteMatch(complexity="low"), preset="fast"),
            ],
        ),
        classifier_response='{"task_type":"chat","complexity":"low","reason":"hi"}',
    )
    decision = await router.resolve_turn_route(
        RoutingContext(user_text="hello", run_kind="chat", session_metadata={}),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )
    assert decision.selected.preset_name == "fast"
    assert decision.reason == "initial_candidate"


@pytest.mark.asyncio
async def test_deterministic_subagent_run_kind_skips_classifier() -> None:
    classifier_provider = _provider("classifier-model")
    classifier_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="should not run"))
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[ModelRouteRule(match=ModelRouteMatch(run_kind="subagent"), preset="deep")],
        ),
    )
    router._refresh_classifier_snapshot = lambda: _snapshot("classifier-model", classifier_provider)  # type: ignore[method-assign]
    decision = await router.resolve_turn_route(
        RoutingContext(user_text="background task", run_kind="subagent"),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )
    classifier_provider.chat_with_retry.assert_not_called()
    assert decision.selected.preset_name == "deep"
    assert decision.reason == "deterministic_rule"


@pytest.mark.asyncio
async def test_irrelevant_semantic_rule_does_not_invoke_classifier() -> None:
    classifier_provider = _provider("classifier-model")
    classifier_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="should not run"))
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[
                ModelRouteRule(
                    match=ModelRouteMatch(run_kind="chat", task_type="coding"),
                    preset="deep",
                ),
                ModelRouteRule(match=ModelRouteMatch(run_kind="cron"), preset="fast"),
            ],
        ),
    )
    router._refresh_classifier_snapshot = lambda: _snapshot("classifier-model", classifier_provider)  # type: ignore[method-assign]

    decision = await router.resolve_turn_route(
        RoutingContext(user_text="nightly cleanup", run_kind="cron"),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )

    classifier_provider.chat_with_retry.assert_not_called()
    assert decision.selected.preset_name == "fast"
    assert decision.reason == "deterministic_rule"


@pytest.mark.asyncio
async def test_known_semantic_mismatch_does_not_invoke_classifier() -> None:
    classifier_provider = _provider("classifier-model")
    classifier_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="should not run"))
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[
                ModelRouteRule(
                    match=ModelRouteMatch(
                        run_kind="cron",
                        task_type="coding",
                        complexity="high",
                    ),
                    preset="deep",
                ),
                ModelRouteRule(match=ModelRouteMatch(run_kind="cron"), preset="fast"),
            ],
        ),
    )
    router._refresh_classifier_snapshot = lambda: _snapshot("classifier-model", classifier_provider)  # type: ignore[method-assign]

    decision = await router.resolve_turn_route(
        RoutingContext(
            user_text="nightly cleanup",
            run_kind="cron",
            complexity="low",
        ),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )

    classifier_provider.chat_with_retry.assert_not_called()
    assert decision.selected.preset_name == "fast"
    assert decision.reason == "deterministic_rule"


@pytest.mark.asyncio
async def test_dream_override_precedes_semantic_classification() -> None:
    classifier_provider = _provider("classifier-model")
    classifier_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="should not run"))
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[
                ModelRouteRule(
                    match=ModelRouteMatch(run_kind="dream", complexity="high"),
                    preset="deep",
                ),
            ],
        ),
        dream=DreamConfig(model_override="override-model"),
    )
    router._refresh_classifier_snapshot = lambda: _snapshot("classifier-model", classifier_provider)  # type: ignore[method-assign]

    decision = await router.resolve_turn_route(
        RoutingContext(user_text="consolidate memory", run_kind="dream"),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )

    classifier_provider.chat_with_retry.assert_not_called()
    assert decision.selected.snapshot.model == "override-model"
    assert decision.selected.run_kind == "dream"
    assert decision.reason == "dream_override"


@pytest.mark.asyncio
async def test_warm_cache_keeps_current_model_when_score_is_below_threshold() -> None:
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[ModelRouteRule(match=ModelRouteMatch(complexity="high"), preset="deep")],
        ),
        classifier_response=(
            '{"task_type":"coding","complexity":"high","confidence":0.7,"reason":"work"}'
        ),
    )
    metadata = {
        MODEL_ROUTING_AFFINITY_KEY: {
            "preset": "fast",
            "cache_identity": "fast-model",
            "prompt_tokens_estimate": 16_000,
            "cached_tokens": 12_000,
            "updated_at": time.time(),
        }
    }
    decision = await router.resolve_turn_route(
        RoutingContext(
            user_text="continue",
            run_kind="chat",
            prompt_tokens_estimate=16_000,
            session_metadata=metadata,
        ),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )
    assert decision.selected.preset_name == "fast"
    assert decision.candidate is not None
    assert decision.candidate.preset_name == "deep"
    assert decision.reason == "kept_for_cache"
    assert decision.switch_score == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_high_benefit_switches_when_cache_is_small() -> None:
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[ModelRouteRule(match=ModelRouteMatch(complexity="high"), preset="deep")],
        ),
        classifier_response=(
            '{"task_type":"coding","complexity":"high","confidence":0.9,"reason":"work"}'
        ),
    )
    metadata = {
        MODEL_ROUTING_AFFINITY_KEY: {
            "preset": "fast",
            "cache_identity": "fast-model",
            "prompt_tokens_estimate": 2_000,
            "cached_tokens": 0,
            "updated_at": time.time(),
        }
    }
    decision = await router.resolve_turn_route(
        RoutingContext(
            user_text="large refactor",
            run_kind="chat",
            prompt_tokens_estimate=4_000,
            session_metadata=metadata,
        ),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )
    assert decision.selected.preset_name == "deep"
    assert decision.reason == "switched_score"
    assert decision.switch_score == pytest.approx(0.81875)


@pytest.mark.asyncio
async def test_same_cache_identity_switches_presets_without_penalty() -> None:
    presets = {
        "fast": ModelPresetConfig(model="shared-model", provider="auto"),
        "deep": ModelPresetConfig(
            model="shared-model",
            provider="auto",
            reasoning_effort="high",
        ),
    }
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[ModelRouteRule(match=ModelRouteMatch(complexity="high"), preset="deep")],
        ),
        presets=presets,
        classifier_response=(
            '{"task_type":"coding","complexity":"high","confidence":0.8,"reason":"work"}'
        ),
    )
    metadata = {
        MODEL_ROUTING_AFFINITY_KEY: {
            "preset": "fast",
            "cache_identity": "shared-model",
            "prompt_tokens_estimate": 40_000,
            "cached_tokens": 30_000,
            "updated_at": time.time(),
        }
    }
    decision = await router.resolve_turn_route(
        RoutingContext(
            user_text="reason more deeply",
            run_kind="chat",
            prompt_tokens_estimate=40_000,
            session_metadata=metadata,
        ),
        baseline_snapshot=_snapshot("shared-model"),
        baseline_preset="fast",
    )
    assert decision.selected.preset_name == "deep"
    assert decision.reason == "same_cache_identity"
    assert decision.cache_penalty is None


@pytest.mark.asyncio
async def test_classifier_failure_keeps_unexpired_affinity() -> None:
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            rules=[ModelRouteRule(match=ModelRouteMatch(complexity="high"), preset="deep")],
        ),
        classifier_response="not json",
    )
    metadata = {
        MODEL_ROUTING_AFFINITY_KEY: {
            "preset": "deep",
            "cache_identity": "deep-model",
            "prompt_tokens_estimate": 8_000,
            "cached_tokens": 4_000,
            "updated_at": time.time(),
        }
    }
    decision = await router.resolve_turn_route(
        RoutingContext(user_text="continue", run_kind="chat", session_metadata=metadata),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )
    assert decision.selected.preset_name == "deep"
    assert decision.reason == "classifier_fallback"


@pytest.mark.asyncio
async def test_expired_affinity_is_removed_before_routing() -> None:
    router = _router(
        routing=ModelRoutingConfig(
            enabled=True,
            classifier_preset="fast",
            affinity_ttl_seconds=10,
            rules=[ModelRouteRule(match=ModelRouteMatch(complexity="high"), preset="deep")],
        ),
    )
    metadata = {
        MODEL_ROUTING_AFFINITY_KEY: {
            "preset": "fast",
            "cache_identity": "fast-model",
            "updated_at": time.time() - 11,
        }
    }
    decision = await router.resolve_turn_route(
        RoutingContext(user_text="refactor", run_kind="chat", session_metadata=metadata),
        baseline_snapshot=_snapshot("fast-model"),
        baseline_preset="fast",
    )
    assert decision.selected.preset_name == "deep"
    assert decision.reason == "initial_candidate"


def test_record_outcome_persists_success_and_ignores_failed_runs() -> None:
    router = _router(routing=ModelRoutingConfig(enabled=True, classifier_preset="fast"))
    route = TurnRoute(
        snapshot=_snapshot("deep-model"),
        preset_name="deep",
        preset=ModelPresetConfig(model="deep-model"),
        run_kind="chat",
        task_type="coding",
        complexity="high",
        confidence=0.9,
    )
    decision = RoutingDecision(
        selected=route,
        candidate=route,
        reason="initial_candidate",
        prompt_tokens_estimate=12_000,
    )
    metadata: dict = {}
    router.record_outcome(
        decision,
        session_metadata=metadata,
        usage={"cached_tokens": 6_000},
        stop_reason="completed",
    )
    state = metadata[MODEL_ROUTING_AFFINITY_KEY]
    assert state["preset"] == "deep"
    assert state["prompt_tokens_estimate"] == 12_000
    assert state["cached_tokens"] == 6_000

    router.record_outcome(
        RoutingDecision(
            selected=TurnRoute(
                snapshot=_snapshot("fast-model"),
                preset_name="fast",
                preset=ModelPresetConfig(model="fast-model"),
                run_kind="chat",
            ),
            candidate=None,
            reason="initial_baseline",
        ),
        session_metadata=metadata,
        usage={},
        stop_reason="error",
    )
    assert metadata[MODEL_ROUTING_AFFINITY_KEY]["preset"] == "deep"

    router.record_outcome(
        RoutingDecision(
            selected=TurnRoute(
                snapshot=_snapshot("fast-model"),
                preset_name="fast",
                preset=ModelPresetConfig(model="fast-model"),
                run_kind="chat",
            ),
            candidate=None,
            reason="initial_baseline",
        ),
        session_metadata=metadata,
        usage={"cached_tokens": 100},
        stop_reason="empty_final_response",
    )
    assert metadata[MODEL_ROUTING_AFFINITY_KEY]["preset"] == "deep"


@pytest.mark.asyncio
async def test_runner_uses_route_provider_without_changing_default() -> None:
    default_provider = _provider("default-model")
    routed_provider = _provider("routed-model")
    routed_provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="routed reply"),
    )
    default_provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="default reply"),
    )
    runner = AgentRunner(default_provider)
    tools = ToolRegistry()
    result = await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        model="routed-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        route_provider=routed_provider,
        routed_preset="deep",
    ))
    routed_provider.chat_with_retry.assert_awaited()
    default_provider.chat_with_retry.assert_not_awaited()
    assert result.final_content == "routed reply"
    assert runner.provider is default_provider
