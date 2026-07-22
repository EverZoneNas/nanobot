#!/usr/bin/env python3
"""Evaluate the configured model-routing classifier against curated labels."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanobot.agent.model_routing import (  # noqa: E402
    _CLASSIFIER_MAX_TOKENS,
    _build_classifier_system,
    _parse_classifier_response,
    _truncate_user_text,
)
from nanobot.config.loader import load_config, resolve_config_env_vars  # noqa: E402
from nanobot.providers.factory import build_provider_snapshot  # noqa: E402

DEFAULT_DATASET = (
    REPO_ROOT / "tests" / "fixtures" / "model_routing" / "classifier_eval.json"
)
DEFAULT_TASK_THRESHOLD = 0.80
DEFAULT_COMPLEXITY_THRESHOLD = 0.80
DEFAULT_JOINT_THRESHOLD = 0.65
COMPLEXITIES = {"low", "medium", "high"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    text: str
    task_type: str
    complexity: str


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    predicted_task_type: str | None
    predicted_complexity: str | None
    error: str | None = None


def _threshold(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real model-routing classifier against the curated, human-labeled "
            "offline dataset."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config.json",
        help="nanobot config path (default: repository config.json)",
    )
    parser.add_argument(
        "--preset",
        help="classifier model preset (default: smartModelRouting.classifierPreset)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="classifier fixture path",
    )
    parser.add_argument(
        "--min-task-type-accuracy",
        type=_threshold,
        default=DEFAULT_TASK_THRESHOLD,
        metavar="FLOAT",
    )
    parser.add_argument(
        "--min-complexity-accuracy",
        type=_threshold,
        default=DEFAULT_COMPLEXITY_THRESHOLD,
        metavar="FLOAT",
    )
    parser.add_argument(
        "--min-joint-accuracy",
        type=_threshold,
        default=DEFAULT_JOINT_THRESHOLD,
        metavar="FLOAT",
    )
    return parser


def _load_dataset(path: Path, valid_task_types: set[str]) -> list[EvalCase]:
    import json

    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read dataset {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("dataset must be a non-empty JSON array")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    required_fields = {"id", "text", "task_type", "complexity"}
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != required_fields:
            raise ValueError(f"dataset row {index} must contain exactly {sorted(required_fields)}")
        if not all(isinstance(item[field], str) and item[field].strip() for field in required_fields):
            raise ValueError(f"dataset row {index} fields must be non-empty strings")
        if item["id"] in seen_ids:
            raise ValueError(f"duplicate dataset id: {item['id']}")
        if item["task_type"] not in valid_task_types:
            raise ValueError(
                f"dataset row {item['id']} uses inactive task type {item['task_type']!r}"
            )
        if item["complexity"] not in COMPLEXITIES:
            raise ValueError(
                f"dataset row {item['id']} uses invalid complexity {item['complexity']!r}"
            )
        seen_ids.add(item["id"])
        cases.append(
            EvalCase(
                id=item["id"],
                text=item["text"],
                task_type=item["task_type"],
                complexity=item["complexity"],
            )
        )
    return cases


async def _evaluate(
    cases: list[EvalCase],
    *,
    provider: Any,
    model: str,
    task_types: dict[str, Any],
) -> list[EvalResult]:
    system_prompt = _build_classifier_system(task_types)
    results: list[EvalResult] = []
    for index, case in enumerate(cases, start=1):
        print(f"\rClassifying {index}/{len(cases)}", end="", flush=True)
        try:
            response = await provider.chat_with_retry(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _truncate_user_text(case.text)},
                ],
                tools=None,
                tool_choice=None,
                max_tokens=_CLASSIFIER_MAX_TOKENS,
                temperature=0.0,
            )
            if response.finish_reason == "error":
                results.append(
                    EvalResult(
                        case,
                        None,
                        None,
                        f"provider error: {(response.content or '')[:200]}",
                    )
                )
                continue
            task_type, complexity, _confidence = _parse_classifier_response(
                response.content,
                task_types,
            )
            error = None
            if task_type is None or complexity is None:
                error = f"unparseable response: {(response.content or '')[:200]}"
            results.append(EvalResult(case, task_type, complexity, error))
        except Exception as exc:
            results.append(EvalResult(case, None, None, f"call failed: {exc}"[:200]))
    print()
    return results


def _accuracy(matches: int, total: int) -> float:
    return matches / total if total else 0.0


def _print_report(
    results: list[EvalResult],
    *,
    preset_name: str,
    model: str,
    task_threshold: float,
    complexity_threshold: float,
    joint_threshold: float,
) -> bool:
    total = len(results)
    task_matches = sum(
        result.predicted_task_type == result.case.task_type for result in results
    )
    complexity_matches = sum(
        result.predicted_complexity == result.case.complexity for result in results
    )
    joint_matches = sum(
        result.predicted_task_type == result.case.task_type
        and result.predicted_complexity == result.case.complexity
        for result in results
    )
    task_accuracy = _accuracy(task_matches, total)
    complexity_accuracy = _accuracy(complexity_matches, total)
    joint_accuracy = _accuracy(joint_matches, total)
    error_count = sum(result.error is not None for result in results)

    print(f"Classifier eval: preset={preset_name} model={model} cases={total}")

    task_confusion = Counter(
        (result.case.task_type, result.predicted_task_type or "<error>")
        for result in results
        if result.predicted_task_type != result.case.task_type
    )
    complexity_confusion = Counter(
        (result.case.complexity, result.predicted_complexity or "<error>")
        for result in results
        if result.predicted_complexity != result.case.complexity
    )
    if task_confusion:
        print("task_type confusion (expected -> predicted):")
        for (expected, predicted), count in sorted(task_confusion.items()):
            print(f"  {expected} -> {predicted}: {count}")
    if complexity_confusion:
        print("complexity confusion (expected -> predicted):")
        for (expected, predicted), count in sorted(complexity_confusion.items()):
            print(f"  {expected} -> {predicted}: {count}")

    failures = [
        result
        for result in results
        if result.predicted_task_type != result.case.task_type
        or result.predicted_complexity != result.case.complexity
    ]
    if failures:
        print("mismatches:")
        for result in failures:
            predicted = (
                f"{result.predicted_task_type or '<error>'}/"
                f"{result.predicted_complexity or '<error>'}"
            )
            detail = f" ({result.error})" if result.error else ""
            print(
                f"  {result.case.id}: expected={result.case.task_type}/"
                f"{result.case.complexity} predicted={predicted}{detail}"
            )

    passed = (
        task_accuracy >= task_threshold
        and complexity_accuracy >= complexity_threshold
        and joint_accuracy >= joint_threshold
    )
    status = "PASS" if passed else "FAIL"
    print(
        f"{status}: thresholds task_type>={task_threshold:.2f}, "
        f"complexity>={complexity_threshold:.2f}, joint>={joint_threshold:.2f}"
    )
    print(f"task_type accuracy: {task_accuracy:.3f} ({task_matches}/{total})")
    print(f"complexity accuracy: {complexity_accuracy:.3f} ({complexity_matches}/{total})")
    print(f"joint exact-match: {joint_accuracy:.3f} ({joint_matches}/{total})")
    print(f"call/parse errors: {error_count}")
    return passed


async def _run(args: argparse.Namespace) -> int:
    try:
        config = resolve_config_env_vars(load_config(args.config))
        routing = config.agents.defaults.smart_model_routing
        preset_name = args.preset or routing.classifier_preset
        task_types = routing.task_type_definitions
        cases = _load_dataset(args.dataset, set(task_types))
        snapshot = build_provider_snapshot(config, preset_name=preset_name)
    except (KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = await _evaluate(
        cases,
        provider=snapshot.provider,
        model=snapshot.model,
        task_types=task_types,
    )
    passed = _print_report(
        results,
        preset_name=preset_name,
        model=snapshot.model,
        task_threshold=args.min_task_type_accuracy,
        complexity_threshold=args.min_complexity_accuracy,
        joint_threshold=args.min_joint_accuracy,
    )
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
