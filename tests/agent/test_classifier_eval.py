"""Offline contract checks for the curated classifier evaluation fixture."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.eval_classifier import EvalCase, EvalResult, _print_report

DATASET_PATH = (
    Path(__file__).parents[1] / "fixtures" / "model_routing" / "classifier_eval.json"
)
TASK_TYPES = {"chat", "admin", "coding", "research", "legal_review"}
COMPLEXITIES = {"low", "medium", "high"}
REQUIRED_FIELDS = {"id", "text", "task_type", "complexity"}


@pytest.fixture(scope="module")
def classifier_cases() -> list[dict[str, str]]:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)
    assert isinstance(dataset, list)
    return dataset


def test_classifier_fixture_has_valid_rows(classifier_cases: list[dict[str, str]]) -> None:
    assert 60 <= len(classifier_cases) <= 80

    for index, case in enumerate(classifier_cases):
        assert isinstance(case, dict), f"row {index} must be an object"
        assert set(case) == REQUIRED_FIELDS, f"row {index} has unexpected fields"
        assert all(isinstance(case[field], str) for field in REQUIRED_FIELDS)
        assert case["id"] == case["id"].strip() and case["id"]
        assert case["text"] == case["text"].strip() and case["text"]
        assert case["task_type"] in TASK_TYPES
        assert case["complexity"] in COMPLEXITIES


def test_classifier_fixture_ids_and_text_are_unique(
    classifier_cases: list[dict[str, str]],
) -> None:
    ids = [case["id"] for case in classifier_cases]
    texts = [case["text"] for case in classifier_cases]

    assert len(ids) == len(set(ids))
    assert len(texts) == len(set(texts))


def test_classifier_fixture_balances_all_label_pairs(
    classifier_cases: list[dict[str, str]],
) -> None:
    task_counts = Counter(case["task_type"] for case in classifier_cases)
    complexity_counts = Counter(case["complexity"] for case in classifier_cases)
    pair_counts = Counter(
        (case["task_type"], case["complexity"]) for case in classifier_cases
    )

    assert set(task_counts) == TASK_TYPES
    assert max(task_counts.values()) - min(task_counts.values()) <= 1
    assert set(complexity_counts) == COMPLEXITIES
    assert max(complexity_counts.values()) - min(complexity_counts.values()) <= 1
    assert set(pair_counts) == {
        (task_type, complexity)
        for task_type in TASK_TYPES
        for complexity in COMPLEXITIES
    }
    assert min(pair_counts.values()) >= 3


def test_classifier_fixture_contains_english_and_chinese(
    classifier_cases: list[dict[str, str]],
) -> None:
    texts = [case["text"] for case in classifier_cases]

    assert any(text.isascii() for text in texts)
    assert any(any("\u4e00" <= char <= "\u9fff" for char in text) for text in texts)


@pytest.mark.parametrize(
    ("threshold", "expected_status", "expected_passed"),
    [(0.50, "PASS", True), (0.51, "FAIL", False)],
)
def test_classifier_report_ends_with_metric_summary(
    capsys: pytest.CaptureFixture[str],
    threshold: float,
    expected_status: str,
    expected_passed: bool,
) -> None:
    results = [
        EvalResult(EvalCase("exact", "hello", "chat", "low"), "chat", "low"),
        EvalResult(
            EvalCase("miss", "fix it", "coding", "high"),
            None,
            None,
            "call failed: unavailable",
        ),
    ]

    passed = _print_report(
        results,
        preset_name="classifier",
        model="test-model",
        task_threshold=threshold,
        complexity_threshold=threshold,
        joint_threshold=threshold,
    )

    lines = capsys.readouterr().out.splitlines()
    assert passed is expected_passed
    assert lines[0] == "Classifier eval: preset=classifier model=test-model cases=2"
    assert "task_type confusion (expected -> predicted):" in lines
    assert "mismatches:" in lines
    assert lines[-5].startswith(f"{expected_status}: thresholds ")
    assert lines[-4:] == [
        "task_type accuracy: 0.500 (1/2)",
        "complexity accuracy: 0.500 (1/2)",
        "joint exact-match: 0.500 (1/2)",
        "call/parse errors: 1",
    ]
