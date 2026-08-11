from __future__ import annotations

from dimabsa.experiment_data import Target, TaskRecord
from dimabsa.training_views import (
    build_relation_expanded_view,
    build_unambiguous_view,
    summarize_training_views,
)


def _record(targets: tuple[Target, ...], task: int = 1) -> TaskRecord:
    return TaskRecord(
        record_id="r1",
        text="battery life is great but sometimes slow",
        task=task,
        language="eng",
        domain="laptop",
        split="train",
        targets=targets,
        source_path="dummy.jsonl",
        source_field="Quadruplet",
    )


def test_relation_expanded_view_preserves_conflicting_targets() -> None:
    records = [
        _record(
            (
                Target("battery life", 7.0, 7.0, source_opinion="great"),
                Target("battery life", 4.0, 5.0, source_opinion="slow"),
            )
        )
    ]
    examples = build_relation_expanded_view(records)
    assert len(examples) == 2
    assert all(example.ambiguous_source_group for example in examples)
    assert {example.target.va for example in examples} == {(7.0, 7.0), (4.0, 5.0)}


def test_unambiguous_view_drops_conflicting_group() -> None:
    records = [
        _record(
            (
                Target("battery life", 7.0, 7.0, source_opinion="great"),
                Target("battery life", 4.0, 5.0, source_opinion="slow"),
            )
        )
    ]
    assert build_unambiguous_view(records) == []


def test_unambiguous_view_safely_collapses_exact_duplicate_targets() -> None:
    records = [
        _record(
            (
                Target("screen", 7.0, 7.0, source_opinion="nice"),
                Target("screen", 7.0, 7.0, source_opinion="pretty"),
            )
        )
    ]
    examples = build_unambiguous_view(records)
    assert len(examples) == 1
    assert examples[0].target.va == (7.0, 7.0)
    assert examples[0].group_size == 2


def test_summary_counts_ambiguous_and_safe_collapses() -> None:
    records = [
        _record(
            (
                Target("battery", 3.0, 6.0, source_opinion="bad"),
                Target("battery", 4.0, 5.0, source_opinion="weak"),
                Target("screen", 7.0, 7.0, source_opinion="nice"),
                Target("screen", 7.0, 7.0, source_opinion="pretty"),
            )
        )
    ]
    summary = summarize_training_views(records)
    assert summary.source_targets == 4
    assert summary.structural_groups == 2
    assert summary.collision_groups == 2
    assert summary.ambiguous_groups == 1
    assert summary.relation_expanded_examples == 4
    assert summary.deterministic_examples == 1
    assert summary.dropped_ambiguous_targets == 2
    assert summary.safely_collapsed_duplicate_targets == 1
