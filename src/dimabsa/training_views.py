from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .experiment_data import TaskRecord, Target


@dataclass(frozen=True)
class TrainingExample:
    record_id: str
    text: str
    task: int
    language: str
    domain: str
    structural_key: tuple[str | None, ...]
    target: Target
    group_size: int
    distinct_va_count: int
    ambiguous_source_group: bool


@dataclass(frozen=True)
class TrainingViewSummary:
    task: int
    language: str
    domain: str
    records: int
    source_targets: int
    structural_groups: int
    collision_groups: int
    ambiguous_groups: int
    relation_expanded_examples: int
    deterministic_examples: int
    dropped_ambiguous_groups: int
    dropped_ambiguous_targets: int
    safely_collapsed_duplicate_targets: int


def _group_record_targets(record: TaskRecord) -> dict[tuple[str | None, ...], list[Target]]:
    groups: dict[tuple[str | None, ...], list[Target]] = defaultdict(list)
    for target in record.targets:
        groups[target.structural_key(record.task)].append(target)
    return dict(groups)


def build_relation_expanded_view(records: list[TaskRecord]) -> list[TrainingExample]:
    """Preserve every projected source relation as a separate training example.

    If two source annotations project to the same task structure but carry
    different VA values, the resulting examples intentionally share the same
    observable structural input while retaining their distinct targets. This is
    the faithful-but-conflicted baseline view.
    """
    examples: list[TrainingExample] = []
    for record in records:
        groups = _group_record_targets(record)
        for key, members in groups.items():
            distinct_va = {member.va for member in members}
            for member in members:
                examples.append(
                    TrainingExample(
                        record_id=record.record_id,
                        text=record.text,
                        task=record.task,
                        language=record.language,
                        domain=record.domain,
                        structural_key=key,
                        target=member,
                        group_size=len(members),
                        distinct_va_count=len(distinct_va),
                        ambiguous_source_group=len(distinct_va) > 1,
                    )
                )
    return examples


def build_unambiguous_view(records: list[TaskRecord]) -> list[TrainingExample]:
    """Create a deterministic task view without inventing an aggregation rule.

    Ambiguous groups with multiple distinct VA labels are dropped entirely.
    Non-ambiguous duplicate groups are safely collapsed to one representative.
    This is intended as a diagnostic baseline, not as a claim about the correct
    semantic aggregation mechanism.
    """
    examples: list[TrainingExample] = []
    for record in records:
        groups = _group_record_targets(record)
        for key, members in groups.items():
            distinct_va = {member.va for member in members}
            if len(distinct_va) > 1:
                continue
            member = members[0]
            examples.append(
                TrainingExample(
                    record_id=record.record_id,
                    text=record.text,
                    task=record.task,
                    language=record.language,
                    domain=record.domain,
                    structural_key=key,
                    target=member,
                    group_size=len(members),
                    distinct_va_count=1,
                    ambiguous_source_group=False,
                )
            )
    return examples


def summarize_training_views(records: list[TaskRecord]) -> TrainingViewSummary:
    if not records:
        raise ValueError("Cannot summarize an empty task record list")

    structural_groups = 0
    collision_groups = 0
    ambiguous_groups = 0
    dropped_ambiguous_targets = 0
    safely_collapsed_duplicate_targets = 0

    for record in records:
        groups = _group_record_targets(record)
        structural_groups += len(groups)
        for members in groups.values():
            if len(members) <= 1:
                continue
            collision_groups += 1
            distinct_va = {member.va for member in members}
            if len(distinct_va) > 1:
                ambiguous_groups += 1
                dropped_ambiguous_targets += len(members)
            else:
                safely_collapsed_duplicate_targets += len(members) - 1

    relation_expanded = build_relation_expanded_view(records)
    deterministic = build_unambiguous_view(records)

    first = records[0]
    return TrainingViewSummary(
        task=first.task,
        language=first.language,
        domain=first.domain,
        records=len(records),
        source_targets=sum(len(record.targets) for record in records),
        structural_groups=structural_groups,
        collision_groups=collision_groups,
        ambiguous_groups=ambiguous_groups,
        relation_expanded_examples=len(relation_expanded),
        deterministic_examples=len(deterministic),
        dropped_ambiguous_groups=ambiguous_groups,
        dropped_ambiguous_targets=dropped_ambiguous_targets,
        safely_collapsed_duplicate_targets=safely_collapsed_duplicate_targets,
    )


def summary_as_dict(summary: TrainingViewSummary) -> dict[str, Any]:
    return {
        field: getattr(summary, field)
        for field in summary.__dataclass_fields__
    }
