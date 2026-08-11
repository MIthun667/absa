from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import discover_dataset_files, parse_va, read_jsonl


@dataclass(frozen=True)
class Target:
    aspect: str
    valence: float
    arousal: float
    opinion: str | None = None
    category: str | None = None
    source_opinion: str | None = None
    source_category: str | None = None
    source_index: int = 0

    @property
    def va(self) -> tuple[float, float]:
        return (self.valence, self.arousal)

    def structural_key(self, task: int) -> tuple[str | None, ...]:
        if task == 1:
            return (self.aspect,)
        if task == 2:
            return (self.aspect, self.opinion)
        if task == 3:
            return (self.aspect, self.opinion, self.category)
        raise ValueError(f"Unsupported task: {task}")


@dataclass(frozen=True)
class TaskRecord:
    record_id: str
    text: str
    task: int
    language: str
    domain: str
    split: str
    targets: tuple[Target, ...]
    source_path: str
    source_field: str


@dataclass(frozen=True)
class ProjectionConflict:
    record_id: str
    task: int
    structural_key: tuple[str | None, ...]
    target_count: int
    distinct_va_count: int
    ambiguous_va: bool


def _raw_term(value: Any) -> str:
    if value is None:
        return "NULL"
    text = str(value)
    return text if text else "NULL"


def locate_dimensional_file(
    raw_root: Path,
    *,
    task: int,
    language: str,
    domain: str,
    split: str,
) -> Path:
    matches = [
        meta.path
        for meta in discover_dataset_files(raw_root)
        if meta.representation == "dimensional"
        and meta.subtask == task
        and meta.language == language
        and meta.domain == domain
        and meta.split == split
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one dimensional dataset file for "
            f"task={task}, language={language}, domain={domain}, split={split}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _target_from_annotation(annotation: dict[str, Any], *, task: int, source_index: int) -> Target:
    if "Aspect" not in annotation or "VA" not in annotation:
        raise ValueError(f"Annotation missing Aspect/VA: {annotation}")

    valence, arousal = parse_va(annotation["VA"])
    source_opinion = _raw_term(annotation.get("Opinion")) if "Opinion" in annotation else None
    source_category = _raw_term(annotation.get("Category")) if "Category" in annotation else None

    if task == 1:
        opinion = None
        category = None
    elif task == 2:
        if "Opinion" not in annotation:
            raise ValueError(f"Task-2 annotation missing Opinion: {annotation}")
        opinion = _raw_term(annotation.get("Opinion"))
        category = None
    elif task == 3:
        if "Opinion" not in annotation or "Category" not in annotation:
            raise ValueError(f"Task-3 annotation missing Opinion/Category: {annotation}")
        opinion = _raw_term(annotation.get("Opinion"))
        category = _raw_term(annotation.get("Category"))
    else:
        raise ValueError(f"Unsupported task: {task}")

    return Target(
        aspect=_raw_term(annotation.get("Aspect")),
        opinion=opinion,
        category=category,
        valence=valence,
        arousal=arousal,
        source_opinion=source_opinion,
        source_category=source_category,
        source_index=source_index,
    )


def _select_source_field(record: dict[str, Any], task: int) -> tuple[str, list[dict[str, Any]]]:
    direct_field = {1: "Aspect_VA", 2: "Triplet", 3: "Quadruplet"}[task]
    direct = record.get(direct_field)
    if isinstance(direct, list):
        return direct_field, direct

    # Official all-task training files store the richest Quadruplet annotation
    # under each subtask directory. For lower tasks we project the structure but
    # retain source opinion/category metadata on each Target. We deliberately do
    # not average or deduplicate conflicting projected VA labels here.
    alltask = record.get("Quadruplet")
    if isinstance(alltask, list):
        return "Quadruplet", alltask

    raise ValueError(
        f"Record {record.get('ID', '<missing>')} has neither {direct_field} nor Quadruplet annotations"
    )


def load_task_records(
    raw_root: Path,
    *,
    task: int,
    language: str,
    domain: str,
    split: str,
) -> list[TaskRecord]:
    """Load one official Track-A slice into a task-comparable representation.

    For task-specific dev/test files the native annotation field is used directly.
    For ``train_alltasks`` files, Task 1 and Task 2 are structural projections of
    the source Quadruplet annotations. Projection duplicates and conflicting VA
    targets are preserved exactly; call ``find_projection_conflicts`` to inspect
    them rather than silently collapsing them.
    """
    raw_root = raw_root.resolve()
    path = locate_dimensional_file(
        raw_root,
        task=task,
        language=language,
        domain=domain,
        split=split,
    )
    records, errors = read_jsonl(path)
    if errors:
        raise ValueError(f"JSONL errors in {path}: {errors[:3]}")

    loaded: list[TaskRecord] = []
    for record in records:
        if "ID" not in record or "Text" not in record:
            raise ValueError(f"Record missing ID/Text in {path}: {record}")
        source_field, annotations = _select_source_field(record, task)
        targets = tuple(
            _target_from_annotation(annotation, task=task, source_index=index)
            for index, annotation in enumerate(annotations)
        )
        loaded.append(
            TaskRecord(
                record_id=str(record["ID"]),
                text=str(record["Text"]),
                task=task,
                language=language,
                domain=domain,
                split=split,
                targets=targets,
                source_path=str(path.relative_to(raw_root)),
                source_field=source_field,
            )
        )
    return loaded


def find_projection_conflicts(records: list[TaskRecord]) -> list[ProjectionConflict]:
    conflicts: list[ProjectionConflict] = []
    for record in records:
        groups: dict[tuple[str | None, ...], list[Target]] = defaultdict(list)
        for target in record.targets:
            groups[target.structural_key(record.task)].append(target)

        for key, members in groups.items():
            if len(members) <= 1:
                continue
            distinct_va = {member.va for member in members}
            conflicts.append(
                ProjectionConflict(
                    record_id=record.record_id,
                    task=record.task,
                    structural_key=key,
                    target_count=len(members),
                    distinct_va_count=len(distinct_va),
                    ambiguous_va=len(distinct_va) > 1,
                )
            )
    return conflicts


def summarize_task_records(records: list[TaskRecord]) -> dict[str, Any]:
    if not records:
        return {
            "records": 0,
            "targets": 0,
            "projection_collision_groups": 0,
            "ambiguous_projection_groups": 0,
        }

    conflicts = find_projection_conflicts(records)
    return {
        "task": records[0].task,
        "language": records[0].language,
        "domain": records[0].domain,
        "split": records[0].split,
        "records": len(records),
        "targets": sum(len(record.targets) for record in records),
        "source_fields": sorted({record.source_field for record in records}),
        "projection_collision_groups": len(conflicts),
        "ambiguous_projection_groups": sum(conflict.ambiguous_va for conflict in conflicts),
        "null_aspects": sum(
            target.aspect.upper() == "NULL"
            for record in records
            for target in record.targets
        ),
        "null_opinions": sum(
            target.opinion is not None and target.opinion.upper() == "NULL"
            for record in records
            for target in record.targets
        ),
    }
