from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import (
    DatasetFile,
    discover_dataset_files,
    iter_annotations,
    normalize_term,
    parse_va,
    read_jsonl,
    sha256_file,
)


@dataclass(frozen=True)
class ProjectionStats:
    target_subtask: int
    source_annotations: int
    projected_groups: int
    collision_groups: int
    safe_collapse_groups: int
    ambiguous_va_groups: int
    invalid_va_groups: int
    null_aspect_groups: int
    null_opinion_groups: int
    examples: tuple[dict[str, Any], ...]


def _structure(annotation: dict[str, Any], target_subtask: int) -> tuple[Any, ...]:
    aspect = normalize_term(annotation.get("Aspect"))
    opinion = normalize_term(annotation.get("Opinion"))
    if target_subtask == 1:
        return (aspect,)
    if target_subtask == 2:
        return (aspect, opinion)
    raise ValueError(f"Only lower-task projections 1 and 2 are supported, got {target_subtask}")


def _safe_va(annotation: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return parse_va(annotation.get("VA"))
    except (TypeError, ValueError):
        return None


def analyze_record_projection(
    record: dict[str, Any],
    target_subtask: int,
) -> ProjectionStats:
    annotations = [a for a in iter_annotations(record) if isinstance(a, dict)]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        groups[_structure(annotation, target_subtask)].append(annotation)

    collision_groups = 0
    safe_collapse_groups = 0
    ambiguous_va_groups = 0
    invalid_va_groups = 0
    null_aspect_groups = 0
    null_opinion_groups = 0
    examples: list[dict[str, Any]] = []

    for structure, members in groups.items():
        aspect = structure[0]
        opinion = structure[1] if target_subtask == 2 else None
        null_aspect_groups += int(aspect is None)
        if target_subtask == 2:
            null_opinion_groups += int(opinion is None)

        valid_vas = {_safe_va(member) for member in members}
        has_invalid = None in valid_vas
        valid_vas.discard(None)

        if has_invalid:
            invalid_va_groups += 1

        if len(members) <= 1:
            continue

        collision_groups += 1
        if len(valid_vas) <= 1 and not has_invalid:
            safe_collapse_groups += 1
        if len(valid_vas) > 1:
            ambiguous_va_groups += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "record_id": record.get("ID"),
                        "text": record.get("Text"),
                        "target_subtask": target_subtask,
                        "structure": list(structure),
                        "source_count": len(members),
                        "vas": [list(va) for va in sorted(valid_vas)],
                        "opinions": sorted(
                            {str(m.get("Opinion")) for m in members}
                        ),
                        "categories": sorted(
                            {str(m.get("Category")) for m in members}
                        ),
                    }
                )

    return ProjectionStats(
        target_subtask=target_subtask,
        source_annotations=len(annotations),
        projected_groups=len(groups),
        collision_groups=collision_groups,
        safe_collapse_groups=safe_collapse_groups,
        ambiguous_va_groups=ambiguous_va_groups,
        invalid_va_groups=invalid_va_groups,
        null_aspect_groups=null_aspect_groups,
        null_opinion_groups=null_opinion_groups,
        examples=tuple(examples),
    )


def canonical_alltask_files(raw_root: Path) -> dict[tuple[str, str], list[DatasetFile]]:
    grouped: dict[tuple[str, str], list[DatasetFile]] = defaultdict(list)
    for meta in discover_dataset_files(raw_root):
        if (
            meta.representation == "dimensional"
            and meta.split == "train"
            and meta.training_scope == "alltasks"
        ):
            grouped[(meta.language, meta.domain)].append(meta)
    return dict(grouped)


def audit_hierarchy(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_root = raw_root.resolve()
    manifest_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    for (language, domain), metas in sorted(canonical_alltask_files(raw_root).items()):
        hashes = {meta.subtask: sha256_file(meta.path) for meta in metas}
        unique_hashes = set(hashes.values())
        manifest_rows.append(
            {
                "language": language,
                "domain": domain,
                "subtasks_present": json.dumps(sorted(hashes)),
                "identical_across_subtasks": len(unique_hashes) == 1,
                "hashes": json.dumps(hashes, sort_keys=True),
                "paths": json.dumps(
                    {meta.subtask: str(meta.path.relative_to(raw_root)) for meta in metas},
                    sort_keys=True,
                ),
            }
        )

        # If the same all-task file is copied into multiple subtask folders, analyze
        # exactly one canonical copy so samples are not triple-counted.
        canonical = sorted(metas, key=lambda m: m.subtask)[0]
        records, errors = read_jsonl(canonical.path)
        if errors:
            raise ValueError(f"JSONL errors in {canonical.path}: {errors[:3]}")

        for target_subtask in (2, 1):
            totals = {
                "records": len(records),
                "source_annotations": 0,
                "projected_groups": 0,
                "collision_groups": 0,
                "safe_collapse_groups": 0,
                "ambiguous_va_groups": 0,
                "invalid_va_groups": 0,
                "null_aspect_groups": 0,
                "null_opinion_groups": 0,
            }

            for record in records:
                stats = analyze_record_projection(record, target_subtask)
                for key in totals:
                    if key != "records":
                        totals[key] += getattr(stats, key)
                for example in stats.examples:
                    example_rows.append(
                        {
                            "language": language,
                            "domain": domain,
                            **example,
                        }
                    )

            projection_rows.append(
                {
                    "language": language,
                    "domain": domain,
                    "source_task": 3,
                    "target_task": target_subtask,
                    **totals,
                    "collision_rate": (
                        totals["collision_groups"] / totals["projected_groups"]
                        if totals["projected_groups"]
                        else 0.0
                    ),
                    "ambiguous_va_rate": (
                        totals["ambiguous_va_groups"] / totals["projected_groups"]
                        if totals["projected_groups"]
                        else 0.0
                    ),
                }
            )

    return manifest_rows, projection_rows, example_rows
