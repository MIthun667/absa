from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import iter_annotations, normalize_term, parse_va, read_jsonl
from .hierarchy import canonical_alltask_files


def _structure(annotation: dict[str, Any], target_task: int) -> tuple[Any, ...]:
    aspect = normalize_term(annotation.get("Aspect"))
    opinion = normalize_term(annotation.get("Opinion"))
    if target_task == 1:
        return (aspect,)
    if target_task == 2:
        return (aspect, opinion)
    raise ValueError(target_task)


def _valid_va(annotation: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return parse_va(annotation.get("VA"))
    except (TypeError, ValueError):
        return None


def _max_pairwise_distance(values: list[tuple[float, float]]) -> float:
    maximum = 0.0
    for i, (v1, a1) in enumerate(values):
        for v2, a2 in values[i + 1 :]:
            maximum = max(maximum, math.hypot(v1 - v2, a1 - a2))
    return maximum


def _conflict_type(members: list[dict[str, Any]]) -> str:
    opinions = {normalize_term(m.get("Opinion")) for m in members}
    categories = {normalize_term(m.get("Category")) for m in members}
    opinion_varies = len(opinions) > 1
    category_varies = len(categories) > 1
    if opinion_varies and category_varies:
        return "opinion_and_category_vary"
    if opinion_varies:
        return "opinion_varies"
    if category_varies:
        return "category_varies"
    return "same_relation_different_va"


def analyze_conflict_anatomy(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_root = raw_root.resolve()
    detail_rows: list[dict[str, Any]] = []

    for (language, domain), metas in sorted(canonical_alltask_files(raw_root).items()):
        canonical = sorted(metas, key=lambda m: m.subtask)[0]
        records, errors = read_jsonl(canonical.path)
        if errors:
            raise ValueError(f"JSONL errors in {canonical.path}: {errors[:3]}")

        for record in records:
            annotations = [a for a in iter_annotations(record) if isinstance(a, dict)]
            for target_task in (2, 1):
                groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
                for annotation in annotations:
                    groups[_structure(annotation, target_task)].append(annotation)

                for structure, members in groups.items():
                    if len(members) <= 1:
                        continue
                    vas = [_valid_va(m) for m in members]
                    valid = [va for va in vas if va is not None]
                    unique_va = sorted(set(valid))
                    if len(unique_va) <= 1:
                        continue

                    opinions = {normalize_term(m.get("Opinion")) for m in members}
                    categories = {normalize_term(m.get("Category")) for m in members}
                    vals = [v for v, _ in unique_va]
                    aros = [a for _, a in unique_va]
                    aspect = structure[0]

                    detail_rows.append(
                        {
                            "language": language,
                            "domain": domain,
                            "record_id": record.get("ID"),
                            "text": record.get("Text"),
                            "target_task": target_task,
                            "aspect": aspect,
                            "aspect_is_null": aspect is None,
                            "source_count": len(members),
                            "distinct_va": len(unique_va),
                            "distinct_opinions": len(opinions),
                            "distinct_categories": len(categories),
                            "conflict_type": _conflict_type(members),
                            "valence_min": min(vals),
                            "valence_max": max(vals),
                            "valence_spread": max(vals) - min(vals),
                            "arousal_min": min(aros),
                            "arousal_max": max(aros),
                            "arousal_spread": max(aros) - min(aros),
                            "max_va_distance": _max_pairwise_distance(unique_va),
                            "opinions": sorted("NULL" if x is None else x for x in opinions),
                            "categories": sorted("NULL" if x is None else x for x in categories),
                            "vas": unique_va,
                        }
                    )

    grouped: dict[tuple[str, str, int, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[
            (
                row["language"],
                row["domain"],
                row["target_task"],
                row["conflict_type"],
                row["aspect_is_null"],
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (language, domain, target_task, conflict_type, aspect_is_null), rows in sorted(grouped.items()):
        distances = [float(r["max_va_distance"]) for r in rows]
        v_spreads = [float(r["valence_spread"]) for r in rows]
        a_spreads = [float(r["arousal_spread"]) for r in rows]
        sizes = Counter(int(r["source_count"]) for r in rows)
        summary_rows.append(
            {
                "language": language,
                "domain": domain,
                "target_task": target_task,
                "conflict_type": conflict_type,
                "aspect_is_null": aspect_is_null,
                "groups": len(rows),
                "group_size_distribution": dict(sorted(sizes.items())),
                "mean_valence_spread": sum(v_spreads) / len(v_spreads),
                "max_valence_spread": max(v_spreads),
                "mean_arousal_spread": sum(a_spreads) / len(a_spreads),
                "max_arousal_spread": max(a_spreads),
                "mean_max_va_distance": sum(distances) / len(distances),
                "max_va_distance": max(distances),
            }
        )

    return detail_rows, summary_rows
