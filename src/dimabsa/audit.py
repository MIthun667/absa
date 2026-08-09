from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import (
    DatasetFile,
    annotation_field,
    discover_dataset_files,
    iter_annotations,
    normalize_text,
    normalized_annotation,
    parse_va,
    projected_key,
    read_jsonl,
    sha256_file,
)


def _safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _safe_std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_id(meta: DatasetFile) -> str:
    return "/".join(
        [
            meta.representation,
            f"subtask_{meta.subtask}",
            meta.language,
            meta.domain,
            meta.split,
        ]
    )


def _expected_field(meta: DatasetFile) -> str:
    if meta.representation == "categorical":
        return "Quadruplet" if meta.split == "train" else (
            "Aspect_Polarity" if meta.subtask == 1 else (
                "Triplet" if meta.subtask == 2 else "Quadruplet"
            )
        )
    if meta.split == "train" and meta.training_scope == "alltasks":
        return "Quadruplet"
    return "Aspect_VA" if meta.subtask == 1 else (
        "Triplet" if meta.subtask == 2 else "Quadruplet"
    )


def audit_dataset(raw_root: Path, output_dir: Path) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_dataset_files(raw_root)
    file_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    va_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    split_overlap_rows: list[dict[str, Any]] = []
    task_consistency_rows: list[dict[str, Any]] = []
    categorical_alignment_rows: list[dict[str, Any]] = []

    loaded: dict[tuple[str, int, str, str, str], tuple[DatasetFile, list[dict[str, Any]]]] = {}

    total_records = 0
    total_annotations = 0
    total_errors = 0

    for meta in files:
        records, parse_errors = read_jsonl(meta.path)
        total_records += len(records)
        total_errors += len(parse_errors)
        loaded[meta.key] = (meta, records)

        rel = str(meta.path.relative_to(raw_root))
        file_rows.append(
            {
                "path": rel,
                "representation": meta.representation,
                "subtask": meta.subtask,
                "language": meta.language,
                "domain": meta.domain,
                "split": meta.split,
                "training_scope": meta.training_scope,
                "records": len(records),
                "sha256": sha256_file(meta.path),
            }
        )

        expected_field = _expected_field(meta)
        schema_counter: Counter[str] = Counter()
        ids: list[str] = []
        texts: list[str] = []
        annotation_count = 0
        implicit_aspect = 0
        implicit_opinion = 0
        va_values: list[tuple[float, float]] = []

        for err in parse_errors:
            validation_rows.append(
                {
                    "path": rel,
                    "line_number": err.line_number,
                    "severity": "error",
                    "kind": "json_decode",
                    "message": err.error,
                }
            )

        for line_idx, record in enumerate(records, start=1):
            record_id = record.get("ID")
            text = record.get("Text")
            if not isinstance(record_id, str) or not record_id.strip():
                validation_rows.append(
                    {
                        "path": rel,
                        "line_number": line_idx,
                        "severity": "error",
                        "kind": "missing_id",
                        "message": repr(record_id),
                    }
                )
            else:
                ids.append(record_id)

            if not isinstance(text, str) or not text.strip():
                validation_rows.append(
                    {
                        "path": rel,
                        "line_number": line_idx,
                        "severity": "error",
                        "kind": "missing_text",
                        "message": repr(text),
                    }
                )
            else:
                texts.append(normalize_text(text))

            try:
                field = annotation_field(record)
            except Exception as exc:
                validation_rows.append(
                    {
                        "path": rel,
                        "line_number": line_idx,
                        "severity": "error",
                        "kind": "annotation_field",
                        "message": str(exc),
                    }
                )
                continue

            schema_counter[str(field)] += 1
            if field != expected_field:
                validation_rows.append(
                    {
                        "path": rel,
                        "line_number": line_idx,
                        "severity": "warning",
                        "kind": "unexpected_annotation_field",
                        "message": f"expected={expected_field}, observed={field}",
                    }
                )

            if field is None:
                continue

            try:
                annotations = list(iter_annotations(record))
            except Exception as exc:
                validation_rows.append(
                    {
                        "path": rel,
                        "line_number": line_idx,
                        "severity": "error",
                        "kind": "annotation_list",
                        "message": str(exc),
                    }
                )
                continue

            annotation_count += len(annotations)
            total_annotations += len(annotations)

            for ann in annotations:
                if not isinstance(ann, dict):
                    validation_rows.append(
                        {
                            "path": rel,
                            "line_number": line_idx,
                            "severity": "error",
                            "kind": "annotation_type",
                            "message": type(ann).__name__,
                        }
                    )
                    continue

                try:
                    normalized = normalized_annotation(ann)
                except Exception as exc:
                    validation_rows.append(
                        {
                            "path": rel,
                            "line_number": line_idx,
                            "severity": "error",
                            "kind": "annotation_value",
                            "message": str(exc),
                        }
                    )
                    continue

                implicit_aspect += int(normalized["aspect"] is None)
                if "Opinion" in ann:
                    implicit_opinion += int(normalized["opinion"] is None)

                if "VA" in ann:
                    try:
                        valence, arousal = parse_va(ann["VA"])
                        va_values.append((valence, arousal))
                        va_rows.append(
                            {
                                "dataset": _dataset_id(meta),
                                "path": rel,
                                "record_id": record_id,
                                "aspect": normalized["aspect"],
                                "opinion": normalized["opinion"],
                                "category": normalized["category"],
                                "valence": valence,
                                "arousal": arousal,
                            }
                        )
                    except Exception as exc:
                        validation_rows.append(
                            {
                                "path": rel,
                                "line_number": line_idx,
                                "severity": "error",
                                "kind": "invalid_va",
                                "message": str(exc),
                            }
                        )

        id_counts = Counter(ids)
        text_counts = Counter(texts)
        for key, count in id_counts.items():
            if count > 1:
                duplicate_rows.append(
                    {
                        "dataset": _dataset_id(meta),
                        "path": rel,
                        "kind": "duplicate_id",
                        "value": key,
                        "count": count,
                    }
                )
        for key, count in text_counts.items():
            if count > 1:
                duplicate_rows.append(
                    {
                        "dataset": _dataset_id(meta),
                        "path": rel,
                        "kind": "duplicate_text",
                        "value": key,
                        "count": count,
                    }
                )

        vals = [v for v, _ in va_values]
        aros = [a for _, a in va_values]
        summary_rows.append(
            {
                "dataset": _dataset_id(meta),
                "path": rel,
                "records": len(records),
                "annotations": annotation_count,
                "annotations_per_record": annotation_count / len(records) if records else None,
                "unique_ids": len(set(ids)),
                "unique_texts": len(set(texts)),
                "implicit_aspects": implicit_aspect,
                "implicit_aspect_rate": implicit_aspect / annotation_count if annotation_count else None,
                "implicit_opinions": implicit_opinion,
                "implicit_opinion_rate": implicit_opinion / annotation_count if annotation_count else None,
                "annotation_fields": json.dumps(dict(schema_counter), ensure_ascii=False, sort_keys=True),
                "va_count": len(va_values),
                "valence_mean": _safe_mean(vals),
                "valence_std": _safe_std(vals),
                "arousal_mean": _safe_mean(aros),
                "arousal_std": _safe_std(aros),
                "va_pearson": _pearson(vals, aros),
            }
        )

    # Exact normalized-text overlap across train/dev/test within representation/language/domain.
    by_group: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for meta, records in loaded.values():
        group = (meta.representation, meta.language, meta.domain)
        for record in records:
            text = record.get("Text")
            if isinstance(text, str) and text.strip():
                by_group[group][meta.split].add(normalize_text(text))

    for (representation, language, domain), split_map in sorted(by_group.items()):
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = split_map.get(left, set()) & split_map.get(right, set())
            split_overlap_rows.append(
                {
                    "representation": representation,
                    "language": language,
                    "domain": domain,
                    "split_a": left,
                    "split_b": right,
                    "count": len(overlap),
                    "examples": json.dumps(sorted(overlap)[:10], ensure_ascii=False),
                }
            )

    # Cross-subtask consistency for dimensional dev/test files where multiple subtasks exist.
    dimensional = {
        key: value
        for key, value in loaded.items()
        if key[0] == "dimensional"
    }
    combos = sorted({(lang, domain, split) for _, _, lang, domain, split in dimensional})

    for language, domain, split in combos:
        task_records: dict[int, list[dict[str, Any]]] = {}
        for task in (1, 2, 3):
            item = dimensional.get(("dimensional", task, language, domain, split))
            if item is not None:
                task_records[task] = item[1]

        if len(task_records) < 2:
            continue

        text_maps: dict[int, dict[str, dict[str, Any]]] = {}
        for task, records in task_records.items():
            text_maps[task] = {
                normalize_text(r.get("Text", "")): r
                for r in records
                if isinstance(r.get("Text"), str)
            }

        for high, low in ((3, 2), (2, 1), (3, 1)):
            if high not in text_maps or low not in text_maps:
                continue
            shared_texts = set(text_maps[high]) & set(text_maps[low])
            exact = 0
            mismatches = 0
            examples: list[dict[str, Any]] = []
            for text in shared_texts:
                high_record = text_maps[high][text]
                low_record = text_maps[low][text]
                try:
                    high_annotations = list(iter_annotations(high_record))
                    low_annotations = list(iter_annotations(low_record))
                    high_keys = Counter(projected_key(a, low) + tuple(parse_va(a["VA"])) for a in high_annotations)
                    low_keys = Counter(projected_key(a, low) + tuple(parse_va(a["VA"])) for a in low_annotations)
                    if high_keys == low_keys:
                        exact += 1
                    else:
                        mismatches += 1
                        if len(examples) < 10:
                            examples.append(
                                {
                                    "text": text,
                                    "higher": list(map(list, high_keys.elements())),
                                    "lower": list(map(list, low_keys.elements())),
                                }
                            )
                except Exception as exc:
                    mismatches += 1
                    if len(examples) < 10:
                        examples.append({"text": text, "error": str(exc)})

            task_consistency_rows.append(
                {
                    "language": language,
                    "domain": domain,
                    "split": split,
                    "higher_task": high,
                    "lower_task": low,
                    "shared_texts": len(shared_texts),
                    "exact_projection_matches": exact,
                    "mismatches": mismatches,
                    "match_rate": exact / len(shared_texts) if shared_texts else None,
                    "examples": json.dumps(examples, ensure_ascii=False),
                }
            )

    # Categorical vs dimensional alignment, matched by normalized text + projected structure.
    categorical = {
        key: value
        for key, value in loaded.items()
        if key[0] == "categorical"
    }
    for _, task, language, domain, split in sorted(dimensional):
        dim_item = dimensional.get(("dimensional", task, language, domain, split))
        cat_item = categorical.get(("categorical", task, language, domain, split))
        if dim_item is None or cat_item is None:
            continue

        dim_records = {
            normalize_text(r.get("Text", "")): r
            for r in dim_item[1]
            if isinstance(r.get("Text"), str)
        }
        cat_records = {
            normalize_text(r.get("Text", "")): r
            for r in cat_item[1]
            if isinstance(r.get("Text"), str)
        }

        matched = 0
        polarity_va: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for text in set(dim_records) & set(cat_records):
            dim_anns = list(iter_annotations(dim_records[text]))
            cat_anns = list(iter_annotations(cat_records[text]))
            cat_by_structure: dict[tuple[Any, ...], list[str]] = defaultdict(list)
            for ann in cat_anns:
                polarity = normalized_annotation(ann)["polarity"]
                if polarity is not None:
                    cat_by_structure[projected_key(ann, task)].append(str(polarity))

            for ann in dim_anns:
                structure = projected_key(ann, task)
                if not cat_by_structure.get(structure):
                    continue
                valence, arousal = parse_va(ann["VA"])
                polarity = cat_by_structure[structure].pop(0)
                polarity_va[polarity].append((valence, arousal))
                matched += 1

        for polarity, pairs in sorted(polarity_va.items()):
            vals = [v for v, _ in pairs]
            aros = [a for _, a in pairs]
            categorical_alignment_rows.append(
                {
                    "subtask": task,
                    "language": language,
                    "domain": domain,
                    "split": split,
                    "polarity": polarity,
                    "matched_annotations": len(pairs),
                    "valence_mean": _safe_mean(vals),
                    "valence_std": _safe_std(vals),
                    "arousal_mean": _safe_mean(aros),
                    "arousal_std": _safe_std(aros),
                }
            )

    _write_csv(output_dir / "file_inventory.csv", file_rows)
    _write_csv(output_dir / "dataset_summary.csv", summary_rows)
    _write_csv(output_dir / "validation_issues.csv", validation_rows)
    _write_csv(output_dir / "va_annotations.csv", va_rows)
    _write_csv(output_dir / "within_file_duplicates.csv", duplicate_rows)
    _write_csv(output_dir / "split_overlap.csv", split_overlap_rows)
    _write_csv(output_dir / "cross_task_consistency.csv", task_consistency_rows)
    _write_csv(output_dir / "categorical_va_alignment.csv", categorical_alignment_rows)

    report = {
        "raw_root": str(raw_root),
        "files": len(files),
        "records": total_records,
        "annotations": total_annotations,
        "validation_issues": len(validation_rows),
        "json_parse_errors": total_errors,
        "artifacts": [
            "file_inventory.csv",
            "dataset_summary.csv",
            "validation_issues.csv",
            "va_annotations.csv",
            "within_file_duplicates.csv",
            "split_overlap.csv",
            "cross_task_consistency.csv",
            "categorical_va_alignment.csv",
        ],
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
