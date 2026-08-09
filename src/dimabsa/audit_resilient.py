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
    normalize_term,
    normalize_text,
    parse_va,
    projected_key,
    read_jsonl,
    sha256_file,
)


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _std(xs: list[float]) -> float | None:
    return statistics.pstdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return None if denom == 0 else sum(x * y for x, y in zip(dx, dy)) / denom


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


def _expected_field(meta: DatasetFile) -> str:
    if meta.representation == "categorical":
        if meta.split == "train":
            return "Quadruplet"
        return "Aspect_Polarity" if meta.subtask == 1 else ("Triplet" if meta.subtask == 2 else "Quadruplet")
    if meta.split == "train" and meta.training_scope == "alltasks":
        return "Quadruplet"
    return "Aspect_VA" if meta.subtask == 1 else ("Triplet" if meta.subtask == 2 else "Quadruplet")


def _dataset_id(meta: DatasetFile) -> str:
    return f"{meta.representation}/subtask_{meta.subtask}/{meta.language}/{meta.domain}/{meta.split}"


def _safe_structure(ann: dict[str, Any], task: int) -> tuple[Any, ...]:
    # projected_key is deliberately independent of VA validity.
    return projected_key(ann, task)


def _polarity(ann: dict[str, Any]) -> str | None:
    value = ann.get("Polarity")
    return value.strip().upper() if isinstance(value, str) else None


def audit_dataset(raw_root: Path, output_dir: Path) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    file_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    va_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    split_overlap_rows: list[dict[str, Any]] = []
    task_consistency_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    loaded: dict[tuple[str, int, str, str, str], tuple[DatasetFile, list[dict[str, Any]]]] = {}
    total_records = 0
    total_annotations = 0
    invalid_va_count = 0
    json_errors = 0

    for meta in discover_dataset_files(raw_root):
        records, parse_errors = read_jsonl(meta.path)
        loaded[meta.key] = (meta, records)
        total_records += len(records)
        json_errors += len(parse_errors)
        rel = str(meta.path.relative_to(raw_root))

        file_rows.append({
            "path": rel,
            "representation": meta.representation,
            "subtask": meta.subtask,
            "language": meta.language,
            "domain": meta.domain,
            "split": meta.split,
            "training_scope": meta.training_scope,
            "records": len(records),
            "sha256": sha256_file(meta.path),
        })

        for err in parse_errors:
            validation_rows.append({
                "path": rel,
                "line_number": err.line_number,
                "severity": "error",
                "kind": "json_decode",
                "message": err.error,
            })

        ids: list[str] = []
        texts: list[str] = []
        va_pairs: list[tuple[float, float]] = []
        ann_count = 0
        implicit_aspects = 0
        implicit_opinions = 0
        field_counts: Counter[str] = Counter()
        expected = _expected_field(meta)

        for line_no, record in enumerate(records, start=1):
            record_id = record.get("ID")
            text = record.get("Text")
            if isinstance(record_id, str) and record_id.strip():
                ids.append(record_id)
            else:
                validation_rows.append({"path": rel, "line_number": line_no, "severity": "error", "kind": "missing_id", "message": repr(record_id)})
            if isinstance(text, str) and text.strip():
                texts.append(normalize_text(text))
            else:
                validation_rows.append({"path": rel, "line_number": line_no, "severity": "error", "kind": "missing_text", "message": repr(text)})

            try:
                field = annotation_field(record)
            except Exception as exc:
                validation_rows.append({"path": rel, "line_number": line_no, "severity": "error", "kind": "annotation_field", "message": str(exc)})
                continue

            field_counts[str(field)] += 1
            if field != expected:
                validation_rows.append({"path": rel, "line_number": line_no, "severity": "warning", "kind": "unexpected_annotation_field", "message": f"expected={expected}, observed={field}"})
            if field is None:
                continue

            try:
                annotations = list(iter_annotations(record))
            except Exception as exc:
                validation_rows.append({"path": rel, "line_number": line_no, "severity": "error", "kind": "annotation_list", "message": str(exc)})
                continue

            ann_count += len(annotations)
            total_annotations += len(annotations)

            for ann_idx, ann in enumerate(annotations):
                if not isinstance(ann, dict):
                    validation_rows.append({"path": rel, "line_number": line_no, "severity": "error", "kind": "annotation_type", "message": type(ann).__name__})
                    continue

                aspect = normalize_term(ann.get("Aspect"))
                opinion = normalize_term(ann.get("Opinion"))
                category = normalize_term(ann.get("Category"))
                implicit_aspects += int(aspect is None)
                if "Opinion" in ann:
                    implicit_opinions += int(opinion is None)

                if "VA" in ann:
                    try:
                        valence, arousal = parse_va(ann["VA"])
                    except Exception as exc:
                        invalid_va_count += 1
                        validation_rows.append({
                            "path": rel,
                            "line_number": line_no,
                            "annotation_index": ann_idx,
                            "record_id": record_id,
                            "severity": "error",
                            "kind": "invalid_va",
                            "raw_va": ann.get("VA"),
                            "message": str(exc),
                        })
                    else:
                        va_pairs.append((valence, arousal))
                        va_rows.append({
                            "dataset": _dataset_id(meta),
                            "path": rel,
                            "record_id": record_id,
                            "aspect": aspect,
                            "opinion": opinion,
                            "category": category,
                            "valence": valence,
                            "arousal": arousal,
                        })

        for value, count in Counter(ids).items():
            if count > 1:
                duplicate_rows.append({"dataset": _dataset_id(meta), "path": rel, "kind": "duplicate_id", "value": value, "count": count})
        for value, count in Counter(texts).items():
            if count > 1:
                duplicate_rows.append({"dataset": _dataset_id(meta), "path": rel, "kind": "duplicate_text", "value": value, "count": count})

        vals = [v for v, _ in va_pairs]
        aros = [a for _, a in va_pairs]
        summary_rows.append({
            "dataset": _dataset_id(meta),
            "path": rel,
            "records": len(records),
            "annotations": ann_count,
            "annotations_per_record": ann_count / len(records) if records else None,
            "unique_ids": len(set(ids)),
            "unique_texts": len(set(texts)),
            "implicit_aspects": implicit_aspects,
            "implicit_aspect_rate": implicit_aspects / ann_count if ann_count else None,
            "implicit_opinions": implicit_opinions,
            "implicit_opinion_rate": implicit_opinions / ann_count if ann_count else None,
            "annotation_fields": json.dumps(dict(field_counts), sort_keys=True),
            "va_count": len(va_pairs),
            "invalid_va_count": sum(1 for row in validation_rows if row.get("path") == rel and row.get("kind") == "invalid_va"),
            "valence_mean": _mean(vals),
            "valence_std": _std(vals),
            "arousal_mean": _mean(aros),
            "arousal_std": _std(aros),
            "va_pearson": _pearson(vals, aros),
        })

    # Split leakage/overlap. Deduplicate repeated task copies by using text sets.
    by_group: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for meta, records in loaded.values():
        group = (meta.representation, meta.language, meta.domain)
        for record in records:
            text = record.get("Text")
            if isinstance(text, str) and text.strip():
                by_group[group][meta.split].add(normalize_text(text))

    for (rep, lang, domain), split_map in sorted(by_group.items()):
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = split_map.get(left, set()) & split_map.get(right, set())
            split_overlap_rows.append({
                "representation": rep,
                "language": lang,
                "domain": domain,
                "split_a": left,
                "split_b": right,
                "count": len(overlap),
                "examples": json.dumps(sorted(overlap)[:10], ensure_ascii=False),
            })

    # Hierarchical projection consistency across tasks. Invalid VA is a mismatch, not a crash.
    dimensional = {k: v for k, v in loaded.items() if k[0] == "dimensional"}
    combos = sorted({(lang, domain, split) for _, _, lang, domain, split in dimensional})
    for lang, domain, split in combos:
        maps: dict[int, dict[str, dict[str, Any]]] = {}
        for task in (1, 2, 3):
            item = dimensional.get(("dimensional", task, lang, domain, split))
            if item:
                maps[task] = {
                    normalize_text(r.get("Text", "")): r
                    for r in item[1]
                    if isinstance(r.get("Text"), str)
                }
        for high, low in ((3, 2), (2, 1), (3, 1)):
            if high not in maps or low not in maps:
                continue
            shared = set(maps[high]) & set(maps[low])
            exact = 0
            mismatches = 0
            invalid_va_records = 0
            examples: list[dict[str, Any]] = []
            for text in shared:
                try:
                    high_anns = list(iter_annotations(maps[high][text]))
                    low_anns = list(iter_annotations(maps[low][text]))
                    high_keys = Counter(_safe_structure(a, low) + tuple(parse_va(a["VA"])) for a in high_anns)
                    low_keys = Counter(_safe_structure(a, low) + tuple(parse_va(a["VA"])) for a in low_anns)
                    if high_keys == low_keys:
                        exact += 1
                    else:
                        mismatches += 1
                        if len(examples) < 10:
                            examples.append({"text": text, "higher": list(map(list, high_keys.elements())), "lower": list(map(list, low_keys.elements()))})
                except Exception as exc:
                    mismatches += 1
                    invalid_va_records += 1
                    if len(examples) < 10:
                        examples.append({"text": text, "error": str(exc)})
            task_consistency_rows.append({
                "language": lang,
                "domain": domain,
                "split": split,
                "higher_task": high,
                "lower_task": low,
                "shared_texts": len(shared),
                "exact_projection_matches": exact,
                "mismatches": mismatches,
                "invalid_va_records": invalid_va_records,
                "match_rate": exact / len(shared) if shared else None,
                "examples": json.dumps(examples, ensure_ascii=False),
            })

    # Categorical-to-VA geometry. Malformed VA labels are skipped but counted.
    categorical = {k: v for k, v in loaded.items() if k[0] == "categorical"}
    for _, task, lang, domain, split in sorted(dimensional):
        dim_item = dimensional.get(("dimensional", task, lang, domain, split))
        cat_item = categorical.get(("categorical", task, lang, domain, split))
        if not dim_item or not cat_item:
            continue
        dim_records = {normalize_text(r.get("Text", "")): r for r in dim_item[1] if isinstance(r.get("Text"), str)}
        cat_records = {normalize_text(r.get("Text", "")): r for r in cat_item[1] if isinstance(r.get("Text"), str)}
        values: dict[str, list[tuple[float, float]]] = defaultdict(list)
        skipped_invalid_va = 0
        for text in set(dim_records) & set(cat_records):
            cat_by_structure: dict[tuple[Any, ...], list[str]] = defaultdict(list)
            for ann in iter_annotations(cat_records[text]):
                pol = _polarity(ann)
                if pol is not None:
                    cat_by_structure[_safe_structure(ann, task)].append(pol)
            for ann in iter_annotations(dim_records[text]):
                structure = _safe_structure(ann, task)
                if not cat_by_structure.get(structure):
                    continue
                try:
                    va = parse_va(ann["VA"])
                except Exception:
                    skipped_invalid_va += 1
                    continue
                pol = cat_by_structure[structure].pop(0)
                values[pol].append(va)
        for polarity, pairs in sorted(values.items()):
            vals = [v for v, _ in pairs]
            aros = [a for _, a in pairs]
            alignment_rows.append({
                "subtask": task,
                "language": lang,
                "domain": domain,
                "split": split,
                "polarity": polarity,
                "matched_annotations": len(pairs),
                "skipped_invalid_va": skipped_invalid_va,
                "valence_mean": _mean(vals),
                "valence_std": _std(vals),
                "arousal_mean": _mean(aros),
                "arousal_std": _std(aros),
            })

    _write_csv(output_dir / "file_inventory.csv", file_rows)
    _write_csv(output_dir / "dataset_summary.csv", summary_rows)
    _write_csv(output_dir / "validation_issues.csv", validation_rows)
    _write_csv(output_dir / "va_annotations.csv", va_rows)
    _write_csv(output_dir / "within_file_duplicates.csv", duplicate_rows)
    _write_csv(output_dir / "split_overlap.csv", split_overlap_rows)
    _write_csv(output_dir / "cross_task_consistency.csv", task_consistency_rows)
    _write_csv(output_dir / "categorical_va_alignment.csv", alignment_rows)

    report = {
        "raw_root": str(raw_root),
        "files": len(file_rows),
        "records": total_records,
        "annotations": total_annotations,
        "validation_issues": len(validation_rows),
        "invalid_va_annotations": invalid_va_count,
        "json_parse_errors": json_errors,
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
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
