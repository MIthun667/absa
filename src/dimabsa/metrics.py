from __future__ import annotations

import math
from typing import Any, Iterable

from .data import parse_va

D_MAX = math.sqrt(128.0)


def _lower(value: Any) -> str:
    return str(value).lower()


def _index_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if "ID" not in record:
            raise ValueError("Record is missing ID")
        record_id = str(record["ID"])
        if record_id in indexed:
            raise ValueError(f"Duplicate record ID: {record_id}")
        indexed[record_id] = record
    return indexed


def _va_unchecked(value: Any) -> tuple[float, float]:
    """Parse a VA pair without enforcing [1, 9].

    The official evaluator computes Task-1 RMSE even when a prediction is outside
    the nominal range; it only emits a warning. Structured tasks assign zero
    continuous credit to an out-of-range prediction. Range handling therefore
    belongs to the task-specific scoring function rather than this parser.
    """
    if not isinstance(value, str) or "#" not in value:
        raise ValueError(f"Invalid VA value: {value!r}")
    left, right = value.split("#", maxsplit=1)
    return float(left), float(right)


def pearson_correlation(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Pearson inputs must have the same length")
    if len(x) < 2:
        return float("nan")

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0.0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / denom


def evaluate_task1(
    gold_records: Iterable[dict[str, Any]],
    pred_records: Iterable[dict[str, Any]],
    *,
    normalize_rmse: bool = False,
) -> dict[str, float | int | bool]:
    """Score DimASR with the official Task-1 semantics.

    RMSE_VA is joint two-dimensional RMSE:
        sqrt(sum_i[(Vhat-V)^2 + (Ahat-A)^2] / N)
    and can optionally be divided by sqrt(128), matching ``--do_norm`` in the
    official evaluator.
    """
    gold = _index_records(gold_records)
    pred = _index_records(pred_records)

    gold_v: list[float] = []
    gold_a: list[float] = []
    pred_v: list[float] = []
    pred_a: list[float] = []
    out_of_range = False

    for record_id, gold_record in gold.items():
        if record_id not in pred:
            raise ValueError(f"Missing prediction record for ID: {record_id}")
        gold_items = gold_record.get("Aspect_VA")
        pred_items = pred[record_id].get("Aspect_VA")
        if not isinstance(gold_items, list) or not isinstance(pred_items, list):
            raise ValueError(f"Aspect_VA must be a list for ID: {record_id}")

        pred_by_aspect: dict[str, dict[str, Any]] = {}
        for item in pred_items:
            if not isinstance(item, dict) or "Aspect" not in item or "VA" not in item:
                raise ValueError(f"Malformed Task-1 prediction for ID: {record_id}")
            aspect = _lower(item["Aspect"])
            if aspect in pred_by_aspect:
                raise ValueError(f"Duplicate Task-1 prediction for ID={record_id}, aspect={aspect!r}")
            pred_by_aspect[aspect] = item

        for gold_item in gold_items:
            if not isinstance(gold_item, dict) or "Aspect" not in gold_item or "VA" not in gold_item:
                raise ValueError(f"Malformed Task-1 gold annotation for ID: {record_id}")
            aspect = _lower(gold_item["Aspect"])
            if aspect not in pred_by_aspect:
                raise ValueError(f"Missing Task-1 VA for ID={record_id}, aspect={aspect!r}")

            gv, ga = _va_unchecked(gold_item["VA"])
            pv, pa = _va_unchecked(pred_by_aspect[aspect]["VA"])
            if not (1.0 <= pv <= 9.0 and 1.0 <= pa <= 9.0):
                out_of_range = True

            gold_v.append(gv)
            gold_a.append(ga)
            pred_v.append(pv)
            pred_a.append(pa)

    if not gold_v:
        raise ValueError("No Task-1 gold annotations to score")

    squared_error = sum((pv - gv) ** 2 for pv, gv in zip(pred_v, gold_v))
    squared_error += sum((pa - ga) ** 2 for pa, ga in zip(pred_a, gold_a))
    rmse = math.sqrt(squared_error / len(gold_v))
    if normalize_rmse:
        rmse /= D_MAX

    return {
        "PCC_V": pearson_correlation(pred_v, gold_v),
        "PCC_A": pearson_correlation(pred_a, gold_a),
        "RMSE_VA": rmse,
        "N": len(gold_v),
        "out_of_range_predictions": out_of_range,
    }


def _match_key(annotation: dict[str, Any], task: int) -> tuple[str, ...]:
    if task == 2:
        required = ("Aspect", "Opinion")
    elif task == 3:
        required = ("Aspect", "Opinion", "Category")
    else:
        raise ValueError("Structured evaluation supports only tasks 2 and 3")

    values: list[str] = []
    for field in required:
        if field not in annotation:
            raise ValueError(f"Structured annotation is missing {field}")
        values.append(_lower(annotation[field]))
    return tuple(values)


def evaluate_structured(
    gold_records: Iterable[dict[str, Any]],
    pred_records: Iterable[dict[str, Any]],
    *,
    task: int,
) -> dict[str, float | int | bool]:
    """Official-compatible cF1 for DimASTE (T2) and DimASQP (T3).

    Structural fields must match exactly after lower-casing. A unique structural
    match receives continuous true-positive credit ``1 - d/sqrt(128)``. If a
    prediction repeats a matching structure more than once, the gold item is
    treated as unmatched and all duplicates remain false positives, mirroring the
    official script.
    """
    if task not in (2, 3):
        raise ValueError("task must be 2 or 3")
    field = "Triplet" if task == 2 else "Quadruplet"

    gold = _index_records(gold_records)
    pred = _index_records(pred_records)
    all_ids = set(gold) | set(pred)

    ctp_total = 0.0
    tp_structural = 0
    fp = 0
    fn = 0
    duplicate_match = False
    out_of_range = False

    for record_id in all_ids:
        gold_items = gold.get(record_id, {}).get(field, [])
        pred_items = pred.get(record_id, {}).get(field, [])
        if not isinstance(gold_items, list) or not isinstance(pred_items, list):
            raise ValueError(f"{field} must be a list for ID: {record_id}")

        for item in [*gold_items, *pred_items]:
            if not isinstance(item, dict) or "VA" not in item:
                raise ValueError(f"Malformed {field} annotation for ID: {record_id}")

        matched_pred_num = 0
        for gold_item in gold_items:
            gold_key = _match_key(gold_item, task)
            matches = [item for item in pred_items if _match_key(item, task) == gold_key]

            if len(matches) > 1:
                duplicate_match = True
                fn += 1
                continue
            if not matches:
                fn += 1
                continue

            pred_item = matches[0]
            matched_pred_num += 1
            tp_structural += 1

            gv, ga = _va_unchecked(gold_item["VA"])
            pv, pa = _va_unchecked(pred_item["VA"])
            if not (1.0 <= pv <= 9.0 and 1.0 <= pa <= 9.0):
                out_of_range = True
                continue

            distance = math.hypot(pv - gv, pa - ga)
            ctp_total += max(0.0, 1.0 - distance / D_MAX)

        fp += len(pred_items) - matched_pred_num

    cprecision = ctp_total / (tp_structural + fp) if (tp_structural + fp) else 0.0
    crecall = ctp_total / (tp_structural + fn) if (tp_structural + fn) else 0.0
    cf1 = (
        2.0 * cprecision * crecall / (cprecision + crecall)
        if (cprecision + crecall)
        else 0.0
    )

    return {
        "TP_structural": tp_structural,
        "cTP": ctp_total,
        "FP": fp,
        "FN": fn,
        "cPrecision": cprecision,
        "cRecall": crecall,
        "cF1": cf1,
        "duplicate_matching_predictions": duplicate_match,
        "out_of_range_predictions": out_of_range,
    }
