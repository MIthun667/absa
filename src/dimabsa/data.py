from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

KNOWN_ANNOTATION_FIELDS = (
    "Quadruplet",
    "Triplet",
    "Aspect_VA",
    "Aspect_Polarity",
)

_SPLIT_RE = re.compile(r"_(train|dev|test)(?:_|\.jsonl$)")


@dataclass(frozen=True)
class DatasetFile:
    path: Path
    representation: str
    subtask: int
    language: str
    domain: str
    split: str
    training_scope: str

    @property
    def key(self) -> tuple[str, int, str, str, str]:
        return (
            self.representation,
            self.subtask,
            self.language,
            self.domain,
            self.split,
        )


@dataclass(frozen=True)
class JsonlError:
    path: str
    line_number: int
    error: str
    raw: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def normalize_term(value: Any) -> str | None:
    if value is None:
        return None
    value = " ".join(str(value).split())
    if not value or value.upper() == "NULL":
        return None
    return value.casefold()


def parse_dataset_file(path: Path, raw_root: Path) -> DatasetFile:
    relative = path.relative_to(raw_root)
    parts = relative.parts

    representation = "categorical" if parts[0] == "categorical" else "dimensional"

    subtask = None
    language = None
    for idx, part in enumerate(parts):
        match = re.fullmatch(r"subtask_(\d+)", part)
        if match:
            subtask = int(match.group(1))
            if idx + 1 < len(parts):
                language = parts[idx + 1]
            break

    if subtask is None or language is None:
        raise ValueError(f"Cannot infer subtask/language from {relative}")

    filename = path.name
    tokens = filename.split("_")
    if len(tokens) < 2:
        raise ValueError(f"Cannot infer domain from {relative}")
    domain = tokens[1]

    split_match = _SPLIT_RE.search(filename)
    if not split_match:
        raise ValueError(f"Cannot infer split from {relative}")
    split = split_match.group(1)

    training_scope = "alltasks" if "alltasks" in filename else f"task{subtask}"

    return DatasetFile(
        path=path,
        representation=representation,
        subtask=subtask,
        language=language,
        domain=domain,
        split=split,
        training_scope=training_scope,
    )


def discover_dataset_files(raw_root: Path) -> list[DatasetFile]:
    return [parse_dataset_file(path, raw_root) for path in sorted(raw_root.rglob("*.jsonl"))]


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[JsonlError]]:
    records: list[dict[str, Any]] = []
    errors: list[JsonlError] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(
                    JsonlError(
                        path=str(path),
                        line_number=line_number,
                        error=str(exc),
                        raw=raw[:500],
                    )
                )
                continue

            if not isinstance(item, dict):
                errors.append(
                    JsonlError(
                        path=str(path),
                        line_number=line_number,
                        error=f"Expected JSON object, got {type(item).__name__}",
                        raw=raw[:500],
                    )
                )
                continue
            records.append(item)

    return records, errors


def annotation_field(record: dict[str, Any]) -> str | None:
    present = [field for field in KNOWN_ANNOTATION_FIELDS if field in record]
    if len(present) > 1:
        raise ValueError(f"Record contains multiple annotation fields: {present}")
    return present[0] if present else None


def iter_annotations(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    field = annotation_field(record)
    if field is None:
        return ()
    annotations = record[field]
    if not isinstance(annotations, list):
        raise TypeError(f"{field} must be a list")
    return annotations


def parse_va(value: Any) -> tuple[float, float]:
    if not isinstance(value, str) or "#" not in value:
        raise ValueError(f"Invalid VA value: {value!r}")
    left, right = value.split("#", maxsplit=1)
    valence = float(left)
    arousal = float(right)
    if not (1.0 <= valence <= 9.0 and 1.0 <= arousal <= 9.0):
        raise ValueError(f"VA outside [1, 9]: {value!r}")
    return valence, arousal


def normalized_structure(annotation: dict[str, Any]) -> dict[str, Any]:
    """Normalize non-VA fields without requiring a valid VA label.

    Structural matching must remain possible even when an official VA annotation is
    malformed or outside the documented range. VA validation is intentionally kept
    separate in ``parse_va``.
    """
    polarity = annotation.get("Polarity")
    if isinstance(polarity, str):
        polarity = polarity.strip().upper()

    return {
        "aspect": normalize_term(annotation.get("Aspect")),
        "opinion": normalize_term(annotation.get("Opinion")),
        "category": normalize_term(annotation.get("Category")),
        "polarity": polarity,
    }


def normalized_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        **normalized_structure(annotation),
        "valence": None,
        "arousal": None,
    }

    if "VA" in annotation:
        valence, arousal = parse_va(annotation["VA"])
        result["valence"] = valence
        result["arousal"] = arousal

    return result


def structural_key(annotation: dict[str, Any], field: str) -> tuple[Any, ...]:
    item = normalized_structure(annotation)
    if field in {"Aspect_VA", "Aspect_Polarity"}:
        return (item["aspect"],)
    if field == "Triplet":
        return (item["aspect"], item["opinion"])
    if field == "Quadruplet":
        return (item["aspect"], item["opinion"], item["category"])
    raise ValueError(f"Unknown annotation field: {field}")


def projected_key(annotation: dict[str, Any], target_subtask: int) -> tuple[Any, ...]:
    item = normalized_structure(annotation)
    if target_subtask == 1:
        return (item["aspect"],)
    if target_subtask == 2:
        return (item["aspect"], item["opinion"])
    if target_subtask == 3:
        return (item["aspect"], item["opinion"], item["category"])
    raise ValueError(f"Unknown target subtask: {target_subtask}")


def annotation_with_va_key(
    annotation: dict[str, Any],
    target_subtask: int,
    decimals: int = 4,
) -> tuple[Any, ...]:
    item = normalized_annotation(annotation)
    structural = projected_key(annotation, target_subtask)
    if item["valence"] is None or item["arousal"] is None:
        return structural
    return structural + (
        round(float(item["valence"]), decimals),
        round(float(item["arousal"]), decimals),
    )
