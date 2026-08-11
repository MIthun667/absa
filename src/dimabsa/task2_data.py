from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .experiment_data import TaskRecord, Target
from .training_views import (
    build_relation_expanded_view,
    build_unambiguous_view,
)


NULL_TERM = "NULL"


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int


@dataclass(frozen=True)
class SurfaceNode:
    text: str
    is_null: bool
    occurrences: tuple[CharSpan, ...]


@dataclass(frozen=True)
class Task2Relation:
    aspect: str
    opinion: str
    valence: float
    arousal: float

    @property
    def key(self) -> tuple[str, str]:
        return (
            self.aspect.casefold(),
            self.opinion.casefold(),
        )


@dataclass(frozen=True)
class Task2Example:
    record_id: str
    text: str
    language: str
    domain: str
    split: str

    aspects: tuple[SurfaceNode, ...]
    opinions: tuple[SurfaceNode, ...]

    relations: tuple[Task2Relation, ...]

    source_path: str


def normalize_surface(value: str | None) -> str:
    if value is None:
        return NULL_TERM

    value = str(value).strip()

    if not value:
        return NULL_TERM

    if value.casefold() == "null":
        return NULL_TERM

    return value


def find_all_occurrences(
    text: str,
    term: str,
) -> tuple[CharSpan, ...]:
    """
    Return all case-insensitive literal occurrences of term in text.

    This Phase-2 implementation targets English Laptop/Restaurant, for which
    the forensic audit established that every explicit annotation occurs
    literally in the sentence.
    """
    if normalize_surface(term) == NULL_TERM:
        return ()

    haystack = text.casefold()
    needle = term.casefold()

    spans: list[CharSpan] = []

    start = 0

    while True:
        index = haystack.find(
            needle,
            start,
        )

        if index < 0:
            break

        spans.append(
            CharSpan(
                start=index,
                end=index + len(needle),
            )
        )

        start = index + len(needle)

    return tuple(spans)


def make_surface_node(
    text: str,
    term: str | None,
) -> SurfaceNode:

    surface = normalize_surface(term)

    if surface == NULL_TERM:
        return SurfaceNode(
            text=NULL_TERM,
            is_null=True,
            occurrences=(),
        )

    occurrences = find_all_occurrences(
        text,
        surface,
    )

    if not occurrences:
        raise ValueError(
            f"Explicit term not found literally in text: "
            f"term={surface!r}, text={text!r}"
        )

    return SurfaceNode(
        text=surface,
        is_null=False,
        occurrences=occurrences,
    )


def _deduplicate_nodes(
    text: str,
    values: Iterable[str | None],
) -> tuple[SurfaceNode, ...]:

    representatives: dict[
        str,
        str
    ] = {}

    for value in values:
        surface = normalize_surface(
            value
        )

        key = surface.casefold()

        if key not in representatives:
            representatives[key] = surface

    # Deterministic order:
    # explicit nodes alphabetically by normalized surface,
    # NULL last.
    ordered = sorted(
        representatives.values(),
        key=lambda value: (
            normalize_surface(value)
            == NULL_TERM,
            normalize_surface(value).casefold(),
        ),
    )

    return tuple(
        make_surface_node(
            text,
            value,
        )
        for value in ordered
    )


def _training_targets_by_record(
    records: list[TaskRecord],
    *,
    view: str,
) -> dict[str, list[Target]]:

    if view == "unambiguous":
        projected = build_unambiguous_view(
            records
        )

    elif view == "relation_expanded":
        projected = (
            build_relation_expanded_view(
                records
            )
        )

    else:
        raise ValueError(
            f"Unsupported Task-2 training view: {view}"
        )

    grouped: dict[
        str,
        list[Target]
    ] = {}

    for item in projected:
        grouped.setdefault(
            item.record_id,
            [],
        ).append(
            item.target
        )

    return grouped


def build_task2_train_examples(
    records: list[TaskRecord],
    *,
    view: str = "unambiguous",
) -> list[Task2Example]:
    """
    Build Task-2 training examples.

    The default view removes projected (Aspect, Opinion) groups that carry
    conflicting VA values and collapses safe duplicate structural relations.

    This matches the Phase-2 primary Task-2 baseline protocol.
    """

    targets_by_record = (
        _training_targets_by_record(
            records,
            view=view,
        )
    )

    examples: list[
        Task2Example
    ] = []

    for record in records:

        targets = targets_by_record.get(
            record.record_id,
            [],
        )

        # A record can become empty if every projected relation was
        # genuinely VA-ambiguous. Skip such records for supervised T2.
        if not targets:
            continue

        relations: list[
            Task2Relation
        ] = []

        for target in targets:

            opinion = normalize_surface(
                target.opinion
            )

            relations.append(
                Task2Relation(
                    aspect=normalize_surface(
                        target.aspect
                    ),
                    opinion=opinion,
                    valence=target.valence,
                    arousal=target.arousal,
                )
            )

        aspects = _deduplicate_nodes(
            record.text,
            (
                relation.aspect
                for relation in relations
            ),
        )

        opinions = _deduplicate_nodes(
            record.text,
            (
                relation.opinion
                for relation in relations
            ),
        )

        examples.append(
            Task2Example(
                record_id=record.record_id,
                text=record.text,
                language=record.language,
                domain=record.domain,
                split=record.split,
                aspects=aspects,
                opinions=opinions,
                relations=tuple(relations),
                source_path=record.source_path,
            )
        )

    return examples


def build_task2_eval_examples(
    records: list[TaskRecord],
) -> list[Task2Example]:
    """
    Build native Task-2 dev/test examples.

    Dev/test use the official Triplet annotations directly.
    """

    examples: list[
        Task2Example
    ] = []

    for record in records:

        relations: list[
            Task2Relation
        ] = []

        for target in record.targets:

            relations.append(
                Task2Relation(
                    aspect=normalize_surface(
                        target.aspect
                    ),
                    opinion=normalize_surface(
                        target.opinion
                    ),
                    valence=target.valence,
                    arousal=target.arousal,
                )
            )

        aspects = _deduplicate_nodes(
            record.text,
            (
                relation.aspect
                for relation in relations
            ),
        )

        opinions = _deduplicate_nodes(
            record.text,
            (
                relation.opinion
                for relation in relations
            ),
        )

        examples.append(
            Task2Example(
                record_id=record.record_id,
                text=record.text,
                language=record.language,
                domain=record.domain,
                split=record.split,
                aspects=aspects,
                opinions=opinions,
                relations=tuple(relations),
                source_path=record.source_path,
            )
        )

    return examples


def candidate_relation_keys(
    example: Task2Example,
    *,
    always_include_null: bool = True,
) -> tuple[
    tuple[str, str],
    ...
]:
    """
    Produce the full aspect x opinion candidate surface space.

    When always_include_null=True, NULL aspect/opinion nodes are added even
    when absent from gold. This is useful for training negative implicit
    candidates.
    """

    aspects = {
        node.text.casefold():
        node.text
        for node in example.aspects
    }

    opinions = {
        node.text.casefold():
        node.text
        for node in example.opinions
    }

    if always_include_null:
        aspects.setdefault(
            NULL_TERM.casefold(),
            NULL_TERM,
        )

        opinions.setdefault(
            NULL_TERM.casefold(),
            NULL_TERM,
        )

    keys = [
        (
            aspect,
            opinion,
        )
        for aspect in aspects.values()
        for opinion in opinions.values()
    ]

    return tuple(
        sorted(
            keys,
            key=lambda pair: (
                pair[0].casefold(),
                pair[1].casefold(),
            ),
        )
    )


def gold_relation_map(
    example: Task2Example,
) -> dict[
    tuple[str, str],
    Task2Relation
]:
    result: dict[
        tuple[str, str],
        Task2Relation
    ] = {}

    for relation in example.relations:

        key = relation.key

        if key in result:
            raise ValueError(
                f"Duplicate Task-2 relation after training-view "
                f"construction: {example.record_id} / {key}"
            )

        result[key] = relation

    return result
