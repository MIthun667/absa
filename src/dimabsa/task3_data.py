from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from dimabsa.experiment_data import TaskRecord, Target
from dimabsa.task2_data import (
    NULL_TERM,
    SurfaceNode,
    make_surface_node,
    normalize_surface,
)


@dataclass(frozen=True)
class Task3Quadruplet:
    aspect: str
    opinion: str
    category: str
    valence: float
    arousal: float


@dataclass(frozen=True)
class Task3Relation:
    aspect: str
    opinion: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class Task3Example:
    record_id: str
    text: str
    domain: str
    aspect_nodes: tuple[SurfaceNode, ...]
    opinion_nodes: tuple[SurfaceNode, ...]
    quadruplets: tuple[Task3Quadruplet, ...]


def normalize_category(category: str) -> str:
    """
    Preserve official category spelling while removing accidental whitespace.
    """
    return str(category).strip()


def _canonical_term(value: str | None) -> str:
    if value is None:
        return NULL_TERM

    value = str(value).strip()

    if not value or value.casefold() == "null":
        return NULL_TERM

    return value


def _target_to_quadruplet(target: Target) -> Task3Quadruplet:
    if target.category is None:
        raise ValueError(
            "Task 3 target is missing category."
        )

    return Task3Quadruplet(
        aspect=_canonical_term(target.aspect),
        opinion=_canonical_term(target.opinion),
        category=normalize_category(target.category),
        valence=float(target.valence),
        arousal=float(target.arousal),
    )


def _deduplicate_nodes(
    nodes: Iterable[SurfaceNode],
) -> tuple[SurfaceNode, ...]:
    """
    Deduplicate nodes by normalized surface form.

    If the same explicit surface occurs multiple times in the sentence,
    make_surface_node already stores all occurrences inside one node.

    NULL is kept at the end for deterministic ordering.
    """
    by_surface: dict[str, SurfaceNode] = {}

    for node in nodes:
        key = normalize_surface(node.text)

        if key not in by_surface:
            by_surface[key] = node

    explicit = [
        node
        for key, node in by_surface.items()
        if key != normalize_surface(NULL_TERM)
    ]

    explicit.sort(
        key=lambda node: normalize_surface(node.text)
    )

    null_nodes = [
        node
        for key, node in by_surface.items()
        if key == normalize_surface(NULL_TERM)
    ]

    return tuple(explicit + null_nodes)


def build_task3_examples(
    records: Iterable[TaskRecord],
) -> list[Task3Example]:
    """
    Build faithful Task-3 examples from official Quadruplet records.

    Important:
    - No averaging across categories.
    - No collapsing category-specific VA.
    - If the same (Aspect, Opinion) has multiple categories, all
      quadruplets are retained.
    """
    examples: list[Task3Example] = []

    for record in records:

        quadruplets = tuple(
            _target_to_quadruplet(target)
            for target in record.targets
        )

        if not quadruplets:
            continue

        aspect_surfaces = {
            normalize_surface(q.aspect): q.aspect
            for q in quadruplets
        }

        opinion_surfaces = {
            normalize_surface(q.opinion): q.opinion
            for q in quadruplets
        }

        aspect_nodes = _deduplicate_nodes(
            make_surface_node(
                record.text,
                surface,
            )
            for surface in aspect_surfaces.values()
        )

        opinion_nodes = _deduplicate_nodes(
            make_surface_node(
                record.text,
                surface,
            )
            for surface in opinion_surfaces.values()
        )

        examples.append(
            Task3Example(
                record_id=record.record_id,
                text=record.text,
                domain=record.domain,
                aspect_nodes=aspect_nodes,
                opinion_nodes=opinion_nodes,
                quadruplets=quadruplets,
            )
        )

    return examples


def gold_quadruplet_map(
    example: Task3Example,
) -> dict[
    tuple[str, str, str],
    tuple[float, float],
]:
    """
    Return:
        (aspect, opinion, category) -> (valence, arousal)

    Structural fields are case-normalized for matching.
    """
    result: dict[
        tuple[str, str, str],
        tuple[float, float],
    ] = {}

    for q in example.quadruplets:

        key = (
            normalize_surface(q.aspect),
            normalize_surface(q.opinion),
            normalize_category(q.category).casefold(),
        )

        value = (
            float(q.valence),
            float(q.arousal),
        )

        if key in result and result[key] != value:
            raise ValueError(
                "Duplicate Task-3 quadruplet key has inconsistent VA: "
                f"{key}: {result[key]} vs {value}"
            )

        result[key] = value

    return result


def relation_category_map(
    example: Task3Example,
) -> dict[
    tuple[str, str],
    tuple[str, ...],
]:
    """
    Return all categories associated with each (Aspect, Opinion).

    This is intentionally multi-label.
    """
    mapping: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    for q in example.quadruplets:

        key = (
            normalize_surface(q.aspect),
            normalize_surface(q.opinion),
        )

        mapping[key].add(
            normalize_category(q.category)
        )

    return {
        key: tuple(sorted(categories))
        for key, categories in mapping.items()
    }


def category_va_map(
    example: Task3Example,
) -> dict[
    tuple[str, str],
    dict[str, tuple[float, float]],
]:
    """
    Return:

        (Aspect, Opinion)
            -> {
                Category_1: (V1, A1),
                Category_2: (V2, A2),
                ...
            }

    This is the core T3 supervision contract.

    It preserves the empirical fact that the same (A,O) relation can
    have multiple categories AND category-specific VA values.
    """
    result: dict[
        tuple[str, str],
        dict[str, tuple[float, float]],
    ] = defaultdict(dict)

    for q in example.quadruplets:

        pair = (
            normalize_surface(q.aspect),
            normalize_surface(q.opinion),
        )

        category = normalize_category(q.category)

        va = (
            float(q.valence),
            float(q.arousal),
        )

        existing = result[pair].get(category)

        if existing is not None and existing != va:
            raise ValueError(
                "Same (Aspect, Opinion, Category) has inconsistent VA: "
                f"{pair}, {category}: {existing} vs {va}"
            )

        result[pair][category] = va

    return dict(result)


def candidate_relation_keys(
    example: Task3Example,
    *,
    always_include_null: bool = True,
) -> tuple[tuple[str, str], ...]:
    """
    Cartesian product of unique Aspect and Opinion surface nodes.

    During model training/inference we may add NULL candidates even if
    they are not present in gold so the model can learn/reject implicit
    structures consistently.
    """
    aspects = {
        normalize_surface(node.text)
        for node in example.aspect_nodes
    }

    opinions = {
        normalize_surface(node.text)
        for node in example.opinion_nodes
    }

    if always_include_null:
        aspects.add(normalize_surface(NULL_TERM))
        opinions.add(normalize_surface(NULL_TERM))

    return tuple(
        sorted(
            (aspect, opinion)
            for aspect in aspects
            for opinion in opinions
        )
    )


def collect_category_vocabulary(
    examples: Iterable[Task3Example],
) -> tuple[str, ...]:
    """
    Deterministic flat category vocabulary for the controlled baseline.
    """
    categories = {
        normalize_category(q.category)
        for example in examples
        for q in example.quadruplets
    }

    return tuple(sorted(categories))


def split_category(
    category: str,
) -> tuple[str, str]:
    """
    Utility for later ontology/compositional experiments.

    Baseline training still uses the flat category string.
    """
    category = normalize_category(category)

    if "#" not in category:
        raise ValueError(
            f"Invalid DimABSA category: {category!r}"
        )

    entity, attribute = category.split("#", 1)

    return entity, attribute
