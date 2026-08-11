from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .task2_data import (
    NULL_TERM,
    SurfaceNode,
    Task2Example,
    gold_relation_map,
)


O = 0
B = 1
I = 2

IGNORE_INDEX = -100


@dataclass(frozen=True)
class EncodedNode:
    text: str
    is_null: bool
    token_indices: tuple[int, ...]


@dataclass(frozen=True)
class RelationCandidate:
    aspect_index: int
    opinion_index: int
    label: int
    va: tuple[float, float] | None


@dataclass
class EncodedTask2Example:
    record_id: str
    text: str
    domain: str

    input_ids: list[int]
    attention_mask: list[int]
    offsets: list[tuple[int, int]]

    aspect_labels: list[int]
    opinion_labels: list[int]

    aspect_nodes: list[EncodedNode]
    opinion_nodes: list[EncodedNode]

    relation_candidates: list[RelationCandidate]


def overlapping_token_indices(
    offsets: list[tuple[int, int]],
    *,
    char_start: int,
    char_end: int,
) -> list[int]:

    result: list[int] = []

    for index, (start, end) in enumerate(offsets):

        # Special tokens generally have (0, 0).
        if start == end:
            continue

        if start < char_end and end > char_start:
            result.append(index)

    return result


def node_token_indices(
    node: SurfaceNode,
    offsets: list[tuple[int, int]],
) -> tuple[int, ...]:

    if node.is_null:
        return ()

    indices: set[int] = set()

    for occurrence in node.occurrences:

        matched = overlapping_token_indices(
            offsets,
            char_start=occurrence.start,
            char_end=occurrence.end,
        )

        if not matched:
            raise ValueError(
                f"No tokenizer overlap for explicit node "
                f"{node.text!r}"
            )

        indices.update(matched)

    return tuple(sorted(indices))


def make_bio_labels(
    *,
    offsets: list[tuple[int, int]],
    nodes: tuple[SurfaceNode, ...],
) -> list[int]:

    labels = [
        IGNORE_INDEX if start == end else O
        for start, end in offsets
    ]

    for node in nodes:

        if node.is_null:
            continue

        for occurrence in node.occurrences:

            token_indices = overlapping_token_indices(
                offsets,
                char_start=occurrence.start,
                char_end=occurrence.end,
            )

            if not token_indices:
                raise ValueError(
                    f"Could not align {node.text!r}"
                )

            labels[token_indices[0]] = B

            for index in token_indices[1:]:
                labels[index] = I

    return labels


def _with_null_node(
    nodes: tuple[SurfaceNode, ...],
) -> list[SurfaceNode]:

    result = list(nodes)

    if not any(node.is_null for node in result):
        result.append(
            SurfaceNode(
                text=NULL_TERM,
                is_null=True,
                occurrences=(),
            )
        )

    return result


def encode_task2_example(
    example: Task2Example,
    *,
    tokenizer,
    max_length: int = 256,
) -> EncodedTask2Example:

    encoded = tokenizer(
        example.text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])

    offsets = [
        tuple(pair)
        for pair in encoded["offset_mapping"]
    ]

    aspect_labels = make_bio_labels(
        offsets=offsets,
        nodes=example.aspects,
    )

    opinion_labels = make_bio_labels(
        offsets=offsets,
        nodes=example.opinions,
    )

    aspect_surface_nodes = _with_null_node(
        example.aspects
    )

    opinion_surface_nodes = _with_null_node(
        example.opinions
    )

    aspect_nodes = [
        EncodedNode(
            text=node.text,
            is_null=node.is_null,
            token_indices=node_token_indices(
                node,
                offsets,
            ),
        )
        for node in aspect_surface_nodes
    ]

    opinion_nodes = [
        EncodedNode(
            text=node.text,
            is_null=node.is_null,
            token_indices=node_token_indices(
                node,
                offsets,
            ),
        )
        for node in opinion_surface_nodes
    ]

    gold = gold_relation_map(
        example
    )

    relation_candidates: list[
        RelationCandidate
    ] = []

    for aspect_index, aspect in enumerate(
        aspect_nodes
    ):
        for opinion_index, opinion in enumerate(
            opinion_nodes
        ):

            key = (
                aspect.text.casefold(),
                opinion.text.casefold(),
            )

            relation = gold.get(key)

            if relation is None:

                relation_candidates.append(
                    RelationCandidate(
                        aspect_index=aspect_index,
                        opinion_index=opinion_index,
                        label=0,
                        va=None,
                    )
                )

            else:

                relation_candidates.append(
                    RelationCandidate(
                        aspect_index=aspect_index,
                        opinion_index=opinion_index,
                        label=1,
                        va=(
                            relation.valence,
                            relation.arousal,
                        ),
                    )
                )

    return EncodedTask2Example(
        record_id=example.record_id,
        text=example.text,
        domain=example.domain,
        input_ids=input_ids,
        attention_mask=attention_mask,
        offsets=offsets,
        aspect_labels=aspect_labels,
        opinion_labels=opinion_labels,
        aspect_nodes=aspect_nodes,
        opinion_nodes=opinion_nodes,
        relation_candidates=relation_candidates,
    )


def decode_bio_spans(
    *,
    labels: list[int],
    offsets: list[tuple[int, int]],
    text: str,
) -> list[str]:
    """
    Convert predicted BIO labels back to unique surface strings.

    Repeated occurrences of the same surface form are deduplicated because
    the official Task-2 evaluator operates on surface (Aspect, Opinion)
    structure rather than character offsets.
    """

    spans: list[tuple[int, int]] = []

    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal current_start, current_end

        if (
            current_start is not None
            and current_end is not None
        ):
            spans.append(
                (
                    current_start,
                    current_end,
                )
            )

        current_start = None
        current_end = None

    for label, (start, end) in zip(
        labels,
        offsets,
    ):

        if start == end:
            continue

        if label == B:

            flush()

            current_start = start
            current_end = end

        elif label == I:

            if current_start is None:
                # Robust BIO decoding:
                # orphan I starts a new span.
                current_start = start

            current_end = end

        else:
            flush()

    flush()

    surfaces: dict[str, str] = {}

    for start, end in spans:

        value = text[start:end].strip()

        if not value:
            continue

        surfaces.setdefault(
            value.casefold(),
            value,
        )

    return list(surfaces.values())
