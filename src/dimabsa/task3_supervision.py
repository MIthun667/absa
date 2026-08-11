from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dimabsa.task2_data import NULL_TERM, SurfaceNode, normalize_surface
from dimabsa.task2_supervision import (
    EncodedNode,
    make_bio_labels,
    node_token_indices,
)
from dimabsa.task3_data import (
    Task3Example,
    normalize_category,
)


@dataclass(frozen=True)
class CategoryVATarget:
    category_index: int
    valence: float
    arousal: float


@dataclass(frozen=True)
class Task3RelationCandidate:
    aspect_index: int
    opinion_index: int

    relation_label: float

    # Multi-label category supervision.
    category_labels: tuple[float, ...]

    # One item per original gold quadruplet.
    # Duplicated A/O/C supervision is deliberately preserved.
    va_targets: tuple[CategoryVATarget, ...]


@dataclass(frozen=True)
class EncodedTask3Example:
    record_id: str
    text: str
    domain: str

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    aspect_bio_labels: tuple[int, ...]
    opinion_bio_labels: tuple[int, ...]

    aspect_nodes: tuple[EncodedNode, ...]
    opinion_nodes: tuple[EncodedNode, ...]

    relation_candidates: tuple[
        Task3RelationCandidate, ...
    ]


def _node_text(node: SurfaceNode) -> str:
    return node.text


def _ensure_null_node(
    nodes: Sequence[SurfaceNode],
) -> tuple[SurfaceNode, ...]:

    if any(node.is_null for node in nodes):
        return tuple(nodes)

    return tuple(nodes) + (
        SurfaceNode(
            text=NULL_TERM,
            is_null=True,
            occurrences=(),
        ),
    )


def _encoded_nodes(
    nodes: Sequence[SurfaceNode],
    offset_mapping: Sequence[tuple[int, int]],
) -> tuple[EncodedNode, ...]:

    encoded = []

    for node in nodes:

        indices = node_token_indices(
            node,
            offset_mapping,
        )

        encoded.append(
            EncodedNode(
                text=node.text,
                is_null=node.is_null,
                token_indices=tuple(indices),
            )
        )

    return tuple(encoded)


def encode_task3_example(
    example: Task3Example,
    tokenizer,
    *,
    category_to_index: dict[str, int],
    max_length: int = 256,
) -> EncodedTask3Example:

    encoded = tokenizer(
        example.text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    input_ids = tuple(encoded["input_ids"])
    attention_mask = tuple(
        encoded["attention_mask"]
    )

    offsets = tuple(
        tuple(x)
        for x in encoded["offset_mapping"]
    )

    aspect_nodes_raw = _ensure_null_node(
        example.aspect_nodes
    )

    opinion_nodes_raw = _ensure_null_node(
        example.opinion_nodes
    )

    aspect_labels = make_bio_labels(
        nodes=aspect_nodes_raw,
        offsets=offsets,
    )

    opinion_labels = make_bio_labels(
        nodes=opinion_nodes_raw,
        offsets=offsets,
    )

    aspect_nodes = _encoded_nodes(
        aspect_nodes_raw,
        offsets,
    )

    opinion_nodes = _encoded_nodes(
        opinion_nodes_raw,
        offsets,
    )

    aspect_index = {
        normalize_surface(node.text): i
        for i, node in enumerate(aspect_nodes)
    }

    opinion_index = {
        normalize_surface(node.text): i
        for i, node in enumerate(opinion_nodes)
    }

    gold_by_pair: dict[
        tuple[str, str],
        list,
    ] = {}

    for q in example.quadruplets:

        pair = (
            normalize_surface(q.aspect),
            normalize_surface(q.opinion),
        )

        gold_by_pair.setdefault(
            pair,
            [],
        ).append(q)

    candidates = []

    for ai, aspect_node in enumerate(aspect_nodes):

        a_key = normalize_surface(
            aspect_node.text
        )

        for oi, opinion_node in enumerate(opinion_nodes):

            o_key = normalize_surface(
                opinion_node.text
            )

            pair = (
                a_key,
                o_key,
            )

            gold_items = gold_by_pair.get(
                pair,
                [],
            )

            relation_label = (
                1.0 if gold_items else 0.0
            )

            category_labels = [
                0.0
                for _ in range(
                    len(category_to_index)
                )
            ]

            va_targets = []

            for q in gold_items:

                category = normalize_category(
                    q.category
                )

                if category not in category_to_index:
                    raise KeyError(
                        "Category missing from vocabulary: "
                        f"{category}"
                    )

                ci = category_to_index[
                    category
                ]

                category_labels[ci] = 1.0

                va_targets.append(
                    CategoryVATarget(
                        category_index=ci,
                        valence=float(q.valence),
                        arousal=float(q.arousal),
                    )
                )

            candidates.append(
                Task3RelationCandidate(
                    aspect_index=ai,
                    opinion_index=oi,
                    relation_label=relation_label,
                    category_labels=tuple(
                        category_labels
                    ),
                    va_targets=tuple(
                        va_targets
                    ),
                )
            )

    return EncodedTask3Example(
        record_id=example.record_id,
        text=example.text,
        domain=example.domain,
        input_ids=input_ids,
        attention_mask=attention_mask,
        aspect_bio_labels=tuple(
            aspect_labels
        ),
        opinion_bio_labels=tuple(
            opinion_labels
        ),
        aspect_nodes=aspect_nodes,
        opinion_nodes=opinion_nodes,
        relation_candidates=tuple(
            candidates
        ),
    )
