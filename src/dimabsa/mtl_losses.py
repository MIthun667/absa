from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def task1_loss(
    *,
    model,
    inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """
    Exact T1 objective:
        sentence + aspect -> first-token representation -> bounded VA
        mean MSE over valence/arousal.
    """

    hidden = model.encode(
        input_ids=inputs[
            "input_ids"
        ],
        attention_mask=inputs[
            "attention_mask"
        ],
        task="t1",
    )

    cls = hidden[
        :,
        0,
    ]

    predictions = (
        model.t1_predict_va(
            cls
        )
    )

    loss = F.mse_loss(
        predictions,
        labels,
        reduction="mean",
    )

    return (
        loss,
        {
            "t1_va": loss.detach(),
        },
    )


def task2_relation_and_va_loss(
    *,
    model,
    hidden: torch.Tensor,
    examples: list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    relation_logits = []
    relation_labels = []

    predicted_va = []
    gold_va = []

    for batch_index, example in enumerate(
        examples
    ):

        example_hidden = hidden[
            batch_index
        ]

        sentence_repr = (
            example_hidden[0]
        )

        aspect_reprs = [
            model.pool_node(
                example_hidden,
                sentence_repr,
                node,
                task="t2",
                node_type="aspect",
            )
            for node
            in example.aspect_nodes
        ]

        opinion_reprs = [
            model.pool_node(
                example_hidden,
                sentence_repr,
                node,
                task="t2",
                node_type="opinion",
            )
            for node
            in example.opinion_nodes
        ]

        for candidate in (
            example.relation_candidates
        ):

            (
                relation_logit,
                va,
            ) = model.t2_score_pair(
                aspect_reprs[
                    candidate.aspect_index
                ],
                opinion_reprs[
                    candidate.opinion_index
                ],
            )

            relation_logits.append(
                relation_logit
            )

            relation_labels.append(
                float(
                    candidate.label
                )
            )

            if candidate.label == 1:

                assert (
                    candidate.va
                    is not None
                )

                predicted_va.append(
                    va
                )

                gold_va.append(
                    candidate.va
                )

    relation_logits_tensor = (
        torch.stack(
            relation_logits
        )
    )

    relation_labels_tensor = (
        torch.tensor(
            relation_labels,
            dtype=torch.float32,
            device=hidden.device,
        )
    )

    relation_loss = (
        F.binary_cross_entropy_with_logits(
            relation_logits_tensor,
            relation_labels_tensor,
        )
    )

    if predicted_va:

        predicted_va_tensor = (
            torch.stack(
                predicted_va
            )
        )

        gold_va_tensor = (
            torch.tensor(
                gold_va,
                dtype=torch.float32,
                device=hidden.device,
            )
        )

        va_loss = F.mse_loss(
            predicted_va_tensor,
            gold_va_tensor,
        )

    else:

        va_loss = (
            hidden.sum()
            * 0.0
        )

    return (
        relation_loss,
        va_loss,
    )


def task2_loss(
    *,
    model,
    batch,
    aspect_weights: torch.Tensor,
    opinion_weights: torch.Tensor,
    ignore_index: int,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:

    input_ids = batch[
        "input_ids"
    ]

    attention_mask = batch[
        "attention_mask"
    ]

    aspect_labels = batch[
        "aspect_labels"
    ]

    opinion_labels = batch[
        "opinion_labels"
    ]

    hidden = model.encode(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task="t2",
    )

    (
        aspect_logits,
        opinion_logits,
    ) = model.t2_token_logits(
        hidden
    )

    aspect_loss = (
        F.cross_entropy(
            aspect_logits.view(
                -1,
                3,
            ),
            aspect_labels.view(
                -1
            ),
            weight=aspect_weights,
            ignore_index=ignore_index,
        )
    )

    opinion_loss = (
        F.cross_entropy(
            opinion_logits.view(
                -1,
                3,
            ),
            opinion_labels.view(
                -1
            ),
            weight=opinion_weights,
            ignore_index=ignore_index,
        )
    )

    (
        relation_loss,
        va_loss,
    ) = task2_relation_and_va_loss(
        model=model,
        hidden=hidden,
        examples=batch[
            "examples"
        ],
    )

    loss = (
        aspect_loss
        + opinion_loss
        + relation_loss
        + va_loss
    )

    return (
        loss,
        {
            "t2_aspect":
                aspect_loss.detach(),
            "t2_opinion":
                opinion_loss.detach(),
            "t2_relation":
                relation_loss.detach(),
            "t2_va":
                va_loss.detach(),
        },
    )


def task3_relation_category_va_loss(
    *,
    model,
    hidden: torch.Tensor,
    examples: list[Any],
    allowed_categories_by_domain,
    allowed_entities_by_domain,
    allowed_attributes_by_domain,
    category_to_entity_index,
    category_to_attribute_index,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    relation_logits = []
    relation_labels = []

    category_losses = []
    entity_losses = []
    attribute_losses = []

    predicted_va = []
    gold_va = []

    for batch_index, example in enumerate(
        examples
    ):

        example_hidden = hidden[
            batch_index
        ]

        sentence_repr = (
            example_hidden[0]
        )

        aspect_reprs = [
            model.pool_node(
                example_hidden,
                sentence_repr,
                node,
                task="t3",
                node_type="aspect",
            )
            for node
            in example.aspect_nodes
        ]

        opinion_reprs = [
            model.pool_node(
                example_hidden,
                sentence_repr,
                node,
                task="t3",
                node_type="opinion",
            )
            for node
            in example.opinion_nodes
        ]

        allowed_category_indices = (
            allowed_categories_by_domain[
                example.domain
            ]
        )

        allowed_entity_indices = (
            allowed_entities_by_domain[
                example.domain
            ]
        )

        allowed_attribute_indices = (
            allowed_attributes_by_domain[
                example.domain
            ]
        )

        category_index_tensor = (
            torch.tensor(
                allowed_category_indices,
                dtype=torch.long,
                device=hidden.device,
            )
        )

        entity_index_tensor = (
            torch.tensor(
                allowed_entity_indices,
                dtype=torch.long,
                device=hidden.device,
            )
        )

        attribute_index_tensor = (
            torch.tensor(
                allowed_attribute_indices,
                dtype=torch.long,
                device=hidden.device,
            )
        )

        for candidate in (
            example.relation_candidates
        ):

            (
                relation_logit,
                category_logit,
                entity_logit,
                attribute_logit,
                relation_repr,
            ) = model.t3_score_pair(
                aspect_reprs[
                    candidate.aspect_index
                ],
                opinion_reprs[
                    candidate.opinion_index
                ],
            )

            relation_logits.append(
                relation_logit
            )

            relation_labels.append(
                float(
                    candidate.relation_label
                )
            )

            if (
                candidate.relation_label
                != 1.0
            ):
                continue

            # ====================================================
            # Flat category supervision
            # ====================================================

            full_category_labels = (
                torch.tensor(
                    candidate.category_labels,
                    dtype=torch.float32,
                    device=hidden.device,
                )
            )

            domain_category_logits = (
                category_logit.index_select(
                    0,
                    category_index_tensor,
                )
            )

            domain_category_labels = (
                full_category_labels.index_select(
                    0,
                    category_index_tensor,
                )
            )

            category_positives = (
                domain_category_labels.sum()
            )

            category_negatives = (
                domain_category_labels.numel()
                - category_positives
            )

            if (
                category_positives.item()
                > 0
            ):

                category_pos_weight = (
                    (
                        category_negatives
                        / category_positives
                    )
                    .clamp(
                        min=1.0
                    )
                )

                category_losses.append(
                    F.binary_cross_entropy_with_logits(
                        domain_category_logits,
                        domain_category_labels,
                        pos_weight=(
                            category_pos_weight
                        ),
                    )
                )

            positive_category_indices = [
                index
                for index, label
                in enumerate(
                    candidate.category_labels
                )
                if label > 0.5
            ]

            gold_entity_indices = {
                category_to_entity_index[
                    index
                ]
                for index
                in positive_category_indices
            }

            gold_attribute_indices = {
                category_to_attribute_index[
                    index
                ]
                for index
                in positive_category_indices
            }

            # ====================================================
            # Entity auxiliary supervision
            # ====================================================

            entity_labels = torch.zeros(
                model.num_t3_entities,
                dtype=torch.float32,
                device=hidden.device,
            )

            for index in (
                gold_entity_indices
            ):
                entity_labels[index] = 1.0

            domain_entity_logits = (
                entity_logit.index_select(
                    0,
                    entity_index_tensor,
                )
            )

            domain_entity_labels = (
                entity_labels.index_select(
                    0,
                    entity_index_tensor,
                )
            )

            entity_positives = (
                domain_entity_labels.sum()
            )

            entity_negatives = (
                domain_entity_labels.numel()
                - entity_positives
            )

            if (
                entity_positives.item()
                > 0
            ):

                entity_pos_weight = (
                    torch.sqrt(
                        (
                            entity_negatives
                            / entity_positives
                        )
                        .clamp(
                            min=1.0
                        )
                    )
                )

                entity_losses.append(
                    F.binary_cross_entropy_with_logits(
                        domain_entity_logits,
                        domain_entity_labels,
                        pos_weight=(
                            entity_pos_weight
                        ),
                    )
                )

            # ====================================================
            # Attribute auxiliary supervision
            # ====================================================

            attribute_labels = (
                torch.zeros(
                    model.num_t3_attributes,
                    dtype=torch.float32,
                    device=hidden.device,
                )
            )

            for index in (
                gold_attribute_indices
            ):
                attribute_labels[
                    index
                ] = 1.0

            domain_attribute_logits = (
                attribute_logit.index_select(
                    0,
                    attribute_index_tensor,
                )
            )

            domain_attribute_labels = (
                attribute_labels.index_select(
                    0,
                    attribute_index_tensor,
                )
            )

            attribute_positives = (
                domain_attribute_labels.sum()
            )

            attribute_negatives = (
                domain_attribute_labels.numel()
                - attribute_positives
            )

            if (
                attribute_positives.item()
                > 0
            ):

                attribute_pos_weight = (
                    torch.sqrt(
                        (
                            attribute_negatives
                            / attribute_positives
                        )
                        .clamp(
                            min=1.0
                        )
                    )
                )

                attribute_losses.append(
                    F.binary_cross_entropy_with_logits(
                        domain_attribute_logits,
                        domain_attribute_labels,
                        pos_weight=(
                            attribute_pos_weight
                        ),
                    )
                )

            # ====================================================
            # Category-conditioned VA
            # ====================================================

            for target in (
                candidate.va_targets
            ):

                va = model.t3_predict_va(
                    relation_repr,
                    torch.tensor(
                        target.category_index,
                        dtype=torch.long,
                        device=hidden.device,
                    ),
                )

                predicted_va.append(
                    va
                )

                gold_va.append(
                    (
                        target.valence,
                        target.arousal,
                    )
                )

    relation_logits_tensor = (
        torch.stack(
            relation_logits
        )
    )

    relation_labels_tensor = (
        torch.tensor(
            relation_labels,
            dtype=torch.float32,
            device=hidden.device,
        )
    )

    relation_loss = (
        F.binary_cross_entropy_with_logits(
            relation_logits_tensor,
            relation_labels_tensor,
        )
    )

    category_loss = (
        torch.stack(
            category_losses
        ).mean()
        if category_losses
        else hidden.sum() * 0.0
    )

    entity_loss = (
        torch.stack(
            entity_losses
        ).mean()
        if entity_losses
        else hidden.sum() * 0.0
    )

    attribute_loss = (
        torch.stack(
            attribute_losses
        ).mean()
        if attribute_losses
        else hidden.sum() * 0.0
    )

    if predicted_va:

        predicted_va_tensor = (
            torch.stack(
                predicted_va
            )
        )

        gold_va_tensor = (
            torch.tensor(
                gold_va,
                dtype=torch.float32,
                device=hidden.device,
            )
        )

        va_loss = F.mse_loss(
            predicted_va_tensor,
            gold_va_tensor,
        )

    else:

        va_loss = (
            hidden.sum()
            * 0.0
        )

    return (
        relation_loss,
        category_loss,
        entity_loss,
        attribute_loss,
        va_loss,
    )


def task3_loss(
    *,
    model,
    batch,
    aspect_weights: torch.Tensor,
    opinion_weights: torch.Tensor,
    ignore_index: int,
    allowed_categories_by_domain,
    allowed_entities_by_domain,
    allowed_attributes_by_domain,
    category_to_entity_index,
    category_to_attribute_index,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:

    input_ids = batch[
        "input_ids"
    ]

    attention_mask = batch[
        "attention_mask"
    ]

    aspect_labels = batch[
        "aspect_labels"
    ]

    opinion_labels = batch[
        "opinion_labels"
    ]

    hidden = model.encode(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task="t3",
    )

    (
        aspect_logits,
        opinion_logits,
    ) = model.t3_token_logits(
        hidden
    )

    aspect_loss = (
        F.cross_entropy(
            aspect_logits.view(
                -1,
                3,
            ),
            aspect_labels.view(
                -1
            ),
            weight=aspect_weights,
            ignore_index=ignore_index,
        )
    )

    opinion_loss = (
        F.cross_entropy(
            opinion_logits.view(
                -1,
                3,
            ),
            opinion_labels.view(
                -1
            ),
            weight=opinion_weights,
            ignore_index=ignore_index,
        )
    )

    (
        relation_loss,
        category_loss,
        entity_loss,
        attribute_loss,
        va_loss,
    ) = task3_relation_category_va_loss(
        model=model,
        hidden=hidden,
        examples=batch[
            "examples"
        ],
        allowed_categories_by_domain=(
            allowed_categories_by_domain
        ),
        allowed_entities_by_domain=(
            allowed_entities_by_domain
        ),
        allowed_attributes_by_domain=(
            allowed_attributes_by_domain
        ),
        category_to_entity_index=(
            category_to_entity_index
        ),
        category_to_attribute_index=(
            category_to_attribute_index
        ),
    )

    loss = (
        aspect_loss
        + opinion_loss
        + relation_loss
        + category_loss
        + 0.5 * entity_loss
        + 0.5 * attribute_loss
        + va_loss
    )

    return (
        loss,
        {
            "t3_aspect":
                aspect_loss.detach(),
            "t3_opinion":
                opinion_loss.detach(),
            "t3_relation":
                relation_loss.detach(),
            "t3_category":
                category_loss.detach(),
            "t3_entity":
                entity_loss.detach(),
            "t3_attribute":
                attribute_loss.detach(),
            "t3_va":
                va_loss.detach(),
        },
    )
