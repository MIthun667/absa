from __future__ import annotations

import torch
import torch.nn as nn


class Task1MTLAdapter(nn.Module):
    """
    Expose NaiveMTLModel through the interface expected by
    train_task1_baseline.evaluate().
    """

    def __init__(
        self,
        shared_model,
    ) -> None:

        super().__init__()

        self.shared_model = (
            shared_model
        )

    def forward(
        self,
        **inputs,
    ) -> torch.Tensor:

        outputs = (
            self.shared_model.encoder(
                **inputs
            )
        )

        cls = (
            outputs.last_hidden_state[
                :,
                0,
            ]
        )

        return (
            self.shared_model
            .t1_predict_va(
                cls
            )
        )


class Task2MTLAdapter(nn.Module):
    """
    Expose the shared model through the exact Task2Model
    inference contract.
    """

    def __init__(
        self,
        shared_model,
    ) -> None:

        super().__init__()

        self.shared_model = (
            shared_model
        )

    def encode(
        self,
        *,
        input_ids,
        attention_mask,
    ):

        hidden = (
            self.shared_model.encode(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
            )
        )

        (
            aspect_logits,
            opinion_logits,
        ) = (
            self.shared_model
            .t2_token_logits(
                hidden
            )
        )

        return (
            hidden,
            aspect_logits,
            opinion_logits,
        )

    def pool_node(
        self,
        hidden,
        sentence_repr,
        node,
        *,
        node_type,
    ):

        return (
            self.shared_model.pool_node(
                hidden,
                sentence_repr,
                node,
                task="t2",
                node_type=node_type,
            )
        )

    def score_pair(
        self,
        aspect_repr,
        opinion_repr,
    ):

        return (
            self.shared_model
            .t2_score_pair(
                aspect_repr,
                opinion_repr,
            )
        )


class Task3MTLAdapter(nn.Module):
    """
    Expose the shared model through the Task3HybridModel
    inference contract.
    """

    def __init__(
        self,
        shared_model,
    ) -> None:

        super().__init__()

        self.shared_model = (
            shared_model
        )

    @property
    def num_categories(
        self,
    ) -> int:

        return (
            self.shared_model
            .num_t3_categories
        )

    @property
    def num_entities(
        self,
    ) -> int:

        return (
            self.shared_model
            .num_t3_entities
        )

    @property
    def num_attributes(
        self,
    ) -> int:

        return (
            self.shared_model
            .num_t3_attributes
        )

    def encode(
        self,
        *,
        input_ids,
        attention_mask,
    ):

        hidden = (
            self.shared_model.encode(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
            )
        )

        (
            aspect_logits,
            opinion_logits,
        ) = (
            self.shared_model
            .t3_token_logits(
                hidden
            )
        )

        return (
            hidden,
            aspect_logits,
            opinion_logits,
        )

    def pool_node(
        self,
        hidden,
        sentence_repr,
        node,
        *,
        node_type,
    ):

        return (
            self.shared_model.pool_node(
                hidden,
                sentence_repr,
                node,
                task="t3",
                node_type=node_type,
            )
        )

    def score_pair(
        self,
        aspect_repr,
        opinion_repr,
    ):

        return (
            self.shared_model
            .t3_score_pair(
                aspect_repr,
                opinion_repr,
            )
        )

    def predict_va(
        self,
        relation_repr,
        category_index,
    ):

        if not torch.is_tensor(
            category_index
        ):

            category_index = (
                torch.tensor(
                    category_index,
                    dtype=torch.long,
                    device=(
                        relation_repr.device
                    ),
                )
            )

        return (
            self.shared_model
            .t3_predict_va(
                relation_repr,
                category_index,
            )
        )
