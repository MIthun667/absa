from __future__ import annotations

import torch
import torch.nn as nn

from transformers import AutoModel

from .task2_supervision import EncodedNode


class PredictedNode:
    def __init__(
        self,
        text: str,
        token_indices: tuple[int, ...],
        is_null: bool = False,
    ):
        self.text = text
        self.token_indices = token_indices
        self.is_null = is_null


class Task3HybridModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        num_categories: int,
        num_entities: int,
        num_attributes: int,
        category_embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden = self.encoder.config.hidden_size

        self.hidden_size = hidden

        self.num_categories = num_categories
        self.num_entities = num_entities
        self.num_attributes = num_attributes

        self.dropout = nn.Dropout(
            dropout
        )

        # ---------------------------------------------------------
        # BIO extraction heads
        # ---------------------------------------------------------

        self.aspect_classifier = nn.Linear(
            hidden,
            3,
        )

        self.opinion_classifier = nn.Linear(
            hidden,
            3,
        )

        # ---------------------------------------------------------
        # NULL representations
        # ---------------------------------------------------------

        self.null_aspect = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.null_opinion = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # ---------------------------------------------------------
        # Shared AO representation
        # ---------------------------------------------------------

        pair_size = hidden * 4

        self.relation_encoder = nn.Sequential(
            nn.Linear(
                pair_size,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---------------------------------------------------------
        # Relation head
        # ---------------------------------------------------------

        self.relation_classifier = nn.Linear(
            hidden,
            1,
        )

        # ---------------------------------------------------------
        # Flat category head
        #
        # Preserve the proven baseline path.
        # ---------------------------------------------------------

        self.category_classifier = nn.Linear(
            hidden,
            num_categories,
        )

        # ---------------------------------------------------------
        # Ontology auxiliary heads
        # ---------------------------------------------------------

        self.entity_classifier = nn.Linear(
            hidden,
            num_entities,
        )

        self.attribute_classifier = nn.Linear(
            hidden,
            num_attributes,
        )

        # ---------------------------------------------------------
        # Category-conditioned VA
        # ---------------------------------------------------------

        self.category_embeddings = nn.Embedding(
            num_categories,
            category_embedding_dim,
        )

        self.va_encoder = nn.Sequential(
            nn.Linear(
                hidden
                + category_embedding_dim,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.va_regressor = nn.Linear(
            hidden,
            2,
        )

    def encode(
        self,
        *,
        input_ids,
        attention_mask,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state

        dropped = self.dropout(
            hidden
        )

        aspect_logits = (
            self.aspect_classifier(
                dropped
            )
        )

        opinion_logits = (
            self.opinion_classifier(
                dropped
            )
        )

        return (
            hidden,
            aspect_logits,
            opinion_logits,
        )

    def pool_node(
        self,
        hidden: torch.Tensor,
        sentence_repr: torch.Tensor,
        node: EncodedNode | PredictedNode,
        *,
        node_type: str,
    ) -> torch.Tensor:

        if node.is_null:

            if node_type == "aspect":
                return self.null_aspect(
                    sentence_repr
                )

            if node_type == "opinion":
                return self.null_opinion(
                    sentence_repr
                )

            raise ValueError(node_type)

        if not node.token_indices:
            raise ValueError(
                "Explicit node has no token indices."
            )

        indices = torch.tensor(
            node.token_indices,
            dtype=torch.long,
            device=hidden.device,
        )

        return hidden.index_select(
            0,
            indices,
        ).mean(dim=0)

    def encode_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> torch.Tensor:

        pair = torch.cat(
            [
                aspect_repr,
                opinion_repr,
                aspect_repr * opinion_repr,
                torch.abs(
                    aspect_repr - opinion_repr
                ),
            ],
            dim=-1,
        )

        return self.relation_encoder(
            pair
        )

    def score_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ):
        relation_repr = self.encode_pair(
            aspect_repr,
            opinion_repr,
        )

        relation_logit = (
            self.relation_classifier(
                relation_repr
            ).squeeze(-1)
        )

        category_logits = (
            self.category_classifier(
                relation_repr
            )
        )

        entity_logits = (
            self.entity_classifier(
                relation_repr
            )
        )

        attribute_logits = (
            self.attribute_classifier(
                relation_repr
            )
        )

        return (
            relation_logit,
            category_logits,
            entity_logits,
            attribute_logits,
            relation_repr,
        )

    def predict_va(
        self,
        relation_repr: torch.Tensor,
        category_index: int | torch.Tensor,
    ) -> torch.Tensor:

        if not torch.is_tensor(
            category_index
        ):
            category_index = torch.tensor(
                category_index,
                dtype=torch.long,
                device=relation_repr.device,
            )

        category_repr = (
            self.category_embeddings(
                category_index
            )
        )

        va_input = torch.cat(
            [
                relation_repr,
                category_repr,
            ],
            dim=-1,
        )

        va_hidden = self.va_encoder(
            va_input
        )

        raw_va = self.va_regressor(
            va_hidden
        )

        return (
            1.0
            + 8.0
            * torch.sigmoid(
                raw_va
            )
        )
