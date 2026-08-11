from __future__ import annotations

import torch
import torch.nn as nn

from transformers import AutoModel


class NaiveMTLModel(nn.Module):
    """
    Shared XLM-R encoder with independent task-specific heads.

    This module deliberately contains no adaptive task weighting,
    gradient surgery, adapters, or cross-task interaction.
    """

    def __init__(
        self,
        model_name: str,
        *,
        num_t3_categories: int,
        num_t3_entities: int,
        num_t3_attributes: int,
        category_embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden = (
            self.encoder.config.hidden_size
        )

        self.hidden_size = hidden

        self.dropout = nn.Dropout(
            dropout
        )

        # ========================================================
        # T1 — DimASR
        # ========================================================

        self.t1_va_regressor = nn.Linear(
            hidden,
            2,
        )

        # ========================================================
        # T2 — DimASTE
        # ========================================================

        self.t2_aspect_classifier = nn.Linear(
            hidden,
            3,
        )

        self.t2_opinion_classifier = nn.Linear(
            hidden,
            3,
        )

        self.t2_null_aspect = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.t2_null_opinion = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.t2_relation_encoder = nn.Sequential(
            nn.Linear(
                hidden * 4,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.t2_relation_classifier = nn.Linear(
            hidden,
            1,
        )

        self.t2_va_regressor = nn.Linear(
            hidden,
            2,
        )

        # ========================================================
        # T3 — DimASQP
        # ========================================================

        self.num_t3_categories = (
            num_t3_categories
        )

        self.num_t3_entities = (
            num_t3_entities
        )

        self.num_t3_attributes = (
            num_t3_attributes
        )

        self.t3_aspect_classifier = nn.Linear(
            hidden,
            3,
        )

        self.t3_opinion_classifier = nn.Linear(
            hidden,
            3,
        )

        self.t3_null_aspect = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.t3_null_opinion = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.t3_relation_encoder = nn.Sequential(
            nn.Linear(
                hidden * 4,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.t3_relation_classifier = nn.Linear(
            hidden,
            1,
        )

        self.t3_category_classifier = nn.Linear(
            hidden,
            num_t3_categories,
        )

        self.t3_entity_classifier = nn.Linear(
            hidden,
            num_t3_entities,
        )

        self.t3_attribute_classifier = nn.Linear(
            hidden,
            num_t3_attributes,
        )

        self.t3_category_embeddings = (
            nn.Embedding(
                num_t3_categories,
                category_embedding_dim,
            )
        )

        self.t3_va_encoder = nn.Sequential(
            nn.Linear(
                hidden
                + category_embedding_dim,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.t3_va_regressor = nn.Linear(
            hidden,
            2,
        )

    # ============================================================
    # Shared encoder
    # ============================================================

    def encode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str | None = None,
    ) -> torch.Tensor:
        """
        Encode with the fully shared backbone.

        ``task`` is intentionally ignored here.  It exists so the
        loss/evaluation code can use one task-aware interface for both
        naive full sharing and partial sharing.
        """

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs.last_hidden_state

    # ============================================================
    # Task-specific token heads
    # ============================================================

    def t2_token_logits(
        self,
        hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        dropped = self.dropout(
            hidden
        )

        return (
            self.t2_aspect_classifier(
                dropped
            ),
            self.t2_opinion_classifier(
                dropped
            ),
        )

    def t3_token_logits(
        self,
        hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        dropped = self.dropout(
            hidden
        )

        return (
            self.t3_aspect_classifier(
                dropped
            ),
            self.t3_opinion_classifier(
                dropped
            ),
        )

    # ============================================================
    # T1
    # ============================================================

    def t1_predict_va(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor:

        raw = self.t1_va_regressor(
            self.dropout(
                representation
            )
        )

        return (
            1.0
            + 8.0
            * torch.sigmoid(raw)
        )

    # ============================================================
    # Node pooling
    # ============================================================

    def pool_node(
        self,
        hidden: torch.Tensor,
        sentence_repr: torch.Tensor,
        node,
        *,
        task: str,
        node_type: str,
    ) -> torch.Tensor:

        if task not in {
            "t2",
            "t3",
        }:
            raise ValueError(
                f"Unsupported task: {task}"
            )

        if node.is_null:

            if (
                task == "t2"
                and node_type == "aspect"
            ):
                return self.t2_null_aspect(
                    sentence_repr
                )

            if (
                task == "t2"
                and node_type == "opinion"
            ):
                return self.t2_null_opinion(
                    sentence_repr
                )

            if (
                task == "t3"
                and node_type == "aspect"
            ):
                return self.t3_null_aspect(
                    sentence_repr
                )

            if (
                task == "t3"
                and node_type == "opinion"
            ):
                return self.t3_null_opinion(
                    sentence_repr
                )

            raise ValueError(
                (
                    task,
                    node_type,
                )
            )

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
        ).mean(
            dim=0
        )

    # ============================================================
    # Shared pair construction utility
    # ============================================================

    @staticmethod
    def pair_features(
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> torch.Tensor:

        return torch.cat(
            [
                aspect_repr,
                opinion_repr,
                aspect_repr
                * opinion_repr,
                torch.abs(
                    aspect_repr
                    - opinion_repr
                ),
            ],
            dim=-1,
        )

    # ============================================================
    # T2 pair representation
    # ============================================================

    def t2_encode_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> torch.Tensor:

        return self.t2_relation_encoder(
            self.pair_features(
                aspect_repr,
                opinion_repr,
            )
        )

    def t2_score_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        relation_repr = (
            self.t2_encode_pair(
                aspect_repr,
                opinion_repr,
            )
        )

        relation_logit = (
            self.t2_relation_classifier(
                relation_repr
            ).squeeze(-1)
        )

        raw_va = (
            self.t2_va_regressor(
                relation_repr
            )
        )

        va = (
            1.0
            + 8.0
            * torch.sigmoid(
                raw_va
            )
        )

        return (
            relation_logit,
            va,
        )

    # ============================================================
    # T3 pair representation
    # ============================================================

    def t3_encode_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> torch.Tensor:

        return self.t3_relation_encoder(
            self.pair_features(
                aspect_repr,
                opinion_repr,
            )
        )

    def t3_score_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        relation_repr = (
            self.t3_encode_pair(
                aspect_repr,
                opinion_repr,
            )
        )

        relation_logit = (
            self.t3_relation_classifier(
                relation_repr
            ).squeeze(-1)
        )

        category_logits = (
            self.t3_category_classifier(
                relation_repr
            )
        )

        entity_logits = (
            self.t3_entity_classifier(
                relation_repr
            )
        )

        attribute_logits = (
            self.t3_attribute_classifier(
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

    def t3_predict_va(
        self,
        relation_repr: torch.Tensor,
        category_index: torch.Tensor,
    ) -> torch.Tensor:

        category_repr = (
            self.t3_category_embeddings(
                category_index
            )
        )

        combined = torch.cat(
            [
                relation_repr,
                category_repr,
            ],
            dim=-1,
        )

        hidden = self.t3_va_encoder(
            combined
        )

        raw = self.t3_va_regressor(
            hidden
        )

        return (
            1.0
            + 8.0
            * torch.sigmoid(raw)
        )
