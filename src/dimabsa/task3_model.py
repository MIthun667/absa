from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoModel

from .task2_supervision import EncodedNode


@dataclass
class PredictedNode:
    text: str
    token_indices: tuple[int, ...]
    is_null: bool = False


class Task3Model(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_categories: int,
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
        self.category_embedding_dim = (
            category_embedding_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # ---------------------------------------------------------
        # BIO heads
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
        # Learned NULL representations
        # ---------------------------------------------------------

        self.null_aspect = nn.Sequential(
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.Tanh(),
        )

        self.null_opinion = nn.Sequential(
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.Tanh(),
        )

        # ---------------------------------------------------------
        # Shared A/O relation representation
        #
        # Exactly the same structural representation as T2:
        #
        # [hA ; hO ; hA*hO ; |hA-hO|]
        # ---------------------------------------------------------

        pair_size = hidden * 4

        self.relation_encoder = nn.Sequential(
            nn.Linear(
                pair_size,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        # ---------------------------------------------------------
        # AO relation existence
        # ---------------------------------------------------------

        self.relation_classifier = nn.Linear(
            hidden,
            1,
        )

        # ---------------------------------------------------------
        # Multi-label category prediction
        #
        # One sigmoid logit per flat category.
        # ---------------------------------------------------------

        self.category_classifier = nn.Linear(
            hidden,
            num_categories,
        )

        # ---------------------------------------------------------
        # Category embeddings
        # ---------------------------------------------------------

        self.category_embeddings = nn.Embedding(
            num_categories,
            category_embedding_dim,
        )

        # ---------------------------------------------------------
        # Category-conditioned VA head
        #
        # VA = f(z_AO, e_category)
        # ---------------------------------------------------------

        va_input_size = (
            hidden
            + category_embedding_dim
        )

        self.va_encoder = nn.Sequential(
            nn.Linear(
                va_input_size,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        self.va_regressor = nn.Linear(
            hidden,
            2,
        )

    # -------------------------------------------------------------
    # Shared XLM-R encoding
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Node pooling
    # -------------------------------------------------------------

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

            raise ValueError(
                node_type
            )

        if not node.token_indices:
            raise ValueError(
                "Explicit node has no tokens: "
                f"{node}"
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

    # -------------------------------------------------------------
    # Pair encoding
    # -------------------------------------------------------------

    def encode_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> torch.Tensor:

        pair = torch.cat(
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

        return self.relation_encoder(
            pair
        )

    # -------------------------------------------------------------
    # Relation + category scoring
    # -------------------------------------------------------------

    def score_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        relation_repr = (
            self.encode_pair(
                aspect_repr,
                opinion_repr,
            )
        )

        relation_logit = (
            self.relation_classifier(
                relation_repr
            )
            .squeeze(-1)
        )

        category_logits = (
            self.category_classifier(
                relation_repr
            )
        )

        return (
            relation_logit,
            category_logits,
            relation_repr,
        )

    # -------------------------------------------------------------
    # Category-conditioned VA
    # -------------------------------------------------------------

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

        # Official DimABSA range:
        #
        # V,A ∈ [1,9]
        return (
            1.0
            + 8.0
            * torch.sigmoid(
                raw_va
            )
        )
