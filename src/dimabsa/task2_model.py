from __future__ import annotations

from collections import defaultdict
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


class Task2Model(nn.Module):
    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.aspect_classifier = nn.Linear(
            hidden,
            3,
        )

        self.opinion_classifier = nn.Linear(
            hidden,
            3,
        )

        self.null_aspect = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.null_opinion = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        pair_size = hidden * 4

        self.relation_encoder = nn.Sequential(
            nn.Linear(pair_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.relation_classifier = nn.Linear(
            hidden,
            1,
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

        dropped = self.dropout(hidden)

        aspect_logits = self.aspect_classifier(
            dropped
        )

        opinion_logits = self.opinion_classifier(
            dropped
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
                f"Explicit node has no tokens: {node}"
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

    def score_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        pair = torch.cat(
            [
                aspect_repr,
                opinion_repr,
                aspect_repr * opinion_repr,
                torch.abs(
                    aspect_repr
                    - opinion_repr
                ),
            ],
            dim=-1,
        )

        relation_repr = self.relation_encoder(
            pair
        )

        relation_logit = (
            self.relation_classifier(
                relation_repr
            )
            .squeeze(-1)
        )

        raw_va = self.va_regressor(
            relation_repr
        )

        va = 1.0 + 8.0 * torch.sigmoid(
            raw_va
        )

        return relation_logit, va
