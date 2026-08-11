import torch

from dimabsa.task3_pair_model import (
    Task3PairModel,
)


def test_pair_category_logits():

    model = Task3PairModel(
        "FacebookAI/xlm-roberta-base",
        num_categories=135,
        num_entities=29,
        num_attributes=11,
        category_embedding_dim=32,
    )

    relation_repr = torch.randn(
        model.hidden_size
    )

    entity_indices = torch.tensor(
        [0, 1, 2, 3]
    )

    attribute_indices = torch.tensor(
        [0, 2, 1, 3]
    )

    logits = model.pair_category_logits(
        relation_repr,
        entity_indices,
        attribute_indices,
    )

    assert logits.shape == (4,)

    assert torch.isfinite(
        logits
    ).all()
