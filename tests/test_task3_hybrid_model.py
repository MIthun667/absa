import torch

from dimabsa.task3_hybrid_model import (
    Task3HybridModel,
)


def test_hybrid_output_shapes():

    model = Task3HybridModel(
        "FacebookAI/xlm-roberta-base",
        num_categories=135,
        num_entities=29,
        num_attributes=11,
        category_embedding_dim=32,
    )

    hidden = model.hidden_size

    aspect = torch.randn(hidden)
    opinion = torch.randn(hidden)

    (
        relation_logit,
        category_logits,
        entity_logits,
        attribute_logits,
        relation_repr,
    ) = model.score_pair(
        aspect,
        opinion,
    )

    assert relation_logit.ndim == 0

    assert category_logits.shape == (
        135,
    )

    assert entity_logits.shape == (
        29,
    )

    assert attribute_logits.shape == (
        11,
    )

    assert relation_repr.shape == (
        hidden,
    )

    va = model.predict_va(
        relation_repr,
        3,
    )

    assert va.shape == (2,)

    assert torch.all(
        va >= 1.0
    )

    assert torch.all(
        va <= 9.0
    )
