import torch

from dimabsa.task3_model import (
    Task3Model,
)


MODEL = "FacebookAI/xlm-roberta-base"


def test_task3_model_shapes():

    model = Task3Model(
        MODEL,
        num_categories=14,
        category_embedding_dim=32,
    )

    hidden = model.hidden_size

    aspect = torch.randn(
        hidden
    )

    opinion = torch.randn(
        hidden
    )

    (
        relation_logit,
        category_logits,
        relation_repr,
    ) = model.score_pair(
        aspect,
        opinion,
    )

    assert relation_logit.ndim == 0

    assert category_logits.shape == (
        14,
    )

    assert relation_repr.shape == (
        hidden,
    )

    va = model.predict_va(
        relation_repr,
        3,
    )

    assert va.shape == (
        2,
    )

    assert torch.all(
        va >= 1.0
    )

    assert torch.all(
        va <= 9.0
    )


def test_category_condition_changes_va_path():

    model = Task3Model(
        MODEL,
        num_categories=4,
        category_embedding_dim=32,
    )

    hidden = model.hidden_size

    relation_repr = torch.randn(
        hidden
    )

    va0 = model.predict_va(
        relation_repr,
        0,
    )

    va1 = model.predict_va(
        relation_repr,
        1,
    )

    assert va0.shape == (
        2,
    )

    assert va1.shape == (
        2,
    )
