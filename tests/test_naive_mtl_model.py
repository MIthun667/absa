from dataclasses import dataclass

import torch

from dimabsa.naive_mtl_model import (
    NaiveMTLModel,
)


@dataclass
class DummyNode:
    token_indices: tuple[int, ...]
    is_null: bool = False


def build_model():

    return NaiveMTLModel(
        "FacebookAI/xlm-roberta-base",
        num_t3_categories=135,
        num_t3_entities=29,
        num_t3_attributes=11,
        category_embedding_dim=32,
    )


def test_t1_va_bounds():

    model = build_model()

    representation = torch.randn(
        model.hidden_size
    )

    prediction = model.t1_predict_va(
        representation
    )

    assert prediction.shape == (2,)

    assert torch.all(
        prediction >= 1.0
    )

    assert torch.all(
        prediction <= 9.0
    )


def test_t2_token_heads():

    model = build_model()

    hidden = torch.randn(
        2,
        7,
        model.hidden_size,
    )

    aspect, opinion = (
        model.t2_token_logits(
            hidden
        )
    )

    assert aspect.shape == (
        2,
        7,
        3,
    )

    assert opinion.shape == (
        2,
        7,
        3,
    )


def test_t3_token_heads():

    model = build_model()

    hidden = torch.randn(
        2,
        7,
        model.hidden_size,
    )

    aspect, opinion = (
        model.t3_token_logits(
            hidden
        )
    )

    assert aspect.shape == (
        2,
        7,
        3,
    )

    assert opinion.shape == (
        2,
        7,
        3,
    )


def test_explicit_node_pooling():

    model = build_model()

    hidden = torch.randn(
        6,
        model.hidden_size,
    )

    node = DummyNode(
        token_indices=(
            1,
            2,
        ),
    )

    pooled = model.pool_node(
        hidden,
        hidden[0],
        node,
        task="t2",
        node_type="aspect",
    )

    expected = hidden[
        [1, 2]
    ].mean(
        dim=0
    )

    assert torch.allclose(
        pooled,
        expected,
    )


def test_t2_score_pair():

    model = build_model()

    aspect = torch.randn(
        model.hidden_size
    )

    opinion = torch.randn(
        model.hidden_size
    )

    relation_logit, va = (
        model.t2_score_pair(
            aspect,
            opinion,
        )
    )

    assert relation_logit.ndim == 0

    assert va.shape == (2,)

    assert torch.all(
        va >= 1.0
    )

    assert torch.all(
        va <= 9.0
    )


def test_t3_score_pair():

    model = build_model()

    aspect = torch.randn(
        model.hidden_size
    )

    opinion = torch.randn(
        model.hidden_size
    )

    (
        relation_logit,
        category_logits,
        entity_logits,
        attribute_logits,
        relation_repr,
    ) = model.t3_score_pair(
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
        model.hidden_size,
    )


def test_t3_category_conditioned_va():

    model = build_model()

    aspect = torch.randn(
        model.hidden_size
    )

    opinion = torch.randn(
        model.hidden_size
    )

    (
        _,
        _,
        _,
        _,
        relation_repr,
    ) = model.t3_score_pair(
        aspect,
        opinion,
    )

    prediction = model.t3_predict_va(
        relation_repr,
        torch.tensor(
            3,
            dtype=torch.long,
        ),
    )

    assert prediction.shape == (2,)

    assert torch.all(
        prediction >= 1.0
    )

    assert torch.all(
        prediction <= 9.0
    )
