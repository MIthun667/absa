import torch

from dimabsa.mtl_eval_adapters import (
    Task1MTLAdapter,
    Task2MTLAdapter,
    Task3MTLAdapter,
)
from dimabsa.naive_mtl_model import (
    NaiveMTLModel,
)


def build_model():

    return NaiveMTLModel(
        "FacebookAI/xlm-roberta-base",
        num_t3_categories=135,
        num_t3_entities=29,
        num_t3_attributes=11,
        category_embedding_dim=32,
    )


def test_task1_adapter():

    model = build_model()

    adapter = Task1MTLAdapter(
        model
    )

    input_ids = torch.ones(
        2,
        8,
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    output = adapter(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert output.shape == (
        2,
        2,
    )


def test_task2_adapter_encode():

    model = build_model()

    adapter = Task2MTLAdapter(
        model
    )

    input_ids = torch.ones(
        2,
        8,
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    hidden, aspect, opinion = (
        adapter.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    )

    assert hidden.shape[:2] == (
        2,
        8,
    )

    assert aspect.shape == (
        2,
        8,
        3,
    )

    assert opinion.shape == (
        2,
        8,
        3,
    )


def test_task3_adapter_score_pair():

    model = build_model()

    adapter = Task3MTLAdapter(
        model
    )

    a = torch.randn(
        model.hidden_size
    )

    o = torch.randn(
        model.hidden_size
    )

    result = adapter.score_pair(
        a,
        o,
    )

    assert len(result) == 5

    assert result[1].shape == (
        135,
    )
