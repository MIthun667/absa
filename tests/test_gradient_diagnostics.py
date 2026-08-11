import torch
import torch.nn as nn

from dimabsa.gradient_diagnostics import (
    cosine_similarity,
    gradient_vector,
    summarize_values,
)


def test_cosine_aligned():

    a = torch.tensor(
        [1.0, 2.0]
    )

    b = torch.tensor(
        [2.0, 4.0]
    )

    value = cosine_similarity(
        a,
        b,
    )

    assert abs(
        value - 1.0
    ) < 1e-6


def test_cosine_conflicting():

    a = torch.tensor(
        [1.0, 0.0]
    )

    b = torch.tensor(
        [-1.0, 0.0]
    )

    value = cosine_similarity(
        a,
        b,
    )

    assert abs(
        value + 1.0
    ) < 1e-6


def test_gradient_vector_does_not_fill_grad():

    layer = nn.Linear(
        3,
        1,
    )

    x = torch.randn(
        4,
        3,
    )

    loss = (
        layer(x)
        .pow(2)
        .mean()
    )

    vector = gradient_vector(
        loss,
        tuple(
            layer.parameters()
        ),
    )

    assert vector.numel() > 0

    for parameter in (
        layer.parameters()
    ):
        assert (
            parameter.grad
            is None
        )


def test_summary():

    result = summarize_values(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    assert result[
        "mean"
    ] == 2.0

    assert result[
        "median"
    ] == 2.0

    assert result[
        "count"
    ] == 3
