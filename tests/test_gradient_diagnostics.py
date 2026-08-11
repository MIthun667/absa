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


def test_layerwise_gradient_groups():

    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(
                [
                    nn.Linear(4, 4)
                    for _ in range(4)
                ]
            )


    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = DummyEncoder()


    from dimabsa.gradient_diagnostics import (
        get_layer_parameter_groups,
    )

    model = DummyModel()

    groups = (
        get_layer_parameter_groups(
            model,
            layers=(1, 4),
        )
    )

    assert tuple(
        groups
    ) == (
        "L1",
        "L4",
    )

    assert len(
        groups["L1"]
    ) > 0

    assert len(
        groups["L4"]
    ) > 0


def test_gradient_vectors_by_group():

    from dimabsa.gradient_diagnostics import (
        gradient_vectors_by_group,
    )

    first = nn.Linear(
        3,
        3,
    )

    second = nn.Linear(
        3,
        1,
    )

    x = torch.randn(
        5,
        3,
    )

    hidden = first(x)

    loss = (
        second(hidden)
        .pow(2)
        .mean()
    )

    vectors = (
        gradient_vectors_by_group(
            loss,
            {
                "L1":
                    tuple(
                        first.parameters()
                    ),
                "L2":
                    tuple(
                        second.parameters()
                    ),
            },
        )
    )

    assert set(
        vectors
    ) == {
        "L1",
        "L2",
    }

    assert (
        vectors["L1"]
        .numel()
        > 0
    )

    assert (
        vectors["L2"]
        .numel()
        > 0
    )

    for parameter in list(
        first.parameters()
    ) + list(
        second.parameters()
    ):

        assert (
            parameter.grad
            is None
        )
