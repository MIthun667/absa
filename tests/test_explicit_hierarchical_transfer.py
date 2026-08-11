import torch

from dimabsa.explicit_hierarchical_transfer_mtl_model import (
    ExplicitHierarchicalTransferMTLModel,
)


def build_model():

    torch.manual_seed(
        123
    )

    return (
        ExplicitHierarchicalTransferMTLModel(
            "FacebookAI/xlm-roberta-base",
            shared_layers=8,
            structured_shared_layers=2,
            active_tasks=(
                "t1",
                "t2",
                "t3",
            ),
            num_t3_categories=7,
            num_t3_entities=3,
            num_t3_attributes=4,
            dropout=0.0,
        )
    )


def test_transfer_is_identity_at_initialization():

    model = build_model()

    model.eval()

    hidden = (
        model.hidden_size
    )

    t3_repr = torch.randn(
        hidden
    )

    t2_repr = torch.randn(
        hidden
    )

    fused = (
        model.fuse_t2_into_t3_relation(
            t3_relation_repr=(
                t3_repr
            ),
            t2_relation_repr=(
                t2_repr
            ),
        )
    )

    assert torch.equal(
        fused,
        t3_repr,
    )


def test_t3_transfer_does_not_backprop_into_t2_relation_encoder():

    model = build_model()

    model.train()

    hidden = (
        model.hidden_size
    )

    aspect = torch.randn(
        hidden,
        requires_grad=True,
    )

    opinion = torch.randn(
        hidden,
        requires_grad=True,
    )

    (
        _,
        category_logits,
        entity_logits,
        attribute_logits,
        fused_relation,
    ) = model.t3_score_pair(
        aspect,
        opinion,
    )

    loss = (
        category_logits.sum()
        + entity_logits.sum()
        + attribute_logits.sum()
        + fused_relation.sum()
    )

    loss.backward()

    # T2 relation encoder is a stop-gradient teacher
    # on a T3 update.
    for parameter in (
        model
        .t2_relation_encoder
        .parameters()
    ):

        assert (
            parameter.grad
            is None
        )

    # T3 relation encoder must still receive gradients.
    assert any(
        parameter.grad
        is not None
        for parameter
        in model
        .t3_relation_encoder
        .parameters()
    )

    # Transfer output layer must learn from T3.
    transfer_output = (
        model.t3_relation_transfer[
            -1
        ]
    )

    assert (
        transfer_output.weight.grad
        is not None
    )


def test_architecture_reports_directed_transfer():

    model = build_model()

    summary = (
        model.architecture_summary()
    )

    assert (
        summary[
            "transfer_direction"
        ]
        == "t2->t3"
    )

    assert (
        summary[
            "transfer_level"
        ]
        == "relation"
    )

    assert (
        summary[
            "transfer_stop_gradient"
        ]
        is True
    )
