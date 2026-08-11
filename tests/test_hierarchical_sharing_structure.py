from unittest.mock import patch

import torch.nn as nn

from dimabsa.naive_mtl_model import (
    NaiveMTLModel,
)
from dimabsa.hierarchical_sharing_mtl_model import (
    HierarchicalSharingMTLModel,
)


class FakeEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embeddings = nn.Linear(
            4,
            4,
        )

        self.encoder = nn.Module()
        self.encoder.layer = (
            nn.ModuleList(
                [
                    nn.Linear(
                        4,
                        4,
                    )
                    for _ in range(12)
                ]
            )
        )

        self.pooler = None

        self.config = type(
            "Config",
            (),
            {
                "hidden_size": 4,
            },
        )()


def fake_naive_init(
    self,
    model_name,
    *,
    num_t3_categories,
    num_t3_entities,
    num_t3_attributes,
    category_embedding_dim=128,
    dropout=0.1,
):

    nn.Module.__init__(self)

    self.encoder = FakeEncoder()

    self.hidden_size = 4

    self.dropout = nn.Dropout(
        dropout
    )

    self.num_t3_categories = (
        num_t3_categories
    )

    self.num_t3_entities = (
        num_t3_entities
    )

    self.num_t3_attributes = (
        num_t3_attributes
    )


def make_model():

    with patch.object(
        NaiveMTLModel,
        "__init__",
        fake_naive_init,
    ):

        return (
            HierarchicalSharingMTLModel(
                "fake",
                shared_layers=8,
                structured_shared_layers=2,
                active_tasks=(
                    "t1",
                    "t2",
                    "t3",
                ),
                num_t3_categories=1,
                num_t3_entities=1,
                num_t3_attributes=1,
            )
        )


def test_all_task_foundation_is_shared():

    model = make_model()

    t1 = model.task_encoders[
        "t1"
    ]

    t2 = model.task_encoders[
        "t2"
    ]

    t3 = model.task_encoders[
        "t3"
    ]

    assert (
        t1.embeddings
        is t2.embeddings
        is t3.embeddings
    )

    for index in range(8):

        assert (
            t1.encoder.layer[index]
            is t2.encoder.layer[index]
            is t3.encoder.layer[index]
        )


def test_structured_layers_shared_only_t2_t3():

    model = make_model()

    t1 = model.task_encoders[
        "t1"
    ]

    t2 = model.task_encoders[
        "t2"
    ]

    t3 = model.task_encoders[
        "t3"
    ]

    for index in (8, 9):

        assert (
            t2.encoder.layer[index]
            is
            t3.encoder.layer[index]
        )

        assert (
            t1.encoder.layer[index]
            is not
            t2.encoder.layer[index]
        )


def test_terminal_layers_are_private():

    model = make_model()

    t1 = model.task_encoders[
        "t1"
    ]

    t2 = model.task_encoders[
        "t2"
    ]

    t3 = model.task_encoders[
        "t3"
    ]

    for index in (10, 11):

        assert (
            t1.encoder.layer[index]
            is not
            t2.encoder.layer[index]
        )

        assert (
            t1.encoder.layer[index]
            is not
            t3.encoder.layer[index]
        )

        assert (
            t2.encoder.layer[index]
            is not
            t3.encoder.layer[index]
        )


def test_parameter_iteration_is_deduplicated():

    model = make_model()

    parameters = list(
        model.parameters()
    )

    ids = [
        id(parameter)
        for parameter
        in parameters
    ]

    assert len(ids) == len(
        set(ids)
    )
