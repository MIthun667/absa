from unittest.mock import patch

import torch.nn as nn

from dimabsa.naive_mtl_model import (
    NaiveMTLModel,
)
from dimabsa.partial_sharing_mtl_model import (
    PartialSharingMTLModel,
)


class FakeEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embeddings = nn.Linear(
            4,
            4,
        )

        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [
                nn.Linear(
                    4,
                    4,
                )
                for _ in range(12)
            ]
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


def test_partial_sharing_object_identity():

    with patch.object(
        NaiveMTLModel,
        "__init__",
        fake_naive_init,
    ):

        model = (
            PartialSharingMTLModel(
                "fake",
                shared_layers=8,
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

    assert (
        model.task_encoders[
            "t1"
        ].embeddings
        is
        model.task_encoders[
            "t2"
        ].embeddings
    )

    for index in range(8):

        assert (
            model.task_encoders[
                "t1"
            ].encoder.layer[
                index
            ]
            is
            model.task_encoders[
                "t2"
            ].encoder.layer[
                index
            ]
        )

        assert (
            model.task_encoders[
                "t2"
            ].encoder.layer[
                index
            ]
            is
            model.task_encoders[
                "t3"
            ].encoder.layer[
                index
            ]
        )

    # First private layer must NOT be shared.
    assert (
        model.task_encoders[
            "t1"
        ].encoder.layer[8]
        is not
        model.task_encoders[
            "t2"
        ].encoder.layer[8]
    )

    assert (
        model.task_encoders[
            "t2"
        ].encoder.layer[8]
        is not
        model.task_encoders[
            "t3"
        ].encoder.layer[8]
    )
