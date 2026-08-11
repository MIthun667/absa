from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .naive_mtl_model import NaiveMTLModel


class PartialSharingMTLModel(NaiveMTLModel):
    """
    Partial encoder-sharing MTL baseline.

    For shared_layers=S:

        shared embeddings
              |
        shared L1 ... LS
              |
       +------+------+
       |      |      |
      T1     T2     T3
       |      |      |
      private pretrained copies of L(S+1) ... L12

    Each task branch remains a normal Hugging Face XLM-R model.
    Therefore Hugging Face itself handles attention-mask creation,
    SDPA/eager dispatch, positions, and all encoder internals.

    No adapters, task weighting, gradient surgery, or cross-task
    communication are introduced.
    """

    VALID_TASKS = (
        "t1",
        "t2",
        "t3",
    )

    def __init__(
        self,
        model_name: str,
        *,
        shared_layers: int,
        active_tasks: tuple[str, ...],
        num_t3_categories: int,
        num_t3_entities: int,
        num_t3_attributes: int,
        category_embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:

        super().__init__(
            model_name,
            num_t3_categories=num_t3_categories,
            num_t3_entities=num_t3_entities,
            num_t3_attributes=num_t3_attributes,
            category_embedding_dim=category_embedding_dim,
            dropout=dropout,
        )

        active_tasks = tuple(
            dict.fromkeys(
                active_tasks
            )
        )

        if not active_tasks:
            raise ValueError(
                "At least one active task is required."
            )

        invalid = (
            set(active_tasks)
            - set(self.VALID_TASKS)
        )

        if invalid:
            raise ValueError(
                f"Unsupported tasks: "
                f"{sorted(invalid)}"
            )

        total_layers = len(
            self.encoder.encoder.layer
        )

        if not (
            1
            <= shared_layers
            <= total_layers
        ):
            raise ValueError(
                "--shared-layers must satisfy "
                f"1 <= S <= {total_layers}; "
                f"received {shared_layers}."
            )

        self.shared_layers = int(
            shared_layers
        )

        self.total_layers = int(
            total_layers
        )

        self.num_private_layers = (
            self.total_layers
            - self.shared_layers
        )

        self.active_tasks = (
            active_tasks
        )

        # --------------------------------------------------------
        # S=12 is deliberately the exact naive baseline.
        # --------------------------------------------------------

        self.task_encoders = (
            nn.ModuleDict()
        )

        if (
            self.shared_layers
            == self.total_layers
        ):
            return

        # --------------------------------------------------------
        # Build complete per-task HF encoders first.
        #
        # Each copy initially contains all pretrained layers.
        # We then replace embeddings + lower S layers with
        # references to the SAME modules from self.encoder.
        #
        # Therefore:
        #   embeddings = shared object
        #   L1..LS     = shared objects
        #   L(S+1)..12 = independent copies
        # --------------------------------------------------------

        for task in (
            self.active_tasks
        ):

            task_encoder = (
                copy.deepcopy(
                    self.encoder
                )
            )

            # Shared embeddings.
            task_encoder.embeddings = (
                self.encoder.embeddings
            )

            # Shared lower transformer blocks.
            for layer_index in range(
                self.shared_layers
            ):

                task_encoder.encoder.layer[
                    layer_index
                ] = (
                    self.encoder
                    .encoder
                    .layer[
                        layer_index
                    ]
                )

            self.task_encoders[
                task
            ] = task_encoder

        # --------------------------------------------------------
        # self.encoder is now only the registered shared trunk.
        #
        # The task encoders already hold their own copies of the
        # original upper layers, so remove the unused original
        # upper layers from the root encoder. This avoids training
        # a fourth unused copy of the private layers.
        # --------------------------------------------------------

        shared_modules = [
            self.encoder
            .encoder
            .layer[index]
            for index in range(
                self.shared_layers
            )
        ]

        self.encoder.encoder.layer = (
            nn.ModuleList(
                shared_modules
            )
        )

        # Pooler is unused throughout this project.
        self.encoder.pooler = None

    def architecture_summary(
        self,
    ) -> dict[str, object]:

        return {
            "total_layers":
                self.total_layers,
            "shared_layers":
                self.shared_layers,
            "private_layers_per_task":
                self.num_private_layers,
            "active_tasks":
                self.active_tasks,
        }

    def encode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str | None = None,
    ) -> torch.Tensor:
        """
        Task-aware encoder forward.

        S=12:
            exact NaiveMTLModel path.

        S<12:
            full Hugging Face forward through the requested task
            encoder. Shared modules are literally shared Parameter
            objects, while upper layers are task-private.
        """

        if (
            self.shared_layers
            == self.total_layers
        ):

            return super().encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
            )

        if task is None:
            raise ValueError(
                "Partial sharing requires "
                "an explicit task."
            )

        if task not in (
            self.task_encoders
        ):
            raise ValueError(
                f"No encoder branch for "
                f"task={task!r}; "
                f"active_tasks="
                f"{self.active_tasks}."
            )

        outputs = (
            self.task_encoders[
                task
            ](
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )

        return (
            outputs.last_hidden_state
        )
