from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .naive_mtl_model import NaiveMTLModel


class HierarchicalSharingMTLModel(
    NaiveMTLModel
):
    """
    Hierarchy-aware selective-sharing MTL baseline.

    XLM-R-base layout:

        embeddings
            |
        L1 ... L8
        shared by T1/T2/T3
            |
        +---+----------------+
        |                    |
        T1                 structured
        L9-L12             L9-L10
        private            shared T2/T3
                              |
                         +----+----+
                         |         |
                       T2         T3
                    L11-L12    L11-L12
                    private    private

    No cross-task features, adapters, task weighting,
    gradient surgery, or auxiliary objectives are added.

    The manipulated variable is the topology of parameter
    sharing itself.
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
        shared_layers: int = 8,
        structured_shared_layers: int = 2,
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
            1 <= shared_layers < total_layers
        ):
            raise ValueError(
                "shared_layers must satisfy "
                f"1 <= S < {total_layers}; "
                f"received {shared_layers}."
            )

        if structured_shared_layers < 0:
            raise ValueError(
                "structured_shared_layers "
                "must be non-negative."
            )

        structured_end = (
            shared_layers
            + structured_shared_layers
        )

        if structured_end >= total_layers:
            raise ValueError(
                "At least one terminal private layer "
                "must remain after the structured "
                "shared block."
            )

        self.total_layers = (
            total_layers
        )

        self.shared_layers = (
            int(shared_layers)
        )

        self.structured_shared_layers = (
            int(
                structured_shared_layers
            )
        )

        self.structured_end = (
            structured_end
        )

        self.active_tasks = (
            active_tasks
        )

        self.task_encoders = (
            nn.ModuleDict()
        )

        # --------------------------------------------------------
        # Start each active task from an identical pretrained
        # 12-layer XLM-R model.
        # --------------------------------------------------------

        for task in active_tasks:

            self.task_encoders[
                task
            ] = copy.deepcopy(
                self.encoder
            )

        # --------------------------------------------------------
        # All-task shared foundation:
        # embeddings + L1 ... L(shared_layers)
        # --------------------------------------------------------

        for task in active_tasks:

            task_encoder = (
                self.task_encoders[
                    task
                ]
            )

            task_encoder.embeddings = (
                self.encoder.embeddings
            )

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

        # --------------------------------------------------------
        # Structured T2/T3 sharing:
        #
        # Use T2's pretrained copies as the canonical shared
        # structured modules and point T3 at the SAME objects.
        #
        # If only one of T2/T3 is active, no extra cross-task
        # sharing is needed.
        # --------------------------------------------------------

        if (
            "t2" in active_tasks
            and "t3" in active_tasks
            and self.structured_shared_layers > 0
        ):

            for layer_index in range(
                self.shared_layers,
                self.structured_end,
            ):

                structured_module = (
                    self.task_encoders[
                        "t2"
                    ].encoder.layer[
                        layer_index
                    ]
                )

                self.task_encoders[
                    "t3"
                ].encoder.layer[
                    layer_index
                ] = structured_module

        # --------------------------------------------------------
        # Remove unused original upper blocks from root encoder.
        #
        # self.encoder remains the canonical registration point
        # for the all-task shared foundation only.
        # --------------------------------------------------------

        root_shared_modules = [
            self.encoder
            .encoder
            .layer[index]
            for index in range(
                self.shared_layers
            )
        ]

        self.encoder.encoder.layer = (
            nn.ModuleList(
                root_shared_modules
            )
        )

        self.encoder.pooler = None

        for task_encoder in (
            self.task_encoders.values()
        ):
            task_encoder.pooler = None

    def architecture_summary(
        self,
    ) -> dict[str, object]:

        terminal_private = (
            self.total_layers
            - self.structured_end
        )

        return {
            "architecture":
                "hierarchical",
            "total_layers":
                self.total_layers,
            "all_task_shared_layers":
                self.shared_layers,
            "structured_shared_layers":
                self.structured_shared_layers,
            "structured_start_layer":
                self.shared_layers + 1,
            "structured_end_layer":
                self.structured_end,
            "t1_private_layers":
                self.total_layers
                - self.shared_layers,
            "t2_terminal_private_layers":
                terminal_private,
            "t3_terminal_private_layers":
                terminal_private,
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

        if task is None:
            raise ValueError(
                "Hierarchical sharing requires "
                "an explicit task."
            )

        if task not in self.task_encoders:
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
