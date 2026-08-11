from __future__ import annotations

import torch
import torch.nn as nn

from .hierarchical_sharing_mtl_model import (
    HierarchicalSharingMTLModel,
)


class ExplicitHierarchicalTransferMTLModel(
    HierarchicalSharingMTLModel
):
    """
    Explicit directed T2 -> T3 relation transfer.

    Encoder topology remains identical to the hierarchical
    selective-sharing baseline:

        all-task shared L1-L8
                  |
            +-----+------+
            |            |
          T1         T2/T3 shared
        L9-L12         L9-L10
                         |
                    +----+----+
                    |         |
                  T2          T3
                L11-L12     L11-L12

    New intervention:

        r3 = T3 relation representation
        r2 = T2 relation representation

        fused_r3 =
            r3 + Transfer([r3 ; stopgrad(r2)])

    The T2 representation is treated as a directed teacher
    feature during T3 updates.

    Important:
    - T3 loss does NOT update the T2 relation encoder through
      this transfer path.
    - T3 relation classification remains based on native r3.
    - Only T3 higher-order predictions use fused_r3:
        category
        entity
        attribute
        category-conditioned VA
    - transfer residual is zero-initialized, making the model
      initially equivalent to the hierarchical baseline.
    """

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
            shared_layers=shared_layers,
            structured_shared_layers=(
                structured_shared_layers
            ),
            active_tasks=active_tasks,
            num_t3_categories=num_t3_categories,
            num_t3_entities=num_t3_entities,
            num_t3_attributes=num_t3_attributes,
            category_embedding_dim=(
                category_embedding_dim
            ),
            dropout=dropout,
        )

        if not (
            "t2" in self.active_tasks
            and "t3" in self.active_tasks
        ):
            raise ValueError(
                "Explicit T2->T3 transfer requires "
                "both t2 and t3 to be active."
            )

        hidden = self.hidden_size

        self.t3_relation_transfer = (
            nn.Sequential(
                nn.Linear(
                    hidden * 2,
                    hidden,
                ),
                nn.GELU(),
                nn.Linear(
                    hidden,
                    hidden,
                ),
            )
        )

        # --------------------------------------------------------
        # Identity initialization.
        #
        # At step 0:
        #
        #   fused_r3 == r3
        #
        # exactly.
        #
        # Therefore any later performance change must be learned
        # rather than caused by random perturbation at init.
        # --------------------------------------------------------

        output_layer = (
            self.t3_relation_transfer[-1]
        )

        nn.init.zeros_(
            output_layer.weight
        )

        nn.init.zeros_(
            output_layer.bias
        )

    def architecture_summary(
        self,
    ) -> dict[str, object]:

        summary = (
            super()
            .architecture_summary()
        )

        summary.update(
            {
                "architecture":
                    "explicit_hierarchical_transfer",
                "transfer_direction":
                    "t2->t3",
                "transfer_level":
                    "relation",
                "transfer_stop_gradient":
                    True,
                "transfer_residual":
                    True,
                "transfer_zero_initialized":
                    True,
                "transfer_stochastic":
                    False,
                "teacher_dropout":
                    False,
            }
        )

        return summary

    def fuse_t2_into_t3_relation(
        self,
        *,
        t3_relation_repr: torch.Tensor,
        t2_relation_repr: torch.Tensor,
    ) -> torch.Tensor:

        transfer_input = (
            torch.cat(
                [
                    t3_relation_repr,
                    t2_relation_repr,
                ],
                dim=-1,
            )
        )

        transfer_delta = (
            self.t3_relation_transfer(
                transfer_input
            )
        )

        return (
            t3_relation_repr
            + transfer_delta
        )

    def t2_teacher_relation_repr(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deterministic T2 relation teacher.

        The normal T2 relation encoder is:

            Linear -> GELU -> Dropout

        For directed feature transfer we use the learned Linear
        transformation and activation but intentionally omit its
        training-time dropout.

        This prevents a T3 forward pass from advancing the global
        RNG state merely because the teacher feature was queried.
        """

        pair_features = (
            self.pair_features(
                aspect_repr,
                opinion_repr,
            )
        )

        hidden = (
            self.t2_relation_encoder[
                0
            ](
                pair_features
            )
        )

        hidden = (
            self.t2_relation_encoder[
                1
            ](
                hidden
            )
        )

        return hidden

    def t3_score_pair(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        T3 candidate scoring with directed T2 relation transfer.

        T2 teacher relation feature is computed from the SAME
        candidate aspect/opinion representations but receives no
        gradient from the T3 loss.
        """

        # --------------------------------------------------------
        # Native T3 relation representation.
        # --------------------------------------------------------

        t3_relation_repr = (
            self.t3_encode_pair(
                aspect_repr,
                opinion_repr,
            )
        )

        # --------------------------------------------------------
        # Directed T2 -> T3 teacher representation.
        #
        # no_grad is intentional:
        # T3 must learn to USE T2 knowledge rather than modify
        # the T2 relation encoder through the transfer pathway.
        # --------------------------------------------------------

        with torch.no_grad():

            t2_relation_repr = (
                self.t2_teacher_relation_repr(
                    aspect_repr,
                    opinion_repr,
                )
            )

        fused_relation_repr = (
            self.fuse_t2_into_t3_relation(
                t3_relation_repr=(
                    t3_relation_repr
                ),
                t2_relation_repr=(
                    t2_relation_repr
                ),
            )
        )

        # --------------------------------------------------------
        # Keep T3 relation detection native.
        #
        # This isolates explicit transfer to the higher-order
        # components that distinguish DimASQP from DimASTE.
        # --------------------------------------------------------

        relation_logit = (
            self.t3_relation_classifier(
                t3_relation_repr
            ).squeeze(-1)
        )

        category_logits = (
            self.t3_category_classifier(
                fused_relation_repr
            )
        )

        entity_logits = (
            self.t3_entity_classifier(
                fused_relation_repr
            )
        )

        attribute_logits = (
            self.t3_attribute_classifier(
                fused_relation_repr
            )
        )

        return (
            relation_logit,
            category_logits,
            entity_logits,
            attribute_logits,
            fused_relation_repr,
        )
