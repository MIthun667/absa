from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable

import torch


def get_shared_parameters(
    model,
    *,
    scope: str = "last_layer",
) -> tuple[torch.nn.Parameter, ...]:
    """
    Select shared encoder parameters used for gradient diagnostics.

    We intentionally default to the final shared transformer block
    rather than flattening the entire XLM-R encoder.

    This measures task conflict in the shared representation layer
    while keeping diagnostic memory practical.
    """

    if scope == "last_layer":

        module = (
            model.encoder
            .encoder
            .layer[-1]
        )

    elif scope == "last_4_layers":

        layers = (
            model.encoder
            .encoder
            .layer[-4:]
        )

        parameters = []

        for layer in layers:
            parameters.extend(
                parameter
                for parameter
                in layer.parameters()
                if parameter.requires_grad
            )

        return tuple(parameters)

    else:
        raise ValueError(
            f"Unsupported gradient diagnostic "
            f"scope: {scope}"
        )

    return tuple(
        parameter
        for parameter
        in module.parameters()
        if parameter.requires_grad
    )


def gradient_vector(
    loss: torch.Tensor,
    parameters: tuple[
        torch.nn.Parameter,
        ...
    ],
) -> torch.Tensor:
    """
    Return a detached FP32 gradient vector without modifying .grad.

    torch.autograd.grad() is used deliberately so the optimizer's
    gradients remain untouched.
    """

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    pieces = []

    for parameter, gradient in zip(
        parameters,
        gradients,
    ):

        if gradient is None:
            pieces.append(
                torch.zeros(
                    parameter.numel(),
                    dtype=torch.float32,
                    device=parameter.device,
                )
            )

        else:
            pieces.append(
                gradient
                .detach()
                .float()
                .reshape(-1)
            )

    if not pieces:
        raise RuntimeError(
            "No parameters selected for "
            "gradient diagnostics."
        )

    vector = torch.cat(
        pieces,
        dim=0,
    )

    if not torch.isfinite(
        vector
    ).all():
        raise RuntimeError(
            "Non-finite diagnostic gradient."
        )

    return vector


def cosine_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:

    first_norm = torch.linalg.vector_norm(
        first
    )

    second_norm = torch.linalg.vector_norm(
        second
    )

    denominator = (
        first_norm
        * second_norm
    )

    if denominator.item() <= 0.0:
        return float("nan")

    value = torch.dot(
        first,
        second,
    ) / denominator

    return float(
        value.item()
    )


def vector_norm(
    vector: torch.Tensor,
) -> float:

    return float(
        torch.linalg.vector_norm(
            vector
        ).item()
    )


def summarize_values(
    values: Iterable[float],
) -> dict[str, float]:

    clean = [
        float(value)
        for value in values
        if math.isfinite(
            float(value)
        )
    ]

    if not clean:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "count": 0,
        }

    return {
        "mean":
            statistics.fmean(
                clean
            ),
        "std":
            (
                statistics.stdev(
                    clean
                )
                if len(clean) > 1
                else 0.0
            ),
        "median":
            statistics.median(
                clean
            ),
        "count":
            len(clean),
    }


def run_gradient_diagnostics(
    *,
    model,
    task_names: tuple[str, ...],
    task_batch_iterators: dict,
    loss_functions: dict[
        str,
        Callable,
    ],
    num_batches: int,
    scope: str = "last_layer",
) -> dict:
    """
    Measure pairwise gradient interaction on fixed task batches.

    loss_functions[task](batch) must return a scalar loss tensor.

    Each diagnostic sample computes one gradient vector per task,
    then compares those vectors in shared-parameter space.
    """

    if num_batches <= 0:
        raise ValueError(
            "num_batches must be positive."
        )

    parameters = get_shared_parameters(
        model,
        scope=scope,
    )

    task_norms = defaultdict(
        list
    )

    pair_cosines = defaultdict(
        list
    )

    task_names = tuple(
        task_names
    )

    for _ in range(
        num_batches
    ):

        vectors = {}

        for task_name in (
            task_names
        ):

            batch = next(
                task_batch_iterators[
                    task_name
                ]
            )

            loss = (
                loss_functions[
                    task_name
                ](
                    batch
                )
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite diagnostic loss "
                    f"for {task_name}."
                )

            vector = gradient_vector(
                loss,
                parameters,
            )

            vectors[
                task_name
            ] = vector

            task_norms[
                task_name
            ].append(
                vector_norm(
                    vector
                )
            )

        for first_index in range(
            len(task_names)
        ):

            for second_index in range(
                first_index + 1,
                len(task_names),
            ):

                first = (
                    task_names[
                        first_index
                    ]
                )

                second = (
                    task_names[
                        second_index
                    ]
                )

                cosine = (
                    cosine_similarity(
                        vectors[first],
                        vectors[second],
                    )
                )

                pair_cosines[
                    f"{first}-{second}"
                ].append(
                    cosine
                )

        del vectors

    result = {
        "scope": scope,
        "num_batches": num_batches,
        "tasks": {},
        "pairs": {},
    }

    for task_name in task_names:

        result["tasks"][
            task_name
        ] = summarize_values(
            task_norms[
                task_name
            ]
        )

    for pair_name, values in (
        pair_cosines.items()
    ):

        summary = summarize_values(
            values
        )

        finite_values = [
            value
            for value in values
            if math.isfinite(
                value
            )
        ]

        if finite_values:

            conflict_rate = (
                sum(
                    value < 0.0
                    for value
                    in finite_values
                )
                / len(
                    finite_values
                )
            )

        else:

            conflict_rate = (
                float("nan")
            )

        summary[
            "conflict_rate"
        ] = conflict_rate

        result["pairs"][
            pair_name
        ] = summary

    return result


def print_gradient_diagnostics(
    *,
    step: int,
    result: dict,
) -> None:

    print()
    print("=" * 68)
    print(
        f"GRADIENT DIAGNOSTICS @ STEP {step}"
    )
    print(
        f"scope={result['scope']} | "
        f"batches={result['num_batches']}"
    )
    print("=" * 68)

    print()
    print("Shared gradient norms")

    for task_name, stats in (
        result["tasks"].items()
    ):

        print(
            f"  {task_name.upper():<4} "
            f"mean={stats['mean']:.4f} | "
            f"std={stats['std']:.4f} | "
            f"median={stats['median']:.4f}"
        )

    print()
    print(
        "Pairwise gradient cosine"
    )

    for pair_name, stats in (
        result["pairs"].items()
    ):

        print(
            f"  {pair_name.upper():<7} "
            f"mean={stats['mean']:+.4f} | "
            f"std={stats['std']:.4f} | "
            f"median={stats['median']:+.4f} | "
            f"conflict="
            f"{100.0 * stats['conflict_rate']:.1f}%"
        )

    print("=" * 68)
    print()


def get_layer_parameter_groups(
    model,
    *,
    layers: tuple[int, ...],
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """
    Return parameter groups for 1-based transformer layer numbers.

    Example:
        layers=(1, 4, 8, 12)

    becomes:
        L1, L4, L8, L12
    """

    encoder_layers = (
        model.encoder
        .encoder
        .layer
    )

    num_layers = len(
        encoder_layers
    )

    groups = {}

    for layer_number in layers:

        if not (
            1 <= layer_number <= num_layers
        ):
            raise ValueError(
                f"Invalid transformer layer "
                f"{layer_number}; encoder has "
                f"{num_layers} layers."
            )

        module = encoder_layers[
            layer_number - 1
        ]

        parameters = tuple(
            parameter
            for parameter
            in module.parameters()
            if parameter.requires_grad
        )

        if not parameters:
            raise RuntimeError(
                f"No trainable parameters in "
                f"layer {layer_number}."
            )

        groups[
            f"L{layer_number}"
        ] = parameters

    return groups


def gradient_vectors_by_group(
    loss: torch.Tensor,
    parameter_groups: dict[
        str,
        tuple[torch.nn.Parameter, ...],
    ],
) -> dict[str, torch.Tensor]:
    """
    Compute gradients for several disjoint parameter groups
    using ONE autograd.grad() call.

    This leaves model .grad buffers untouched.
    """

    flat_parameters = []
    group_slices = {}

    start = 0

    for group_name, parameters in (
        parameter_groups.items()
    ):

        flat_parameters.extend(
            parameters
        )

        end = (
            start
            + len(parameters)
        )

        group_slices[
            group_name
        ] = (
            start,
            end,
            parameters,
        )

        start = end

    gradients = torch.autograd.grad(
        loss,
        tuple(flat_parameters),
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    vectors = {}

    for group_name, (
        start,
        end,
        parameters,
    ) in group_slices.items():

        pieces = []

        group_gradients = gradients[
            start:end
        ]

        for parameter, gradient in zip(
            parameters,
            group_gradients,
        ):

            if gradient is None:

                pieces.append(
                    torch.zeros(
                        parameter.numel(),
                        dtype=torch.float32,
                        device=parameter.device,
                    )
                )

            else:

                pieces.append(
                    gradient
                    .detach()
                    .float()
                    .reshape(-1)
                )

        vector = torch.cat(
            pieces,
            dim=0,
        )

        if not torch.isfinite(
            vector
        ).all():

            raise RuntimeError(
                "Non-finite diagnostic "
                f"gradient in {group_name}."
            )

        vectors[
            group_name
        ] = vector

    return vectors


def run_layerwise_gradient_diagnostics(
    *,
    model,
    task_names: tuple[str, ...],
    task_batch_iterators: dict,
    loss_functions: dict[
        str,
        Callable,
    ],
    num_batches: int,
    layers: tuple[int, ...],
) -> dict:
    """
    Record-aligned layer-wise gradient geometry.

    For every aligned batch:
      1. compute each task loss once;
      2. obtain gradients for all requested layers together;
      3. compute task norms and pairwise cosine independently
         within each transformer layer.
    """

    if num_batches <= 0:
        raise ValueError(
            "num_batches must be positive."
        )

    if not layers:
        raise ValueError(
            "At least one diagnostic layer "
            "must be provided."
        )

    layers = tuple(
        dict.fromkeys(
            int(layer)
            for layer in layers
        )
    )

    task_names = tuple(
        task_names
    )

    parameter_groups = (
        get_layer_parameter_groups(
            model,
            layers=layers,
        )
    )

    task_norms = {
        layer_name:
            defaultdict(list)
        for layer_name
        in parameter_groups
    }

    pair_cosines = {
        layer_name:
            defaultdict(list)
        for layer_name
        in parameter_groups
    }

    for _ in range(
        num_batches
    ):

        task_vectors = {}

        for task_name in (
            task_names
        ):

            batch = next(
                task_batch_iterators[
                    task_name
                ]
            )

            loss = (
                loss_functions[
                    task_name
                ](
                    batch
                )
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite diagnostic "
                    f"loss for {task_name}."
                )

            vectors = (
                gradient_vectors_by_group(
                    loss,
                    parameter_groups,
                )
            )

            task_vectors[
                task_name
            ] = vectors

            for (
                layer_name,
                vector,
            ) in vectors.items():

                task_norms[
                    layer_name
                ][
                    task_name
                ].append(
                    vector_norm(
                        vector
                    )
                )

        for first_index in range(
            len(task_names)
        ):

            for second_index in range(
                first_index + 1,
                len(task_names),
            ):

                first = task_names[
                    first_index
                ]

                second = task_names[
                    second_index
                ]

                pair_name = (
                    f"{first}-{second}"
                )

                for layer_name in (
                    parameter_groups
                ):

                    cosine = (
                        cosine_similarity(
                            task_vectors[
                                first
                            ][
                                layer_name
                            ],
                            task_vectors[
                                second
                            ][
                                layer_name
                            ],
                        )
                    )

                    pair_cosines[
                        layer_name
                    ][
                        pair_name
                    ].append(
                        cosine
                    )

        del task_vectors

    result = {
        "scope": "layerwise",
        "layers": {},
        "num_batches": num_batches,
    }

    for layer_name in (
        parameter_groups
    ):

        layer_result = {
            "tasks": {},
            "pairs": {},
        }

        for task_name in (
            task_names
        ):

            layer_result[
                "tasks"
            ][
                task_name
            ] = summarize_values(
                task_norms[
                    layer_name
                ][
                    task_name
                ]
            )

        for pair_name, values in (
            pair_cosines[
                layer_name
            ].items()
        ):

            summary = (
                summarize_values(
                    values
                )
            )

            finite_values = [
                value
                for value in values
                if math.isfinite(
                    value
                )
            ]

            if finite_values:

                conflict_rate = (
                    sum(
                        value < 0.0
                        for value
                        in finite_values
                    )
                    / len(
                        finite_values
                    )
                )

            else:

                conflict_rate = (
                    float("nan")
                )

            summary[
                "conflict_rate"
            ] = conflict_rate

            layer_result[
                "pairs"
            ][
                pair_name
            ] = summary

        result[
            "layers"
        ][
            layer_name
        ] = layer_result

    return result


def print_layerwise_gradient_diagnostics(
    *,
    step: int,
    result: dict,
) -> None:

    print()
    print("=" * 84)
    print(
        "LAYER-WISE GRADIENT "
        f"DIAGNOSTICS @ STEP {step}"
    )
    print(
        f"batches={result['num_batches']} | "
        f"layers="
        f"{','.join(result['layers'])}"
    )
    print("=" * 84)

    for layer_name, layer_result in (
        result["layers"].items()
    ):

        print()
        print(
            f"[{layer_name}] "
            "gradient norms"
        )

        for task_name, stats in (
            layer_result[
                "tasks"
            ].items()
        ):

            print(
                f"  {task_name.upper():<4} "
                f"mean={stats['mean']:.4f} | "
                f"median="
                f"{stats['median']:.4f}"
            )

        print(
            f"[{layer_name}] "
            "pairwise cosine"
        )

        for pair_name, stats in (
            layer_result[
                "pairs"
            ].items()
        ):

            print(
                f"  {pair_name.upper():<7} "
                f"mean="
                f"{stats['mean']:+.4f} | "
                f"std="
                f"{stats['std']:.4f} | "
                f"median="
                f"{stats['median']:+.4f} | "
                f"conflict="
                f"{100.0 * stats['conflict_rate']:.1f}%"
            )

    print("=" * 84)
    print()
