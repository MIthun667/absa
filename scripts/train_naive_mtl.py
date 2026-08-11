from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import train_task1_baseline as t1base
import train_task2_baseline as t2base
import train_task3_hybrid as t3base

from dimabsa.experiment_data import (
    load_task_records,
)
from dimabsa.mtl_losses import (
    task1_loss,
    task2_loss,
    task3_loss,
)
from dimabsa.gradient_diagnostics import (
    print_gradient_diagnostics,
    run_gradient_diagnostics,
)
from dimabsa.mtl_eval_adapters import (
    Task1MTLAdapter,
    Task2MTLAdapter,
    Task3MTLAdapter,
)
from dimabsa.mtl_scheduler import (
    BalancedTaskScheduler,
)
from dimabsa.naive_mtl_model import (
    NaiveMTLModel,
)
from dimabsa.task3_data import (
    split_category,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        default="FacebookAI/xlm-roberta-base",
    )

    parser.add_argument(
        "--raw-root",
        default="data/raw/dimabsa",
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        default=[
            "laptop",
            "restaurant",
        ],
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[
            "t1",
            "t2",
            "t3",
        ],
        default=[
            "t1",
            "t2",
            "t3",
        ],
        help=(
            "Tasks participating in MTL. "
            "Examples: --tasks t1 t2, "
            "--tasks t1 t3, "
            "--tasks t2 t3."
        ),
    )

    parser.add_argument(
        "--language",
        default="eng",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--t2-relation-threshold",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--t3-relation-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--t3-category-threshold",
        type=float,
        default=0.93,
    )

    parser.add_argument(
        "--gradient-diagnostics",
        action="store_true",
        help=(
            "Measure shared-layer task "
            "gradient cosine statistics."
        ),
    )

    parser.add_argument(
        "--gradient-diag-every",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--gradient-diag-batches",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--gradient-diag-scope",
        choices=[
            "last_layer",
            "last_4_layers",
        ],
        default="last_layer",
    )

    parser.add_argument(
        "--gradient-diag-records-per-batch",
        type=int,
        default=8,
        help=(
            "Number of shared record IDs per "
            "aligned gradient diagnostic batch."
        ),
    )

    parser.add_argument(
        "--output-root",
        default="outputs/phase3/naive_mtl_smoke",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(
    seed: int,
):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(
        seed
    )

    if torch.cuda.is_available():

        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True


def amp_context(
    use_amp: bool,
    dtype: torch.dtype | None,
):

    if not use_amp:
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


def endless_next(
    *,
    iterator,
    loader,
):

    try:

        batch = next(
            iterator
        )

    except StopIteration:

        iterator = iter(
            loader
        )

        batch = next(
            iterator
        )

    return batch, iterator


def encoder_grad_norm(
    model,
) -> float:

    total_sq = 0.0

    found = False

    for parameter in (
        model.encoder.parameters()
    ):

        if parameter.grad is None:
            continue

        found = True

        grad = (
            parameter.grad
            .detach()
            .float()
        )

        total_sq += (
            grad.pow(2)
            .sum()
            .item()
        )

    if not found:
        return 0.0

    return math.sqrt(
        total_sq
    )


def move_t1_batch(
    batch,
    device,
):

    inputs = {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value
        in batch[
            "inputs"
        ].items()
    }

    labels = (
        batch["labels"]
        .to(
            device,
            non_blocking=True,
        )
    )

    return inputs, labels


def move_structured_batch(
    batch,
    device,
):

    result = dict(
        batch
    )

    for key in (
        "input_ids",
        "attention_mask",
        "aspect_labels",
        "opinion_labels",
    ):

        result[key] = (
            result[key]
            .to(
                device,
                non_blocking=True,
            )
        )

    return result


def make_loader(
    dataset,
    *,
    batch_size,
    collate_fn,
    num_workers,
    seed,
    device,
):

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            num_workers > 0
        ),
        generator=generator,
    )


def make_diagnostic_loader(
    dataset,
    *,
    batch_size,
    collate_fn,
    num_workers,
    device,
):
    """
    Deterministic, non-shuffled loader used only for
    gradient diagnostics.

    Recreating an iterator always yields the same batches.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            num_workers > 0
        ),
    )


def main():

    args = parse_args()

    selected_tasks = tuple(
        dict.fromkeys(
            args.tasks
        )
    )

    if len(selected_tasks) < 2:
        raise ValueError(
            "MTL requires at least two tasks. "
            "Use one of: "
            "--tasks t1 t2, "
            "--tasks t1 t3, "
            "--tasks t2 t3, "
            "or --tasks t1 t2 t3."
        )

    set_seed(
        args.seed
    )

    raw_root = Path(
        args.raw_root
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    amp_dtype = None

    if use_amp:

        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    print()
    print("=" * 80)
    print("NAIVE MULTI-TASK LEARNING")
    print("=" * 80)

    print(
        "tasks         :",
        " + ".join(
            task.upper()
            for task
            in selected_tasks
        ),
    )

    print(
        "device        :",
        device,
    )

    print(
        "AMP           :",
        use_amp,
    )

    print(
        "AMP dtype     :",
        amp_dtype,
    )

    if device.type == "cuda":

        print(
            "GPU           :",
            torch.cuda.get_device_name(
                0
            ),
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=True,
        )
    )

    # ------------------------------------------------------------
    # Optional task state.
    #
    # Only selected task datasets are materialized.
    # ------------------------------------------------------------

    t1_train_examples = []
    t1_domain_counts = Counter()
    t1_loader = None
    t1_dev_loader = None
    t1_dev_records_by_domain = {}

    t2_train_encoded = []
    t2_domain_counts = Counter()
    t2_loader = None
    t2_dev_by_domain = {}
    t2_aspect_weights = None
    t2_opinion_weights = None

    t3_train_encoded = []
    t3_domain_counts = Counter()
    t3_loader = None
    t3_dev_by_domain = {}

    category_names = ()
    entity_names = ()
    attribute_names = ()

    category_to_index = {}
    category_to_entity_index = []
    category_to_attribute_index = []

    allowed_categories_by_domain = {}
    allowed_entities_by_domain = {}
    allowed_attributes_by_domain = {}

    if "t1" in selected_tasks:

        # ============================================================
        # T1
        # ============================================================

        t1_train_examples = []

        t1_domain_counts = Counter()

        for domain in args.domains:

            records = load_task_records(
                raw_root,
                task=1,
                language=args.language,
                domain=domain,
                split="train",
            )

            examples = (
                t1base.flatten_train_records(
                    records,
                    view="relation_expanded",
                )
            )

            t1_train_examples.extend(
                examples
            )

            t1_domain_counts[
                domain
            ] = len(
                examples
            )

        t1_collator = (
            t1base.Task1Collator(
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
        )

        t1_loader = make_loader(
            t1base.Task1Dataset(
                t1_train_examples
            ),
            batch_size=(
                args.train_batch_size
            ),
            collate_fn=t1_collator,
            num_workers=args.num_workers,
            seed=args.seed + 1,
            device=device,
        )

        # ------------------------------------------------------------
        # T1 development data
        # ------------------------------------------------------------

        t1_dev_records_by_domain = {}
        t1_dev_examples = []

        for domain in args.domains:

            records = load_task_records(
                raw_root,
                task=1,
                language=args.language,
                domain=domain,
                split="dev",
            )

            t1_dev_records_by_domain[
                domain
            ] = records

            t1_dev_examples.extend(
                t1base.flatten_dev_records(
                    records
                )
            )

        t1_dev_loader = DataLoader(
            t1base.Task1Dataset(
                t1_dev_examples
            ),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=t1_collator,
            num_workers=args.num_workers,
            pin_memory=(
                device.type == "cuda"
            ),
            persistent_workers=(
                args.num_workers > 0
            ),
        )


    if "t2" in selected_tasks:

        # ============================================================
        # T2
        # ============================================================

        t2_train_encoded = []

        t2_domain_counts = Counter()

        for domain in args.domains:

            records = load_task_records(
                raw_root,
                task=2,
                language=args.language,
                domain=domain,
                split="train",
            )

            examples = (
                t2base.build_task2_train_examples(
                    records
                )
            )

            t2_domain_counts[
                domain
            ] = len(
                examples
            )

            for example in examples:

                t2_train_encoded.append(
                    t2base.encode_task2_example(
                        example,
                        tokenizer=tokenizer,
                        max_length=(
                            args.max_length
                        ),
                    )
                )

        t2_aspect_weights = (
            t2base.compute_bio_weights(
                t2_train_encoded,
                "aspect_labels",
            ).to(device)
        )

        t2_opinion_weights = (
            t2base.compute_bio_weights(
                t2_train_encoded,
                "opinion_labels",
            ).to(device)
        )

        t2_loader = make_loader(
            t2base.EncodedDataset(
                t2_train_encoded
            ),
            batch_size=(
                args.train_batch_size
            ),
            collate_fn=(
                t2base.Collator(
                    tokenizer
                )
            ),
            num_workers=args.num_workers,
            seed=args.seed + 2,
            device=device,
        )

        # ------------------------------------------------------------
        # T2 development data
        # ------------------------------------------------------------

        t2_dev_by_domain = {}

        for domain in args.domains:

            records = load_task_records(
                raw_root,
                task=2,
                language=args.language,
                domain=domain,
                split="dev",
            )

            t2_dev_by_domain[
                domain
            ] = (
                t2base.build_task2_eval_examples(
                    records
                )
            )


    if "t3" in selected_tasks:

        # ============================================================
        # T3
        # ============================================================

        t3_examples_by_domain = {}

        t3_domain_counts = Counter()

        for domain in args.domains:

            records = load_task_records(
                raw_root,
                task=3,
                language=args.language,
                domain=domain,
                split="train",
            )

            examples = (
                t3base.build_task3_examples(
                    records
                )
            )

            t3_examples_by_domain[
                domain
            ] = examples

            t3_domain_counts[
                domain
            ] = len(
                examples
            )

        category_names = tuple(
            sorted(
                {
                    category
                    for domain
                    in args.domains
                    for category
                    in t3base.collect_category_vocabulary(
                        t3_examples_by_domain[
                            domain
                        ]
                    )
                }
            )
        )

        category_to_index = {
            category: index
            for index, category
            in enumerate(
                category_names
            )
        }

        entity_names = tuple(
            sorted(
                {
                    split_category(
                        category
                    )[0]
                    for category
                    in category_names
                }
            )
        )

        attribute_names = tuple(
            sorted(
                {
                    split_category(
                        category
                    )[1]
                    for category
                    in category_names
                }
            )
        )

        entity_to_index = {
            entity: index
            for index, entity
            in enumerate(
                entity_names
            )
        }

        attribute_to_index = {
            attribute: index
            for index, attribute
            in enumerate(
                attribute_names
            )
        }

        category_to_entity_index = []

        category_to_attribute_index = []

        for category in category_names:

            entity, attribute = (
                split_category(
                    category
                )
            )

            category_to_entity_index.append(
                entity_to_index[
                    entity
                ]
            )

            category_to_attribute_index.append(
                attribute_to_index[
                    attribute
                ]
            )

        allowed_categories_by_domain = {}

        for domain in args.domains:

            domain_categories = set(
                t3base.collect_category_vocabulary(
                    t3_examples_by_domain[
                        domain
                    ]
                )
            )

            allowed_categories_by_domain[
                domain
            ] = tuple(
                category_to_index[
                    category
                ]
                for category
                in category_names
                if category
                in domain_categories
            )

        allowed_entities_by_domain = {}

        allowed_attributes_by_domain = {}

        for domain in args.domains:

            category_indices = (
                allowed_categories_by_domain[
                    domain
                ]
            )

            allowed_entities_by_domain[
                domain
            ] = tuple(
                sorted(
                    {
                        category_to_entity_index[
                            index
                        ]
                        for index
                        in category_indices
                    }
                )
            )

            allowed_attributes_by_domain[
                domain
            ] = tuple(
                sorted(
                    {
                        category_to_attribute_index[
                            index
                        ]
                        for index
                        in category_indices
                    }
                )
            )

        t3_train_encoded = []

        for domain in args.domains:

            for example in (
                t3_examples_by_domain[
                    domain
                ]
            ):

                t3_train_encoded.append(
                    t3base.encode_task3_example(
                        example,
                        tokenizer=tokenizer,
                        category_to_index=(
                            category_to_index
                        ),
                        max_length=(
                            args.max_length
                        ),
                    )
                )

        t3_aspect_weights = (
            t3base.compute_bio_weights(
                t3_train_encoded,
                "aspect_bio_labels",
            ).to(device)
        )

        t3_opinion_weights = (
            t3base.compute_bio_weights(
                t3_train_encoded,
                "opinion_bio_labels",
            ).to(device)
        )

        t3_loader = make_loader(
            t3base.EncodedDataset(
                t3_train_encoded
            ),
            batch_size=(
                args.train_batch_size
            ),
            collate_fn=(
                t3base.Collator(
                    tokenizer
                )
            ),
            num_workers=args.num_workers,
            seed=args.seed + 3,
            device=device,
        )

        # ------------------------------------------------------------
        # T3 development data
        # ------------------------------------------------------------

        t3_dev_by_domain = {}

        for domain in args.domains:

            records = load_task_records(
                raw_root,
                task=3,
                language=args.language,
                domain=domain,
                split="dev",
            )

            t3_dev_by_domain[
                domain
            ] = (
                t3base.build_task3_examples(
                    records
                )
            )


    # ------------------------------------------------------------
    # Record-aligned gradient diagnostic batches.
    #
    # Every task receives supervision derived from the SAME
    # underlying record IDs. This separates task-objective
    # interaction from ordinary sentence/content differences.
    # ------------------------------------------------------------

    aligned_gradient_batches = {}

    gradient_diag_record_ids = ()

    if args.gradient_diagnostics:

        if (
            args.gradient_diag_batches
            <= 0
        ):
            raise ValueError(
                "--gradient-diag-batches "
                "must be positive."
            )

        if (
            args.gradient_diag_records_per_batch
            <= 0
        ):
            raise ValueError(
                "--gradient-diag-records-per-batch "
                "must be positive."
            )

        # --------------------------------------------------------
        # Build record -> supervision maps for active tasks.
        # --------------------------------------------------------

        task_record_ids = {}

        t1_by_record = {}

        if "t1" in selected_tasks:

            grouped = defaultdict(
                list
            )

            for example in (
                t1_train_examples
            ):

                grouped[
                    example.record_id
                ].append(
                    example
                )

            t1_by_record = dict(
                grouped
            )

            task_record_ids[
                "t1"
            ] = set(
                t1_by_record
            )

        t2_by_record = {}

        if "t2" in selected_tasks:

            t2_by_record = {
                example.record_id:
                    example
                for example
                in t2_train_encoded
            }

            if (
                len(t2_by_record)
                != len(t2_train_encoded)
            ):
                raise RuntimeError(
                    "Task 2 diagnostic alignment "
                    "requires one encoded example "
                    "per record ID."
                )

            task_record_ids[
                "t2"
            ] = set(
                t2_by_record
            )

        t3_by_record = {}

        if "t3" in selected_tasks:

            t3_by_record = {
                example.record_id:
                    example
                for example
                in t3_train_encoded
            }

            if (
                len(t3_by_record)
                != len(t3_train_encoded)
            ):
                raise RuntimeError(
                    "Task 3 diagnostic alignment "
                    "requires one encoded example "
                    "per record ID."
                )

            task_record_ids[
                "t3"
            ] = set(
                t3_by_record
            )

        # --------------------------------------------------------
        # Intersection across all active tasks.
        # --------------------------------------------------------

        common_ids = None

        for task_name in (
            selected_tasks
        ):

            ids = task_record_ids[
                task_name
            ]

            common_ids = (
                set(ids)
                if common_ids is None
                else common_ids & ids
            )

        common_ids = sorted(
            common_ids
        )

        required_records = (
            args.gradient_diag_batches
            * args.gradient_diag_records_per_batch
        )

        if (
            len(common_ids)
            < required_records
        ):
            raise RuntimeError(
                "Not enough aligned records for "
                "gradient diagnostics: "
                f"need={required_records}, "
                f"available={len(common_ids)}"
            )

        # Fixed deterministic diagnostic subset.
        diagnostic_rng = random.Random(
            args.seed + 91073
        )

        diagnostic_rng.shuffle(
            common_ids
        )

        gradient_diag_record_ids = tuple(
            common_ids[
                :required_records
            ]
        )

        record_id_batches = []

        for start_index in range(
            0,
            required_records,
            args.gradient_diag_records_per_batch,
        ):

            record_id_batches.append(
                gradient_diag_record_ids[
                    start_index:
                    start_index
                    + args.gradient_diag_records_per_batch
                ]
            )

        # --------------------------------------------------------
        # Collators.
        # --------------------------------------------------------

        if "t2" in selected_tasks:

            t2_diag_collator = (
                t2base.Collator(
                    tokenizer
                )
            )

        if "t3" in selected_tasks:

            t3_diag_collator = (
                t3base.Collator(
                    tokenizer
                )
            )

        # --------------------------------------------------------
        # Pre-collate CPU batches.
        #
        # Reused at every diagnostic checkpoint so comparisons
        # over training time use exactly the same records.
        # --------------------------------------------------------

        if "t1" in selected_tasks:

            aligned_gradient_batches[
                "t1"
            ] = []

        if "t2" in selected_tasks:

            aligned_gradient_batches[
                "t2"
            ] = []

        if "t3" in selected_tasks:

            aligned_gradient_batches[
                "t3"
            ] = []

        for record_ids in (
            record_id_batches
        ):

            if "t1" in selected_tasks:

                t1_batch_examples = []

                for record_id in (
                    record_ids
                ):

                    t1_batch_examples.extend(
                        t1_by_record[
                            record_id
                        ]
                    )

                aligned_gradient_batches[
                    "t1"
                ].append(
                    t1_collator(
                        t1_batch_examples
                    )
                )

            if "t2" in selected_tasks:

                aligned_gradient_batches[
                    "t2"
                ].append(
                    t2_diag_collator(
                        [
                            t2_by_record[
                                record_id
                            ]
                            for record_id
                            in record_ids
                        ]
                    )
                )

            if "t3" in selected_tasks:

                aligned_gradient_batches[
                    "t3"
                ].append(
                    t3_diag_collator(
                        [
                            t3_by_record[
                                record_id
                            ]
                            for record_id
                            in record_ids
                        ]
                    )
                )

        print()
        print("GRADIENT DIAGNOSTIC DATA")
        print("-" * 80)
        print(
            "aligned records available :",
            len(common_ids),
        )
        print(
            "aligned records selected  :",
            len(
                gradient_diag_record_ids
            ),
        )
        print(
            "records / batch           :",
            args.gradient_diag_records_per_batch,
        )
        print(
            "diagnostic batches        :",
            args.gradient_diag_batches,
        )
        print(
            "scope                     :",
            args.gradient_diag_scope,
        )

    # ============================================================
    # Dataset summary
    # ============================================================

    print()
    print("DATA")
    print("-" * 80)

    if "t1" in selected_tasks:
        print(
            "T1 examples    :",
            len(t1_train_examples),
            dict(t1_domain_counts),
        )

    if "t2" in selected_tasks:
        print(
            "T2 examples    :",
            len(t2_train_encoded),
            dict(t2_domain_counts),
        )

    if "t3" in selected_tasks:
        print(
            "T3 examples    :",
            len(t3_train_encoded),
            dict(t3_domain_counts),
        )

        print(
            "T3 categories  :",
            len(category_names),
        )

        print(
            "T3 entities    :",
            len(entity_names),
        )

        print(
            "T3 attributes  :",
            len(attribute_names),
        )

    # ============================================================
    # Shared model
    # ============================================================

    model = NaiveMTLModel(
        args.model_name,
        num_t3_categories=(
            len(category_names)
            if "t3" in selected_tasks
            else 1
        ),
        num_t3_entities=(
            len(entity_names)
            if "t3" in selected_tasks
            else 1
        ),
        num_t3_attributes=(
            len(attribute_names)
            if "t3" in selected_tasks
            else 1
        ),
        dropout=args.dropout,
    ).to(device)

    # ------------------------------------------------------------
    # Evaluation adapters.
    #
    # All three point to the SAME NaiveMTLModel.
    # ------------------------------------------------------------

    t1_eval_model = Task1MTLAdapter(
        model
    )

    t2_eval_model = Task2MTLAdapter(
        model
    )

    t3_eval_model = Task3MTLAdapter(
        model
    )

    # ------------------------------------------------------------
    # Freeze excluded task heads.
    #
    # The shared encoder always remains trainable.
    # ------------------------------------------------------------

    for name, parameter in (
        model.named_parameters()
    ):

        if (
            name.startswith("t1_")
            and "t1" not in selected_tasks
        ):
            parameter.requires_grad = False

        elif (
            name.startswith("t2_")
            and "t2" not in selected_tasks
        ):
            parameter.requires_grad = False

        elif (
            name.startswith("t3_")
            and "t3" not in selected_tasks
        ):
            parameter.requires_grad = False

    # ============================================================
    # One optimizer over shared encoder + all heads
    # ============================================================

    no_decay_terms = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
    )

    decay_params = []

    no_decay_params = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if any(
            term in name
            for term
            in no_decay_terms
        ):

            no_decay_params.append(
                parameter
            )

        else:

            decay_params.append(
                parameter
            )

    optimizer = AdamW(
        [
            {
                "params":
                    decay_params,
                "weight_decay":
                    args.weight_decay,
            },
            {
                "params":
                    no_decay_params,
                "weight_decay":
                    0.0,
            },
        ],
        lr=args.learning_rate,
    )

    warmup_steps = int(
        args.max_steps
        * args.warmup_ratio
    )

    lr_scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(
                warmup_steps
            ),
            num_training_steps=(
                args.max_steps
            ),
        )
    )

    scaler = None

    if (
        use_amp
        and amp_dtype
        == torch.float16
    ):

        scaler = (
            torch.amp.GradScaler(
                "cuda"
            )
        )

    # ============================================================
    # Balanced task scheduler
    # ============================================================

    task_scheduler = iter(
        BalancedTaskScheduler(
            seed=args.seed,
            task_names=selected_tasks,
        )
    )

    task_counts = Counter()

    running_loss = defaultdict(
        float
    )

    t1_iterator = (
        iter(t1_loader)
        if t1_loader is not None
        else None
    )

    t2_iterator = (
        iter(t2_loader)
        if t2_loader is not None
        else None
    )

    t3_iterator = (
        iter(t3_loader)
        if t3_loader is not None
        else None
    )

    dev_history = []

    def diagnostic_loss_t1(
        batch,
    ):

        inputs, labels = (
            move_t1_batch(
                batch,
                device,
            )
        )

        loss, _ = task1_loss(
            model=model,
            inputs=inputs,
            labels=labels,
        )

        return loss


    def diagnostic_loss_t2(
        batch,
    ):

        batch = (
            move_structured_batch(
                batch,
                device,
            )
        )

        loss, _ = task2_loss(
            model=model,
            batch=batch,
            aspect_weights=(
                t2_aspect_weights
            ),
            opinion_weights=(
                t2_opinion_weights
            ),
            ignore_index=(
                t2base.IGNORE_INDEX
            ),
        )

        return loss


    def diagnostic_loss_t3(
        batch,
    ):

        batch = (
            move_structured_batch(
                batch,
                device,
            )
        )

        loss, _ = task3_loss(
            model=model,
            batch=batch,
            aspect_weights=(
                t3_aspect_weights
            ),
            opinion_weights=(
                t3_opinion_weights
            ),
            ignore_index=(
                t3base.IGNORE_INDEX
            ),
            allowed_categories_by_domain=(
                allowed_categories_by_domain
            ),
            allowed_entities_by_domain=(
                allowed_entities_by_domain
            ),
            allowed_attributes_by_domain=(
                allowed_attributes_by_domain
            ),
            category_to_entity_index=(
                category_to_entity_index
            ),
            category_to_attribute_index=(
                category_to_attribute_index
            ),
        )

        return loss


    diagnostic_loss_functions = {
        "t1": diagnostic_loss_t1,
        "t2": diagnostic_loss_t2,
        "t3": diagnostic_loss_t3,
    }

    gradient_history = []

    # ============================================================
    # Training
    # ============================================================

    print()
    print("TRAINING")
    print("-" * 80)

    for step in range(
        1,
        args.max_steps + 1,
    ):

        model.train()

        task_name = next(
            task_scheduler
        )

        task_counts[
            task_name
        ] += 1

        optimizer.zero_grad(
            set_to_none=True
        )

        with amp_context(
            use_amp,
            amp_dtype,
        ):

            if task_name == "t1":

                batch, t1_iterator = (
                    endless_next(
                        iterator=(
                            t1_iterator
                        ),
                        loader=t1_loader,
                    )
                )

                inputs, labels = (
                    move_t1_batch(
                        batch,
                        device,
                    )
                )

                loss, components = (
                    task1_loss(
                        model=model,
                        inputs=inputs,
                        labels=labels,
                    )
                )

            elif task_name == "t2":

                batch, t2_iterator = (
                    endless_next(
                        iterator=(
                            t2_iterator
                        ),
                        loader=t2_loader,
                    )
                )

                batch = (
                    move_structured_batch(
                        batch,
                        device,
                    )
                )

                loss, components = (
                    task2_loss(
                        model=model,
                        batch=batch,
                        aspect_weights=(
                            t2_aspect_weights
                        ),
                        opinion_weights=(
                            t2_opinion_weights
                        ),
                        ignore_index=(
                            t2base.IGNORE_INDEX
                        ),
                    )
                )

            elif task_name == "t3":

                batch, t3_iterator = (
                    endless_next(
                        iterator=(
                            t3_iterator
                        ),
                        loader=t3_loader,
                    )
                )

                batch = (
                    move_structured_batch(
                        batch,
                        device,
                    )
                )

                loss, components = (
                    task3_loss(
                        model=model,
                        batch=batch,
                        aspect_weights=(
                            t3_aspect_weights
                        ),
                        opinion_weights=(
                            t3_opinion_weights
                        ),
                        ignore_index=(
                            t3base.IGNORE_INDEX
                        ),
                        allowed_categories_by_domain=(
                            allowed_categories_by_domain
                        ),
                        allowed_entities_by_domain=(
                            allowed_entities_by_domain
                        ),
                        allowed_attributes_by_domain=(
                            allowed_attributes_by_domain
                        ),
                        category_to_entity_index=(
                            category_to_entity_index
                        ),
                        category_to_attribute_index=(
                            category_to_attribute_index
                        ),
                    )
                )

            else:

                raise ValueError(
                    task_name
                )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite loss at "
                f"step {step}: "
                f"{loss.item()}"
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

        else:

            loss.backward()

        enc_grad = (
            encoder_grad_norm(
                model
            )
        )

        if (
            not math.isfinite(
                enc_grad
            )
            or enc_grad <= 0.0
        ):

            raise RuntimeError(
                f"Invalid encoder gradient "
                f"at step={step}, "
                f"task={task_name}: "
                f"{enc_grad}"
            )

        total_grad = (
            clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
        )

        if scaler is not None:

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            optimizer.step()

        lr_scheduler.step()

        running_loss[
            task_name
        ] += (
            loss.detach()
            .float()
            .item()
        )

        if (
            step == 1
            or step % args.log_every == 0
        ):

            was_clipped = (
                float(total_grad)
                > args.max_grad_norm
            )

            print(
                f"[TRAIN] "
                f"step={step}/{args.max_steps} | "
                f"task={task_name.upper()} | "
                f"loss={loss.item():.4f} | "
                f"encoder_grad={enc_grad:.2f} | "
                f"clipped={'YES' if was_clipped else 'NO'} | "
                f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
            )

        # ========================================================
        # Shared-layer gradient diagnostics
        # ========================================================

        if (
            args.gradient_diagnostics
            and args.gradient_diag_every > 0
            and step
            % args.gradient_diag_every
            == 0
        ):

            # Diagnostic gradients are computed separately with
            # autograd.grad(), so optimizer .grad buffers remain
            # untouched.
            model.eval()

            diagnostic_iterators = {
                task_name:
                    iter(
                        aligned_gradient_batches[
                            task_name
                        ]
                    )
                for task_name
                in selected_tasks
            }

            with amp_context(
                use_amp,
                amp_dtype,
            ):

                gradient_result = (
                    run_gradient_diagnostics(
                        model=model,
                        task_names=(
                            selected_tasks
                        ),
                        task_batch_iterators=(
                            diagnostic_iterators
                        ),
                        loss_functions=(
                            diagnostic_loss_functions
                        ),
                        num_batches=(
                            args.gradient_diag_batches
                        ),
                        scope=(
                            args.gradient_diag_scope
                        ),
                    )
                )

            gradient_result[
                "step"
            ] = step

            gradient_history.append(
                gradient_result
            )

            print_gradient_diagnostics(
                step=step,
                result=gradient_result,
            )

            compact_pairs = " | ".join(
                (
                    f"{pair.upper()}="
                    f"{stats['mean']:+.4f}"
                    f" "
                    f"(conflict "
                    f"{100.0 * stats['conflict_rate']:.0f}%)"
                )
                for pair, stats
                in gradient_result[
                    "pairs"
                ].items()
            )

            print(
                f"[GRAD] step={step} | "
                f"{compact_pairs}"
            )

        # ========================================================
        # Fixed-protocol development evaluation.
        #
        # No threshold tuning is performed here.
        # ========================================================

        if (
            args.eval_every > 0
            and step % args.eval_every == 0
        ):

            print()
            print(
                f"[DEV @ step {step}]"
            )

            record = {
                "step": step,
            }

            print()
            print("=" * 60)
            print(
                f"DEV @ STEP {step}"
            )
            print("=" * 60)

            if "t1" in selected_tasks:

                (
                    t1_metrics,
                    _,
                ) = t1base.evaluate(
                    t1_eval_model,
                    t1_dev_loader,
                    device,
                    use_amp,
                    amp_dtype,
                    t1_dev_records_by_domain,
                )

                record["t1"] = (
                    t1_metrics
                )

                print("T1 RMSE ↓")
                print(
                    f"  Laptop      "
                    f"{t1_metrics['laptop']['RMSE_VA']:.4f}"
                )
                print(
                    f"  Restaurant  "
                    f"{t1_metrics['restaurant']['RMSE_VA']:.4f}"
                )
                print(
                    f"  Macro       "
                    f"{t1_metrics['macro']['RMSE_VA']:.4f}"
                )
                print()

            if "t2" in selected_tasks:

                (
                    t2_metrics,
                    _,
                ) = t2base.evaluate(
                    model=t2_eval_model,
                    tokenizer=tokenizer,
                    dev_by_domain=(
                        t2_dev_by_domain
                    ),
                    device=device,
                    max_length=(
                        args.max_length
                    ),
                    threshold=(
                        args.t2_relation_threshold
                    ),
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )

                record["t2"] = (
                    t2_metrics
                )

                print("T2 cF1 ↑")
                print(
                    f"  Laptop      "
                    f"{t2_metrics['laptop']['cF1']:.4f}"
                )
                print(
                    f"  Restaurant  "
                    f"{t2_metrics['restaurant']['cF1']:.4f}"
                )
                print(
                    f"  Macro       "
                    f"{t2_metrics['macro']['cF1']:.4f}"
                )
                print()

            if "t3" in selected_tasks:

                t3_eval_output = (
                    t3base.evaluate(
                        model=t3_eval_model,
                        tokenizer=tokenizer,
                        dev_by_domain=(
                            t3_dev_by_domain
                        ),
                        category_names=(
                            category_names
                        ),
                        allowed_categories_by_domain=(
                            allowed_categories_by_domain
                        ),
                        device=device,
                        max_length=(
                            args.max_length
                        ),
                        relation_threshold=(
                            args.t3_relation_threshold
                        ),
                        category_threshold=(
                            args.t3_category_threshold
                        ),
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                    )
                )

                (
                    t3_macro_cf1,
                    t3_metrics,
                    _,
                ) = t3_eval_output

                record["t3"] = (
                    t3_metrics
                )

                record[
                    "t3_macro_cf1"
                ] = t3_macro_cf1

                print("T3 cF1 ↑")

                for domain in (
                    "laptop",
                    "restaurant",
                ):

                    print(
                        f"  "
                        f"{domain.capitalize():<11} "
                        f"{t3_metrics[domain]['cF1']:.4f}"
                    )

                print(
                    f"  {'Macro':<11} "
                    f"{t3_macro_cf1:.4f}"
                )

                print()

            dev_history.append(
                record
            )

            print("=" * 60)
            print()

    # ============================================================
    # Final diagnostics
    # ============================================================

    print()
    print("=" * 80)
    print("TASK UPDATE COUNTS")
    print("=" * 80)

    for task_name in selected_tasks:

        print(
            f"{task_name}: "
            f"{task_counts[task_name]}"
        )

    num_tasks = len(
        selected_tasks
    )

    expected = (
        args.max_steps
        // num_tasks
    )

    if (
        args.max_steps
        % num_tasks
        == 0
    ):

        for task_name in (
            selected_tasks
        ):

            assert (
                task_counts[
                    task_name
                ]
                == expected
            )

    print()
    print("=" * 80)
    print("MEAN TRAIN LOSS BY TASK")
    print("=" * 80)

    for task_name in selected_tasks:

        mean_loss = (
            running_loss[
                task_name
            ]
            / max(
                task_counts[
                    task_name
                ],
                1,
            )
        )

        print(
            f"{task_name}: "
            f"{mean_loss:.6f}"
        )

    # ============================================================
    # Save smoke checkpoint
    # ============================================================

    output_dir = (
        Path(
            args.output_root
        )
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "smoke_checkpoint.pt"
    )

    (
        output_dir
        / "dev_history.json"
    ).write_text(
        json.dumps(
            dev_history,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.gradient_diagnostics:

        (
            output_dir
            / "gradient_diagnostic_records.json"
        ).write_text(
            json.dumps(
                {
                    "tasks":
                        selected_tasks,
                    "seed":
                        args.seed,
                    "records_per_batch":
                        args.gradient_diag_records_per_batch,
                    "num_batches":
                        args.gradient_diag_batches,
                    "record_ids":
                        gradient_diag_record_ids,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    (
        output_dir
        / "gradient_history.json"
    ).write_text(
        json.dumps(
            gradient_history,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    torch.save(
        {
            "step":
                args.max_steps,
            "selected_tasks":
                selected_tasks,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "task_counts":
                dict(
                    task_counts
                ),
            "category_names":
                category_names,
            "entity_names":
                entity_names,
            "attribute_names":
                attribute_names,
            "allowed_categories_by_domain":
                allowed_categories_by_domain,
        },
        checkpoint_path,
    )

    print()
    print(
        "Saved:",
        checkpoint_path,
    )

    print()
    print(
        "NAIVE MTL SMOKE TEST PASSED."
    )


if __name__ == "__main__":
    main()
