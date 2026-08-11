from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import (
    DataLoader,
    Dataset,
)

import transformers

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from dimabsa.experiment_data import (
    load_task_records,
)

from dimabsa.metrics import (
    evaluate_structured,
)

from dimabsa.task2_data import (
    NULL_TERM,
)
from dimabsa.task3_data import (
    Task3Example,
    build_task3_examples,
    collect_category_vocabulary,
    split_category,
)

from dimabsa.task3_pair_model import (
    PredictedNode,
    Task3PairModel,
)

from dimabsa.task2_supervision import (
    B,
    I,
    O,
    IGNORE_INDEX,
)
from dimabsa.task3_supervision import (
    EncodedTask3Example,
    encode_task3_example,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def current_git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
        )

    except Exception:
        return None


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------


def safe_json(value):

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, dict):
        return {
            key: safe_json(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            safe_json(item)
            for item in value
        ]

    return value


def save_json(
    path: Path,
    value,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            safe_json(value),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


class EncodedDataset(Dataset):
    def __init__(
        self,
        examples: list[
            EncodedTask3Example
        ],
    ):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class Collator:
    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(
        self,
        batch: list[
            EncodedTask3Example
        ],
    ):

        max_len = max(
            len(x.input_ids)
            for x in batch
        )

        input_ids = []
        attention_mask = []
        aspect_labels = []
        opinion_labels = []

        for example in batch:

            length = len(
                example.input_ids
            )

            pad = max_len - length

            input_ids.append(
                list(example.input_ids)
                + [
                    self.tokenizer.pad_token_id
                ]
                * pad
            )

            attention_mask.append(
                list(example.attention_mask)
                + [0] * pad
            )

            aspect_labels.append(
                list(example.aspect_bio_labels)
                + [IGNORE_INDEX] * pad
            )

            opinion_labels.append(
                list(example.opinion_bio_labels)
                + [IGNORE_INDEX] * pad
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "aspect_labels": torch.tensor(
                aspect_labels,
                dtype=torch.long,
            ),
            "opinion_labels": torch.tensor(
                opinion_labels,
                dtype=torch.long,
            ),
            "examples": batch,
        }


# ---------------------------------------------------------------------
# BIO class weighting
# ---------------------------------------------------------------------


def compute_bio_weights(
    examples: list[
        EncodedTask3Example
    ],
    attribute: str,
) -> torch.Tensor:

    counts = Counter()

    for example in examples:

        labels = getattr(
            example,
            attribute,
        )

        for label in labels:

            if label != IGNORE_INDEX:
                counts[label] += 1

    frequencies = torch.tensor(
        [
            counts[O],
            counts[B],
            counts[I],
        ],
        dtype=torch.float32,
    )

    total = frequencies.sum()

    # Inverse-square-root frequency:
    #
    #     w_c = sqrt(N / n_c)
    #
    # Normalize to mean weight 1.
    weights = torch.sqrt(
        total / frequencies
    )

    weights = (
        weights
        / weights.mean()
    )

    return weights


# ---------------------------------------------------------------------
# Predicted BIO nodes
# ---------------------------------------------------------------------


def predicted_nodes_from_bio(
    *,
    labels: list[int],
    offsets: list[
        tuple[int, int]
    ],
    text: str,
) -> list[PredictedNode]:

    occurrences = []

    current_indices = []

    def flush():

        nonlocal current_indices

        if not current_indices:
            return

        valid_offsets = [
            offsets[index]
            for index in current_indices
            if offsets[index][0]
            != offsets[index][1]
        ]

        if not valid_offsets:
            current_indices = []
            return

        char_start = min(
            start
            for start, _ in valid_offsets
        )

        char_end = max(
            end
            for _, end in valid_offsets
        )

        surface = text[
            char_start:char_end
        ].strip()

        if surface:
            occurrences.append(
                (
                    surface,
                    tuple(
                        current_indices
                    ),
                )
            )

        current_indices = []

    for index, label in enumerate(
        labels
    ):

        start, end = offsets[index]

        if start == end:
            continue

        if label == B:

            flush()

            current_indices = [
                index
            ]

        elif label == I:

            if not current_indices:
                current_indices = [
                    index
                ]
            else:
                current_indices.append(
                    index
                )

        else:
            flush()

    flush()

    grouped = defaultdict(
        lambda: {
            "surface": None,
            "indices": set(),
        }
    )

    for surface, token_indices in (
        occurrences
    ):

        key = surface.casefold()

        grouped[key]["surface"] = surface

        grouped[key]["indices"].update(
            token_indices
        )

    nodes = []

    for value in grouped.values():

        nodes.append(
            PredictedNode(
                text=value["surface"],
                token_indices=tuple(
                    sorted(
                        value["indices"]
                    )
                ),
                is_null=False,
            )
        )

    nodes.sort(
        key=lambda node:
        node.text.casefold()
    )

    return nodes


def add_null_node(
    nodes: list[PredictedNode],
) -> list[PredictedNode]:

    result = list(nodes)

    result.append(
        PredictedNode(
            text=NULL_TERM,
            token_indices=(),
            is_null=True,
        )
    )

    return result


# ---------------------------------------------------------------------
# AMP
# ---------------------------------------------------------------------


def amp_context(
    use_amp,
    dtype,
):

    if not use_amp:
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


# ---------------------------------------------------------------------
# Training relation loss
# ---------------------------------------------------------------------


def relation_category_va_loss(
    *,
    model,
    hidden,
    examples,
    allowed_categories_by_domain,
    allowed_entities_by_domain,
    allowed_attributes_by_domain,
    category_to_entity_index,
    category_to_attribute_index,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    relation_logits = []
    relation_labels = []

    category_losses = []
    pair_category_losses = []
    entity_losses = []
    attribute_losses = []

    predicted_va = []
    gold_va = []

    for batch_index, example in enumerate(
        examples
    ):

        example_hidden = hidden[
            batch_index
        ]

        sentence_repr = (
            example_hidden[0]
        )

        aspect_reprs = [
            model.pool_node(
                example_hidden,
                sentence_repr,
                node,
                node_type="aspect",
            )
            for node in example.aspect_nodes
        ]

        opinion_reprs = [
            model.pool_node(
                example_hidden,
                sentence_repr,
                node,
                node_type="opinion",
            )
            for node in example.opinion_nodes
        ]

        allowed_category_indices = (
            allowed_categories_by_domain[
                example.domain
            ]
        )

        allowed_entity_indices = (
            allowed_entities_by_domain[
                example.domain
            ]
        )

        allowed_attribute_indices = (
            allowed_attributes_by_domain[
                example.domain
            ]
        )

        category_index_tensor = torch.tensor(
            allowed_category_indices,
            dtype=torch.long,
            device=hidden.device,
        )

        entity_index_tensor = torch.tensor(
            allowed_entity_indices,
            dtype=torch.long,
            device=hidden.device,
        )

        attribute_index_tensor = torch.tensor(
            allowed_attribute_indices,
            dtype=torch.long,
            device=hidden.device,
        )

        for candidate in (
            example.relation_candidates
        ):

            (
                relation_logit,
                category_logit,
                entity_logit,
                attribute_logit,
                relation_repr,
            ) = model.score_pair(
                aspect_reprs[
                    candidate.aspect_index
                ],
                opinion_reprs[
                    candidate.opinion_index
                ],
            )

            relation_logits.append(
                relation_logit
            )

            relation_labels.append(
                float(
                    candidate.relation_label
                )
            )

            if candidate.relation_label == 1.0:

                # =================================================
                # Flat category supervision
                # =================================================

                full_category_labels = torch.tensor(
                    candidate.category_labels,
                    dtype=torch.float32,
                    device=hidden.device,
                )

                domain_category_logits = (
                    category_logit.index_select(
                        0,
                        category_index_tensor,
                    )
                )

                domain_category_labels = (
                    full_category_labels.index_select(
                        0,
                        category_index_tensor,
                    )
                )

                category_positives = (
                    domain_category_labels.sum()
                )

                category_negatives = (
                    domain_category_labels.numel()
                    - category_positives
                )

                if category_positives.item() > 0:

                    category_pos_weight = (
                        category_negatives
                        / category_positives
                    ).clamp(
                        min=1.0
                    )

                    category_losses.append(
                        F.binary_cross_entropy_with_logits(
                            domain_category_logits,
                            domain_category_labels,
                            pos_weight=(
                                category_pos_weight
                            ),
                        )
                    )

                    # =============================================
                    # Compositional pair-category supervision.
                    #
                    # Score each valid category from its
                    # Entity/Attribute prototype.
                    # =============================================

                    pair_entity_indices = torch.tensor(
                        [
                            category_to_entity_index[
                                category_index
                            ]
                            for category_index
                            in allowed_category_indices
                        ],
                        dtype=torch.long,
                        device=hidden.device,
                    )

                    pair_attribute_indices = torch.tensor(
                        [
                            category_to_attribute_index[
                                category_index
                            ]
                            for category_index
                            in allowed_category_indices
                        ],
                        dtype=torch.long,
                        device=hidden.device,
                    )

                    pair_category_logits = (
                        model.pair_category_logits(
                            relation_repr,
                            pair_entity_indices,
                            pair_attribute_indices,
                        )
                    )

                    pair_category_losses.append(
                        F.binary_cross_entropy_with_logits(
                            pair_category_logits,
                            domain_category_labels,
                            pos_weight=(
                                category_pos_weight
                            ),
                        )
                    )

                # =================================================
                # Derive compositional gold labels from gold flat
                # categories.
                # =================================================

                positive_category_indices = [
                    index
                    for index, label
                    in enumerate(
                        candidate.category_labels
                    )
                    if label > 0.5
                ]

                gold_entity_indices = {
                    category_to_entity_index[
                        index
                    ]
                    for index in (
                        positive_category_indices
                    )
                }

                gold_attribute_indices = {
                    category_to_attribute_index[
                        index
                    ]
                    for index in (
                        positive_category_indices
                    )
                }

                # =================================================
                # Entity auxiliary supervision
                # =================================================

                entity_labels = torch.zeros(
                    model.num_entities,
                    dtype=torch.float32,
                    device=hidden.device,
                )

                for index in gold_entity_indices:
                    entity_labels[index] = 1.0

                domain_entity_logits = (
                    entity_logit.index_select(
                        0,
                        entity_index_tensor,
                    )
                )

                domain_entity_labels = (
                    entity_labels.index_select(
                        0,
                        entity_index_tensor,
                    )
                )

                entity_positives = (
                    domain_entity_labels.sum()
                )

                entity_negatives = (
                    domain_entity_labels.numel()
                    - entity_positives
                )

                if entity_positives.item() > 0:

                    # Softer than the flat-category weighting.
                    entity_pos_weight = torch.sqrt(
                        (
                            entity_negatives
                            / entity_positives
                        ).clamp(
                            min=1.0
                        )
                    )

                    entity_losses.append(
                        F.binary_cross_entropy_with_logits(
                            domain_entity_logits,
                            domain_entity_labels,
                            pos_weight=(
                                entity_pos_weight
                            ),
                        )
                    )

                # =================================================
                # Attribute auxiliary supervision
                # =================================================

                attribute_labels = torch.zeros(
                    model.num_attributes,
                    dtype=torch.float32,
                    device=hidden.device,
                )

                for index in gold_attribute_indices:
                    attribute_labels[index] = 1.0

                domain_attribute_logits = (
                    attribute_logit.index_select(
                        0,
                        attribute_index_tensor,
                    )
                )

                domain_attribute_labels = (
                    attribute_labels.index_select(
                        0,
                        attribute_index_tensor,
                    )
                )

                attribute_positives = (
                    domain_attribute_labels.sum()
                )

                attribute_negatives = (
                    domain_attribute_labels.numel()
                    - attribute_positives
                )

                if attribute_positives.item() > 0:

                    attribute_pos_weight = torch.sqrt(
                        (
                            attribute_negatives
                            / attribute_positives
                        ).clamp(
                            min=1.0
                        )
                    )

                    attribute_losses.append(
                        F.binary_cross_entropy_with_logits(
                            domain_attribute_logits,
                            domain_attribute_labels,
                            pos_weight=(
                                attribute_pos_weight
                            ),
                        )
                    )

                # =================================================
                # Original category-conditioned VA supervision.
                # =================================================

                for target in (
                    candidate.va_targets
                ):

                    va = model.predict_va(
                        relation_repr,
                        target.category_index,
                    )

                    predicted_va.append(
                        va
                    )

                    gold_va.append(
                        (
                            target.valence,
                            target.arousal,
                        )
                    )

    relation_logits_tensor = torch.stack(
        relation_logits
    )

    relation_labels_tensor = torch.tensor(
        relation_labels,
        dtype=torch.float32,
        device=hidden.device,
    )

    relation_loss = (
        F.binary_cross_entropy_with_logits(
            relation_logits_tensor,
            relation_labels_tensor,
        )
    )

    category_loss = (
        torch.stack(
            category_losses
        ).mean()
        if category_losses
        else hidden.sum() * 0.0
    )

    pair_category_loss = (
        torch.stack(
            pair_category_losses
        ).mean()
        if pair_category_losses
        else hidden.sum() * 0.0
    )

    entity_loss = (
        torch.stack(
            entity_losses
        ).mean()
        if entity_losses
        else hidden.sum() * 0.0
    )

    attribute_loss = (
        torch.stack(
            attribute_losses
        ).mean()
        if attribute_losses
        else hidden.sum() * 0.0
    )

    if predicted_va:

        predicted_va_tensor = torch.stack(
            predicted_va
        )

        gold_va_tensor = torch.tensor(
            gold_va,
            dtype=torch.float32,
            device=hidden.device,
        )

        va_loss = F.mse_loss(
            predicted_va_tensor,
            gold_va_tensor,
        )

    else:

        va_loss = (
            hidden.sum() * 0.0
        )

    return (
        relation_loss,
        category_loss,
        pair_category_loss,
        entity_loss,
        attribute_loss,
        va_loss,
    )


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------


@torch.no_grad()
def predict_example(
    *,
    model,
    tokenizer,
    example: Task3Example,
    category_names,
    allowed_category_indices,
    device,
    max_length,
    relation_threshold,
    category_threshold,
    use_amp,
    amp_dtype,
):

    encoded = tokenizer(
        example.text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets = [
        tuple(x)
        for x in encoded[
            "offset_mapping"
        ][0].tolist()
    ]

    model_inputs = {
        "input_ids":
        encoded["input_ids"].to(
            device
        ),
        "attention_mask":
        encoded[
            "attention_mask"
        ].to(device),
    }

    with amp_context(
        use_amp,
        amp_dtype,
    ):

        (
            hidden,
            aspect_logits,
            opinion_logits,
        ) = model.encode(
            **model_inputs
        )

    aspect_labels = (
        aspect_logits[0]
        .argmax(dim=-1)
        .cpu()
        .tolist()
    )

    opinion_labels = (
        opinion_logits[0]
        .argmax(dim=-1)
        .cpu()
        .tolist()
    )

    aspects = predicted_nodes_from_bio(
        labels=aspect_labels,
        offsets=offsets,
        text=example.text,
    )

    opinions = predicted_nodes_from_bio(
        labels=opinion_labels,
        offsets=offsets,
        text=example.text,
    )

    aspects = add_null_node(
        aspects
    )

    opinions = add_null_node(
        opinions
    )

    example_hidden = hidden[0]

    sentence_repr = (
        example_hidden[0]
    )

    aspect_reprs = [
        model.pool_node(
            example_hidden,
            sentence_repr,
            node,
            node_type="aspect",
        )
        for node in aspects
    ]

    opinion_reprs = [
        model.pool_node(
            example_hidden,
            sentence_repr,
            node,
            node_type="opinion",
        )
        for node in opinions
    ]

    predictions = []

    for a_index, aspect in enumerate(
        aspects
    ):

        for o_index, opinion in enumerate(
            opinions
        ):

            with amp_context(
                use_amp,
                amp_dtype,
            ):

                (
                    relation_logit,
                    category_logits,
                    _,
                    _,
                    relation_repr,
                ) = model.score_pair(
                    aspect_reprs[a_index],
                    opinion_reprs[o_index],
                )

            relation_probability = (
                torch.sigmoid(
                    relation_logit
                ).item()
            )

            if (
                relation_probability
                < relation_threshold
            ):
                continue

            category_probabilities = (
                torch.sigmoid(
                    category_logits
                )
            )

            for category_index in (
                allowed_category_indices
            ):

                category_probability = (
                    category_probabilities[
                        category_index
                    ].item()
                )

                if (
                    category_probability
                    < category_threshold
                ):
                    continue

                with amp_context(
                    use_amp,
                    amp_dtype,
                ):

                    va = model.predict_va(
                        relation_repr,
                        category_index,
                    )

                va = (
                    va.float()
                    .cpu()
                    .tolist()
                )

                predictions.append(
                    {
                        "Aspect":
                            aspect.text,
                        "Opinion":
                            opinion.text,
                        "Category":
                            category_names[
                                category_index
                            ],
                        "VA":
                            (
                                f"{va[0]:.8f}"
                                f"#{va[1]:.8f}"
                            ),
                    }
                )

    return predictions


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    *,
    model,
    tokenizer,
    dev_by_domain,
    category_names,
    allowed_categories_by_domain,
    device,
    max_length,
    relation_threshold,
    category_threshold,
    use_amp,
    amp_dtype,
):

    model.eval()

    results = {}
    prediction_files = {}

    for domain, examples in (
        dev_by_domain.items()
    ):

        gold = []
        pred = []

        allowed_category_indices = (
            allowed_categories_by_domain[
                domain
            ]
        )

        for example in examples:

            quadruplets = predict_example(
                model=model,
                tokenizer=tokenizer,
                example=example,
                category_names=category_names,
                allowed_category_indices=(
                    allowed_category_indices
                ),
                device=device,
                max_length=max_length,
                relation_threshold=(
                    relation_threshold
                ),
                category_threshold=(
                    category_threshold
                ),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            pred.append(
                {
                    "ID":
                        example.record_id,
                    "Quadruplet":
                        quadruplets,
                }
            )

            gold.append(
                {
                    "ID":
                        example.record_id,
                    "Quadruplet":
                        [
                            {
                                "Aspect":
                                    q.aspect,
                                "Opinion":
                                    q.opinion,
                                "Category":
                                    q.category,
                                "VA":
                                    (
                                        f"{q.valence}"
                                        f"#{q.arousal}"
                                    ),
                            }
                            for q in (
                                example.quadruplets
                            )
                        ],
                }
            )

        metrics = evaluate_structured(
            gold,
            pred,
            task=3,
        )

        results[domain] = metrics
        prediction_files[domain] = pred

    macro_cf1 = statistics.mean(
        result["cF1"]
        for result in results.values()
    )

    return (
        macro_cf1,
        results,
        prediction_files,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        default="data/raw/dimabsa",
    )

    parser.add_argument(
        "--model-name",
        default=(
            "FacebookAI/"
            "xlm-roberta-base"
        ),
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
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
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
        "--relation-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--category-threshold",
        type=float,
        default=0.5,
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
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/phase2/task3_pair"
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    set_seed(args.seed)

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

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=True,
        )
    )

    raw_root = Path(
        args.raw_root
    )

    train_examples_by_domain = {}

    dev_by_domain = {}

    train_counts = {}

    # ------------------------------------------------------------
    # First pass:
    # load Task-3 training/dev examples WITHOUT encoding.
    #
    # We need all training examples first so that the category
    # vocabulary is defined strictly from training supervision.
    # ------------------------------------------------------------

    for domain in args.domains:

        train_records = (
            load_task_records(
                raw_root,
                task=3,
                language="eng",
                domain=domain,
                split="train",
            )
        )

        train_examples = (
            build_task3_examples(
                train_records
            )
        )

        train_examples_by_domain[
            domain
        ] = train_examples

        train_counts[domain] = (
            len(train_examples)
        )

        dev_records = (
            load_task_records(
                raw_root,
                task=3,
                language="eng",
                domain=domain,
                split="dev",
            )
        )

        dev_by_domain[domain] = (
            build_task3_examples(
                dev_records
            )
        )

    # ------------------------------------------------------------
    # Flat category vocabulary:
    # union over TRAIN only.
    #
    # Do not add dev-only labels.
    # ------------------------------------------------------------

    all_train_examples = [
        example
        for domain in args.domains
        for example in (
            train_examples_by_domain[
                domain
            ]
        )
    ]

    category_names = (
        collect_category_vocabulary(
            all_train_examples
        )
    )

    category_to_index = {
        category: index
        for index, category
        in enumerate(category_names)
    }

    # ------------------------------------------------------------
    # Compositional ontology vocabularies.
    # ------------------------------------------------------------

    category_parts = {
        category: split_category(
            category
        )
        for category in category_names
    }

    entity_names = tuple(
        sorted(
            {
                entity
                for entity, _
                in category_parts.values()
            }
        )
    )

    attribute_names = tuple(
        sorted(
            {
                attribute
                for _, attribute
                in category_parts.values()
            }
        )
    )

    entity_to_index = {
        entity: index
        for index, entity
        in enumerate(entity_names)
    }

    attribute_to_index = {
        attribute: index
        for index, attribute
        in enumerate(attribute_names)
    }

    category_to_entity_index = tuple(
        entity_to_index[
            category_parts[category][0]
        ]
        for category in category_names
    )

    category_to_attribute_index = tuple(
        attribute_to_index[
            category_parts[category][1]
        ]
        for category in category_names
    )

    # ------------------------------------------------------------
    # Domain masks.
    #
    # Laptop inference should not emit restaurant categories and
    # vice versa. Again, these masks are learned from TRAIN only.
    # ------------------------------------------------------------

    allowed_categories_by_domain = {}

    for domain in args.domains:

        domain_categories = set(
            collect_category_vocabulary(
                train_examples_by_domain[
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
            for category in category_names
            if category in domain_categories
        )

    # ------------------------------------------------------------
    # Valid Entity/Attribute spaces by domain.
    # ------------------------------------------------------------

    allowed_entities_by_domain = {}
    allowed_attributes_by_domain = {}

    for domain in args.domains:

        domain_category_indices = (
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
                        category_index
                    ]
                    for category_index
                    in domain_category_indices
                }
            )
        )

        allowed_attributes_by_domain[
            domain
        ] = tuple(
            sorted(
                {
                    category_to_attribute_index[
                        category_index
                    ]
                    for category_index
                    in domain_category_indices
                }
            )
        )

    # ------------------------------------------------------------
    # Second pass:
    # encode training examples using fixed training vocabulary.
    # ------------------------------------------------------------

    train_encoded = []

    for domain in args.domains:

        for example in (
            train_examples_by_domain[
                domain
            ]
        ):

            train_encoded.append(
                encode_task3_example(
                    example,
                    tokenizer,
                    category_to_index=(
                        category_to_index
                    ),
                    max_length=(
                        args.max_length
                    ),
                )
            )

    aspect_weights = (
        compute_bio_weights(
            train_encoded,
            "aspect_bio_labels",
        )
        .to(device)
    )

    opinion_weights = (
        compute_bio_weights(
            train_encoded,
            "opinion_bio_labels",
        )
        .to(device)
    )

    print()
    print("=" * 72)
    print("TASK 3 BASELINE")
    print("=" * 72)

    print(
        "train examples :",
        len(train_encoded),
    )

    print(
        "domain counts  :",
        train_counts,
    )

    print(
        "aspect weights :",
        aspect_weights.tolist(),
    )

    print(
        "opinion weights:",
        opinion_weights.tolist(),
    )

    print(
        "device         :",
        device,
    )

    print(
        "AMP dtype      :",
        amp_dtype,
    )

    if device.type == "cuda":
        print(
            "GPU            :",
            torch.cuda.get_device_name(0),
        )

    print("=" * 72)
    print()

    dataset = EncodedDataset(
        train_encoded
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=(
            args.train_batch_size
        ),
        shuffle=True,
        collate_fn=Collator(
            tokenizer
        ),
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
        generator=generator,
    )

    model = Task3PairModel(
        args.model_name,
        num_categories=len(
            category_names
        ),
        num_entities=len(
            entity_names
        ),
        num_attributes=len(
            attribute_names
        ),
        dropout=args.dropout,
    ).to(device)

    no_decay = (
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
            key in name
            for key in no_decay
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

    scheduler = (
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

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    output_dir = (
        Path(args.output_root)
        / "unambiguous"
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        output_dir
        / "run_config.json",
        {
            **vars(args),
            "git_commit":
            current_git_commit(),
            "torch_version":
            torch.__version__,
            "transformers_version":
            transformers.__version__,
            "aspect_weights":
            aspect_weights.tolist(),
            "opinion_weights":
            opinion_weights.tolist(),
            "train_examples":
            len(train_encoded),
            "train_domain_counts":
            train_counts,
            "num_categories":
            len(category_names),
            "num_entities":
            len(entity_names),
            "num_attributes":
            len(attribute_names),
            "entity_names":
            list(entity_names),
            "attribute_names":
            list(attribute_names),
            "hybrid_entity_loss_weight":
            0.5,
            "hybrid_attribute_loss_weight":
            0.5,
            "pair_category_loss_weight":
            0.5,
            "pair_category_decode":
            "flat_only",
            "hybrid_decode":
            "flat_only",
            "category_loss":
            "domain_masked_balanced_bce",
            "category_names":
            list(category_names),
            "allowed_categories_by_domain":
            {
                domain: [
                    category_names[index]
                    for index in indices
                ]
                for domain, indices
                in allowed_categories_by_domain.items()
            },
        },
    )

    train_iterator = iter(loader)

    best_cf1 = -1.0
    best_step = None

    history = []

    running = defaultdict(float)
    running_batches = 0

    checkpoint_path = (
        output_dir
        / "best_checkpoint.pt"
    )

    start_time = time.time()

    for step in range(
        1,
        args.max_steps + 1,
    ):

        model.train()

        try:
            batch = next(
                train_iterator
            )

        except StopIteration:
            train_iterator = iter(
                loader
            )

            batch = next(
                train_iterator
            )

        input_ids = (
            batch["input_ids"]
            .to(
                device,
                non_blocking=True,
            )
        )

        attention_mask = (
            batch[
                "attention_mask"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        aspect_labels = (
            batch["aspect_labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        opinion_labels = (
            batch["opinion_labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with amp_context(
            use_amp,
            amp_dtype,
        ):

            (
                hidden,
                aspect_logits,
                opinion_logits,
            ) = model.encode(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
            )

            aspect_loss = (
                F.cross_entropy(
                    aspect_logits.view(
                        -1,
                        3,
                    ),
                    aspect_labels.view(
                        -1
                    ),
                    weight=aspect_weights,
                    ignore_index=(
                        IGNORE_INDEX
                    ),
                )
            )

            opinion_loss = (
                F.cross_entropy(
                    opinion_logits.view(
                        -1,
                        3,
                    ),
                    opinion_labels.view(
                        -1
                    ),
                    weight=opinion_weights,
                    ignore_index=(
                        IGNORE_INDEX
                    ),
                )
            )

            (
                relation_loss,
                category_loss,
                pair_category_loss,
                entity_loss,
                attribute_loss,
                va_loss,
            ) = relation_category_va_loss(
                model=model,
                hidden=hidden,
                examples=(
                    batch["examples"]
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

            loss = (
                aspect_loss
                + opinion_loss
                + relation_loss
                + category_loss
                + 0.5 * pair_category_loss
                + 0.5 * entity_loss
                + 0.5 * attribute_loss
                + va_loss
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

            optimizer.step()

        scheduler.step()

        running["total"] += (
            loss.item()
        )

        running["aspect"] += (
            aspect_loss.item()
        )

        running["opinion"] += (
            opinion_loss.item()
        )

        running["relation"] += (
            relation_loss.item()
        )

        running["category"] += (
            category_loss.item()
        )

        running["pair_category"] += (
            pair_category_loss.item()
        )

        running["entity"] += (
            entity_loss.item()
        )

        running["attribute"] += (
            attribute_loss.item()
        )

        running["va"] += (
            va_loss.item()
        )

        running_batches += 1

        if (
            step
            % args.log_every
            == 0
        ):

            print(
                f"step={step:4d} "
                f"loss="
                f"{running['total']/running_batches:.4f} "
                f"A="
                f"{running['aspect']/running_batches:.4f} "
                f"O="
                f"{running['opinion']/running_batches:.4f} "
                f"R="
                f"{running['relation']/running_batches:.4f} "
                f"C="
                f"{running['category']/running_batches:.4f} "
                f"Pair="
                f"{running['pair_category']/running_batches:.4f} "
                f"E="
                f"{running['entity']/running_batches:.4f} "
                f"Attr="
                f"{running['attribute']/running_batches:.4f} "
                f"VA="
                f"{running['va']/running_batches:.4f} "
                f"lr="
                f"{scheduler.get_last_lr()[0]:.3e}"
            )

            running.clear()
            running_batches = 0

        should_eval = (
            step
            % args.eval_every
            == 0
            or step
            == args.max_steps
        )

        if not should_eval:
            continue

        (
            macro_cf1,
            metrics,
            predictions,
        ) = evaluate(
            model=model,
            tokenizer=tokenizer,
            dev_by_domain=(
                dev_by_domain
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
                args.relation_threshold
            ),
            category_threshold=(
                args.category_threshold
            ),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        history.append(
            {
                "step": step,
                "metrics": metrics,
            }
        )

        save_json(
            output_dir
            / "history.json",
            history,
        )

        print()
        print(
            f"[DEV @ step {step}]"
        )

        for domain in args.domains:

            m = metrics[domain]

            print(
                f"{domain:10s} "
                f"cF1={m['cF1']:.6f} "
                f"cP={m['cPrecision']:.6f} "
                f"cR={m['cRecall']:.6f} "
                f"TP={m['TP_structural']} "
                f"FP={m['FP']} "
                f"FN={m['FN']}"
            )

        print(
            f"macro      "
            f"cF1={macro_cf1:.6f}"
        )

        print()

        if macro_cf1 > best_cf1:

            best_cf1 = (
                macro_cf1
            )

            best_step = step

            torch.save(
                {
                    "model_state_dict":
                    model.state_dict(),
                    "step":
                    step,
                    "macro_cf1":
                    macro_cf1,
                    "seed":
                    args.seed,
                },
                checkpoint_path,
            )

            save_json(
                output_dir
                / "best_metrics.json",
                {
                    "step": step,
                    "metrics": metrics,
                },
            )

            pred_dir = (
                output_dir
                / "predictions"
            )

            pred_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for domain, rows in (
                predictions.items()
            ):

                with (
                    pred_dir
                    / (
                        f"eng_{domain}"
                        "_dev_task2.jsonl"
                    )
                ).open(
                    "w",
                    encoding="utf-8",
                ) as handle:

                    for row in rows:

                        handle.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            print(
                f"NEW BEST: "
                f"step={step}, "
                f"macro cF1="
                f"{macro_cf1:.6f}"
            )

            print()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    (
        final_macro_cf1,
        final_metrics,
        _,
    ) = evaluate(
        model=model,
        tokenizer=tokenizer,
        dev_by_domain=dev_by_domain,
        category_names=category_names,
        allowed_categories_by_domain=(
            allowed_categories_by_domain
        ),
        device=device,
        max_length=args.max_length,
        relation_threshold=(
            args.relation_threshold
        ),
        category_threshold=(
            args.category_threshold
        ),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )


    save_json(
        output_dir
        / "final_report.json",
        {
            "seed":
            args.seed,
            "best_step":
            best_step,
            "best_macro_cf1":
            best_cf1,
            "metrics":
            final_metrics,
            "elapsed_seconds":
            time.time()
            - start_time,
        },
    )

    print()
    print("=" * 72)
    print("FINAL BEST RESULT")
    print("=" * 72)

    print(
        "best step :",
        best_step,
    )

    print(
        "macro cF1 :",
        f"{best_cf1:.6f}",
    )

    for domain in args.domains:

        m = final_metrics[domain]

        print(
            f"{domain:10s} "
            f"cF1={m['cF1']:.6f} "
            f"cP={m['cPrecision']:.6f} "
            f"cR={m['cRecall']:.6f}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
