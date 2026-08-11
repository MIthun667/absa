from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dimabsa.experiment_data import load_task_records
from dimabsa.task2_data import (
    NULL_TERM,
    SurfaceNode,
    normalize_surface,
)
from dimabsa.task2_supervision import (
    EncodedNode,
    node_token_indices,
)
from dimabsa.task3_data import (
    build_task3_examples,
    normalize_category,
)
from dimabsa.task3_model import Task3Model


ROOT = Path("data/raw/dimabsa")

RUN_DIR = Path(
    "outputs/phase2/task3_sqrt_balanced_pilot/"
    "unambiguous/seed_42"
)

CONFIG_PATH = RUN_DIR / "run_config.json"
CHECKPOINT_PATH = RUN_DIR / "best_checkpoint.pt"

DOMAINS = ("laptop", "restaurant")


def ensure_null(
    nodes: tuple[SurfaceNode, ...],
) -> tuple[SurfaceNode, ...]:

    if any(x.is_null for x in nodes):
        return nodes

    return nodes + (
        SurfaceNode(
            text=NULL_TERM,
            is_null=True,
            occurrences=(),
        ),
    )


def summary(values):
    if not values:
        return {
            "n": 0,
        }

    x = torch.tensor(
        values,
        dtype=torch.float32,
    )

    return {
        "n": len(values),
        "mean": x.mean().item(),
        "median": x.median().item(),
        "p10": torch.quantile(
            x,
            0.10,
        ).item(),
        "p90": torch.quantile(
            x,
            0.90,
        ).item(),
        "gt_030": (
            (x >= 0.30)
            .float()
            .mean()
            .item()
        ),
        "gt_050": (
            (x >= 0.50)
            .float()
            .mean()
            .item()
        ),
    }


config = json.loads(
    CONFIG_PATH.read_text()
)

category_names = tuple(
    config["category_names"]
)

category_to_index = {
    category: i
    for i, category
    in enumerate(category_names)
}

allowed_by_domain = {}

for domain, names in (
    config[
        "allowed_categories_by_domain"
    ].items()
):

    allowed_by_domain[domain] = tuple(
        category_to_index[name]
        for name in names
    )


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    config["model_name"],
    use_fast=True,
)

model = Task3Model(
    config["model_name"],
    num_categories=len(category_names),
    dropout=config["dropout"],
).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("=" * 72)
print("TASK 3 ORACLE-HEAD DIAGNOSTIC")
print("=" * 72)

print("checkpoint step :", checkpoint["step"])
print("num categories  :", len(category_names))
print("device          :", device)
print()


all_results = {}


for domain in DOMAINS:

    records = load_task_records(
        ROOT,
        task=3,
        language="eng",
        domain=domain,
        split="dev",
    )

    examples = build_task3_examples(
        records
    )

    relation_positive = []
    relation_negative = []

    category_positive = []
    category_negative = []

    category_ranks = []

    known_gold_categories = 0
    unseen_gold_categories = 0

    positive_pairs = 0

    with torch.no_grad():

        for example in examples:

            encoded = tokenizer(
                example.text,
                add_special_tokens=True,
                truncation=True,
                max_length=config[
                    "max_length"
                ],
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            offsets = [
                tuple(x)
                for x in encoded[
                    "offset_mapping"
                ][0].tolist()
            ]

            input_ids = encoded[
                "input_ids"
            ].to(device)

            attention_mask = encoded[
                "attention_mask"
            ].to(device)

            (
                hidden,
                _,
                _,
            ) = model.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            h = hidden[0]

            sentence_repr = h[0]

            raw_aspects = ensure_null(
                example.aspect_nodes
            )

            raw_opinions = ensure_null(
                example.opinion_nodes
            )

            aspects = [
                EncodedNode(
                    text=node.text,
                    is_null=node.is_null,
                    token_indices=(
                        node_token_indices(
                            node,
                            offsets,
                        )
                    ),
                )
                for node in raw_aspects
            ]

            opinions = [
                EncodedNode(
                    text=node.text,
                    is_null=node.is_null,
                    token_indices=(
                        node_token_indices(
                            node,
                            offsets,
                        )
                    ),
                )
                for node in raw_opinions
            ]

            aspect_reprs = [
                model.pool_node(
                    h,
                    sentence_repr,
                    node,
                    node_type="aspect",
                )
                for node in aspects
            ]

            opinion_reprs = [
                model.pool_node(
                    h,
                    sentence_repr,
                    node,
                    node_type="opinion",
                )
                for node in opinions
            ]

            gold_by_pair = defaultdict(set)

            for q in example.quadruplets:

                pair = (
                    normalize_surface(
                        q.aspect
                    ),
                    normalize_surface(
                        q.opinion
                    ),
                )

                category = (
                    normalize_category(
                        q.category
                    )
                )

                gold_by_pair[pair].add(
                    category
                )

            for ai, aspect in enumerate(
                aspects
            ):

                for oi, opinion in enumerate(
                    opinions
                ):

                    pair = (
                        normalize_surface(
                            aspect.text
                        ),
                        normalize_surface(
                            opinion.text
                        ),
                    )

                    (
                        relation_logit,
                        category_logits,
                        _,
                    ) = model.score_pair(
                        aspect_reprs[ai],
                        opinion_reprs[oi],
                    )

                    rp = (
                        torch.sigmoid(
                            relation_logit
                        )
                        .float()
                        .item()
                    )

                    if pair in gold_by_pair:

                        relation_positive.append(
                            rp
                        )

                        positive_pairs += 1

                        probs = (
                            torch.sigmoid(
                                category_logits
                            )
                            .float()
                            .cpu()
                        )

                        gold_categories = (
                            gold_by_pair[pair]
                        )

                        known_indices = []

                        for category in (
                            gold_categories
                        ):

                            if (
                                category
                                not in category_to_index
                            ):
                                unseen_gold_categories += 1
                                continue

                            ci = (
                                category_to_index[
                                    category
                                ]
                            )

                            known_indices.append(
                                ci
                            )

                            known_gold_categories += 1

                            category_positive.append(
                                probs[ci].item()
                            )

                            allowed = (
                                allowed_by_domain[
                                    domain
                                ]
                            )

                            allowed_scores = [
                                (
                                    probs[j].item(),
                                    j,
                                )
                                for j in allowed
                            ]

                            allowed_scores.sort(
                                reverse=True
                            )

                            rank = next(
                                rank
                                for rank, (
                                    _,
                                    index,
                                )
                                in enumerate(
                                    allowed_scores,
                                    start=1,
                                )
                                if index == ci
                            )

                            category_ranks.append(
                                rank
                            )

                        known_set = set(
                            known_indices
                        )

                        for ci in (
                            allowed_by_domain[
                                domain
                            ]
                        ):

                            if ci in known_set:
                                continue

                            category_negative.append(
                                probs[ci].item()
                            )

                    else:

                        relation_negative.append(
                            rp
                        )

    print()
    print("=" * 72)
    print(domain.upper())
    print("=" * 72)

    print(
        "positive AO pairs        :",
        positive_pairs,
    )

    print(
        "known gold categories    :",
        known_gold_categories,
    )

    print(
        "unseen gold categories   :",
        unseen_gold_categories,
    )

    print()
    print("RELATION POSITIVE")
    print(summary(relation_positive))

    print()
    print("RELATION NEGATIVE")
    print(summary(relation_negative))

    print()
    print("CATEGORY POSITIVE")
    print(summary(category_positive))

    print()
    print("CATEGORY NEGATIVE")
    print(summary(category_negative))

    if category_ranks:

        ranks = torch.tensor(
            category_ranks,
            dtype=torch.float32,
        )

        print()
        print("CATEGORY RANK")

        print(
            "mean rank               :",
            ranks.mean().item(),
        )

        print(
            "gold category top-1     :",
            (ranks <= 1)
            .float()
            .mean()
            .item(),
        )

        print(
            "gold category top-3     :",
            (ranks <= 3)
            .float()
            .mean()
            .item(),
        )

        print(
            "gold category top-5     :",
            (ranks <= 5)
            .float()
            .mean()
            .item(),
        )
