from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dimabsa.experiment_data import load_task_records
from dimabsa.task2_data import normalize_surface
from dimabsa.task3_data import (
    build_task3_examples,
    normalize_category,
    split_category,
)
from dimabsa.task3_pair_model import (
    Task3PairModel,
)
from dimabsa.task3_supervision import (
    _encoded_nodes,
    _ensure_null_node,
)

from train_task3_pair_frozen import (
    amp_context,
)


RUN_DIR = Path(
    "outputs/phase2/task3_pair_frozen_400/"
    "unambiguous/seed_42"
)

CHECKPOINT_PATH = (
    RUN_DIR
    / "final_pair_checkpoint.pt"
)

RAW_ROOT = Path(
    "data/raw/dimabsa"
)

DOMAINS = (
    "laptop",
    "restaurant",
)


config = json.loads(
    (RUN_DIR / "run_config.json").read_text()
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
)

category_names = tuple(
    config["category_names"]
)

entity_names = tuple(
    config["entity_names"]
)

attribute_names = tuple(
    config["attribute_names"]
)

category_to_index = {
    name: index
    for index, name
    in enumerate(category_names)
}

entity_to_index = {
    name: index
    for index, name
    in enumerate(entity_names)
}

attribute_to_index = {
    name: index
    for index, name
    in enumerate(attribute_names)
}


# ------------------------------------------------------------
# Category -> compositional factor maps
# ------------------------------------------------------------

category_entity_index = []
category_attribute_index = []

for category in category_names:

    entity, attribute = split_category(
        category
    )

    category_entity_index.append(
        entity_to_index[
            entity
        ]
    )

    category_attribute_index.append(
        attribute_to_index[
            attribute
        ]
    )


# ------------------------------------------------------------
# Domain-valid category spaces
# ------------------------------------------------------------

allowed_categories_by_domain = {
    domain: tuple(
        category_to_index[
            category
        ]
        for category in categories
    )
    for domain, categories
    in config[
        "allowed_categories_by_domain"
    ].items()
}


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

use_amp = (
    device.type == "cuda"
)

amp_dtype = None

if use_amp:

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )


tokenizer = AutoTokenizer.from_pretrained(
    config["model_name"],
    use_fast=True,
)


model = Task3PairModel(
    config["model_name"],
    num_categories=len(
        category_names
    ),
    num_entities=len(
        entity_names
    ),
    num_attributes=len(
        attribute_names
    ),
    dropout=config["dropout"],
).to(device)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


@torch.no_grad()
def evaluate_domain(
    domain: str,
) -> None:

    records = load_task_records(
        RAW_ROOT,
        task=3,
        language="eng",
        domain=domain,
        split="dev",
    )

    examples = build_task3_examples(
        records
    )

    valid_category_indices = (
        allowed_categories_by_domain[
            domain
        ]
    )

    valid_category_tensor = torch.tensor(
        valid_category_indices,
        dtype=torch.long,
        device=device,
    )

    pair_entity_tensor = torch.tensor(
        [
            category_entity_index[
                category_index
            ]
            for category_index
            in valid_category_indices
        ],
        dtype=torch.long,
        device=device,
    )

    pair_attribute_tensor = torch.tensor(
        [
            category_attribute_index[
                category_index
            ]
            for category_index
            in valid_category_indices
        ],
        dtype=torch.long,
        device=device,
    )

    global_to_local = {
        global_index: local_index
        for local_index, global_index
        in enumerate(
            valid_category_indices
        )
    }

    stats = defaultdict(
        float
    )

    seen_stats = defaultdict(
        float
    )

    unseen_complete_categories = defaultdict(
        int
    )

    positive_probs = []
    negative_probs = []

    ranks = []

    for example in examples:

        encoded = tokenizer(
            example.text,
            truncation=True,
            max_length=config["max_length"],
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        input_ids = torch.tensor(
            encoded["input_ids"],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        attention_mask = torch.tensor(
            encoded["attention_mask"],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        offsets = tuple(
            tuple(x)
            for x in encoded[
                "offset_mapping"
            ]
        )

        aspect_nodes = _encoded_nodes(
            _ensure_null_node(
                example.aspect_nodes
            ),
            offsets,
        )

        opinion_nodes = _encoded_nodes(
            _ensure_null_node(
                example.opinion_nodes
            ),
            offsets,
        )

        aspect_index = {
            normalize_surface(
                node.text
            ): index
            for index, node
            in enumerate(
                aspect_nodes
            )
        }

        opinion_index = {
            normalize_surface(
                node.text
            ): index
            for index, node
            in enumerate(
                opinion_nodes
            )
        }

        with amp_context(
            use_amp,
            amp_dtype,
        ):

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

        aspect_reprs = [
            model.pool_node(
                h,
                sentence_repr,
                node,
                node_type="aspect",
            )
            for node in aspect_nodes
        ]

        opinion_reprs = [
            model.pool_node(
                h,
                sentence_repr,
                node,
                node_type="opinion",
            )
            for node in opinion_nodes
        ]

        gold_by_pair = defaultdict(
            list
        )

        for q in example.quadruplets:

            pair = (
                normalize_surface(
                    q.aspect
                ),
                normalize_surface(
                    q.opinion
                ),
            )

            gold_by_pair[
                pair
            ].append(q)

        for (
            aspect_key,
            opinion_key,
        ), gold_items in (
            gold_by_pair.items()
        ):

            if (
                aspect_key
                not in aspect_index
                or opinion_key
                not in opinion_index
            ):
                continue

            ai = aspect_index[
                aspect_key
            ]

            oi = opinion_index[
                opinion_key
            ]

            (
                _,
                _,
                _,
                _,
                relation_repr,
            ) = model.score_pair(
                aspect_reprs[ai],
                opinion_reprs[oi],
            )

            pair_logits = (
                model.pair_category_logits(
                    relation_repr,
                    pair_entity_tensor,
                    pair_attribute_tensor,
                )
            )

            pair_probs = torch.sigmoid(
                pair_logits.float()
            )

            sorted_local_indices = (
                torch.argsort(
                    pair_probs,
                    descending=True,
                )
            )

            rank_lookup = {
                local_index.item():
                    rank + 1
                for rank, local_index
                in enumerate(
                    sorted_local_indices
                )
            }

            gold_local_indices = []

            for q in gold_items:

                category = normalize_category(
                    q.category
                )

                if (
                    category
                    not in category_to_index
                ):

                    unseen_complete_categories[
                        category
                    ] += 1

                    continue

                global_category_index = (
                    category_to_index[
                        category
                    ]
                )

                if (
                    global_category_index
                    not in global_to_local
                ):
                    continue

                local_category_index = (
                    global_to_local[
                        global_category_index
                    ]
                )

                gold_local_indices.append(
                    local_category_index
                )

                rank = rank_lookup[
                    local_category_index
                ]

                ranks.append(
                    rank
                )

                stats["total"] += 1

                seen_stats["total"] += 1

                if rank <= 1:
                    stats["top1"] += 1
                    seen_stats["top1"] += 1

                if rank <= 3:
                    stats["top3"] += 1
                    seen_stats["top3"] += 1

                if rank <= 5:
                    stats["top5"] += 1
                    seen_stats["top5"] += 1

                if rank <= 10:
                    stats["top10"] += 1
                    seen_stats["top10"] += 1

                positive_probs.append(
                    pair_probs[
                        local_category_index
                    ].item()
                )

            gold_local_set = set(
                gold_local_indices
            )

            for local_index in range(
                pair_probs.numel()
            ):

                if (
                    local_index
                    in gold_local_set
                ):
                    continue

                negative_probs.append(
                    pair_probs[
                        local_index
                    ].item()
                )

    total = int(
        stats["total"]
    )

    print()
    print("=" * 78)
    print(domain.upper())
    print("=" * 78)

    print(
        "seen category targets :",
        total,
    )

    if total:

        print(
            "pair top-1           :",
            f"{safe_ratio(int(stats['top1']), total):.4f}",
        )

        print(
            "pair top-3           :",
            f"{safe_ratio(int(stats['top3']), total):.4f}",
        )

        print(
            "pair top-5           :",
            f"{safe_ratio(int(stats['top5']), total):.4f}",
        )

        print(
            "pair top-10          :",
            f"{safe_ratio(int(stats['top10']), total):.4f}",
        )

        print(
            "mean gold rank       :",
            f"{sum(ranks)/len(ranks):.4f}",
        )

        print(
            "median gold rank     :",
            f"{sorted(ranks)[len(ranks)//2]}",
        )

    if positive_probs:

        print(
            "gold mean prob       :",
            f"{sum(positive_probs)/len(positive_probs):.4f}",
        )

    if negative_probs:

        print(
            "negative mean prob   :",
            f"{sum(negative_probs)/len(negative_probs):.4f}",
        )

    print()
    print(
        "unseen complete targets:",
        sum(
            unseen_complete_categories.values()
        ),
    )

    if unseen_complete_categories:

        for category, count in sorted(
            unseen_complete_categories.items()
        ):

            print(
                f"  {category:45s} "
                f"{count:4d}"
            )


print("=" * 78)
print("TASK 3 FROZEN PAIR-CATEGORY ORACLE DIAGNOSTIC")
print("=" * 78)

print(
    "checkpoint:",
    CHECKPOINT_PATH,
)

print(
    "pair training step:",
    checkpoint["step"],
)

for domain in DOMAINS:

    evaluate_domain(
        domain
    )
