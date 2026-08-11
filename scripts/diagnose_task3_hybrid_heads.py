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
from dimabsa.task3_hybrid_model import (
    Task3HybridModel,
)
from dimabsa.task3_supervision import (
    _encoded_nodes,
    _ensure_null_node,
)

from train_task3_hybrid import (
    amp_context,
)


RUN_DIR = Path(
    "outputs/phase2/task3_hybrid_aux_1200/"
    "unambiguous/seed_42"
)

RAW_ROOT = Path("data/raw/dimabsa")

DOMAINS = (
    "laptop",
    "restaurant",
)


config = json.loads(
    (RUN_DIR / "run_config.json").read_text()
)

checkpoint = torch.load(
    RUN_DIR / "best_checkpoint.pt",
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

category_set = set(
    category_names
)

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
# Domain-valid ontology spaces.
#
# Entity/Attribute training is domain-masked, so diagnostics
# must rank only labels that are valid for that domain.
# ------------------------------------------------------------

allowed_categories_by_domain = {
    domain: tuple(categories)
    for domain, categories
    in config[
        "allowed_categories_by_domain"
    ].items()
}

allowed_entities_by_domain = {}
allowed_attributes_by_domain = {}

for domain, categories in (
    allowed_categories_by_domain.items()
):

    domain_entities = set()
    domain_attributes = set()

    for category in categories:

        entity, attribute = split_category(
            category
        )

        domain_entities.add(
            entity_to_index[entity]
        )

        domain_attributes.add(
            attribute_to_index[attribute]
        )

    allowed_entities_by_domain[
        domain
    ] = tuple(
        sorted(domain_entities)
    )

    allowed_attributes_by_domain[
        domain
    ] = tuple(
        sorted(domain_attributes)
    )


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


model = Task3HybridModel(
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
    checkpoint["model_state_dict"]
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

    stats = {
        "all": defaultdict(float),
        "seen_category": defaultdict(float),
        "unseen_category": defaultdict(float),
    }

    unseen_categories = defaultdict(
        int
    )

    unseen_entity_component = defaultdict(
        int
    )

    unseen_attribute_component = defaultdict(
        int
    )

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

        aspect_nodes_raw = (
            _ensure_null_node(
                example.aspect_nodes
            )
        )

        opinion_nodes_raw = (
            _ensure_null_node(
                example.opinion_nodes
            )
        )

        aspect_nodes = _encoded_nodes(
            aspect_nodes_raw,
            offsets,
        )

        opinion_nodes = _encoded_nodes(
            opinion_nodes_raw,
            offsets,
        )

        aspect_index = {
            normalize_surface(
                node.text
            ): index
            for index, node
            in enumerate(aspect_nodes)
        }

        opinion_index = {
            normalize_surface(
                node.text
            ): index
            for index, node
            in enumerate(opinion_nodes)
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

        # --------------------------------------------------------
        # Group original gold quadruplets by AO pair.
        #
        # Crucially, we do NOT require the full category to exist
        # in the flat training vocabulary.
        # --------------------------------------------------------

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
                entity_logits,
                attribute_logits,
                _,
            ) = model.score_pair(
                aspect_reprs[ai],
                opinion_reprs[oi],
            )

            entity_probs = torch.sigmoid(
                entity_logits.float()
            )

            attribute_probs = torch.sigmoid(
                attribute_logits.float()
            )

            # ----------------------------------------------------
            # Rank ONLY within the ontology valid for this domain.
            # This matches the domain-masked training objective.
            # ----------------------------------------------------

            valid_entity_indices = torch.tensor(
                allowed_entities_by_domain[
                    domain
                ],
                dtype=torch.long,
                device=device,
            )

            valid_attribute_indices = torch.tensor(
                allowed_attributes_by_domain[
                    domain
                ],
                dtype=torch.long,
                device=device,
            )

            valid_entity_probs = (
                entity_probs.index_select(
                    0,
                    valid_entity_indices,
                )
            )

            valid_attribute_probs = (
                attribute_probs.index_select(
                    0,
                    valid_attribute_indices,
                )
            )

            entity_local_rank = torch.argsort(
                valid_entity_probs,
                descending=True,
            )

            attribute_local_rank = torch.argsort(
                valid_attribute_probs,
                descending=True,
            )

            entity_rank = (
                valid_entity_indices.index_select(
                    0,
                    entity_local_rank,
                )
            )

            attribute_rank = (
                valid_attribute_indices.index_select(
                    0,
                    attribute_local_rank,
                )
            )

            entity_top1_prediction = (
                entity_rank[0].item()
            )

            attribute_top1_prediction = (
                attribute_rank[0].item()
            )

            entity_top3_predictions = set(
                entity_rank[:3]
                .cpu()
                .tolist()
            )

            attribute_top3_predictions = set(
                attribute_rank[:3]
                .cpu()
                .tolist()
            )

            # ----------------------------------------------------
            # One evaluation item per original quadruplet/category.
            # ----------------------------------------------------

            for q in gold_items:

                category = (
                    normalize_category(
                        q.category
                    )
                )

                entity, attribute = (
                    split_category(
                        category
                    )
                )

                category_is_seen = (
                    category
                    in category_set
                )

                if not category_is_seen:

                    unseen_categories[
                        category
                    ] += 1

                # ------------------------------------------------
                # Full category may be unseen while both of its
                # components remain predictable.
                # ------------------------------------------------

                if (
                    entity
                    not in entity_to_index
                ):

                    unseen_entity_component[
                        entity
                    ] += 1

                    continue

                if (
                    attribute
                    not in attribute_to_index
                ):

                    unseen_attribute_component[
                        attribute
                    ] += 1

                    continue

                ei = entity_to_index[
                    entity
                ]

                ati = attribute_to_index[
                    attribute
                ]

                group_names = [
                    "all",
                    (
                        "seen_category"
                        if category_is_seen
                        else "unseen_category"
                    ),
                ]

                for group_name in (
                    group_names
                ):

                    s = stats[
                        group_name
                    ]

                    s["total"] += 1

                    s[
                        "entity_prob_sum"
                    ] += (
                        entity_probs[
                            ei
                        ].item()
                    )

                    s[
                        "attribute_prob_sum"
                    ] += (
                        attribute_probs[
                            ati
                        ].item()
                    )

                    if (
                        entity_top1_prediction
                        == ei
                    ):

                        s[
                            "entity_top1"
                        ] += 1

                    if (
                        ei
                        in entity_top3_predictions
                    ):

                        s[
                            "entity_top3"
                        ] += 1

                    if (
                        attribute_top1_prediction
                        == ati
                    ):

                        s[
                            "attribute_top1"
                        ] += 1

                    if (
                        ati
                        in attribute_top3_predictions
                    ):

                        s[
                            "attribute_top3"
                        ] += 1

                    if (
                        entity_top1_prediction
                        == ei
                        and
                        attribute_top1_prediction
                        == ati
                    ):

                        s[
                            "joint_top1"
                        ] += 1

                    if (
                        ei
                        in entity_top3_predictions
                        and
                        ati
                        in attribute_top3_predictions
                    ):

                        s[
                            "joint_top3"
                        ] += 1

    print()
    print("=" * 78)
    print(domain.upper())
    print("=" * 78)

    for group_name, title in (
        (
            "all",
            "ALL COMPOSITIONALLY EVALUABLE TARGETS",
        ),
        (
            "seen_category",
            "SEEN COMPLETE CATEGORIES",
        ),
        (
            "unseen_category",
            "UNSEEN COMPLETE CATEGORIES",
        ),
    ):

        s = stats[
            group_name
        ]

        total = int(
            s["total"]
        )

        print()
        print(title)
        print("-" * 78)

        print(
            "targets                  :",
            total,
        )

        if total == 0:
            continue

        print(
            "entity top-1             :",
            f"{safe_ratio(int(s['entity_top1']), total):.4f}",
        )

        print(
            "entity top-3             :",
            f"{safe_ratio(int(s['entity_top3']), total):.4f}",
        )

        print(
            "attribute top-1          :",
            f"{safe_ratio(int(s['attribute_top1']), total):.4f}",
        )

        print(
            "attribute top-3          :",
            f"{safe_ratio(int(s['attribute_top3']), total):.4f}",
        )

        print(
            "joint E+A top-1          :",
            f"{safe_ratio(int(s['joint_top1']), total):.4f}",
        )

        print(
            "joint E+A top-3          :",
            f"{safe_ratio(int(s['joint_top3']), total):.4f}",
        )

        print(
            "gold entity mean prob    :",
            f"{s['entity_prob_sum']/total:.4f}",
        )

        print(
            "gold attribute mean prob :",
            f"{s['attribute_prob_sum']/total:.4f}",
        )

    print()
    print("UNSEEN COMPLETE CATEGORY COUNTS")
    print("-" * 78)

    if unseen_categories:

        for category, count in sorted(
            unseen_categories.items()
        ):

            print(
                f"{category:45s} "
                f"{count:4d}"
            )

    else:

        print("none")

    print()
    print("UNSEEN COMPONENTS")
    print("-" * 78)

    print(
        "unseen entities   :",
        dict(
            sorted(
                unseen_entity_component.items()
            )
        ),
    )

    print(
        "unseen attributes :",
        dict(
            sorted(
                unseen_attribute_component.items()
            )
        ),
    )


print("=" * 78)
print("TASK 3 HYBRID ORACLE COMPOSITIONAL-HEAD DIAGNOSTIC")
print("=" * 78)

print(
    "checkpoint step :",
    checkpoint["step"]
)

print(
    "train categories:",
    len(category_names),
)

print(
    "train entities  :",
    len(entity_names),
)

print(
    "train attributes:",
    len(attribute_names),
)

for domain in DOMAINS:

    evaluate_domain(
        domain
    )
