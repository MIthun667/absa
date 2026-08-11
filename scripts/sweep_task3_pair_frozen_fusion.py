from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dimabsa.experiment_data import load_task_records
from dimabsa.task3_data import (
    build_task3_examples,
    split_category,
)
from dimabsa.task3_pair_model import (
    Task3PairModel,
)

from train_task3_pair_frozen import (
    add_null_node,
    amp_context,
    predicted_nodes_from_bio,
)

from dimabsa.metrics import evaluate_structured


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

RELATION_THRESHOLD = 0.50

BETAS = (
    0.00,
    0.40,
)

CATEGORY_THRESHOLDS = (
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
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
    category: index
    for index, category
    in enumerate(category_names)
}

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


# ------------------------------------------------------------
# Category -> Entity/Attribute maps
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


domain_global_to_local = {
    domain: {
        global_index: local_index
        for local_index, global_index
        in enumerate(
            allowed_categories_by_domain[
                domain
            ]
        )
    }
    for domain in DOMAINS
}


pair_entity_indices_by_domain = {
    domain: torch.tensor(
        [
            category_entity_index[
                category_index
            ]
            for category_index
            in allowed_categories_by_domain[
                domain
            ]
        ],
        dtype=torch.long,
    )
    for domain in DOMAINS
}


pair_attribute_indices_by_domain = {
    domain: torch.tensor(
        [
            category_attribute_index[
                category_index
            ]
            for category_index
            in allowed_categories_by_domain[
                domain
            ]
        ],
        dtype=torch.long,
    )
    for domain in DOMAINS
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


for domain in DOMAINS:

    pair_entity_indices_by_domain[
        domain
    ] = (
        pair_entity_indices_by_domain[
            domain
        ].to(device)
    )

    pair_attribute_indices_by_domain[
        domain
    ] = (
        pair_attribute_indices_by_domain[
            domain
        ].to(device)
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


dev_by_domain = {}

for domain in DOMAINS:

    records = load_task_records(
        RAW_ROOT,
        task=3,
        language="eng",
        domain=domain,
        split="dev",
    )

    dev_by_domain[
        domain
    ] = build_task3_examples(
        records
    )


def gold_rows(
    examples,
):

    rows = []

    for example in examples:

        rows.append(
            {
                "ID":
                    example.record_id,
                "Quadruplet":
                    [
                        {
                            "Aspect":
                                quadruplet.aspect,
                            "Opinion":
                                quadruplet.opinion,
                            "Category":
                                quadruplet.category,
                            "VA":
                                (
                                    f"{quadruplet.valence}"
                                    f"#{quadruplet.arousal}"
                                ),
                        }
                        for quadruplet
                        in example.quadruplets
                    ],
            }
        )

    return rows


gold_by_domain = {
    domain: gold_rows(
        dev_by_domain[
            domain
        ]
    )
    for domain in DOMAINS
}


@torch.no_grad()
def predict_example(
    example,
    domain,
    beta,
    category_threshold,
):

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
            attention_mask=attention_mask,
        )

    aspects = predicted_nodes_from_bio(
        labels=(
            aspect_logits[0]
            .argmax(-1)
            .cpu()
            .tolist()
        ),
        offsets=offsets,
        text=example.text,
    )

    opinions = predicted_nodes_from_bio(
        labels=(
            opinion_logits[0]
            .argmax(-1)
            .cpu()
            .tolist()
        ),
        offsets=offsets,
        text=example.text,
    )

    aspects = add_null_node(
        aspects
    )

    opinions = add_null_node(
        opinions
    )

    hidden_example = hidden[0]
    sentence_repr = hidden_example[0]

    aspect_reprs = [
        model.pool_node(
            hidden_example,
            sentence_repr,
            node,
            node_type="aspect",
        )
        for node in aspects
    ]

    opinion_reprs = [
        model.pool_node(
            hidden_example,
            sentence_repr,
            node,
            node_type="opinion",
        )
        for node in opinions
    ]

    predictions = []

    pair_entity_indices = (
        pair_entity_indices_by_domain[
            domain
        ]
    )

    pair_attribute_indices = (
        pair_attribute_indices_by_domain[
            domain
        ]
    )

    for aspect_index, aspect in enumerate(
        aspects
    ):

        for opinion_index, opinion in enumerate(
            opinions
        ):

            (
                relation_logit,
                flat_category_logits,
                _,
                _,
                relation_repr,
            ) = model.score_pair(
                aspect_reprs[
                    aspect_index
                ],
                opinion_reprs[
                    opinion_index
                ],
            )

            relation_probability = (
                torch.sigmoid(
                    relation_logit.float()
                ).item()
            )

            if (
                relation_probability
                < RELATION_THRESHOLD
            ):
                continue

            # ----------------------------------------------------
            # Pair-category branch over the SAME domain-valid
            # category inventory used by the flat branch.
            # ----------------------------------------------------

            pair_logits = (
                model.pair_category_logits(
                    relation_repr,
                    pair_entity_indices,
                    pair_attribute_indices,
                )
            )

            pair_probabilities = (
                torch.sigmoid(
                    pair_logits.float()
                )
            )

            for global_category_index in (
                allowed_categories_by_domain[
                    domain
                ]
            ):

                local_category_index = (
                    domain_global_to_local[
                        domain
                    ][
                        global_category_index
                    ]
                )

                flat_probability = (
                    torch.sigmoid(
                        flat_category_logits[
                            global_category_index
                        ].float()
                    ).item()
                )

                pair_probability = (
                    pair_probabilities[
                        local_category_index
                    ].item()
                )

                # ================================================
                # SAFE probability-space residual mixture.
                #
                # beta = 0.0 exactly reproduces flat-only decoding.
                # ================================================

                fused_probability = (
                    (1.0 - beta)
                    * flat_probability
                    + beta
                    * pair_probability
                )

                if (
                    fused_probability
                    < category_threshold
                ):
                    continue

                va = model.predict_va(
                    relation_repr,
                    global_category_index,
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
                                global_category_index
                            ],
                        "VA":
                            (
                                f"{va[0]:.8f}"
                                f"#{va[1]:.8f}"
                            ),
                    }
                )

    return predictions


results = []


print("=" * 120)
print(
    "TASK 3 FROZEN FLAT + PAIR PROBABILITY-FUSION SWEEP"
)
print("=" * 120)

print(
    "checkpoint step :",
    checkpoint["step"],
)

print(
    "relation thresh :",
    RELATION_THRESHOLD,
)

print()


for beta in BETAS:

    for category_threshold in (
        CATEGORY_THRESHOLDS
    ):

        metrics = {}

        for domain in DOMAINS:

            prediction_rows = []

            for example in (
                dev_by_domain[
                    domain
                ]
            ):

                prediction_rows.append(
                    {
                        "ID":
                            example.record_id,
                        "Quadruplet":
                            predict_example(
                                example,
                                domain,
                                beta,
                                category_threshold,
                            ),
                    }
                )

            metrics[
                domain
            ] = evaluate_structured(
                gold_by_domain[
                    domain
                ],
                prediction_rows,
                task=3,
            )

        macro_cf1 = (
            sum(
                metrics[
                    domain
                ]["cF1"]
                for domain in DOMAINS
            )
            / len(DOMAINS)
        )

        result = {
            "beta":
                beta,
            "category_threshold":
                category_threshold,
            "macro_cf1":
                macro_cf1,
            "metrics":
                metrics,
        }

        results.append(
            result
        )

        laptop = metrics[
            "laptop"
        ]

        restaurant = metrics[
            "restaurant"
        ]

        print(
            f"b={beta:4.2f} "
            f"C={category_threshold:4.2f} | "
            f"macro={macro_cf1:.6f} | "
            f"L={laptop['cF1']:.6f} "
            f"(TP={laptop['TP_structural']}, "
            f"FP={laptop['FP']}) | "
            f"R={restaurant['cF1']:.6f} "
            f"(TP={restaurant['TP_structural']}, "
            f"FP={restaurant['FP']})"
        )


best_macro = max(
    results,
    key=lambda result:
        result[
            "macro_cf1"
        ],
)


# ------------------------------------------------------------
# Also identify the best setting that does NOT degrade either
# domain relative to beta=0 baseline by more than tiny tolerance.
# ------------------------------------------------------------

baseline_candidates = [
    result
    for result in results
    if result["beta"] == 0.0
]

baseline = max(
    baseline_candidates,
    key=lambda result:
        result[
            "macro_cf1"
        ],
)

baseline_laptop = (
    baseline[
        "metrics"
    ][
        "laptop"
    ][
        "cF1"
    ]
)

baseline_restaurant = (
    baseline[
        "metrics"
    ][
        "restaurant"
    ][
        "cF1"
    ]
)

SAFE_TOLERANCE = 0.002

safe_results = [
    result
    for result in results
    if (
        result[
            "metrics"
        ][
            "laptop"
        ][
            "cF1"
        ]
        >= (
            baseline_laptop
            - SAFE_TOLERANCE
        )
        and
        result[
            "metrics"
        ][
            "restaurant"
        ][
            "cF1"
        ]
        >= (
            baseline_restaurant
            - SAFE_TOLERANCE
        )
    )
]

best_safe = (
    max(
        safe_results,
        key=lambda result:
            result[
                "macro_cf1"
            ],
    )
    if safe_results
    else None
)


print()
print("=" * 120)
print("BEST MACRO")
print("=" * 120)

print(
    "beta      :",
    best_macro[
        "beta"
    ],
)

print(
    "threshold :",
    best_macro[
        "category_threshold"
    ],
)

print(
    "macro cF1 :",
    f"{best_macro['macro_cf1']:.6f}",
)

for domain in DOMAINS:

    metric = best_macro[
        "metrics"
    ][
        domain
    ]

    print(
        f"{domain:10s} "
        f"cF1={metric['cF1']:.6f} "
        f"cP={metric['cPrecision']:.6f} "
        f"cR={metric['cRecall']:.6f} "
        f"TP={metric['TP_structural']} "
        f"FP={metric['FP']} "
        f"FN={metric['FN']}"
    )


print()
print("=" * 120)
print("BASELINE beta=0")
print("=" * 120)

print(
    "threshold :",
    baseline[
        "category_threshold"
    ],
)

print(
    "macro cF1 :",
    f"{baseline['macro_cf1']:.6f}",
)

print(
    "laptop    :",
    f"{baseline_laptop:.6f}",
)

print(
    "restaurant:",
    f"{baseline_restaurant:.6f}",
)


if best_safe is not None:

    print()
    print("=" * 120)
    print("BEST SAFE BOTH-DOMAINS SETTING")
    print("=" * 120)

    print(
        "beta      :",
        best_safe[
            "beta"
        ],
    )

    print(
        "threshold :",
        best_safe[
            "category_threshold"
        ],
    )

    print(
        "macro cF1 :",
        f"{best_safe['macro_cf1']:.6f}",
    )

    for domain in DOMAINS:

        metric = best_safe[
            "metrics"
        ][
            domain
        ]

        print(
            f"{domain:10s} "
            f"cF1={metric['cF1']:.6f}"
        )


output_path = (
    RUN_DIR
    / "frozen_pair_probability_fusion_sweep.json"
)

output_path.write_text(
    json.dumps(
        results,
        indent=2,
    )
)

print()
print(
    "Saved:",
    output_path,
)
