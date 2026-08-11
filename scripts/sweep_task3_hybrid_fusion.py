from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dimabsa.experiment_data import load_task_records
from dimabsa.task2_data import NULL_TERM
from dimabsa.task3_data import (
    build_task3_examples,
    split_category,
)
from dimabsa.task3_hybrid_model import (
    Task3HybridModel,
)

from train_task3_hybrid import (
    add_null_node,
    amp_context,
    predicted_nodes_from_bio,
)

from dimabsa.metrics import evaluate_structured


RUN_DIR = Path(
    "outputs/phase2/task3_hybrid_aux_400/"
    "unambiguous/seed_42"
)

RAW_ROOT = Path("data/raw/dimabsa")

DOMAINS = (
    "laptop",
    "restaurant",
)

ALPHAS = (
    0.0,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
)

CATEGORY_THRESHOLDS = (
    0.70,
    0.75,
    0.80,
    0.84,
    0.88,
    0.90,
)

RELATION_THRESHOLD = 0.50


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

category_to_index = {
    x: i
    for i, x in enumerate(category_names)
}

entity_to_index = {
    x: i
    for i, x in enumerate(entity_names)
}

attribute_to_index = {
    x: i
    for i, x in enumerate(attribute_names)
}


category_entity_index = []
category_attribute_index = []

for category in category_names:

    entity, attribute = split_category(
        category
    )

    category_entity_index.append(
        entity_to_index[entity]
    )

    category_attribute_index.append(
        attribute_to_index[attribute]
    )


allowed_categories_by_domain = {
    domain: tuple(
        category_to_index[x]
        for x in categories
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

use_amp = device.type == "cuda"

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
    num_categories=len(category_names),
    num_entities=len(entity_names),
    num_attributes=len(attribute_names),
    dropout=config["dropout"],
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
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

    dev_by_domain[domain] = (
        build_task3_examples(records)
    )


@torch.no_grad()
def predict(
    example,
    domain,
    alpha,
    category_threshold,
):

    encoded = tokenizer(
        example.text,
        add_special_tokens=True,
        truncation=True,
        max_length=config["max_length"],
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

    aspects = add_null_node(aspects)
    opinions = add_null_node(opinions)

    h = hidden[0]
    sentence_repr = h[0]

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

    predictions = []

    for ai, aspect in enumerate(aspects):

        for oi, opinion in enumerate(opinions):

            (
                relation_logit,
                flat_logits,
                entity_logits,
                attribute_logits,
                relation_repr,
            ) = model.score_pair(
                aspect_reprs[ai],
                opinion_reprs[oi],
            )

            relation_probability = (
                torch.sigmoid(
                    relation_logit
                ).item()
            )

            if (
                relation_probability
                < RELATION_THRESHOLD
            ):
                continue

            for ci in (
                allowed_categories_by_domain[
                    domain
                ]
            ):

                ei = category_entity_index[ci]
                ati = category_attribute_index[ci]

                # -------------------------------------------------
                # Logit-space fusion.
                #
                # alpha=0 gives the original flat decoder exactly.
                # -------------------------------------------------

                fused_logit = (
                    flat_logits[ci]
                    + alpha
                    * entity_logits[ei]
                    + alpha
                    * attribute_logits[ati]
                )

                probability = (
                    torch.sigmoid(
                        fused_logit
                    ).item()
                )

                if (
                    probability
                    < category_threshold
                ):
                    continue

                va = model.predict_va(
                    relation_repr,
                    ci,
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
                            category_names[ci],
                        "VA":
                            (
                                f"{va[0]:.8f}"
                                f"#{va[1]:.8f}"
                            ),
                    }
                )

    return predictions


def gold_rows(examples):

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

    return rows


gold_by_domain = {
    domain: gold_rows(
        dev_by_domain[domain]
    )
    for domain in DOMAINS
}


results = []

print("=" * 120)
print("TASK 3 HYBRID FLAT + COMPOSITIONAL FUSION SWEEP")
print("=" * 120)
print("checkpoint step:", checkpoint["step"])
print()


for alpha in ALPHAS:

    for threshold in CATEGORY_THRESHOLDS:

        metrics = {}

        for domain in DOMAINS:

            pred = []

            for example in (
                dev_by_domain[domain]
            ):

                pred.append(
                    {
                        "ID":
                            example.record_id,
                        "Quadruplet":
                            predict(
                                example,
                                domain,
                                alpha,
                                threshold,
                            ),
                    }
                )

            metrics[domain] = (
                evaluate_structured(
                    gold_by_domain[domain],
                    pred,
                    task=3,
                )
            )

        macro = sum(
            metrics[d]["cF1"]
            for d in DOMAINS
        ) / len(DOMAINS)

        results.append(
            {
                "alpha":
                    alpha,
                "threshold":
                    threshold,
                "macro_cf1":
                    macro,
                "metrics":
                    metrics,
            }
        )

        l = metrics["laptop"]
        r = metrics["restaurant"]

        print(
            f"a={alpha:4.2f} "
            f"C={threshold:4.2f} | "
            f"macro={macro:.6f} | "
            f"L={l['cF1']:.6f} "
            f"(TP={l['TP_structural']}, "
            f"FP={l['FP']}) | "
            f"R={r['cF1']:.6f} "
            f"(TP={r['TP_structural']}, "
            f"FP={r['FP']})"
        )


best = max(
    results,
    key=lambda x:
        x["macro_cf1"],
)

print()
print("=" * 120)
print("BEST HYBRID FUSION")
print("=" * 120)

print(
    "alpha     :",
    best["alpha"],
)

print(
    "threshold :",
    best["threshold"],
)

print(
    "macro cF1 :",
    f"{best['macro_cf1']:.6f}",
)

for domain in DOMAINS:

    m = best["metrics"][domain]

    print(
        f"{domain:10s} "
        f"cF1={m['cF1']:.6f} "
        f"cP={m['cPrecision']:.6f} "
        f"cR={m['cRecall']:.6f} "
        f"TP={m['TP_structural']} "
        f"FP={m['FP']} "
        f"FN={m['FN']}"
    )


output = (
    RUN_DIR
    / "hybrid_fusion_sweep.json"
)

output.write_text(
    json.dumps(
        results,
        indent=2,
    )
)

print()
print("Saved:", output)
