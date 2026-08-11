from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dimabsa.experiment_data import load_task_records
from dimabsa.task3_data import build_task3_examples
from dimabsa.task3_model import Task3Model

from train_task3_baseline import evaluate


RUN_DIR = Path(
    "outputs/phase2/task3_balanced_800/"
    "unambiguous/seed_42"
)

RAW_ROOT = Path(
    "data/raw/dimabsa"
)

DOMAINS = (
    "laptop",
    "restaurant",
)

CATEGORY_THRESHOLDS = (
    0.70,
    0.72,
    0.74,
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
    0.92,
)

RELATION_THRESHOLD = 0.50


config = json.loads(
    (
        RUN_DIR
        / "run_config.json"
    ).read_text()
)

checkpoint = torch.load(
    RUN_DIR / "best_checkpoint.pt",
    map_location="cpu",
)

category_names = tuple(
    config["category_names"]
)

category_to_index = {
    category: index
    for index, category
    in enumerate(category_names)
}

allowed_categories_by_domain = {
    domain: tuple(
        category_to_index[category]
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
        build_task3_examples(
            records
        )
    )


model = Task3Model(
    config["model_name"],
    num_categories=len(
        category_names
    ),
    dropout=config["dropout"],
).to(device)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print("=" * 100)
print("TASK 3 CATEGORY-THRESHOLD SWEEP")
print("=" * 100)

print(
    "checkpoint step      :",
    checkpoint["step"],
)

print(
    "relation threshold   :",
    RELATION_THRESHOLD,
)

print(
    "category loss        :",
    config.get(
        "category_loss",
        "unknown",
    ),
)

print(
    "number of categories :",
    len(category_names),
)

print("=" * 100)

results = []


for category_threshold in (
    CATEGORY_THRESHOLDS
):

    (
        macro_cf1,
        metrics,
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
        max_length=config["max_length"],
        relation_threshold=(
            RELATION_THRESHOLD
        ),
        category_threshold=(
            category_threshold
        ),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    row = {
        "category_threshold":
            category_threshold,
        "macro_cf1":
            macro_cf1,
        "domains":
            metrics,
    }

    results.append(
        row
    )

    print()
    print(
        f"CATEGORY THRESHOLD = "
        f"{category_threshold:.2f}"
    )

    for domain in DOMAINS:

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


best = max(
    results,
    key=lambda row:
        row["macro_cf1"],
)


print()
print("=" * 100)
print("BEST COMMON CATEGORY THRESHOLD")
print("=" * 100)

print(
    "threshold :",
    best[
        "category_threshold"
    ],
)

print(
    "macro cF1:",
    f"{best['macro_cf1']:.6f}",
)

for domain in DOMAINS:

    m = best[
        "domains"
    ][domain]

    print(
        f"{domain:10s} "
        f"cF1={m['cF1']:.6f} "
        f"cP={m['cPrecision']:.6f} "
        f"cR={m['cRecall']:.6f} "
        f"TP={m['TP_structural']} "
        f"FP={m['FP']} "
        f"FN={m['FN']}"
    )


output_path = (
    RUN_DIR
    / "category_threshold_sweep.json"
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
