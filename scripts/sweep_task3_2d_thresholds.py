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
    "outputs/phase2/task3_balanced_400/"
    "unambiguous/seed_42"
)

RAW_ROOT = Path("data/raw/dimabsa")

DOMAINS = (
    "laptop",
    "restaurant",
)

RELATION_THRESHOLDS = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
)

CATEGORY_THRESHOLDS = (
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
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
        build_task3_examples(records)
    )


model = Task3Model(
    config["model_name"],
    num_categories=len(category_names),
    dropout=config["dropout"],
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


results = []

print("=" * 110)
print("TASK 3 RELATION × CATEGORY THRESHOLD SWEEP")
print("=" * 110)
print("checkpoint step:", checkpoint["step"])
print()


for relation_threshold in RELATION_THRESHOLDS:

    for category_threshold in CATEGORY_THRESHOLDS:

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
                relation_threshold
            ),
            category_threshold=(
                category_threshold
            ),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        row = {
            "relation_threshold":
                relation_threshold,
            "category_threshold":
                category_threshold,
            "macro_cf1":
                macro_cf1,
            "domains":
                metrics,
        }

        results.append(row)

        laptop = metrics["laptop"]
        restaurant = metrics["restaurant"]

        print(
            f"R={relation_threshold:.2f} "
            f"C={category_threshold:.2f} | "
            f"macro={macro_cf1:.6f} | "
            f"L={laptop['cF1']:.6f} "
            f"(TP={laptop['TP_structural']}, "
            f"FP={laptop['FP']}) | "
            f"Rst={restaurant['cF1']:.6f} "
            f"(TP={restaurant['TP_structural']}, "
            f"FP={restaurant['FP']})"
        )


best = max(
    results,
    key=lambda x: x["macro_cf1"],
)

print()
print("=" * 110)
print("BEST SHARED THRESHOLD PAIR")
print("=" * 110)

print(
    "relation threshold :",
    best["relation_threshold"],
)

print(
    "category threshold :",
    best["category_threshold"],
)

print(
    "macro cF1          :",
    f"{best['macro_cf1']:.6f}",
)

for domain in DOMAINS:

    m = best["domains"][domain]

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
    / "relation_category_threshold_sweep.json"
)

output.write_text(
    json.dumps(
        results,
        indent=2,
    )
)

print()
print("Saved:", output)
