from collections import Counter, defaultdict
from pathlib import Path

from dimabsa.experiment_data import load_task_records
from dimabsa.task3_data import (
    build_task3_examples,
    collect_category_vocabulary,
    split_category,
)

RAW_ROOT = Path("data/raw/dimabsa")
DOMAINS = ("laptop", "restaurant")

all_categories = set()

for domain in DOMAINS:

    records = load_task_records(
        RAW_ROOT,
        task=3,
        language="eng",
        domain=domain,
        split="train",
    )

    examples = build_task3_examples(
        records
    )

    categories = collect_category_vocabulary(
        examples
    )

    entities = Counter()
    attributes = Counter()

    pairs = defaultdict(list)

    for category in categories:

        entity, attribute = split_category(
            category
        )

        entities[entity] += 1
        attributes[attribute] += 1

        pairs[entity].append(
            attribute
        )

        all_categories.add(
            category
        )

    print()
    print("=" * 72)
    print(domain.upper())
    print("=" * 72)

    print(
        "flat categories :",
        len(categories),
    )

    print(
        "entities        :",
        len(entities),
    )

    print(
        "attributes      :",
        len(attributes),
    )

    print(
        "cartesian size  :",
        len(entities)
        * len(attributes),
    )

    print()
    print("ENTITIES")

    for name, count in sorted(
        entities.items()
    ):
        print(
            f"{name:30s} "
            f"{count:3d} attributes"
        )

    print()
    print("ATTRIBUTES")

    for name, count in sorted(
        attributes.items()
    ):
        print(
            f"{name:30s} "
            f"{count:3d} entities"
        )


print()
print("=" * 72)
print("GLOBAL")
print("=" * 72)

global_entities = set()
global_attributes = set()

for category in sorted(
    all_categories
):

    entity, attribute = split_category(
        category
    )

    global_entities.add(
        entity
    )

    global_attributes.add(
        attribute
    )

print(
    "flat categories :",
    len(all_categories),
)

print(
    "entities        :",
    len(global_entities),
)

print(
    "attributes      :",
    len(global_attributes),
)
