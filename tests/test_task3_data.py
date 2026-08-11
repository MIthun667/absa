from dimabsa.experiment_data import TaskRecord, Target
from dimabsa.task2_data import NULL_TERM
from dimabsa.task3_data import (
    build_task3_examples,
    candidate_relation_keys,
    category_va_map,
    collect_category_vocabulary,
    gold_quadruplet_map,
    relation_category_map,
    split_category,
)


def make_record(*targets):
    return TaskRecord(
        record_id="example-1",
        text="The food was good for the price.",
        task=3,
        language="eng",
        domain="restaurant",
        split="train",
        targets=tuple(targets),
        source_path="dummy.jsonl",
        source_field="Quadruplet",
    )


def make_target(
    *,
    aspect,
    opinion,
    category,
    valence,
    arousal,
):
    return Target(
        aspect=aspect,
        opinion=opinion,
        category=category,
        valence=valence,
        arousal=arousal,
        source_opinion=opinion,
        source_category=category,
        source_index=0,
    )


def test_preserves_multiple_categories_for_same_relation():
    record = make_record(
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#QUALITY",
            valence=7.0,
            arousal=7.2,
        ),
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#PRICES",
            valence=6.5,
            arousal=6.8,
        ),
    )

    example = build_task3_examples([record])[0]

    mapping = relation_category_map(example)

    assert set(
        mapping[("food", "good")]
    ) == {
        "FOOD#QUALITY",
        "FOOD#PRICES",
    }


def test_preserves_category_specific_va():
    record = make_record(
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#QUALITY",
            valence=7.0,
            arousal=7.2,
        ),
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#PRICES",
            valence=6.5,
            arousal=6.8,
        ),
    )

    example = build_task3_examples([record])[0]

    mapping = category_va_map(example)

    assert mapping[
        ("food", "good")
    ]["FOOD#QUALITY"] == (
        7.0,
        7.2,
    )

    assert mapping[
        ("food", "good")
    ]["FOOD#PRICES"] == (
        6.5,
        6.8,
    )


def test_gold_quadruplet_key_contains_category():
    record = make_record(
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#QUALITY",
            valence=7.0,
            arousal=7.2,
        )
    )

    example = build_task3_examples([record])[0]

    gold = gold_quadruplet_map(example)

    assert (
        "food",
        "good",
        "food#quality",
    ) in gold


def test_null_candidates_are_added():
    record = make_record(
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#QUALITY",
            valence=7.0,
            arousal=7.2,
        )
    )

    example = build_task3_examples([record])[0]

    candidates = set(
        candidate_relation_keys(
            example,
            always_include_null=True,
        )
    )

    assert ("food", "good") in candidates
    assert ("food", NULL_TERM) in candidates
    assert (NULL_TERM, "good") in candidates
    assert (NULL_TERM, NULL_TERM) in candidates


def test_category_vocabulary_is_deterministic():
    record = make_record(
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#QUALITY",
            valence=7.0,
            arousal=7.2,
        ),
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#PRICES",
            valence=6.5,
            arousal=6.8,
        ),
    )

    examples = build_task3_examples([record])

    assert collect_category_vocabulary(
        examples
    ) == (
        "FOOD#PRICES",
        "FOOD#QUALITY",
    )


def test_split_category():
    assert split_category(
        "FOOD#QUALITY"
    ) == (
        "FOOD",
        "QUALITY",
    )
