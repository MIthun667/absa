from transformers import AutoTokenizer

from dimabsa.experiment_data import TaskRecord, Target
from dimabsa.task3_data import (
    build_task3_examples,
)
from dimabsa.task3_supervision import (
    encode_task3_example,
)


TOKENIZER = AutoTokenizer.from_pretrained(
    "FacebookAI/xlm-roberta-base",
    use_fast=True,
)


def make_target(
    *,
    aspect,
    opinion,
    category,
    valence,
    arousal,
    source_index,
):
    return Target(
        aspect=aspect,
        opinion=opinion,
        category=category,
        valence=valence,
        arousal=arousal,
        source_opinion=opinion,
        source_category=category,
        source_index=source_index,
    )


def make_record(*targets):
    return TaskRecord(
        record_id="t3-test",
        text="The food was good for the price.",
        task=3,
        language="eng",
        domain="restaurant",
        split="train",
        targets=tuple(targets),
        source_path="dummy.jsonl",
        source_field="Quadruplet",
    )


def test_multilabel_category_supervision():

    record = make_record(
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#QUALITY",
            valence=7.0,
            arousal=7.2,
            source_index=0,
        ),
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#PRICES",
            valence=6.5,
            arousal=6.8,
            source_index=1,
        ),
    )

    example = build_task3_examples(
        [record]
    )[0]

    vocab = {
        "FOOD#PRICES": 0,
        "FOOD#QUALITY": 1,
    }

    encoded = encode_task3_example(
        example,
        TOKENIZER,
        category_to_index=vocab,
    )

    positive = [
        c
        for c in encoded.relation_candidates
        if c.relation_label == 1.0
    ]

    assert len(positive) == 1

    candidate = positive[0]

    assert candidate.category_labels == (
        1.0,
        1.0,
    )

    assert len(
        candidate.va_targets
    ) == 2


def test_category_specific_va_is_preserved():

    record = make_record(
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#QUALITY",
            valence=7.0,
            arousal=7.2,
            source_index=0,
        ),
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#PRICES",
            valence=6.5,
            arousal=6.8,
            source_index=1,
        ),
    )

    example = build_task3_examples(
        [record]
    )[0]

    vocab = {
        "FOOD#PRICES": 0,
        "FOOD#QUALITY": 1,
    }

    encoded = encode_task3_example(
        example,
        TOKENIZER,
        category_to_index=vocab,
    )

    candidate = next(
        c
        for c in encoded.relation_candidates
        if c.relation_label == 1.0
    )

    targets = {
        t.category_index: (
            t.valence,
            t.arousal,
        )
        for t in candidate.va_targets
    }

    assert targets[0] == (
        6.5,
        6.8,
    )

    assert targets[1] == (
        7.0,
        7.2,
    )


def test_duplicate_aoc_va_is_not_averaged():

    record = TaskRecord(
        record_id="duplicate-va",
        text="I love this computer.",
        task=3,
        language="eng",
        domain="laptop",
        split="train",
        targets=(
            make_target(
                aspect="computer",
                opinion="love",
                category="LAPTOP#GENERAL",
                valence=6.75,
                arousal=6.38,
                source_index=0,
            ),
            make_target(
                aspect="computer",
                opinion="love",
                category="LAPTOP#GENERAL",
                valence=6.88,
                arousal=6.50,
                source_index=1,
            ),
        ),
        source_path="dummy.jsonl",
        source_field="Quadruplet",
    )

    example = build_task3_examples(
        [record]
    )[0]

    vocab = {
        "LAPTOP#GENERAL": 0,
    }

    encoded = encode_task3_example(
        example,
        TOKENIZER,
        category_to_index=vocab,
    )

    candidate = next(
        c
        for c in encoded.relation_candidates
        if c.relation_label == 1.0
    )

    assert len(
        candidate.va_targets
    ) == 2

    values = [
        (
            t.valence,
            t.arousal,
        )
        for t in candidate.va_targets
    ]

    assert values == [
        (6.75, 6.38),
        (6.88, 6.50),
    ]


def test_negative_pair_has_no_category_or_va():

    record = make_record(
        make_target(
            aspect="food",
            opinion="good",
            category="FOOD#QUALITY",
            valence=7.0,
            arousal=7.2,
            source_index=0,
        )
    )

    example = build_task3_examples(
        [record]
    )[0]

    vocab = {
        "FOOD#QUALITY": 0,
    }

    encoded = encode_task3_example(
        example,
        TOKENIZER,
        category_to_index=vocab,
    )

    negatives = [
        c
        for c in encoded.relation_candidates
        if c.relation_label == 0.0
    ]

    assert negatives

    for candidate in negatives:
        assert candidate.category_labels == (
            0.0,
        )
        assert candidate.va_targets == ()
