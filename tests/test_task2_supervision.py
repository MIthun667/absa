from transformers import AutoTokenizer

from dimabsa.experiment_data import TaskRecord, Target
from dimabsa.task2_data import build_task2_eval_examples
from dimabsa.task2_supervision import (
    B,
    I,
    O,
    decode_bio_spans,
    encode_task2_example,
)


MODEL = "FacebookAI/xlm-roberta-base"


def make_record():
    return TaskRecord(
        record_id="x",
        text="The screen is great but the keyboard is bad.",
        task=2,
        language="eng",
        domain="laptop",
        split="dev",
        targets=(
            Target(
                aspect="screen",
                opinion="great",
                category=None,
                valence=7.0,
                arousal=6.0,
                source_opinion="great",
                source_category=None,
                source_index=0,
            ),
            Target(
                aspect="keyboard",
                opinion="bad",
                category=None,
                valence=3.0,
                arousal=5.0,
                source_opinion="bad",
                source_category=None,
                source_index=1,
            ),
        ),
        source_path="dummy.jsonl",
        source_field="Triplet",
    )


def test_task2_encoding():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        use_fast=True,
    )

    example = build_task2_eval_examples(
        [make_record()]
    )[0]

    encoded = encode_task2_example(
        example,
        tokenizer=tokenizer,
        max_length=256,
    )

    assert len(encoded.input_ids) == len(
        encoded.aspect_labels
    )

    assert len(encoded.input_ids) == len(
        encoded.opinion_labels
    )

    assert any(
        label == B
        for label in encoded.aspect_labels
    )

    assert any(
        label == B
        for label in encoded.opinion_labels
    )


def test_null_candidates_are_present():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        use_fast=True,
    )

    example = build_task2_eval_examples(
        [make_record()]
    )[0]

    encoded = encode_task2_example(
        example,
        tokenizer=tokenizer,
    )

    assert any(
        node.is_null
        for node in encoded.aspect_nodes
    )

    assert any(
        node.is_null
        for node in encoded.opinion_nodes
    )


def test_gold_relations_and_negative_pairs():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        use_fast=True,
    )

    example = build_task2_eval_examples(
        [make_record()]
    )[0]

    encoded = encode_task2_example(
        example,
        tokenizer=tokenizer,
    )

    positives = [
        x
        for x in encoded.relation_candidates
        if x.label == 1
    ]

    negatives = [
        x
        for x in encoded.relation_candidates
        if x.label == 0
    ]

    assert len(positives) == 2
    assert len(negatives) > 0

    assert all(
        x.va is not None
        for x in positives
    )

    assert all(
        x.va is None
        for x in negatives
    )


def test_decode_bio_surface():
    text = "great screen"

    offsets = [
        (0, 0),
        (0, 5),
        (6, 12),
        (0, 0),
    ]

    labels = [
        -100,
        B,
        O,
        -100,
    ]

    spans = decode_bio_spans(
        labels=labels,
        offsets=offsets,
        text=text,
    )

    assert spans == ["great"]


def test_orphan_i_is_robustly_decoded():
    text = "great screen"

    offsets = [
        (0, 0),
        (0, 5),
        (6, 12),
        (0, 0),
    ]

    labels = [
        -100,
        I,
        O,
        -100,
    ]

    spans = decode_bio_spans(
        labels=labels,
        offsets=offsets,
        text=text,
    )

    assert spans == ["great"]
