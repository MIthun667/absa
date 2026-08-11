from dimabsa.experiment_data import (
    TaskRecord,
    Target,
)
from dimabsa.task2_data import (
    NULL_TERM,
    build_task2_eval_examples,
    build_task2_train_examples,
    candidate_relation_keys,
    find_all_occurrences,
    gold_relation_map,
)


def make_record(
    *,
    targets,
    text="The screen is great but the keyboard is bad.",
):
    return TaskRecord(
        record_id="x",
        text=text,
        task=2,
        language="eng",
        domain="laptop",
        split="train",
        targets=tuple(targets),
        source_path="dummy.jsonl",
        source_field="Quadruplet",
    )


def make_target(
    aspect,
    opinion,
    v,
    a,
    index=0,
):
    return Target(
        aspect=aspect,
        opinion=opinion,
        category=None,
        valence=v,
        arousal=a,
        source_opinion=opinion,
        source_category=None,
        source_index=index,
    )


def test_find_all_occurrences():
    spans = find_all_occurrences(
        "good screen, good keyboard",
        "good",
    )

    assert len(spans) == 2
    assert spans[0].start == 0
    assert spans[1].start == 13


def test_explicit_nodes_keep_all_occurrences():
    record = make_record(
        text="screen is good and screen looks good",
        targets=[
            make_target(
                "screen",
                "good",
                7.0,
                6.0,
            )
        ],
    )

    example = build_task2_eval_examples(
        [record]
    )[0]

    assert len(
        example.aspects[0].occurrences
    ) == 2

    assert len(
        example.opinions[0].occurrences
    ) == 2


def test_null_nodes_supported():
    record = make_record(
        targets=[
            make_target(
                NULL_TERM,
                "great",
                7.0,
                6.0,
            )
        ],
    )

    example = build_task2_eval_examples(
        [record]
    )[0]

    assert example.aspects[0].is_null
    assert (
        example.aspects[0].occurrences
        == ()
    )


def test_candidate_cartesian_product_adds_null():
    record = make_record(
        targets=[
            make_target(
                "screen",
                "great",
                7.0,
                6.0,
            )
        ],
    )

    example = build_task2_eval_examples(
        [record]
    )[0]

    candidates = set(
        candidate_relation_keys(
            example,
            always_include_null=True,
        )
    )

    assert (
        "screen",
        "great",
    ) in candidates

    assert (
        "screen",
        "NULL",
    ) in candidates

    assert (
        "NULL",
        "great",
    ) in candidates

    assert (
        "NULL",
        "NULL",
    ) in candidates


def test_unambiguous_training_drops_conflicting_va():
    record = make_record(
        text="screen is good",
        targets=[
            make_target(
                "screen",
                "good",
                7.0,
                6.0,
                0,
            ),
            make_target(
                "screen",
                "good",
                4.0,
                5.0,
                1,
            ),
        ],
    )

    examples = build_task2_train_examples(
        [record],
        view="unambiguous",
    )

    assert examples == []


def test_safe_duplicate_collapses():
    record = make_record(
        text="screen is good",
        targets=[
            make_target(
                "screen",
                "good",
                7.0,
                6.0,
                0,
            ),
            make_target(
                "screen",
                "good",
                7.0,
                6.0,
                1,
            ),
        ],
    )

    examples = build_task2_train_examples(
        [record],
        view="unambiguous",
    )

    assert len(examples) == 1
    assert len(
        examples[0].relations
    ) == 1


def test_gold_relation_map():
    record = make_record(
        targets=[
            make_target(
                "screen",
                "great",
                7.0,
                6.0,
            ),
            make_target(
                "keyboard",
                "bad",
                3.0,
                5.0,
            ),
        ],
    )

    example = build_task2_eval_examples(
        [record]
    )[0]

    mapping = gold_relation_map(
        example
    )

    assert len(mapping) == 2

    assert (
        "screen",
        "great",
    ) in mapping

    assert (
        "keyboard",
        "bad",
    ) in mapping
