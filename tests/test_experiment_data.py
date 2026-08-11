from pathlib import Path

from dimabsa.experiment_data import (
    TaskRecord,
    Target,
    find_projection_conflicts,
    load_task_records,
    summarize_task_records,
)


def test_find_projection_conflicts_marks_ambiguous_va() -> None:
    record = TaskRecord(
        record_id="r1",
        text="screen is bright and clear",
        task=1,
        language="eng",
        domain="laptop",
        split="train",
        source_path="x.jsonl",
        source_field="Quadruplet",
        targets=(
            Target("screen", 7.0, 7.0, source_opinion="bright"),
            Target("screen", 7.5, 7.2, source_opinion="clear"),
        ),
    )

    conflicts = find_projection_conflicts([record])
    assert len(conflicts) == 1
    assert conflicts[0].ambiguous_va is True
    assert conflicts[0].distinct_va_count == 2


def test_find_projection_conflicts_safe_duplicate() -> None:
    record = TaskRecord(
        record_id="r1",
        text="great price",
        task=2,
        language="eng",
        domain="laptop",
        split="train",
        source_path="x.jsonl",
        source_field="Quadruplet",
        targets=(
            Target("laptop", 7.0, 7.0, opinion="great", source_category="LAPTOP#GENERAL"),
            Target("laptop", 7.0, 7.0, opinion="great", source_category="LAPTOP#PRICE"),
        ),
    )

    conflicts = find_projection_conflicts([record])
    assert len(conflicts) == 1
    assert conflicts[0].ambiguous_va is False


def test_summarize_empty_records() -> None:
    summary = summarize_task_records([])
    assert summary["records"] == 0
    assert summary["targets"] == 0


def test_load_task1_dev_native_schema(tmp_path: Path) -> None:
    path = tmp_path / "subtask_1" / "eng"
    path.mkdir(parents=True)
    dataset = path / "eng_laptop_dev_task1.jsonl"
    dataset.write_text(
        '{"ID":"1","Text":"screen is good","Aspect_VA":[{"Aspect":"screen","VA":"7#7"}]}\n',
        encoding="utf-8",
    )

    records = load_task_records(
        tmp_path,
        task=1,
        language="eng",
        domain="laptop",
        split="dev",
    )
    assert len(records) == 1
    assert records[0].source_field == "Aspect_VA"
    assert records[0].targets[0].aspect == "screen"
    assert records[0].targets[0].source_opinion is None


def test_load_task1_train_preserves_quadruplet_conflict(tmp_path: Path) -> None:
    path = tmp_path / "subtask_1" / "eng"
    path.mkdir(parents=True)
    dataset = path / "eng_laptop_train_alltasks.jsonl"
    dataset.write_text(
        '{"ID":"1","Text":"screen bright but dim later","Quadruplet":['
        '{"Aspect":"screen","Opinion":"bright","Category":"DISPLAY#QUALITY","VA":"7#7"},'
        '{"Aspect":"screen","Opinion":"dim","Category":"DISPLAY#QUALITY","VA":"3#6"}'
        ']}\n',
        encoding="utf-8",
    )

    records = load_task_records(
        tmp_path,
        task=1,
        language="eng",
        domain="laptop",
        split="train",
    )
    assert records[0].source_field == "Quadruplet"
    assert len(records[0].targets) == 2
    assert {target.source_opinion for target in records[0].targets} == {"bright", "dim"}
    conflicts = find_projection_conflicts(records)
    assert len(conflicts) == 1
    assert conflicts[0].ambiguous_va is True
