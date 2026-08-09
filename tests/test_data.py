from pathlib import Path

import pytest

from dimabsa.data import (
    normalize_term,
    parse_dataset_file,
    parse_va,
    projected_key,
)


def test_parse_dimensional_file_metadata() -> None:
    root = Path("data/raw/dimabsa")
    path = root / "subtask_3" / "eng" / "eng_restaurant_train_alltasks.jsonl"
    meta = parse_dataset_file(path, root)

    assert meta.representation == "dimensional"
    assert meta.subtask == 3
    assert meta.language == "eng"
    assert meta.domain == "restaurant"
    assert meta.split == "train"
    assert meta.training_scope == "alltasks"


def test_parse_categorical_file_metadata() -> None:
    root = Path("data/raw/dimabsa")
    path = root / "categorical" / "subtask_1" / "zho" / "zho_finance_dev_task1_polarity.jsonl"
    meta = parse_dataset_file(path, root)

    assert meta.representation == "categorical"
    assert meta.subtask == 1
    assert meta.language == "zho"
    assert meta.domain == "finance"
    assert meta.split == "dev"


def test_null_terms_are_normalized_to_none() -> None:
    assert normalize_term("NULL") is None
    assert normalize_term(" null ") is None
    assert normalize_term(None) is None
    assert normalize_term("Battery Life") == "battery life"


def test_parse_va() -> None:
    assert parse_va("7.50#7.62") == (7.5, 7.62)


@pytest.mark.parametrize("value", ["0.99#5.00", "5.00#9.01", "bad", "5.0"])
def test_invalid_va_raises(value: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_va(value)


def test_hierarchical_projection() -> None:
    quad = {
        "Aspect": "Food",
        "Opinion": "great",
        "Category": "FOOD#QUALITY",
        "VA": "7.67#7.83",
    }

    assert projected_key(quad, 1) == ("food",)
    assert projected_key(quad, 2) == ("food", "great")
    assert projected_key(quad, 3) == ("food", "great", "food#quality")
