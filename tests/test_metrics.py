from __future__ import annotations

import math

import pytest

from dimabsa.metrics import D_MAX, evaluate_structured, evaluate_task1


def test_task1_perfect_prediction() -> None:
    gold = [
        {
            "ID": "1",
            "Aspect_VA": [
                {"Aspect": "screen", "VA": "7.0#6.0"},
                {"Aspect": "battery", "VA": "3.0#5.0"},
            ],
        }
    ]
    pred = [
        {
            "ID": "1",
            "Aspect_VA": [
                {"Aspect": "SCREEN", "VA": "7.0#6.0"},
                {"Aspect": "battery", "VA": "3.0#5.0"},
            ],
        }
    ]

    result = evaluate_task1(gold, pred)
    assert result["RMSE_VA"] == pytest.approx(0.0)
    assert result["PCC_V"] == pytest.approx(1.0)
    assert result["PCC_A"] == pytest.approx(1.0)
    assert result["N"] == 2


def test_task1_joint_rmse_matches_official_formula() -> None:
    gold = [{"ID": "1", "Aspect_VA": [{"Aspect": "x", "VA": "5#5"}]}]
    pred = [{"ID": "1", "Aspect_VA": [{"Aspect": "x", "VA": "6#7"}]}]

    raw = evaluate_task1(gold, pred, normalize_rmse=False)
    normalized = evaluate_task1(gold, pred, normalize_rmse=True)

    assert raw["RMSE_VA"] == pytest.approx(math.sqrt(5.0))
    assert normalized["RMSE_VA"] == pytest.approx(math.sqrt(5.0) / D_MAX)


def test_task1_missing_aspect_fails() -> None:
    gold = [{"ID": "1", "Aspect_VA": [{"Aspect": "screen", "VA": "5#5"}]}]
    pred = [{"ID": "1", "Aspect_VA": []}]
    with pytest.raises(ValueError, match="Missing Task-1 VA"):
        evaluate_task1(gold, pred)


def test_task1_duplicate_prediction_fails_explicitly() -> None:
    gold = [{"ID": "1", "Aspect_VA": [{"Aspect": "screen", "VA": "5#5"}]}]
    pred = [
        {
            "ID": "1",
            "Aspect_VA": [
                {"Aspect": "screen", "VA": "5#5"},
                {"Aspect": "SCREEN", "VA": "5#5"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="Duplicate Task-1 prediction"):
        evaluate_task1(gold, pred)


def test_task2_perfect_prediction() -> None:
    gold = [
        {
            "ID": "1",
            "Triplet": [
                {"Aspect": "screen", "Opinion": "great", "VA": "7#7"},
            ],
        }
    ]
    pred = [
        {
            "ID": "1",
            "Triplet": [
                {"Aspect": "SCREEN", "Opinion": "GREAT", "VA": "7#7"},
            ],
        }
    ]
    result = evaluate_structured(gold, pred, task=2)
    assert result["cF1"] == pytest.approx(1.0)
    assert result["TP_structural"] == 1
    assert result["FP"] == 0
    assert result["FN"] == 0


def test_task2_accepts_quadruplet_gold_like_official_reader() -> None:
    gold = [
        {
            "ID": "1",
            "Quadruplet": [
                {
                    "Aspect": "screen",
                    "Opinion": "great",
                    "Category": "DISPLAY#GENERAL",
                    "VA": "7#7",
                }
            ],
        }
    ]
    pred = [
        {
            "ID": "1",
            "Triplet": [
                {"Aspect": "screen", "Opinion": "great", "VA": "7#7"},
            ],
        }
    ]
    result = evaluate_structured(gold, pred, task=2)
    assert result["cF1"] == pytest.approx(1.0)


def test_task3_continuous_credit() -> None:
    gold = [
        {
            "ID": "1",
            "Quadruplet": [
                {
                    "Aspect": "screen",
                    "Opinion": "great",
                    "Category": "DISPLAY#GENERAL",
                    "VA": "5#5",
                }
            ],
        }
    ]
    pred = [
        {
            "ID": "1",
            "Quadruplet": [
                {
                    "Aspect": "screen",
                    "Opinion": "great",
                    "Category": "display#general",
                    "VA": "6#5",
                }
            ],
        }
    ]
    result = evaluate_structured(gold, pred, task=3)
    expected = 1.0 - 1.0 / D_MAX
    assert result["cTP"] == pytest.approx(expected)
    assert result["cF1"] == pytest.approx(expected)


def test_structured_wrong_structure_gets_no_continuous_credit() -> None:
    gold = [
        {
            "ID": "1",
            "Triplet": [
                {"Aspect": "screen", "Opinion": "great", "VA": "7#7"},
            ],
        }
    ]
    pred = [
        {
            "ID": "1",
            "Triplet": [
                {"Aspect": "screen", "Opinion": "bad", "VA": "7#7"},
            ],
        }
    ]
    result = evaluate_structured(gold, pred, task=2)
    assert result["cTP"] == 0.0
    assert result["FP"] == 1
    assert result["FN"] == 1
    assert result["cF1"] == 0.0


def test_structured_duplicate_matching_prediction_is_penalized() -> None:
    gold = [
        {
            "ID": "1",
            "Triplet": [
                {"Aspect": "screen", "Opinion": "great", "VA": "7#7"},
            ],
        }
    ]
    pred = [
        {
            "ID": "1",
            "Triplet": [
                {"Aspect": "screen", "Opinion": "great", "VA": "7#7"},
                {"Aspect": "screen", "Opinion": "great", "VA": "6#6"},
            ],
        }
    ]
    result = evaluate_structured(gold, pred, task=2)
    assert result["duplicate_matching_predictions"] is True
    assert result["TP_structural"] == 0
    assert result["FN"] == 1
    assert result["FP"] == 2
    assert result["cF1"] == 0.0


def test_structured_out_of_range_prediction_gets_zero_ctp() -> None:
    gold = [
        {
            "ID": "1",
            "Triplet": [
                {"Aspect": "screen", "Opinion": "great", "VA": "7#7"},
            ],
        }
    ]
    pred = [
        {
            "ID": "1",
            "Triplet": [
                {"Aspect": "screen", "Opinion": "great", "VA": "10#7"},
            ],
        }
    ]
    result = evaluate_structured(gold, pred, task=2)
    assert result["TP_structural"] == 1
    assert result["cTP"] == 0.0
    assert result["out_of_range_predictions"] is True
    assert result["cF1"] == 0.0
