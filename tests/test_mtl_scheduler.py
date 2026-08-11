from collections import Counter

from dimabsa.mtl_scheduler import (
    BalancedTaskScheduler,
    TASK_NAMES,
)


def test_each_cycle_contains_all_tasks_once():

    scheduler = BalancedTaskScheduler(
        seed=42
    )

    for _ in range(100):

        cycle = scheduler.next_cycle()

        assert len(cycle) == 3

        assert set(cycle) == set(
            TASK_NAMES
        )


def test_long_run_is_exactly_balanced():

    scheduler = BalancedTaskScheduler(
        seed=42
    )

    iterator = iter(
        scheduler
    )

    counts = Counter(
        next(iterator)
        for _ in range(300)
    )

    assert counts == {
        "t1": 100,
        "t2": 100,
        "t3": 100,
    }


def test_scheduler_is_deterministic_given_seed():

    first = BalancedTaskScheduler(
        seed=42
    )

    second = BalancedTaskScheduler(
        seed=42
    )

    first_iterator = iter(
        first
    )

    second_iterator = iter(
        second
    )

    first_sequence = [
        next(first_iterator)
        for _ in range(30)
    ]

    second_sequence = [
        next(second_iterator)
        for _ in range(30)
    ]

    assert (
        first_sequence
        == second_sequence
    )


def test_different_seeds_change_order():

    first = BalancedTaskScheduler(
        seed=42
    )

    second = BalancedTaskScheduler(
        seed=100
    )

    first_iterator = iter(
        first
    )

    second_iterator = iter(
        second
    )

    first_sequence = [
        next(first_iterator)
        for _ in range(30)
    ]

    second_sequence = [
        next(second_iterator)
        for _ in range(30)
    ]

    assert (
        first_sequence
        != second_sequence
    )
