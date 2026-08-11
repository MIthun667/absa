from __future__ import annotations

import random
from collections.abc import Iterator


TASK_NAMES = (
    "t1",
    "t2",
    "t3",
)


class BalancedTaskScheduler:
    """
    Infinite task scheduler for naive MTL.

    Each cycle contains every task exactly once.
    The order is independently shuffled per cycle.

    Therefore task exposure remains exactly balanced while
    avoiding a deterministic T1 -> T2 -> T3 ordering.
    """

    def __init__(
        self,
        *,
        seed: int,
        task_names: tuple[str, ...] = TASK_NAMES,
    ) -> None:

        if not task_names:
            raise ValueError(
                "task_names must not be empty"
            )

        if len(set(task_names)) != len(
            task_names
        ):
            raise ValueError(
                "task_names must be unique"
            )

        self.task_names = tuple(
            task_names
        )

        self.rng = random.Random(
            seed
        )

        self.cycle_index = 0

    def next_cycle(
        self,
    ) -> tuple[str, ...]:

        tasks = list(
            self.task_names
        )

        self.rng.shuffle(
            tasks
        )

        self.cycle_index += 1

        return tuple(
            tasks
        )

    def __iter__(
        self,
    ) -> Iterator[str]:

        while True:

            for task_name in (
                self.next_cycle()
            ):

                yield task_name
