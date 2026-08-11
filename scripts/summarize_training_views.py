from __future__ import annotations

import argparse
from pathlib import Path

from dimabsa.experiment_data import load_task_records
from dimabsa.training_views import summarize_training_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/dimabsa"))
    parser.add_argument("--language", default="eng")
    parser.add_argument("--domains", nargs="+", default=["laptop", "restaurant"])
    args = parser.parse_args()

    for domain in args.domains:
        print(f"\n===== {args.language.upper()} / {domain.upper()} =====")
        for task in (1, 2, 3):
            records = load_task_records(
                args.raw_root,
                task=task,
                language=args.language,
                domain=domain,
                split="train",
            )
            summary = summarize_training_views(records)
            print(
                f"T{task} | source={summary.source_targets:5d} | "
                f"groups={summary.structural_groups:5d} | "
                f"collisions={summary.collision_groups:4d} | "
                f"ambiguous={summary.ambiguous_groups:4d} | "
                f"expanded={summary.relation_expanded_examples:5d} | "
                f"unambiguous={summary.deterministic_examples:5d} | "
                f"drop_targets={summary.dropped_ambiguous_targets:4d} | "
                f"safe_dedup={summary.safely_collapsed_duplicate_targets:4d}"
            )


if __name__ == "__main__":
    main()
