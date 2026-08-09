from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dimabsa.conflict_anatomy import analyze_conflict_anatomy


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze semantic anatomy of Track A hierarchy conflicts.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "dimabsa",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "conflict_anatomy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details, summary = analyze_conflict_anatomy(args.raw_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "conflict_details.csv", details)
    _write_csv(args.output_dir / "conflict_summary.csv", summary)

    report = {
        "ambiguous_groups": len(details),
        "target1_groups": sum(1 for row in details if row["target_task"] == 1),
        "target2_groups": sum(1 for row in details if row["target_task"] == 2),
        "null_aspect_groups": sum(1 for row in details if row["aspect_is_null"]),
        "artifacts": ["conflict_details.csv", "conflict_summary.csv"],
    }
    (args.output_dir / "conflict_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
