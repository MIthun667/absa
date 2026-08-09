from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dimabsa.hierarchy import audit_hierarchy


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    raw_root = PROJECT_ROOT / "data" / "raw" / "dimabsa"
    output_dir = PROJECT_ROOT / "artifacts" / "hierarchy_audit"
    manifest, projections, examples = audit_hierarchy(raw_root)
    write_csv(output_dir / "alltasks_manifest.csv", manifest)
    write_csv(output_dir / "projection_collisions.csv", projections)
    write_csv(output_dir / "ambiguous_projection_examples.csv", examples)

    report = {
        "alltask_datasets": len(manifest),
        "all_identical_across_subtasks": all(
            str(row["identical_across_subtasks"]).lower() == "true" for row in manifest
        ),
        "ambiguous_projection_groups": sum(int(row["ambiguous_va_groups"]) for row in projections),
        "collision_groups": sum(int(row["collision_groups"]) for row in projections),
        "artifacts": [
            "alltasks_manifest.csv",
            "projection_collisions.csv",
            "ambiguous_projection_examples.csv",
        ],
    }
    (output_dir / "hierarchy_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
