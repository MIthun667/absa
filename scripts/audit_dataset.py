from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dimabsa.audit_resilient import audit_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the official DimABSA Track A dataset.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "dimabsa",
        help="Path to immutable raw Track A data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_audit",
        help="Directory for audit artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dataset(args.raw_root, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
