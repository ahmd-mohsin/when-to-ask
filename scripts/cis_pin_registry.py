"""Regenerate data/cis_registry_pins.json (decisions/029 §3.3).

Pins every blocker resolution's sha256, the id order per task, and the list
of ids whose resolution carries a literal escape / the JSON-terminator leak.
The contract test harness/contract/test_cis_registry.py compares the
vendored registry against this file, so a benchmark update or an accidental
unescaping fails loudly. Run only when the vendored benchmark is
deliberately changed, and commit the diff with the reason.

    python scripts/cis_pin_registry.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.cis_registry import load_resolutions, pins  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="data/cis_registry_pins.json")
    args = ap.parse_args()
    res = load_resolutions(args.tasks_dir, args.classes)
    p = pins(res)
    Path(args.out).write_text(json.dumps(p, indent=1), encoding="utf-8")
    print(f"pinned {p['n_tasks']} tasks / {p['n_blockers']} blockers / "
          f"{len(p['defective_ids'])} defective ids -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
