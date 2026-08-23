"""T7 smoke gate (decisions/028 Amendment A item 9) — mechanical PASS/FAIL.

The gate is pre-registered so the go/no-go on a second model family is not a
judgment call made after seeing the data:

  (a) no chat-template / protocol breakage and no reasoning-trace leakage
  (b) >=1 mutating action in >=50% of smoke runs
  (c) median reads/run >= 10

PASS -> launch the full run. FAIL on the primary -> try the fallback ONCE.
FAIL on both -> record the NO-GO in 028 and drop T7. Never tune the protocol
to make a model comply: that would make the families non-comparable.

    python scripts/xfam_smoke_gate.py --a0 data/xfam_<slug>_smoke
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.labeling import _is_mutating  # noqa: E402

LEAK = re.compile(r"<think>|</think>|<\|channel\|>|<\|start\|>|<reasoning>", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.a0)
    runs, n_leak, n_mut_runs, reads = [], 0, 0, []
    for td in sorted(p for p in root.iterdir() if p.is_dir()):
        for jf in sorted(td.glob("*.json")):
            rid = jf.stem
            if rid.endswith(".segments") or rid.startswith("collection_manifest"):
                continue
            d = json.loads(jf.read_text(encoding="utf-8"))
            runs.append(rid)
            acts = d.get("actions") or []
            if any(_is_mutating(a.get("action_text") or "") for a in acts):
                n_mut_runs += 1
            reads.append(len(d.get("reads") or []))
            txt = td / f"{rid}.txt"
            seg = td / f"{rid}.segments.json"
            body = ""
            if seg.exists():
                body = "\n\n".join(json.loads(seg.read_text(encoding="utf-8")))
            elif txt.exists():
                body = txt.read_text(encoding="utf-8", errors="replace")
            if LEAK.search(body):
                n_leak += 1

    n = len(runs)
    if n == 0:
        print("FAIL: no runs found — collection produced nothing")
        return 1
    med_reads = float(np.median(reads))
    frac_mut = n_mut_runs / n

    a_ok = n_leak == 0
    b_ok = frac_mut >= 0.50
    c_ok = med_reads >= 10

    print(f"smoke runs                 : {n}")
    print(f"(a) leakage-free           : {'PASS' if a_ok else 'FAIL'} "
          f"({n_leak} runs with reasoning-trace markers)")
    print(f"(b) >=1 mutating action    : {'PASS' if b_ok else 'FAIL'} "
          f"({n_mut_runs}/{n} = {frac_mut:.2f}, bar 0.50)")
    print(f"(c) median reads/run >= 10 : {'PASS' if c_ok else 'FAIL'} "
          f"({med_reads:.1f})")
    verdict = a_ok and b_ok and c_ok
    print(f"\nGATE: {'PASS -> launch the full run' if verdict else 'FAIL -> fallback model, or NO-GO'}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "a0": str(root), "n_runs": n, "n_leak": n_leak,
            "frac_runs_with_mutating": frac_mut, "median_reads": med_reads,
            "criteria": {"leak_free": a_ok, "mutating": b_ok, "reads": c_ok},
            "passed": bool(verdict)}, indent=1), encoding="utf-8")
    return 0 if verdict else 2


if __name__ == "__main__":
    raise SystemExit(main())
