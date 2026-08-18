"""Run-level permutation of gate 5 in A2's learned L space — the HANDOFF §1b
"missing step" (decisions/026 §5).

The raw-h permutation (gate5_permutation_test.py) says what the labels carry
in the untrained representation; §1's headline 0.894 lives in the LEAN space
of a trained A2, evaluated per k-fold. This script runs the SAME exact/MC
run-level permutation on those fold-eval lean embeddings, so the kfold gate5
ratio can finally be stated in permutation terms.

Inputs: the fold dumps written by `run_full_gates.py --kfold N` (decisions/026)
into <out>/lhe_folds/fold*.npz — each holds the fold's held-out labeled reads:
l (n, lean_dim), decision, cls, run_idx.

    python scripts/gate5_lhe_permutation.py --folds models/v3_32b_fixed/lhe_folds

Per (fold, decision) cell: exact enumeration of the run-class assignment
space when small (else MC), per-cell p + testability floor, and a GLOBAL
Stouffer test pooled over all informative cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate5_permutation_test import decision_null, ratio  # same-dir import


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="models/v3_32b_fixed/lhe_folds",
                    help="dir of fold*.npz written by run_full_gates --kfold")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--exact-cap", type=int, default=20000)
    ap.add_argument("--global-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gate5_lhe_permutation.json")
    args = ap.parse_args()

    fold_files = sorted(Path(args.folds).glob("fold*.npz"))
    if not fold_files:
        raise SystemExit(f"no fold*.npz under {args.folds} -- run "
                         "run_full_gates.py --kfold first (026 dumps folds)")

    rng = np.random.default_rng(args.seed)
    rows, nulls = [], []
    for ff in fold_files:
        z = np.load(ff, allow_pickle=False)
        L, dec, cls, run = z["l"], z["decision"], z["cls"], z["run_idx"]
        for d in np.unique(dec[dec >= 0]):
            m = dec == d
            if m.sum() < 4 or len(np.unique(cls[m])) < 2:
                continue
            Ld, cd, rd = L[m], cls[m], run[m]
            o = ratio(Ld, cd)
            null, p, method, n_assign, min_p = decision_null(
                Ld, cd, rd, rng, args.perms, args.exact_cap)
            nulls.append(null)
            rows.append({"fold": ff.stem, "decision": int(d),
                         "n_reads": int(m.sum()),
                         "n_runs": int(len(np.unique(rd))), "obs": o,
                         "run_null": float(null.mean()),
                         "null_sd": float(null.std()),
                         "p_run": p, "p_method": method,
                         "n_assignments": n_assign, "min_p": min_p,
                         "testable": bool(min_p < 0.05)})

    if not rows:
        raise SystemExit("no gate5-eligible (fold, decision) cells")

    obs = np.array([r["obs"] for r in rows])
    rn_ = np.array([r["run_null"] for r in rows])
    ps = np.array([r["p_run"] for r in rows])
    testable = np.array([r["testable"] for r in rows], dtype=bool)

    live = [i for i, r in enumerate(rows) if r["null_sd"] > 0]
    S_obs = float(sum((obs[i] - rows[i]["run_null"]) / rows[i]["null_sd"]
                      for i in live))
    grng = np.random.default_rng(args.seed + 1)
    S_null = np.array([
        sum((nulls[i][grng.integers(len(nulls[i]))]
             - rows[i]["run_null"]) / rows[i]["null_sd"] for i in live)
        for _ in range(args.global_draws)])
    p_global = (1 + int((S_null >= S_obs).sum())) / (1 + args.global_draws)

    print(f"(fold, decision) cells: {len(rows)} over {len(fold_files)} folds")
    print(f"  observed L-space ratio  mean {obs.mean():.3f}")
    print(f"  RUN-level null          mean {rn_.mean():.3f}")
    print(f"  observed / run-null     {obs.mean() / rn_.mean():.3f}x")
    print(f"  TESTABLE at alpha=0.05: {int(testable.sum())} / {len(rows)}")
    print(f"  cells with p < 0.05: {int((ps < 0.05).sum())} of "
          f"{int(testable.sum())} testable")
    print(f"  GLOBAL Stouffer test:   S_obs {S_obs:+.3f}  p {p_global:.4f}   "
          f"({len(live)} informative cells)   <- the headline for l_he")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"n_cells": len(rows), "n_folds": len(fold_files),
                    "observed_mean": float(obs.mean()),
                    "run_null_mean": float(rn_.mean()),
                    "n_sig_p05": int((ps < 0.05).sum()),
                    "n_testable": int(testable.sum()),
                    "global_stouffer": {"S_obs": S_obs, "p": p_global,
                                        "n_informative": len(live),
                                        "draws": args.global_draws},
                    "rows": rows}, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
