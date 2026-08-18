"""What does gate 5 read when the labels carry NO signal? (2026-08-18)

gate5_lean_separation reports between/within with the note "want ratio >> 1",
implying 1.0 is the no-signal reference. It is not. For n reads per class the
statistic has a pure-noise expectation of

    E[ratio | random labels]  ~=  sqrt(2 / (n - 1))

because the between-class term is a distance between CENTROIDS (which shrink
as 1/sqrt(n)) while the within term is a distance between POINTS (which does
not shrink). Analytically, for iid isotropic points:

    between ~= sqrt(2d/n),  within ~= sqrt(d(1 - 1/n))
    ratio   ~= sqrt(2/n) / sqrt(1 - 1/n) = sqrt(2/(n-1))

Consequences for the gate as written (a4_gates.py:225 admits a decision at
m.sum() >= 4 with >= 2 classes, i.e. as few as 2 reads per class):

    n=2 -> 1.414   n=4 -> 0.816   n=8 -> 0.535   n=20 -> 0.324

So a decision with 2 reads per class scores ~1.41 on PURE NOISE and clears
the stated bar, while a decision with 20 reads per class must beat only 0.32.
The reported average mixes decisions of different n, so it is not comparable
to any single threshold -- and never to 1.0.

This script verifies the closed form against the REAL gate5 code by feeding it
random labels, and (with --labels) reports the floor implied by the actual
reads-per-class of a stored labels.npz.

    python scripts/gate5_noise_floor.py
    python scripts/gate5_noise_floor.py --labels models/v3_32b/labels.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.a4_gates import gate5_lean_separation  # noqa: E402


def monte_carlo(n_per_class: int, dim: int, n_classes: int = 2,
                trials: int = 400, seed: int = 0) -> float:
    """Feed the real gate5 random labels; return the mean ratio."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(trials):
        n = n_per_class * n_classes
        L = rng.standard_normal((n, dim))          # one distribution, no signal
        cls = np.repeat(np.arange(n_classes), n_per_class)  # arbitrary labels
        top = np.zeros(n, dtype=int)
        r = gate5_lean_separation(L, top, cls)
        if r.numbers.get("n_decisions"):
            vals.append(r.numbers["between_within_ratio"])
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None,
                    help="labels.npz to report the empirical floor for")
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()

    print("closed form  E[ratio] = sqrt(2/(n-1))   vs   Monte Carlo on the "
          "real gate5 code")
    print(f"{'n/class':>8} {'closed':>8} {'d=64':>8} {'d=512':>8} {'d=4096':>8}")
    for n in (2, 3, 4, 5, 8, 12, 20):
        closed = np.sqrt(2.0 / (n - 1))
        row = [monte_carlo(n, d, trials=args.trials) for d in (64, 512, 4096)]
        print(f"{n:>8} {closed:>8.3f} {row[0]:>8.3f} {row[1]:>8.3f} "
              f"{row[2]:>8.3f}")

    print("\n3-class (forks can be 3-way):")
    print(f"{'n/class':>8} {'closed':>8} {'MC d=512':>10}")
    for n in (2, 4, 8):
        print(f"{n:>8} {np.sqrt(2.0/(n-1)):>8.3f} "
              f"{monte_carlo(n, 512, n_classes=3, trials=args.trials):>10.3f}")

    if args.labels:
        z = np.load(args.labels, allow_pickle=False)
        dec, cls = z["decision"], z["cls"]
        floors, ns = [], []
        for d in np.unique(dec[dec >= 0]):
            m = (dec == d) & (cls >= 0)
            if m.sum() < 4:
                continue
            classes, counts = np.unique(cls[m], return_counts=True)
            if len(classes) < 2:
                continue
            n_eff = float(np.mean(counts))          # mean reads per class
            ns.append(n_eff)
            floors.append(np.sqrt(2.0 / max(n_eff - 1, 1e-9)))
        if floors:
            floors = np.array(floors)
            print(f"\n{args.labels}: {len(floors)} gate5-eligible decisions")
            print(f"  reads/class  median {np.median(ns):.1f}  "
                  f"min {min(ns):.1f}  max {max(ns):.1f}")
            print(f"  implied pure-noise floor: mean {floors.mean():.3f}  "
                  f"median {np.median(floors):.3f}")
            print("  -> compare THIS to the reported between/within, not 1.0")
        else:
            print(f"\n{args.labels}: no gate5-eligible decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
