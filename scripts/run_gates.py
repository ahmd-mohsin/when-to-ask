"""Run the A4 gates on REAL held-out activations (AWS box) and print the report.

    python scripts/run_gates.py --data data/a0_features.npz --a2 models/a2.pt

Expects an npz with: h (n, H) held-out reads, topic (n,) decision-identity
labels, cls (n,) interpretation-class labels (-1 = unlabeled), and optionally
ood_h/ood_topic for gate 6 and conflation_pairs (m, 2) for gate 4 -- produced
by the offline labeling step from A0 logs + registries.

The numbers are printed UNFILTERED and the run stops there: owner review is
the checkpoint (spec A4, decisions/011). Nothing here trains or tunes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.a2_autoencoder import A2Model  # noqa: E402
from wta.a4_gates import (  # noqa: E402
    gate1_topic_leakage, gate2_decision_recovery, gate3_fork_collocation,
    gate4_conflation, gate5_lean_separation, gate6_ood_transfer,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="held-out labeled features npz")
    ap.add_argument("--a2", required=True, help="trained A2 checkpoint")
    ap.add_argument("--probe-frac", type=float, default=0.5,
                    help="fraction of held-out data used to FIT gate probes")
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    model = A2Model.load(args.a2)
    h, topic, cls = data["h"], data["topic"], data["cls"]

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(h))
    cut = int(args.probe_frac * len(idx))
    fit, ev = idx[:cut], idx[cut:]
    t_fit, t_ev = model.encode_topic(h[fit]), model.encode_topic(h[ev])
    l_ev = model.encode_lean(h[ev])

    results = [
        gate1_topic_leakage(t_fit, cls[fit], t_ev, cls[ev]),
        gate2_decision_recovery(t_fit, topic[fit], t_ev, topic[ev]),
    ]
    g3 = gate3_fork_collocation(t_ev, topic[ev], cls[ev])
    results.append(g3)
    if "conflation_pairs" in data:
        results.append(gate4_conflation(model.encode_topic(h),
                                        data["conflation_pairs"], g3.numbers["theta"]))
    results.append(gate5_lean_separation(l_ev, topic[ev], cls[ev]))
    if "ood_h" in data:
        results.append(gate6_ood_transfer(model.encode_topic(data["ood_h"]),
                                          data["ood_topic"], g3.numbers["theta"]))

    print("\n=== A4 GATES (held-out real data) — report to owner, do not tune ===")
    for g in results:
        print(" ", g)
    print("\nGate 7 (lead-time) runs from trajectory logs:"
          " see scripts/run_pipeline_smoke.py for the fixture version;"
          " the real-data variant needs per-run logs + action reads.")
    print("\nSTOP: owner review required before trusting any Part B result"
          " (decisions/011).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
