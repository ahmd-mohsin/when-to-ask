"""Layer + eps/window sweeps (decisions/014) -- pure CPU, no AWS re-run.

    python scripts/sweep.py --layers 0,1,2,3 --eps 0.4,0.6,0.8 --window 3,5

LAYER sweep: for each captured layer, rebuild labels selecting that layer,
train A2, and report the two representation gates that depend on WHICH layer
we read -- gate 1 (topic-invariance, within decision) and gate 5 (lean
separation). Ranks layers by gate 5 (the make-or-break -- does the lean
subspace separate interpretations). Single-split for speed; run
`run_full_gates.py --kfold 5 --layer <best>` afterwards for the trustworthy
number.

EPS/WINDOW sweep: on the best layer's trained A2 (lean vectors are fixed once
A2 is trained), loop eps x window through A3 -> commitment settle rate + gate 7
lead-time. This is what fixes the 7%-settle-rate finding.

With single-layer legacy data the layer sweep just reports one row (the stored
layer), so this script is safe to dry-run before the multi-layer data lands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.a4_gates import gate1_topic_leakage, gate5_lean_separation  # noqa: E402
from wta.labeling import build_labels  # noqa: E402

# reuse the pipeline pieces so there is ONE implementation
from run_full_gates import (  # noqa: E402
    compute_gate7, fit_a1, fit_a2, fit_a3, prepare_split,
)


def layer_sweep(a0, classes, layers, n_ood, held_seeds, epochs) -> list[dict]:
    rows = []
    for layer in layers:
        ds = build_labels(a0, classes, layer=layer)
        split = prepare_split(ds, n_ood, held_seeds)
        model = fit_a2(ds, split["train"], epochs)
        t_tr = model.encode_topic(ds.h[split["train"]])
        t_ev = model.encode_topic(ds.h[split["evalm"]])
        g1 = gate1_topic_leakage(t_tr, ds.cls[split["train"]], t_ev,
                                 ds.cls[split["evalm"]],
                                 dec_he=ds.decision[split["evalm"]]).numbers
        g5 = gate5_lean_separation(model.encode_lean(ds.h[split["evalm"]]),
                                   ds.decision[split["evalm"]],
                                   ds.cls[split["evalm"]]).numbers
        rows.append({"layer": layer,
                     "g1_eta2": g1.get("mean_partial_eta2", float("nan")),
                     "g1_ndec": g1.get("n_decisions", 0),
                     "g5_ratio": g5.get("between_within_ratio", float("nan")),
                     "g5_sil": g5.get("silhouette", float("nan")),
                     "g5_ndec": g5.get("n_decisions", 0)})
    return rows


def eps_window_sweep(a0, classes, layer, epss, windows, n_ood, held_seeds, epochs):
    ds = build_labels(a0, classes, layer=layer)
    split = prepare_split(ds, n_ood, held_seeds)
    model = fit_a2(ds, split["train"], epochs)
    rows = []
    for eps in epss:
        for window in windows:
            calib, ref, n_pairs, n_seq = fit_a3(ds, model, split["train"], eps, window)
            g7 = compute_gate7(ds, model, split["evalm"], calib, window)
            rows.append({"eps": eps, "window": window,
                         "settle_rate": calib.n_points / max(n_seq, 1),
                         "n_settled": calib.n_points, "n_seq": n_seq,
                         "tau": calib.tau, "benign_ref": ref, "benign_pairs": n_pairs,
                         "gate7_medK": (g7.numbers["median_K"] if g7 else float("nan")),
                         "gate7_ndec": (g7.numbers["n_decisions"] if g7 else 0)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--layers", default="0", help="layer positions to sweep")
    ap.add_argument("--eps", default="0.4,0.6,0.8")
    ap.add_argument("--window", default="3,5")
    ap.add_argument("--n-ood", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--held-seeds", default="s6,s7")
    args = ap.parse_args()
    held = tuple(args.held_seeds.split(","))
    layers = [int(x) for x in args.layers.split(",")]
    epss = [float(x) for x in args.eps.split(",")]
    windows = [int(x) for x in args.window.split(",")]

    print("=== LAYER SWEEP (gate1 invariance + gate5 lean-separation) ===")
    print("layer  g1_eta2(want~0)  g1_ndec   g5_ratio(want>>1)  g5_sil   g5_ndec")
    rows = layer_sweep(args.a0, args.classes, layers, args.n_ood, held, args.epochs)
    for r in rows:
        print(f"{r['layer']:>5}  {r['g1_eta2']:>13.4f}  {r['g1_ndec']:>7}  "
              f"{r['g5_ratio']:>16.3f}  {r['g5_sil']:>6.3f}  {r['g5_ndec']:>7}")
    ranked = [r for r in rows if r["g5_ndec"] > 0]
    best_layer = (max(ranked, key=lambda r: r["g5_ratio"])["layer"]
                  if ranked else layers[0])
    print(f"\nbest layer by gate5 ratio: {best_layer}")

    print(f"\n=== EPS x WINDOW SWEEP (layer {best_layer}) ===")
    print("eps   window  settle_rate  tau     benign_ref(pairs)  gate7_medK  ndec")
    ew = eps_window_sweep(args.a0, args.classes, best_layer, epss, windows,
                          args.n_ood, held, args.epochs)
    for r in ew:
        print(f"{r['eps']:<5} {r['window']:<6}  {r['settle_rate']:>10.0%}  "
              f"{r['tau']:>6.3f}  {r['benign_ref']:>8.3f} ({r['benign_pairs']:>3})   "
              f"{r['gate7_medK']:>9}  {r['gate7_ndec']:>4}")
    best_ew = max(ew, key=lambda r: r["settle_rate"])
    print(f"\nbest A3 by settle-rate: eps={best_ew['eps']}, window={best_ew['window']} "
          f"({best_ew['settle_rate']:.0%})")
    print(f"\n=> for the trustworthy gate numbers run:\n   python scripts/run_full_gates.py "
          f"--layer {best_layer} --eps-settle {best_ew['eps']} "
          f"--window {best_ew['window']} --kfold 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
