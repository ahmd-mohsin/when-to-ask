"""T3 (decisions/028): probe-robustness appendix — closes the last gate-2 escape.

Gate-2 task (which held-out decision does this read belong to), FIXED labels,
same held-seed split as the text control (s6,s7; train filtered to test
classes; max 40k train / 20k test, rng seed 0 — the split code is replicated
verbatim from scripts/gate2_text_control.py), three activation probes:

  1. 256-d JL linear      — consistency check vs the recorded 0.2745 (026 B)
  2. full-dim 5120 linear — the escape "projection crippled the probe"
  3. 2-layer MLP, hidden 512 (full-dim input) — the ONE nonlinear family,
     pre-declared in 028; the escape "linear probes are too weak"

Reference to beat: the causal + anchors-masked TEXT baseline 0.730 on the
same task and split (results/gate2_text_control_masked_causal.json).
Whatever the numbers, they go in the paper as-run.

Runs on the box (needs models/v3_32b_fixed/labels.npz, ~6.5 GB h streamed):

    python scripts/gate2_probe_robustness.py \
        --labels models/v3_32b_fixed/labels.npz \
        --out results/gate2_probe_robustness.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gate2_text_control import stream_project  # noqa: E402 (same dir)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="models/v3_32b_fixed/labels.npz")
    ap.add_argument("--held-seeds", default="6,7")
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--max-test", type=int, default=20000)
    ap.add_argument("--mlp-hidden", type=int, default=512)
    ap.add_argument("--out", default="results/gate2_probe_robustness.json")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    npz = Path(args.labels)
    log("loading label arrays ...")
    z = np.load(npz, allow_pickle=False)
    dec = z["decision"]
    run_idx = z["run_idx"]
    meta = json.loads(str(z["meta"]))
    runs = [tuple(r) for r in meta["runs"]]
    log(f"reads={len(dec)} decisions={len(meta['decisions'])} runs={len(runs)}")
    # input-identity guard: the 0.730 / 0.2745 references are only valid for
    # the FIXED npz (026). The old models/v3_32b/labels.npz has a different
    # read count; refuse to burn hours on the wrong input.
    if len(dec) != 787281:
        log(f"!! reads={len(dec)} != 787281 — this is NOT the fixed-label "
            f"npz (models/v3_32b_fixed). Aborting before the probes.")
        return 1

    # --- split: verbatim gate2_text_control.py logic ----------------------
    held = {int(s) for s in args.held_seeds.split(",")}
    seed_of_row = np.array([int(runs[i][1].rsplit("-s", 1)[1]) for i in run_idx])
    m = dec >= 0
    is_held = np.isin(seed_of_row, list(held))
    tr = np.where(m & ~is_held)[0]
    te = np.where(m & is_held)[0]
    tr = tr[np.isin(dec[tr], np.unique(dec[te]))]
    rng = np.random.default_rng(0)
    if len(tr) > args.max_train:
        tr = np.sort(rng.choice(tr, args.max_train, replace=False))
    if len(te) > args.max_test:
        te = np.sort(rng.choice(te, args.max_test, replace=False))
    n_cls = len(np.unique(dec[te]))
    log(f"train={len(tr)} test={len(te)} classes={n_cls} chance={1 / n_cls:.4f}")
    if len(te) != 9180 or n_cls != 110:
        log(f"!! split mismatch: expected test=9180 classes=110 on the fixed "
            f"labels (reconstructed from the fixed debug trail). Aborting.")
        return 1

    sel = np.concatenate([tr, te])
    idx = {int(r): k for k, r in enumerate(sel)}
    tr_pos = [idx[int(i)] for i in tr]
    te_pos = [idx[int(i)] for i in te]
    out = {"n_train": int(len(tr)), "n_test": int(len(te)),
           "n_classes": int(n_cls), "chance": 1 / n_cls,
           "references": {"text_masked_causal": 0.730,
                          "activ_256d_recorded": 0.2745,
                          "gate2_reported_T": 0.4102}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        # after each probe: a late crash must not discard finished numbers
        out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # --- probe 1: 256-d JL linear (consistency check) ---------------------
    log("streaming h -> 256d JL ...")
    H = stream_project(npz, 256, sel, seed=0)
    Htr, Hte = H[tr_pos], H[te_pos]
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000)
    clf.fit((Htr - mu) / sd, dec[tr])
    out["acc_256d_linear"] = float(clf.score((Hte - mu) / sd, dec[te]))
    log(f"256d linear : acc={out['acc_256d_linear']:.4f} "
        f"(recorded 0.2745)")
    persist()
    del H, Htr, Hte

    # --- full-dim stream (probes 2 and 3 share it) ------------------------
    log("streaming h FULL-DIM (identity) ...")
    H = stream_project(npz, 1 << 30, sel, seed=0)  # dim >= d -> identity
    Htr, Hte = H[tr_pos], H[te_pos]
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    Xtr, Xte = (Htr - mu) / sd, (Hte - mu) / sd
    del H, Htr, Hte

    log("full-dim LINEAR probe ...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, dec[tr])
    out["acc_fulldim_linear"] = float(clf.score(Xte, dec[te]))
    log(f"full-dim linear : acc={out['acc_fulldim_linear']:.4f} "
        f"(old labels scored 0.5037)")
    persist()

    log(f"MLP probe (hidden {args.mlp_hidden}, relu, adam, "
        f"early stopping at sklearn-default patience, random_state 0) ...")
    # every hyperparameter 028 Amendment A does not pin stays at the sklearn
    # default (pre-run review: an unpinned non-default patience would reopen
    # the 'nonlinear probe was under-trained' escape T3 exists to close)
    mlp = MLPClassifier(hidden_layer_sizes=(args.mlp_hidden,),
                        activation="relu", alpha=1e-4, batch_size=256,
                        learning_rate_init=1e-3, max_iter=100,
                        early_stopping=True, random_state=0)
    mlp.fit(Xtr, dec[tr])
    out["acc_fulldim_mlp"] = float(mlp.score(Xte, dec[te]))
    out["mlp_config"] = {"hidden": args.mlp_hidden, "activation": "relu",
                         "alpha": 1e-4, "batch_size": 256,
                         "learning_rate_init": 1e-3, "max_iter": 100,
                         "early_stopping": True,
                         "n_iter_no_change": int(mlp.n_iter_no_change),
                         "validation_fraction": float(mlp.validation_fraction),
                         "random_state": 0,
                         "n_iter_": int(mlp.n_iter_)}
    log(f"full-dim MLP : acc={out['acc_fulldim_mlp']:.4f}")

    print()
    print(f"chance {out['chance']:.4f} | 256d {out['acc_256d_linear']:.4f} | "
          f"full-linear {out['acc_fulldim_linear']:.4f} | "
          f"full-MLP {out['acc_fulldim_mlp']:.4f} | "
          f"TEXT (causal+masked) 0.730")
    persist()
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
