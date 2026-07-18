"""Offline pipeline on REAL A0 data: labels -> A1 d -> A2 -> A3 -> gate features.

    python scripts/train_offline.py --a0 data/a0 --out models/

At the 3-task sample scale this is a DRY RUN of the machinery: every number it
prints is indicative only. The real gate run needs the full A0 collection and
is an owner stop-point (decisions/011). Nothing here tunes on held-out data:
the split is by SEED (runs s0-s5 train, s6-s7 held out); gate 6 (OOD) needs a
held-out task family and is skipped until the full collection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

from wta.a1_direction import ambiguity_signal, auroc, build_direction  # noqa: E402
from wta.a2_autoencoder import A2Config, grl_diagnostic, train_a2  # noqa: E402
from wta.a3_commitment import (  # noqa: E402
    benign_spread_reference, calibrate_tau, s_reference,
)
from wta.a4_gates import (  # noqa: E402
    gate1_topic_leakage, gate2_decision_recovery, gate3_fork_collocation,
    gate5_lean_separation,
)
from wta.labeling import build_labels, coverage_table  # noqa: E402

HELD_OUT_SEEDS = ("s6", "s7")


def class_balance_table(ds) -> str:
    """Per-decision class-label counts -- lopsided rows explain weak lean
    supervision before anyone blames the autoencoder."""
    import numpy as np

    lines = ["decision (task/blocker)                              class counts"]
    for did, (task, blocker) in enumerate(ds.vocab.decisions):
        counts = {}
        for cid in ds.vocab.class_of_decision[did]:
            n = int((ds.cls == cid).sum())
            if n:
                counts[ds.vocab.classes[cid][2]] = n
        if counts:
            lines.append(f"{task}/{blocker[:42]:<44} {counts}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--eps-settle", type=float, default=0.6)
    args = ap.parse_args()
    out = Path(args.out)

    out.mkdir(parents=True, exist_ok=True)
    print("=== labels (offline teacher; per-decision audit trail in "
          "labels_debug.jsonl) ===")
    ds = build_labels(args.a0, args.classes, debug_path=out / "labels_debug.jsonl")
    print(coverage_table(ds))
    print()
    print(class_balance_table(ds))
    ds.save(out / "labels.npz")

    held = np.array([ds.runs[r][1].endswith(HELD_OUT_SEEDS) for r in ds.run_idx])
    lab = ds.decision >= 0
    tr, he = lab & ~held, lab & held
    print(f"\nreads: {len(ds.h)} total, {tr.sum()} train / {he.sum()} held (decision-labeled)")

    print("\n=== A1: matched contrast (should-ask vs settled, same decisions) ===")
    d = build_direction(ds.h[tr & (ds.phase == 0)], ds.h[tr & (ds.phase == 1)])
    np.save(out / "a1_direction.npy", d)
    s_pos = ambiguity_signal(ds.h[he & (ds.phase == 0)], d)
    s_neg = ambiguity_signal(ds.h[he & (ds.phase == 1)], d)
    score = auroc(s_pos, s_neg) if len(np.atleast_1d(s_pos)) and len(np.atleast_1d(s_neg)) else float("nan")
    s_ref = s_reference(ambiguity_signal(ds.h[tr & (ds.phase == 0)], d),
                        ambiguity_signal(ds.h[tr & (ds.phase == 1)], d))
    print(f"held-out AUROC(s) = {score:.3f}  (n_pos={np.size(s_pos)}, n_neg={np.size(s_neg)}); s_ref={s_ref:.3f}")

    print("\n=== A2: disentangling autoencoder (loss curves -> a2_history.jsonl) ===")
    cfg = A2Config(in_dim=ds.h.shape[1], n_topics=len(ds.vocab.decisions),
                   n_classes=len(ds.vocab.classes), epochs=args.epochs, seed=0)
    hist = out / "a2_history.jsonl"
    hist.unlink(missing_ok=True)
    model = train_a2(ds.h[tr], ds.decision[tr], ds.cls[tr], cfg,
                     log_every=max(args.epochs // 5, 1), history_path=hist)
    model.save(out / "a2.pt")
    diag = grl_diagnostic(model, ds.h[he], ds.cls[he])
    print("GRL treadmill check (held-out):", json.dumps(diag))
    if diag.get("treadmill_suspected"):
        print("  ^^ WARNING: fresh probe reads the class from T while the "
              "adversary cannot -- invariance did NOT happen; expect gate 1 red.")

    print("\n=== A3: conformal commitment (window =", args.window, ") ===")
    seqs, groups_by_dec, seq_lens, too_short = [], {}, [], 0
    for r in range(len(ds.runs)):
        for dec in set(ds.decision[(ds.run_idx == r) & lab].tolist()):
            m = (ds.run_idx == r) & (ds.decision == dec)
            if m.sum() < args.window + 1:
                too_short += 1
                continue
            # generation order == row order; token_idx restarts per v2 segment
            # (decisions/017), so sorting by it would interleave turns.
            order = np.arange(int(m.sum()))
            r_seq = model.encode_lean(ds.h[m][order])
            seqs.append(r_seq)
            seq_lens.append(int(m.sum()))
            run_cls = ds.cls[m]
            run_cls = run_cls[run_cls >= 0]
            if len(run_cls):
                groups_by_dec.setdefault(dec, []).append((r_seq, int(run_cls[0])))
    calib = calibrate_tau(seqs, window=args.window, eps_settle=args.eps_settle)
    settled_frac = calib.n_points / max(len(seqs), 1)
    print(f"{len(seqs)} usable (run, decision) sequences "
          f"(+{too_short} too short; lengths min/med/max "
          f"{min(seq_lens or [0])}/{int(np.median(seq_lens or [0]))}/{max(seq_lens or [0])})")
    print(f"tau={calib.tau:.3f} (l_scale={calib.l_scale:.3f}) from {calib.n_points} "
          f"points; {calib.n_skipped_never_settled} never settled "
          f"({settled_frac:.0%} settle rate -- low rate means traces end before "
          f"the lean stabilizes: raise --max-new-tokens or eps)")
    ref, n_pairs = benign_spread_reference(list(groups_by_dec.values()),
                                           args.window, args.eps_settle)
    print(f"benign spread reference = {ref:.3f} from {n_pairs} same-class pairs "
          f"(Part B's CUSUM reference; NaN = no same-class settled pairs yet)")
    np.savez(out / "a3_calibration.npz", tau=calib.tau, l_scale=calib.l_scale,
             s_ref=s_ref, window=args.window, benign_reference=ref,
             benign_n_pairs=n_pairs)

    print("\n" + "=" * 62)
    print("DRY-RUN GATE PREVIEW — n=3 tasks, INDICATIVE ONLY, NOT the gate run")
    print("=" * 62)
    t_tr, t_he = model.encode_topic(ds.h[tr]), model.encode_topic(ds.h[he])
    print(" ", gate1_topic_leakage(t_tr, ds.cls[tr], t_he, ds.cls[he]))
    print(" ", gate2_decision_recovery(t_tr, ds.decision[tr], t_he, ds.decision[he]))
    print(" ", gate3_fork_collocation(t_he, ds.decision[he], ds.cls[he]))
    print(" ", gate5_lean_separation(model.encode_lean(ds.h[he]),
                                     ds.decision[he], ds.cls[he]))
    print("  gate6_ood_transfer: SKIPPED (needs a held-out task family; full collection)")
    print("  gate7_lead_time: SKIPPED at sample scale (needs per-decision fork coverage)")
    print("\nArtifacts in", out.resolve(), "- the real gate run is scripts/run_gates.py "
          "on the full collection, then STOP for owner review (decisions/011).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
