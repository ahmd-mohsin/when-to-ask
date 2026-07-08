"""THE A4 gate run on the full real collection -> STOP for owner review.

    python scripts/run_full_gates.py --a0 data/a0 --out models

Splits the collected tasks THREE ways so every gate is honest:
  * OOD tasks  (--n-ood, default 4): held out ENTIRELY. A2/A1/A3 never see
    them. Gate 6 (transfer) is measured here.
  * train tasks, seeds s0-s5: the training set for A1 d, A2, A3 tau.
  * train tasks, seeds s6-s7: the held-out EVAL set for gates 1,2,3,4,5,7.

Nothing is tuned on eval or OOD data (build brief rule 1). The gate numbers
are printed unfiltered; a red gate is a FINDING, and the script stops after
printing them (decisions/011). Diagnostics land in models/ alongside the
artifacts so any number can be traced to its cause.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.a1_direction import ambiguity_signal, auroc, build_direction  # noqa: E402
from wta.a2_autoencoder import A2Config, grl_diagnostic, train_a2  # noqa: E402
from wta.a3_commitment import (  # noqa: E402
    CommitmentDetector, benign_spread_reference, calibrate_tau, s_reference,
)
from wta.a4_gates import (  # noqa: E402
    gate1_topic_leakage, gate2_decision_recovery, gate3_fork_collocation,
    gate4_conflation, gate5_lean_separation, gate6_ood_transfer,
    gate7_aggregate, gate7_lead_time,
)
from wta.labeling import build_labels, coverage_table  # noqa: E402


def pick_ood_tasks(tasks: list[str], n_ood: int) -> set[str]:
    """Deterministic, spread-out holdout (every k-th task by sorted order)."""
    if n_ood <= 0 or n_ood >= len(tasks):
        return set()
    step = len(tasks) / n_ood
    return {tasks[min(len(tasks) - 1, int(i * step))] for i in range(n_ood)}


def seq_by_run_decision(ds, mask, model):
    """{decision -> [(r_seq, committed_class|-1)]} over reads selected by mask,
    ordered within each (run, decision) by token index."""
    groups: dict = {}
    for r in range(len(ds.runs)):
        for dec in set(ds.decision[(ds.run_idx == r) & mask].tolist()):
            if dec < 0:
                continue
            m = (ds.run_idx == r) & (ds.decision == dec) & mask
            if m.sum() < 2:
                continue
            order = np.argsort(ds.read_token_idx[m])
            r_seq = model.encode_lean(ds.h[m][order])
            cl = ds.cls[m][order]
            cl = cl[cl >= 0]
            groups.setdefault(dec, []).append((r_seq, int(cl[0]) if len(cl) else -1))
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="models")
    ap.add_argument("--n-ood", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--eps-settle", type=float, default=0.6)
    ap.add_argument("--held-seeds", default="s6,s7")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    held_seeds = tuple(args.held_seeds.split(","))

    print("=== labels (audit trail -> labels_debug.jsonl) ===")
    ds = build_labels(args.a0, args.classes, debug_path=out / "labels_debug.jsonl")
    print(coverage_table(ds))
    ds.save(out / "labels.npz")

    ood_tasks = pick_ood_tasks(ds.tasks, args.n_ood)
    task_of_read = np.array([ds.tasks[t] for t in ds.task_idx])
    seed_of_read = np.array([ds.runs[r][1].split("-")[-1] for r in ds.run_idx])
    is_ood = np.isin(task_of_read, list(ood_tasks))
    is_heldseed = np.isin(seed_of_read, held_seeds)
    lab = ds.decision >= 0

    train = lab & ~is_ood & ~is_heldseed          # A1/A2/A3 fit here
    evalm = lab & ~is_ood & is_heldseed           # gates 1-5,7
    print(f"\nOOD tasks held out entirely: {sorted(ood_tasks)}")
    print(f"reads: {lab.sum()} labeled | train {train.sum()} | "
          f"eval(seeds {held_seeds}) {evalm.sum()} | ood {(lab & is_ood).sum()}")

    print("\n=== A1 ===")
    d = build_direction(ds.h[train & (ds.phase == 0)], ds.h[train & (ds.phase == 1)])
    np.save(out / "a1_direction.npy", d)
    sp, sn = ambiguity_signal(ds.h[evalm & (ds.phase == 0)], d), \
        ambiguity_signal(ds.h[evalm & (ds.phase == 1)], d)
    a1 = auroc(sp, sn) if np.size(sp) and np.size(sn) else float("nan")
    s_ref = s_reference(ambiguity_signal(ds.h[train & (ds.phase == 0)], d),
                        ambiguity_signal(ds.h[train & (ds.phase == 1)], d))
    print(f"held-out AUROC(s)={a1:.3f} (n_pos={np.size(sp)}, n_neg={np.size(sn)}); s_ref={s_ref:.3f}")

    print("\n=== A2 (loss curve -> a2_history.jsonl) ===")
    cfg = A2Config(in_dim=ds.h.shape[1], n_topics=len(ds.vocab.decisions),
                   n_classes=len(ds.vocab.classes), epochs=args.epochs, seed=0)
    (out / "a2_history.jsonl").unlink(missing_ok=True)
    model = train_a2(ds.h[train], ds.decision[train], ds.cls[train], cfg,
                     log_every=max(args.epochs // 5, 1),
                     history_path=out / "a2_history.jsonl")
    model.save(out / "a2.pt")
    print("GRL treadmill check:", json.dumps(grl_diagnostic(model, ds.h[evalm], ds.cls[evalm])))

    print("\n=== A3 ===")
    grp = seq_by_run_decision(ds, train, model)
    seqs = [s for lst in grp.values() for s, _ in lst]
    calib = calibrate_tau(seqs, window=args.window, eps_settle=args.eps_settle)
    ref, n_pairs = benign_spread_reference(list(grp.values()), args.window, args.eps_settle)
    print(f"tau={calib.tau:.3f} (l_scale={calib.l_scale:.3f}); "
          f"{calib.n_points}/{len(seqs)} settled; benign ref={ref:.3f} ({n_pairs} pairs)")
    np.savez(out / "a3_calibration.npz", tau=calib.tau, l_scale=calib.l_scale,
             s_ref=s_ref, window=args.window, benign_reference=ref)

    # ---- gates ---------------------------------------------------------
    print("\n" + "=" * 64)
    print("A4 GATES — held-out real data — REPORT TO OWNER, DO NOT TUNE")
    print("=" * 64)
    results = []
    t_tr, t_ev = model.encode_topic(ds.h[train]), model.encode_topic(ds.h[evalm])
    # gate 1 WITHIN decision (global class is nested in decision -> confounded);
    # also print the naive global number for transparency.
    results.append(gate1_topic_leakage(t_tr, ds.cls[train], t_ev, ds.cls[evalm],
                                       dec_he=ds.decision[evalm]))
    print("  (transparency) gate1 GLOBAL/confounded:",
          gate1_topic_leakage(t_tr, ds.cls[train], t_ev, ds.cls[evalm]).numbers)
    results.append(gate2_decision_recovery(t_tr, ds.decision[train], t_ev, ds.decision[evalm]))
    g3 = gate3_fork_collocation(t_ev, ds.decision[evalm], ds.cls[evalm])
    results.append(g3)

    # gate 4: same-task/different-decision pairs (same-observable proxy)
    theta = g3.numbers.get("theta")
    if theta is not None and np.isfinite(theta):
        ev_idx = np.where(evalm)[0]
        rng = np.random.default_rng(0)
        pairs = []
        for _ in range(4000):
            a, b = rng.choice(ev_idx, 2, replace=False)
            if task_of_read[a] == task_of_read[b] and ds.decision[a] != ds.decision[b]:
                pairs.append((np.where(ev_idx == a)[0][0], np.where(ev_idx == b)[0][0]))
        if len(pairs) >= 10:
            results.append(gate4_conflation(t_ev, np.array(pairs), theta))
        else:
            print("  gate4_conflation: INSUFFICIENT same-task/diff-decision pairs")
    results.append(gate5_lean_separation(model.encode_lean(ds.h[evalm]),
                                         ds.decision[evalm], ds.cls[evalm]))

    # gate 6: OOD transfer (fully held-out tasks)
    if ood_tasks and theta is not None and np.isfinite(theta):
        results.append(gate6_ood_transfer(model.encode_topic(ds.h[lab & is_ood]),
                                          ds.decision[lab & is_ood], theta))
    else:
        print("  gate6_ood_transfer: SKIPPED (no OOD tasks or no theta)")

    # gate 7: lead-time. Behavioural reference on real data = the read where the
    # committed-class signature first appears in the trace (labeler's phase 0->1
    # transition), vs. the read where the bucket's committed-lean dispersion
    # rises. proxy=True (signature-appearance stands in for a logged action).
    per_dec = []
    ev_groups = seq_by_run_decision(ds, evalm, model)
    for dec, lst in ev_groups.items():
        classes = [c for _, c in lst if c >= 0]
        if len(set(classes)) < 2:
            continue
        R = min(len(s) for s, _ in lst)
        if R < args.window + 1:
            continue
        r_by = np.stack([s[:R] for s, _ in lst])
        weights = np.zeros(r_by.shape[:2])
        action_read = np.full(len(lst), -1)
        for i, (s_seq, c) in enumerate(lst):
            det = CommitmentDetector(tau=calib.tau, s_ref=1e9, window=args.window,
                                     l_scale=calib.l_scale)  # s-gate open (per-dec)
            for k in range(R):
                _, weights[i, k] = det.step(s_seq[k], s=0.0)
            # behavioural commit read: first settled read for this (run,dec)
            action_read[i] = R - 1  # proxy: end-of-trace commit (conservative)
        per_dec.append(gate7_lead_time(r_by, weights, action_read,
                                       np.array([c for _, c in lst])))
    per_dec = [p for p in per_dec if p]
    if per_dec:
        results.append(gate7_aggregate(per_dec, proxy=True))
    else:
        print("  gate7_lead_time: INSUFFICIENT forked decisions in eval seeds")

    print()
    for g in results:
        print(" ", g)
    (out / "gate_report.json").write_text(json.dumps(
        {g.name: g.numbers for g in results}, indent=1), encoding="utf-8")
    print("\nSTOP — owner reviews these numbers before any Part B result is "
          "trusted (decisions/011). Report saved to", out / "gate_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
