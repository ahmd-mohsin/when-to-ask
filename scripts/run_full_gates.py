"""THE A4 gate run on the full real collection -> STOP for owner review.

    python scripts/run_full_gates.py --a0 data/a0 --out models
    python scripts/run_full_gates.py --layer 17 --eps-settle 0.4,0.6,0.8 --window 3,5
    python scripts/run_full_gates.py --kfold 5           # power fix (decisions/014)

Splits the collected tasks THREE ways so every gate is honest:
  * OOD tasks  (--n-ood, default 4): held out ENTIRELY -> gate 6 (transfer).
  * train tasks, seeds s0-s5: training set for A1 d, A2, A3 tau.
  * train tasks, seeds s6-s7: held-out EVAL for gates 1,2,3,4,5,7.

--kfold N replaces the single seed-holdout with N-fold cross-fitting over
(task, seed) groups (retrain A2 per fold, pool gates 1/5/7) for statistical
power. Nothing is tuned on eval/OOD (build brief rule 1); numbers print
unfiltered and the script STOPS (decisions/011). The pipeline pieces are
importable so scripts/sweep.py can drive layer/eps sweeps without duplication.
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
    gate7_aggregate, gate7_lead_time, kfold_group_indices,
)
from wta.labeling import (LabeledDataset, build_labels, coverage_table,  # noqa: E402
                          resolve_tokenizer)


# ---------------------------------------------------------------------------
# reusable pipeline pieces (imported by sweep.py)
# ---------------------------------------------------------------------------


def pick_ood_tasks(tasks: list[str], n_ood: int) -> set[str]:
    """Deterministic, spread-out holdout (every k-th task by sorted order)."""
    if n_ood <= 0 or n_ood >= len(tasks):
        return set()
    step = len(tasks) / n_ood
    return {tasks[min(len(tasks) - 1, int(i * step))] for i in range(n_ood)}


def prepare_split(ds, n_ood: int, held_seeds: tuple[str, ...]) -> dict:
    ood = pick_ood_tasks(ds.tasks, n_ood)
    task_of_read = np.array([ds.tasks[t] for t in ds.task_idx])
    seed_of_read = np.array([ds.runs[r][1].split("-")[-1] for r in ds.run_idx])
    is_ood = np.isin(task_of_read, list(ood))
    is_held = np.isin(seed_of_read, held_seeds)
    lab = ds.decision >= 0
    return dict(ood_tasks=ood, task_of_read=task_of_read, seed_of_read=seed_of_read,
                is_ood=is_ood, is_held=is_held, lab=lab,
                train=lab & ~is_ood & ~is_held, evalm=lab & ~is_ood & is_held)


def fit_a1(ds, train):
    d = build_direction(ds.h[train & (ds.phase == 0)], ds.h[train & (ds.phase == 1)])
    s_ref = s_reference(ambiguity_signal(ds.h[train & (ds.phase == 0)], d),
                        ambiguity_signal(ds.h[train & (ds.phase == 1)], d))
    return d, s_ref


def fit_a2(ds, train, epochs: int, out: Path | None = None):
    cfg = A2Config(in_dim=ds.h.shape[1], n_topics=len(ds.vocab.decisions),
                   n_classes=len(ds.vocab.classes), epochs=epochs, seed=0)
    hist = None
    if out is not None:
        hist = out / "a2_history.jsonl"
        hist.unlink(missing_ok=True)
    return train_a2(ds.h[train], ds.decision[train], ds.cls[train], cfg,
                    log_every=(max(epochs // 5, 1) if out else 0), history_path=hist)


def seq_by_run_decision(ds, mask, model):
    """{decision -> [(r_seq, committed_class|-1)]} over masked reads, in
    generation order. build_labels appends a run's reads chronologically, so
    row order IS generation order; sorting by read_token_idx would interleave
    v2 turns (token_idx restarts per segment, decisions/017)."""
    groups: dict = {}
    for r in range(len(ds.runs)):
        for dec in set(ds.decision[(ds.run_idx == r) & mask].tolist()):
            if dec < 0:
                continue
            m = (ds.run_idx == r) & (ds.decision == dec) & mask
            if m.sum() < 2:
                continue
            order = np.arange(int(m.sum()))
            cl = ds.cls[m][order]
            cl = cl[cl >= 0]
            groups.setdefault(dec, []).append(
                (model.encode_lean(ds.h[m][order]), int(cl[0]) if len(cl) else -1, r))
    return groups


def real_action_reads(ds, a0_dir, classes_path) -> dict:
    """{(run_idx, decision) -> read index of the FIRST ActionEvent whose
    command matches the run's committed-class signatures; -1 = never acted}.

    The v2 replacement for gate7's end-of-trace proxy (decisions/017): the
    behavioural commitment moment is the tool call that writes the choice,
    located by (segment_idx, token_idx) against the (run, decision) read
    sequence. Runs without logged actions (v1 data) produce no entries."""
    from wta.labeling import _norm, load_class_artifact

    art = load_class_artifact(classes_path)
    sigs_of: dict[int, list[str]] = {}
    for did, (task, blocker) in enumerate(ds.vocab.decisions):
        spec = art[task][blocker]
        for local, gcls in enumerate(ds.vocab.class_of_decision[did]):
            sigs_of[gcls] = [_norm(s) for s in spec["classes"][local]["signatures"]]

    out: dict = {}
    for r, (task, run_id) in enumerate(ds.runs):
        meta = json.loads((Path(a0_dir) / task / f"{run_id}.json")
                          .read_text(encoding="utf-8"))
        actions = meta.get("actions") or []
        if not actions:
            continue
        # ds rows for run r are meta["reads"] in order (build_labels appends
        # every read), so ordinal k here == ordinal position in the row block
        pos_of_read = [(rd.get("segment_idx", 0), rd["token_idx"])
                       for rd in meta["reads"]]
        rows = np.where(ds.run_idx == r)[0]
        for dec in set(ds.decision[rows].tolist()):
            if dec < 0:
                continue
            in_dec = ds.decision[rows] == dec
            cls_vals = ds.cls[rows[in_dec]]
            cls_vals = cls_vals[cls_vals >= 0]
            if not len(cls_vals):
                continue
            sigs = [s for s in sigs_of.get(int(cls_vals[0]), []) if s]
            a_pos = next(((a.get("segment_idx", 0), a["token_idx"])
                          for a in actions
                          if any(s in _norm(a.get("action_text", "")) for s in sigs)),
                         None)
            if a_pos is None:
                out[(r, dec)] = -1
                continue
            seq_pos = [pos_of_read[k] for k in np.where(in_dec)[0]]
            n_before = sum(1 for p in seq_pos if p <= a_pos)
            out[(r, dec)] = max(n_before - 1, 0)
    return out


def fit_a3(ds, model, train, eps: float, window: int):
    grp = seq_by_run_decision(ds, train, model)
    seqs = [s for lst in grp.values() for s, *_ in lst]
    calib = calibrate_tau(seqs, window=window, eps_settle=eps)
    ref, n_pairs = benign_spread_reference(
        [[(s, c) for s, c, _ in lst] for lst in grp.values()], window, eps)
    return calib, ref, n_pairs, len(seqs)


def compute_gate7(ds, model, evalm, calib, window,
                  action_reads: dict | None = None) -> "object | None":
    """action_reads: real behavioural-commit indices from real_action_reads();
    None -> the v1 end-of-trace proxy."""
    per_dec = []
    for dec, lst in seq_by_run_decision(ds, evalm, model).items():
        if len({c for _, c, _ in lst if c >= 0}) < 2:
            continue
        R = min(len(s) for s, *_ in lst)
        if R < window + 1:
            continue
        r_by = np.stack([s[:R] for s, *_ in lst])
        weights = np.zeros(r_by.shape[:2])
        if action_reads is None:
            action_read = np.full(len(lst), R - 1)  # proxy: end-of-trace commit
        else:
            action_read = np.array([min(action_reads.get((r, dec), -1), R - 1)
                                    for _, _, r in lst])
        for i, (s_seq, *_) in enumerate(lst):
            det = CommitmentDetector(tau=calib.tau, s_ref=1e9, window=window,
                                     l_scale=calib.l_scale)
            for k in range(R):
                _, weights[i, k] = det.step(s_seq[k], s=0.0)
        g = gate7_lead_time(r_by, weights, action_read,
                            np.array([c for _, c, _ in lst]))
        if g:
            per_dec.append(g)
    return (gate7_aggregate(per_dec, proxy=action_reads is None)
            if per_dec else None)


def gates_1to6(ds, model, split, out_notes: list):
    """Gates 1-6 on a single split. Returns (results, theta)."""
    train, evalm = split["train"], split["evalm"]
    task_of_read, is_ood, lab = split["task_of_read"], split["is_ood"], split["lab"]
    t_tr, t_ev = model.encode_topic(ds.h[train]), model.encode_topic(ds.h[evalm])
    results = [
        gate1_topic_leakage(t_tr, ds.cls[train], t_ev, ds.cls[evalm],
                            dec_he=ds.decision[evalm]),
        gate2_decision_recovery(t_tr, ds.decision[train], t_ev, ds.decision[evalm]),
    ]
    out_notes.append("gate1 GLOBAL/confounded: " + json.dumps(
        gate1_topic_leakage(t_tr, ds.cls[train], t_ev, ds.cls[evalm]).numbers))
    g3 = gate3_fork_collocation(t_ev, ds.decision[evalm], ds.cls[evalm])
    results.append(g3)
    theta = g3.numbers.get("theta")

    if theta is not None and np.isfinite(theta):
        ev_idx = np.where(evalm)[0]
        rng = np.random.default_rng(0)
        pairs = []
        for _ in range(4000):
            a, b = rng.choice(ev_idx, 2, replace=False)
            if task_of_read[a] == task_of_read[b] and ds.decision[a] != ds.decision[b]:
                pairs.append((int(np.where(ev_idx == a)[0][0]),
                              int(np.where(ev_idx == b)[0][0])))
        if len(pairs) >= 10:
            results.append(gate4_conflation(t_ev, np.array(pairs), theta))
        else:
            out_notes.append("gate4_conflation: INSUFFICIENT same-task/diff-decision pairs")
    results.append(gate5_lean_separation(model.encode_lean(ds.h[evalm]),
                                         ds.decision[evalm], ds.cls[evalm]))
    if split["ood_tasks"] and theta is not None and np.isfinite(theta):
        results.append(gate6_ood_transfer(model.encode_topic(ds.h[lab & is_ood]),
                                          ds.decision[lab & is_ood], theta))
    else:
        out_notes.append("gate6_ood_transfer: SKIPPED (no OOD tasks or no theta)")
    return results, theta


def kfold_gates(ds, split, k: int, epochs: int, eps: float, window: int,
                seed: int = 0, action_reads: dict | None = None,
                lhe_dump: Path | None = None) -> dict:
    """N-fold cross-fit over non-OOD (task, seed) groups; pool gates
    1/2/3/5/6/7 (decisions/014; 2/3/6 added by decisions/026 §5 — at 24 seeds
    the s6,s7 single split had zero fork pairs, so gates 3 and 6 had NO 32B
    measurement while this path only pooled 1/5/7). Retrains A2 per fold."""
    base = split["lab"] & ~split["is_ood"]
    base_idx = np.where(base)[0]
    gid = np.array([f"{split['task_of_read'][i]}::{split['seed_of_read'][i]}"
                    for i in base_idx])
    ood = split["lab"] & split["is_ood"]
    acc = {"g1_acc": [], "g1_chance": [], "g1_eta2": [], "g1_ndec": [],
           "g2_acc": [], "g2_chance": [],
           "g3_same": [], "g3_diff": [], "g3_theta": [], "g3_nsame": [],
           "g5_ratio": [], "g5_sil": [], "g5_ndec": [],
           "g6_purity": [], "g6_nbuckets": [], "g6_ndec": [],
           "g7_medK": [], "g7_fracpos": [], "g7_ndec": [], "folds": 0}
    for tr_local, te_local in kfold_group_indices(gid, k, seed):
        train = np.zeros(len(ds.h), bool); train[base_idx[tr_local]] = True
        evalm = np.zeros(len(ds.h), bool); evalm[base_idx[te_local]] = True
        model = fit_a2(ds, train, epochs)
        t_tr, t_ev = model.encode_topic(ds.h[train]), model.encode_topic(ds.h[evalm])
        g1 = gate1_topic_leakage(t_tr, ds.cls[train], t_ev, ds.cls[evalm],
                                 dec_he=ds.decision[evalm]).numbers
        g2 = gate2_decision_recovery(t_tr, ds.decision[train],
                                     t_ev, ds.decision[evalm]).numbers
        g3 = gate3_fork_collocation(t_ev, ds.decision[evalm],
                                    ds.cls[evalm]).numbers
        l_ev = model.encode_lean(ds.h[evalm])
        g5 = gate5_lean_separation(l_ev, ds.decision[evalm],
                                   ds.cls[evalm]).numbers
        if lhe_dump is not None:
            # decisions/026 §5: persist each fold's held-out LEAN embeddings
            # so gate5_lhe_permutation.py can run the run-level permutation
            # on the exact space gate5's kfold ratio is computed in (the
            # HANDOFF 1b "missing step"). Small: eval reads x lean dim.
            lhe_dump.mkdir(parents=True, exist_ok=True)
            keep = ds.cls[evalm] >= 0
            np.savez_compressed(
                lhe_dump / f"fold{acc['folds']}.npz",
                l=l_ev[keep].astype(np.float32),
                decision=ds.decision[evalm][keep],
                cls=ds.cls[evalm][keep],
                run_idx=ds.run_idx[evalm][keep])
        calib, *_ = fit_a3(ds, model, train, eps, window)
        g7 = compute_gate7(ds, model, evalm, calib, window, action_reads)
        acc["folds"] += 1
        if "within_decision_class_from_T_acc" in g1:
            acc["g1_acc"].append(g1["within_decision_class_from_T_acc"])
            acc["g1_chance"].append(g1["within_decision_chance"])
            acc["g1_eta2"].append(g1["mean_partial_eta2"])
            acc["g1_ndec"].append(g1["n_decisions"])
        if "topic_from_T_acc" in g2:
            acc["g2_acc"].append(g2["topic_from_T_acc"])
            acc["g2_chance"].append(g2["chance"])
        theta = g3.get("theta")
        if theta is not None and np.isfinite(theta):
            acc["g3_same"].append(g3["mean_same_decision_cos"])
            acc["g3_diff"].append(g3["mean_diff_decision_cos"])
            acc["g3_theta"].append(theta)
            acc["g3_nsame"].append(g3["n_same"])
            if ood.any():
                g6 = gate6_ood_transfer(model.encode_topic(ds.h[ood]),
                                        ds.decision[ood], theta).numbers
                acc["g6_purity"].append(g6["bucket_purity"])
                acc["g6_nbuckets"].append(g6["n_buckets"])
                acc["g6_ndec"].append(g6["n_decisions"])
        if "between_within_ratio" in g5:
            acc["g5_ratio"].append(g5["between_within_ratio"])
            acc["g5_sil"].append(g5["silhouette"])
            acc["g5_ndec"].append(g5["n_decisions"])
        if g7 is not None and not np.isnan(g7.numbers.get("median_K", np.nan)):
            acc["g7_medK"].append(g7.numbers["median_K"])
            acc["g7_fracpos"].append(g7.numbers["frac_positive"])
            acc["g7_ndec"].append(g7.numbers["n_decisions"])
    return acc


def _ms(xs):
    a = np.array(xs, dtype=float)
    return (float("nan"), float("nan"), 0) if not len(a) else \
        (float(a.mean()), float(a.std()), int(a.sum()) if a.dtype == float else len(a))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _floats(s):
    return [float(x) for x in str(s).split(",")]


def _ints(s):
    return [int(x) for x in str(s).split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="models")
    ap.add_argument("--n-ood", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--layer", default=None,
                    help="layer index/position for multi-layer data (select-at-load)")
    ap.add_argument("--window", default="3", help="comma list -> A3/gate7 sweep")
    ap.add_argument("--eps-settle", default="0.6", help="comma list -> A3/gate7 sweep")
    ap.add_argument("--held-seeds", default="s6,s7")
    ap.add_argument("--kfold", type=int, default=0,
                    help="N>0 -> N-fold cross-fit gates 1/5/7 (power fix)")
    ap.add_argument("--labels-npz", default=None,
                    help="load a prebuilt labels.npz instead of rebuilding "
                         "(decisions/026: skips the ~2h/16GB label pass; the "
                         "npz must have been built with the SAME --layer)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layer = None if args.layer is None else int(args.layer)
    epss, windows = _floats(args.eps_settle), _ints(args.window)

    if args.labels_npz:
        print(f"=== labels (prebuilt: {args.labels_npz}; "
              f"debug trail not regenerated) ===")
        ds = LabeledDataset.load(args.labels_npz)
    else:
        print("=== labels (audit trail -> labels_debug.jsonl) ===")
        ds = build_labels(args.a0, args.classes,
                          debug_path=out / "labels_debug.jsonl",
                          layer=layer,
                          tokenizer_name=resolve_tokenizer(args.a0, "auto"))
        ds.save(out / "labels.npz")
    print(coverage_table(ds))
    split = prepare_split(ds, args.n_ood, tuple(args.held_seeds.split(",")))
    print(f"\nOOD tasks: {sorted(split['ood_tasks'])} | layer={layer}")
    print(f"reads: {split['lab'].sum()} labeled | train {split['train'].sum()} | "
          f"eval {split['evalm'].sum()} | ood {(split['lab'] & split['is_ood']).sum()}")

    action_reads = real_action_reads(ds, args.a0, args.classes)
    if action_reads:
        acted = sum(1 for v in action_reads.values() if v >= 0)
        print(f"real action-commit reads (gate7, decisions/017): "
              f"{acted}/{len(action_reads)} committed (run, decision) pairs "
              f"matched to an ActionEvent; unmatched -> never-acted")
    else:
        action_reads = None
        print("no ActionEvents in this collection -> gate7 uses the "
              "end-of-trace PROXY")

    if args.kfold > 0:
        print(f"\n=== {args.kfold}-FOLD CROSS-FIT GATES (eps={epss[0]}, window={windows[0]}) ===")
        acc = kfold_gates(ds, split, args.kfold, args.epochs, epss[0], windows[0],
                          action_reads=action_reads,
                          lhe_dump=out / "lhe_folds")
        g1 = _ms(acc["g1_acc"]); g1c = _ms(acc["g1_chance"]); g1e = _ms(acc["g1_eta2"])
        g2 = _ms(acc["g2_acc"]); g2c = _ms(acc["g2_chance"])
        g3s = _ms(acc["g3_same"]); g3d = _ms(acc["g3_diff"]); g3t = _ms(acc["g3_theta"])
        g5r = _ms(acc["g5_ratio"]); g5s = _ms(acc["g5_sil"])
        g6p = _ms(acc["g6_purity"])
        g7 = _ms(acc["g7_medK"]); g7f = _ms(acc["g7_fracpos"])
        print(f"folds={acc['folds']}")
        print(f"  gate1 within-decision: acc {g1[0]:.3f}+-{g1[1]:.3f} vs chance "
              f"{g1c[0]:.3f}; partial eta2 {g1e[0]:.3f}+-{g1e[1]:.3f} "
              f"(pooled decisions {sum(acc['g1_ndec'])})")
        print(f"  gate2 decision-recovery: acc {g2[0]:.3f}+-{g2[1]:.3f} vs "
              f"chance {g2c[0]:.4f} ({len(acc['g2_acc'])} folds)")
        print(f"  gate3 fork-collocation: same {g3s[0]:.3f}+-{g3s[1]:.3f} vs "
              f"diff {g3d[0]:.3f}+-{g3d[1]:.3f}, theta {g3t[0]:.3f} "
              f"(pooled same-pairs {sum(acc['g3_nsame'])}, "
              f"{len(acc['g3_theta'])} folds with theta)")
        print(f"  gate5 lean-sep: ratio {g5r[0]:.3f}+-{g5r[1]:.3f}, silhouette "
              f"{g5s[0]:.3f}+-{g5s[1]:.3f} (pooled decisions {sum(acc['g5_ndec'])})")
        print(f"  gate6 OOD purity: {g6p[0]:.3f}+-{g6p[1]:.3f} "
              f"({len(acc['g6_purity'])} folds; pooled OOD decisions "
              f"{sum(acc['g6_ndec'])})")
        print(f"  gate7 lead-time: median_K {g7[0]:.2f}+-{g7[1]:.2f}, frac_pos "
              f"{g7f[0]:.2f} (pooled decisions {sum(acc['g7_ndec'])})")
        (out / "gate_report_kfold.json").write_text(json.dumps(acc, indent=1), encoding="utf-8")
        print("\nSTOP -- owner reviews (decisions/011). Saved gate_report_kfold.json")
        return 0

    # single-split path (default)
    d, s_ref = fit_a1(ds, split["train"])
    np.save(out / "a1_direction.npy", d)
    sp = ambiguity_signal(ds.h[split["evalm"] & (ds.phase == 0)], d)
    sn = ambiguity_signal(ds.h[split["evalm"] & (ds.phase == 1)], d)
    a1 = auroc(sp, sn) if np.size(sp) and np.size(sn) else float("nan")
    print(f"\n=== A1 === held-out AUROC(s)={a1:.3f} (n_pos={np.size(sp)}, "
          f"n_neg={np.size(sn)}); s_ref={s_ref:.3f}")

    print("\n=== A2 (loss -> a2_history.jsonl) ===")
    model = fit_a2(ds, split["train"], args.epochs, out=out)
    model.save(out / "a2.pt")
    print("GRL treadmill:", json.dumps(grl_diagnostic(model, ds.h[split["evalm"]],
                                                       ds.cls[split["evalm"]])))

    print("\n=== A3 (eps x window sweep) ===")
    best = None
    for eps in epss:
        for window in windows:
            calib, ref, n_pairs, n_seq = fit_a3(ds, model, split["train"], eps, window)
            rate = calib.n_points / max(n_seq, 1)
            g7 = compute_gate7(ds, model, split["evalm"], calib, window, action_reads)
            k = g7.numbers["median_K"] if g7 else float("nan")
            print(f"  eps={eps} window={window}: settle {calib.n_points}/{n_seq} "
                  f"({rate:.0%}), tau={calib.tau:.3f}, benign ref={ref:.3f} "
                  f"({n_pairs}p), gate7 medK={k}")
            if best is None or rate > best[0]:
                best = (rate, eps, window, calib, ref, g7)
    _, beps, bwin, bcalib, bref, bg7 = best
    np.savez(out / "a3_calibration.npz", tau=bcalib.tau, l_scale=bcalib.l_scale,
             s_ref=s_ref, window=bwin, eps_settle=beps, benign_reference=bref)

    print("\n" + "=" * 64)
    print("A4 GATES -- held-out real data -- REPORT TO OWNER, DO NOT TUNE")
    print(f"(A3 best by settle-rate: eps={beps}, window={bwin})")
    print("=" * 64)
    notes: list = []
    results, _ = gates_1to6(ds, model, split, notes)
    if bg7 is not None:
        results.append(bg7)
    else:
        notes.append("gate7_lead_time: INSUFFICIENT forked decisions in eval")
    print()
    for g in results:
        print(" ", g)
    for n in notes:
        print("  (note)", n)
    (out / "gate_report.json").write_text(json.dumps(
        {**{g.name: g.numbers for g in results}, "_notes": notes}, indent=1),
        encoding="utf-8")
    print("\nSTOP -- owner reviews before any Part B result (decisions/011). "
          "Report -> " + str(out / "gate_report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
