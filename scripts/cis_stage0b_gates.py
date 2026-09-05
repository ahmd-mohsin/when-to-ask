"""Stage-0b gate evaluator for the 029.3 lean readout (decisions/029 Am.029.3).
CPU only. Every gate is printed beside its frozen bar; any failing gate ->
NO-GO and Stage 2 is not fit.

    python scripts/cis_stage0b_gates.py --pilot /ssd3/wta-lean/pilot \
        --replicate /ssd3/wta-lean/pilot_rep --fullinfo /ssd3/wta-lean/pilot_fi \
        --a0 /ssd/wta_data/a0_v3_32b --labels-debug models/v3_32b_fixed/labels_debug.jsonl \
        --out results/cis_lean_stage0b_pilot.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cis_stage0_gates import (binom_p_ge, clustered_ci, lexicon_commitments,  # noqa: E402
                              load_work, paired_auroc, spearman)

BARS = {"G0b.2_nats": 0.25, "G0b.2_rho": 0.99, "G0b.3": 0.75, "G0b.5_chance": 0.38,
        "G0b.6": 0.65, "G0b.7": 0.95}
# swe_0 planted contrast (028 Am.H.1 + STATUS gap 8(c)): every baseline run
# chose the WRONG class on these three blockers and the RIGHT class on the
# fourth. The fifth blocker had no baseline commitments and is excluded.
SWE0_WRONG = ["non_linux_distribution_source_precedence_on_conflict",
              "non_linux_distribution_version_source",
              "non_linux_distribution_detection_scope"]
SWE0_RIGHT = "non_linux_distribution_normalization"


def unit_rows(runs):
    rows = []
    for rid, d in runs.items():
        for u in d["units"]:
            if u.get("unscored_context_cap") or not u.get("blockers"):
                continue
            rows.append({"run": rid, "task": d["task"], "k": u["k"],
                         "is_mutating": bool(u["is_mutating"]),
                         "blockers": u["blockers"], "foreign": u.get("foreign", {}),
                         "relevance": u.get("relevance")})
    return rows


def last_mutating_k(runs, rid):
    ks = [u["k"] for u in runs[rid]["units"] if u.get("is_mutating") and u.get("blockers")]
    return max(ks) if ks else None


def gate_from_scratch(runs):
    diffs, xs, ys = [], [], []
    for d in runs.values():
        for c in d.get("from_scratch_checks", []):
            by_b = defaultdict(list)
            for b, text, lp_b, lp_s in c["rows"]:
                by_b[b].append((lp_b, lp_s)); xs.append(lp_b); ys.append(lp_s)
            for b, rows in by_b.items():
                for i in range(len(rows)):
                    for j in range(i + 1, len(rows)):
                        db = rows[i][0] - rows[j][0]; ds = rows[i][1] - rows[j][1]
                        diffs.append(abs(db - ds))
    if not xs:
        return {"n": 0, "pass": False, "note": "run with --check-from-scratch"}
    mx = float(max(diffs)) if diffs else 0.0
    rho = spearman(xs, ys)
    return {"n_statements": len(xs), "n_option_pairs": len(diffs),
            "max_abs_option_difference_delta_nats": mx, "spearman_lp": rho,
            "bar": {"nats": BARS["G0b.2_nats"], "rho": BARS["G0b.2_rho"]},
            "pass": mx <= BARS["G0b.2_nats"] and rho >= BARS["G0b.2_rho"]}


def gate_swe0_contrast(runs):
    wins, n, detail = 0, 0, []
    for rid, d in runs.items():
        if d["task"] != "swe_0":
            continue
        k = last_mutating_k(runs, rid)
        if k is None:
            continue
        u = next(u for u in d["units"] if u["k"] == k)
        b = u["blockers"]
        if SWE0_RIGHT not in b or not all(w in b for w in SWE0_WRONG):
            continue
        right = b[SWE0_RIGHT]["p_canonical"]
        wrong = float(np.mean([b[w]["p_canonical"] for w in SWE0_WRONG]))
        n += 1; wins += int(right > wrong)
        detail.append({"run": rid, "k": k, "p_canon_right": right, "p_canon_wrong_mean": wrong})
    frac = wins / n if n else None
    p = binom_p_ge(wins, n) if n else None
    return {"n_runs": n, "wins": wins, "frac": frac, "binomial_p_vs_half": p,
            "median_p_canon_right": float(np.median([x["p_canon_right"] for x in detail])) if detail else None,
            "median_p_canon_wrong": float(np.median([x["p_canon_wrong_mean"] for x in detail])) if detail else None,
            "bar": BARS["G0b.3"], "pass": frac is not None and frac >= BARS["G0b.3"] and p < 0.05}


def gate_fullinfo_shift(rows_base, rows_fi):
    def per_blocker(rows):
        acc = defaultdict(list)
        for r in rows:
            if not r["is_mutating"]:
                continue
            for b, v in r["blockers"].items():
                acc[(r["task"], b)].append(v["p_canonical"])
        return {k: float(np.median(v)) for k, v in acc.items()}
    mb, mf = per_blocker(rows_base), per_blocker(rows_fi)
    keys = sorted(set(mb) & set(mf))
    if not keys:
        return {"pass": False, "n_blockers": 0}
    diffs = {k: mf[k] - mb[k] for k in keys}
    pos = sum(1 for v in diffs.values() if v > 0)
    by_task = defaultdict(list)
    for k, v in diffs.items():
        by_task[k[0]].append(v)
    ci = clustered_ci(by_task, lambda v: float(np.median(v)))
    return {"n_blockers": len(keys), "n_positive": pos, "frac_positive": pos / len(keys),
            "median_shift": float(np.median(list(diffs.values()))), "ci95_clustered": ci,
            "per_blocker": {f"{k[0]}/{k[1]}": round(v, 4) for k, v in diffs.items()},
            "bar": "ci excludes 0 and >= 75% blockers positive",
            "pass": ci[0] is not None and ci[0] > 0 and pos / len(keys) >= 0.75}


def gate_lexicon_agreement(runs, commits):
    agree, n = 0, 0
    by_run = {rid: {u["k"]: u for u in d["units"]} for rid, d in runs.items()}
    for (rid, b), (cls, canon, seg) in commits.items():
        u = by_run.get(rid, {}).get(seg)
        if not u or b not in u.get("blockers", {}):
            continue
        n += 1; agree += int(u["blockers"][b]["argmax"] == cls)
    frac = agree / n if n else None
    p = binom_p_ge(agree, n, BARS["G0b.5_chance"]) if n else None
    return {"n": n, "agree": agree, "frac": frac, "chance": BARS["G0b.5_chance"],
            "binomial_p_vs_chance": p, "pass": frac is not None and frac > BARS["G0b.5_chance"] and p < 0.05}


def gate_relevance(rows):
    m = [r for r in rows if r["is_mutating"] and r["relevance"] is not None]
    if not m:
        return {"n": 0, "pass": False}
    own = [float(np.mean([v["pmi"][0] for v in r["blockers"].values()])) for r in m]
    frn = [float(np.mean([v["pmi"] for v in r["foreign"].values()])) for r in m]
    au = paired_auroc(own, frn)
    by_task = defaultdict(list)
    for r, o, f in zip(m, own, frn):
        by_task[r["task"]].append(o - f)
    ci = clustered_ci(by_task, lambda v: float((np.array(v) > 0).mean() + 0.5 * (np.array(v) == 0).mean()))
    return {"n_mutating_units": len(m), "paired_auroc": au, "ci95_clustered": ci,
            "median_own_pmi": float(np.median(own)), "median_foreign_pmi": float(np.median(frn)),
            "bar": BARS["G0b.6"], "pass": au >= BARS["G0b.6"] and ci[0] is not None and ci[0] > 0.5}


def gate_replicate(rows_a, rows_b):
    ib = {(r["run"], r["k"]): r for r in rows_b}
    xs, ys = [], []
    for r in rows_a:
        o = ib.get((r["run"], r["k"]))
        if not o:
            continue
        for b, v in r["blockers"].items():
            if b in o["blockers"]:
                xs.append(v["p_canonical"]); ys.append(o["blockers"][b]["p_canonical"])
    rho = spearman(xs, ys)
    return {"n_pairs": len(xs), "spearman_p_canonical": rho,
            "median_abs_delta": float(np.median(np.abs(np.array(xs) - np.array(ys)))) if xs else None,
            "bar": BARS["G0b.7"], "pass": bool(xs) and rho >= BARS["G0b.7"]}


def anatomy(rows, commits):
    """Stage-1b descriptive: P(canonical), entropy, relevance vs (k - commit_k)."""
    cseg = {}
    for (rid, b), (cls, canon, seg) in commits.items():
        cseg.setdefault(rid, {})[b] = seg
    buckets = defaultdict(lambda: {"p_canon": [], "entropy": [], "relevance": []})
    for r in rows:
        for b, v in r["blockers"].items():
            s = cseg.get(r["run"], {}).get(b)
            if s is None:
                continue
            d = max(-10, min(10, r["k"] - s))
            buckets[d]["p_canon"].append(v["p_canonical"]); buckets[d]["entropy"].append(v["entropy"])
            if r["relevance"] is not None:
                buckets[d]["relevance"].append(r["relevance"])
    return {str(d): {k: (float(np.median(v)) if v else None) for k, v in b.items()} | {"n": len(b["p_canon"])}
            for d, b in sorted(buckets.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--replicate", default=None)
    ap.add_argument("--fullinfo", default=None)
    ap.add_argument("--a0", default="/ssd/wta_data/a0_v3_32b")
    ap.add_argument("--labels-debug", default="models/v3_32b_fixed/labels_debug.jsonl")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="results/cis_lean_stage0b_pilot.json")
    args = ap.parse_args()
    t0 = time.time()
    runs = load_work(Path(args.pilot)); rows = unit_rows(runs)
    tasks = sorted({d["task"] for d in runs.values()})
    print(f"pilot: {len(runs)} runs, {len(rows)} units with a complete menu, tasks {tasks}")
    commits = lexicon_commitments(Path(args.a0), Path(args.labels_debug), Path(args.classes), tasks)
    g = {}
    g["G0b.2_from_scratch"] = gate_from_scratch(runs)
    g["G0b.3_swe0_planted_contrast"] = gate_swe0_contrast(runs)
    if args.fullinfo:
        g["G0b.4_fullinfo_shift"] = gate_fullinfo_shift(rows, unit_rows(load_work(Path(args.fullinfo))))
    g["G0b.5_lexicon_agreement"] = gate_lexicon_agreement(runs, commits)
    g["G0b.6_relevance"] = gate_relevance(rows)
    if args.replicate:
        g["G0b.7_replicate"] = gate_replicate(rows, unit_rows(load_work(Path(args.replicate))))
    fid = [d["fidelity"]["rate"] for d in runs.values() if d.get("fidelity", {}).get("rate") is not None]
    res = {"note": "decisions/029 Am.029.3 Stage-0b gates, as-run.", "inputs": vars(args),
           "n_runs": len(runs), "n_units": len(rows), "fidelity_mean": float(np.mean(fid)) if fid else None,
           "gates": g, "stage1b_anatomy_vs_commitment": anatomy(rows, commits),
           "cost": {"wall_score_s": sum(d.get("wall_score_s", 0) for d in runs.values()),
                    "units": len(rows)}}
    required = [k for k in g if k != "G0b.5_lexicon_agreement"]   # G0b.5 is diagnostic-strength
    failed = [k for k in required if not g[k].get("pass")]
    res["verdict"] = "GO: all Stage-0b validity gates pass" if not failed else f"NO-GO: {failed}"
    res["elapsed_s"] = round(time.time() - t0, 1)
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in g.items():
        print(f"  {k:30s} pass={v.get('pass')}  " + ", ".join(
            f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}"
            for kk, vv in v.items() if kk not in ("pass", "bar", "note", "per_blocker") and not isinstance(vv, (list, dict))))
    print(res["verdict"]); print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
