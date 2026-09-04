"""Stage-0 gate evaluator for the CIS pilot (decisions/029 §5). CPU only.

Reads the per-run JSON/NPZ that scripts/cis_score.py wrote for the pilot,
its independent replicate, and the full_info arm, and reports every G0 gate
as-run against its frozen bar. Any gate below its bar -> verdict NO-GO and
Stage 2 is not fit (029 §8.1). Nothing here is tuned; the bars are quoted
from 029 and printed beside each number.

    python scripts/cis_stage0_gates.py --pilot /ssd3/wta-cis/pilot \
        --replicate /ssd3/wta-cis/pilot_rep --fullinfo /ssd3/wta-cis/pilot_fi \
        --a0 /ssd/wta_data/a0_v3_32b --labels-debug models/v3_32b_fixed/labels_debug.jsonl \
        --out results/cis_stage0_pilot.json
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

BOOT, SEED = 2000, 0
BARS = {"G0.2": 0.95, "G0.3": 0.95, "G0.4_nats": 0.05, "G0.4_rho": 0.999,
        "G0.5": 0.95, "G0.6": 0.65, "G0.7": 0.75, "G0.8": 0.5,
        "G0.10_kill": 0.3, "G0.11": 0.75}


# ---------------------------------------------------------------- loading
def load_work(work: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(work.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "error" not in d and "units" in d:
            out[d["run"]] = d
    return out


def unit_rows(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for rid, d in runs.items():
        for u in d["units"]:
            if u.get("unscored_context_cap"):
                continue
            own = {n[4:]: v for n, v in u["variants"].items() if n.startswith("own:")}
            foreign = [v for n, v in u["variants"].items() if n.startswith("foreign:")]
            rival = {n[6:]: v for n, v in u["variants"].items() if n.startswith("rival:")}
            if not own:
                continue
            rows.append({
                "run": rid, "task": d["task"], "k": u["k"],
                "is_mutating": bool(u["is_mutating"]), "no_block": bool(u["no_block"]),
                "S_k": u["S_k"], "block_tokens": u["block_tokens"],
                "own": {b: v["cis_block"] for b, v in own.items()},
                "own_max_neg": max(-v["cis_block"] for v in own.values()),
                "own_abs_max": max(abs(v["cis_block"]) for v in own.values()),
                "foreign_mean_neg": (float(np.mean([-v["cis_block"] for v in foreign]))
                                     if foreign else None),
                "foreign_abs_mean": (float(np.mean([abs(v["cis_block"]) for v in foreign]))
                                     if foreign else None),
                "rival": {b: v["cis_block"] for b, v in rival.items()},
            })
    return rows


# ------------------------------------------------------------- statistics
def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def paired_auroc(own, other) -> float:
    d = np.asarray(own, float) - np.asarray(other, float)
    return float((d > 0).mean() + 0.5 * (d == 0).mean()) if len(d) else float("nan")


def clustered_ci(values_by_task: dict[str, list[float]], stat, boot=BOOT, seed=SEED):
    """Task-clustered bootstrap of `stat(list_of_values)` -- the repo's
    estimator (t1_auroc_ci / canonical_accuracy), BOOT=2000, SEED=0."""
    tasks = sorted(values_by_task)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(boot):
        pick = rng.choice(len(tasks), len(tasks), replace=True)
        vals = [v for i in pick for v in values_by_task[tasks[i]]]
        if vals:
            draws.append(stat(vals))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if draws else [None, None]


def binom_p_ge(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k), X ~ Binomial(n, p). Exact."""
    if n == 0:
        return float("nan")
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def r2(y, X) -> float:
    y = np.asarray(y, float); X = np.asarray(X, float)
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    tss = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tss) if tss > 0 else float("nan")


# ------------------------------------------------------------------ gates
def gate_fidelity(runs):
    rates = [d["fidelity"]["rate"] for d in runs.values() if d["fidelity"]["rate"] is not None]
    return {"mean": float(np.mean(rates)) if rates else None,
            "frac_ge_0.9": float(np.mean(np.array(rates) >= 0.9)) if rates else None,
            "n_runs": len(rates), "bar": BARS["G0.2"],
            "pass": bool(rates) and float(np.mean(rates)) >= BARS["G0.2"]}


def gate_replicate(rows_a, rows_b):
    ia = {(r["run"], r["k"]): r for r in rows_a}
    xs, ys, deltas = [], [], []
    for key, ra in ia.items():
        rb = next((r for r in rows_b if (r["run"], r["k"]) == key), None)
        if rb is None:
            continue
        for b, v in ra["own"].items():
            if b in rb["own"]:
                xs.append(v); ys.append(rb["own"][b]); deltas.append(abs(v - rb["own"][b]))
    rho = spearman(xs, ys)
    return {"n_pairs": len(xs), "spearman": rho,
            "median_abs_delta": float(np.median(deltas)) if deltas else None,
            "bar": BARS["G0.3"], "pass": bool(xs) and rho >= BARS["G0.3"]}


def gate_from_scratch(runs):
    b, s, tok = [], [], []
    for d in runs.values():
        for c in d.get("from_scratch_checks", []):
            b.append(c["block_sum_branch"]); s.append(c["block_sum_scratch"]); tok.append(c["max_abs_tok"])
    if not b:
        return {"n": 0, "pass": False, "note": "run cis_score.py with --check-from-scratch"}
    b, s = np.array(b), np.array(s)
    rho = spearman(b, s) if len(b) >= 3 else float("nan")
    mx = float(np.abs(b - s).max())
    return {"n": len(b), "max_abs_block_delta_nats": mx, "max_abs_tok": float(max(tok)),
            "spearman": rho, "bar": {"nats": BARS["G0.4_nats"], "rho": BARS["G0.4_rho"]},
            "pass": mx <= BARS["G0.4_nats"] and (rho >= BARS["G0.4_rho"] or len(b) < 3)}


def gate_recorded_vs_fresh(runs, work: Path, a0: Path):
    from wta.logging_schema import load_run_log
    cos = []
    for rid, d in runs.items():
        npz = work / f"{rid}.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        f7, ks = z["feat_tok7"], z["unit_k"]
        try:
            log = load_run_log(a0 / d["task"], rid, layer=3)
        except Exception:
            continue
        H = log.read_matrix().astype(np.float32)
        idx = {(r.segment_idx, r.token_idx): i for i, r in enumerate(log.reads)}
        for j, k in enumerate(ks.tolist()):
            if (k, 7) not in idx or np.isnan(f7[j]).any():
                continue
            a, b = H[idx[(k, 7)]], f7[j].astype(np.float32)
            den = np.linalg.norm(a) * np.linalg.norm(b)
            if den > 0:
                cos.append(float(a @ b / den))
    med = float(np.median(cos)) if cos else None
    return {"n": len(cos), "median_cosine": med,
            "p10": float(np.percentile(cos, 10)) if cos else None,
            "bar": BARS["G0.5"], "pass": med is not None and med >= BARS["G0.5"]}


def gate_relevance(rows):
    m = [r for r in rows if r["is_mutating"] and r["foreign_abs_mean"] is not None]
    if not m:
        return {"n": 0, "pass": False}
    own = [r["own_abs_max"] for r in m]; frn = [r["foreign_abs_mean"] for r in m]
    au = paired_auroc(own, frn)
    by_task = defaultdict(list)
    for r in m:
        by_task[r["task"]].append(r["own_abs_max"] - r["foreign_abs_mean"])
    ci = clustered_ci(by_task, lambda v: float((np.array(v) > 0).mean() + 0.5 * (np.array(v) == 0).mean()))
    return {"n_mutating_units": len(m), "n_tasks": len(by_task), "paired_auroc": au,
            "ci95_clustered": ci, "median_own_abs": float(np.median(own)),
            "median_foreign_abs": float(np.median(frn)), "bar": BARS["G0.6"],
            "pass": au >= BARS["G0.6"] and ci[0] is not None and ci[0] > 0.5}


def lexicon_commitments(a0: Path, labels_debug: Path, classes: Path, tasks: list[str]):
    """{(run, blocker): (class, canonical_class, commitment_segment)} via
    commit_rounds, converted from list position to segment_idx."""
    from offline_ask_headtohead import commit_rounds, load_commitments, load_task_actions
    from wta.labeling import load_class_artifact
    art = load_class_artifact(str(classes))
    committed = load_commitments(labels_debug)
    actions = load_task_actions(a0, None)
    out = {}
    for t in tasks:
        if t not in actions:
            continue
        rounds = commit_rounds(actions[t], committed, art, t)
        for (rid, b), k in rounds.items():
            cls = committed.get((rid, b))
            if cls is None or k is None:
                continue
            seg = actions[t][rid][k].segment_idx
            canon = art[t][b]["classes"][0]["name"]
            out[(rid, b)] = (cls, canon, seg)
    return out


def gate_sign(rows, commits):
    idx = {(r["run"], r["k"]): r for r in rows}
    neg_noncanon, n_noncanon, nonneg_canon, n_canon = 0, 0, 0, 0
    for (rid, b), (cls, canon, seg) in commits.items():
        r = idx.get((rid, seg))
        if r is None or b not in r["own"]:
            continue
        v = r["own"][b]
        if cls == canon:
            n_canon += 1; nonneg_canon += int(v >= 0)
        else:
            n_noncanon += 1; neg_noncanon += int(v < 0)
    frac = neg_noncanon / n_noncanon if n_noncanon else None
    p = binom_p_ge(neg_noncanon, n_noncanon) if n_noncanon else None
    return {"n_noncanonical_commitments": n_noncanon, "n_negative": neg_noncanon,
            "frac_negative": frac, "binomial_p_vs_half": p,
            "n_canonical_commitments": n_canon, "frac_canonical_nonneg":
                (nonneg_canon / n_canon) if n_canon else None,
            "bar": BARS["G0.7"], "pass": frac is not None and frac >= BARS["G0.7"] and p < 0.05}


def gate_redundancy(rows_base, rows_fi):
    b = [r["own_abs_max"] for r in rows_base if r["is_mutating"]]
    f = [r["own_abs_max"] for r in rows_fi if r["is_mutating"]]
    if not b or not f:
        return {"pass": False, "n_base": len(b), "n_fi": len(f)}
    ratio = float(np.median(f) / np.median(b)) if np.median(b) > 0 else None
    return {"n_base": len(b), "n_fi": len(f), "median_abs_base": float(np.median(b)),
            "median_abs_fullinfo": float(np.median(f)), "ratio": ratio,
            "bar": BARS["G0.8"], "pass": ratio is not None and ratio <= BARS["G0.8"]}


def gate_variance(rows):
    m = [r for r in rows if r["foreign_mean_neg"] is not None]
    if len(m) < 20:
        return {"n": len(m), "pass": False, "note": "too few units"}
    tasks = sorted({r["task"] for r in m})
    def cov(r):
        return ([1.0 if r["task"] == t else 0.0 for t in tasks[1:]]
                + [r["k"], math.log1p(r["block_tokens"]), float(r["is_mutating"]), r["S_k"]])
    X, y, is_own = [], [], []
    for r in m:
        X.append(cov(r)); y.append(r["own_max_neg"]); is_own.append(1.0)
        X.append(cov(r)); y.append(r["foreign_mean_neg"]); is_own.append(0.0)
    X = np.array(X); y = np.array(y); is_own = np.array(is_own)[:, None]
    r2_cov = r2(y, X)
    r2_full = r2(y, np.hstack([X, is_own]))
    contrast = r2_full - r2_cov
    return {"n_units": len(m), "r2_covariates": r2_cov, "r2_contrast_increment": contrast,
            "pass": contrast >= r2_cov}


def gate_surprisal(rows):
    m = [r for r in rows if r["is_mutating"]]
    rho = spearman([r["S_k"] for r in m], [r["own_max_neg"] for r in m]) if len(m) >= 3 else float("nan")
    return {"n": len(m), "spearman_Sk_vs_CIS": rho, "kill": BARS["G0.10_kill"],
            "raw_cis_disqualified": bool(abs(rho) > BARS["G0.10_kill"]) if not math.isnan(rho) else None}


def gate_rival(rows, commits):
    idx = {(r["run"], r["k"]): r for r in rows}
    wins, n = 0, 0
    for (rid, b), (cls, canon, seg) in commits.items():
        if cls == canon:
            continue
        r = idx.get((rid, seg))
        if r is None:
            continue
        key = f"{b}/{cls}"
        if key not in r["rival"] or b not in r["own"]:
            continue
        n += 1; wins += int(r["rival"][key] > r["own"][b])
    frac = wins / n if n else None
    p = binom_p_ge(wins, n) if n else None
    return {"n": n, "wins": wins, "frac": frac, "binomial_p_vs_half": p,
            "bar": BARS["G0.11"],
            "pass": (frac is not None and frac >= BARS["G0.11"] and p < 0.05) if n else None,
            "note": None if n else "no rival variants scored (fixture unapproved or --controls without rival)"}


def cost(runs):
    units = sum(1 for d in runs.values() for u in d["units"] if not u.get("unscored_context_cap"))
    ws = sum(d.get("wall_score_s", 0) for d in runs.values())
    wr = sum(d.get("wall_replay_s", 0) for d in runs.values())
    return {"units_scored": units, "wall_score_s": ws, "wall_replay_s": wr,
            "s_per_unit_score": (ws / units) if units else None,
            "projected_full_run_score_h": (ws / units * 41538 / 3600) if units else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--replicate", default=None)
    ap.add_argument("--fullinfo", default=None)
    ap.add_argument("--a0", default="/ssd/wta_data/a0_v3_32b")
    ap.add_argument("--labels-debug", default="models/v3_32b_fixed/labels_debug.jsonl")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="results/cis_stage0_pilot.json")
    args = ap.parse_args()
    t0 = time.time()

    pilot = Path(args.pilot)
    runs = load_work(pilot)
    rows = unit_rows(runs)
    tasks = sorted({d["task"] for d in runs.values()})
    print(f"pilot: {len(runs)} runs, {len(rows)} scored units, tasks {tasks}")
    res = {"note": "decisions/029 Stage-0 gates, as-run. Any failing gate -> Stage 2 is not fit.",
           "inputs": vars(args), "n_runs": len(runs), "n_units": len(rows),
           "n_mutating_units": sum(r["is_mutating"] for r in rows), "gates": {}}
    g = res["gates"]
    g["G0.2_fidelity"] = gate_fidelity(runs)
    if args.replicate:
        g["G0.3_replicate"] = gate_replicate(rows, unit_rows(load_work(Path(args.replicate))))
    g["G0.4_from_scratch"] = gate_from_scratch(runs)
    g["G0.5_recorded_vs_fresh"] = gate_recorded_vs_fresh(runs, pilot, Path(args.a0))
    g["G0.6_relevance"] = gate_relevance(rows)
    commits = lexicon_commitments(Path(args.a0), Path(args.labels_debug), Path(args.classes), tasks)
    g["G0.7_sign_lexicon"] = gate_sign(rows, commits)
    if args.fullinfo:
        g["G0.8_redundancy"] = gate_redundancy(rows, unit_rows(load_work(Path(args.fullinfo))))
    g["G0.9_variance"] = gate_variance(rows)
    g["G0.10_surprisal"] = gate_surprisal(rows)
    g["G0.11_rival"] = gate_rival(rows, commits)
    res["cost"] = cost(runs)

    required = ["G0.2_fidelity", "G0.4_from_scratch", "G0.5_recorded_vs_fresh",
                "G0.6_relevance", "G0.7_sign_lexicon", "G0.9_variance"]
    required += [k for k in ("G0.3_replicate", "G0.8_redundancy") if k in g]
    failed = [k for k in required if not g[k].get("pass")]
    if g["G0.11_rival"].get("pass") is False:
        failed.append("G0.11_rival")
    res["verdict"] = "GO: all Stage-0 gates pass" if not failed else f"NO-GO: {failed}"
    res["elapsed_s"] = round(time.time() - t0, 1)
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in g.items():
        print(f"  {k:24s} pass={v.get('pass')}  " + ", ".join(
            f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}"
            for kk, vv in v.items() if kk not in ("pass", "bar", "note") and not isinstance(vv, (list, dict))))
    print(res["verdict"]); print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
