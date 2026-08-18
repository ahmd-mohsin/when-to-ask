"""The no-signal reference gate 5 actually needs: a RUN-LEVEL permutation test.

Two facts, established in scripts/gate5_noise_floor.py and here:

1. gate5's between/within has a pure-noise expectation of sqrt(2/(n-1)) for n
   points per class, NOT 1.0. The note "want ratio >> 1" is comparing to the
   wrong reference.

2. That closed form assumes INDEPENDENT points. Gate 5's points are reads, and
   reads are not independent: dozens come from the same run, adjacent in one
   generation. The independent unit is the RUN. So the naive floor computed
   from reads/class (~90 -> floor ~0.15) is far too permissive, while the floor
   from runs/class (~4 -> ~0.82) is the honest one.

The defensible null is therefore a permutation test that respects run
structure: hold the activations fixed, shuffle each decision's class labels
BETWEEN RUNS (never within a run, which would break the very correlation that
matters), recompute the statistic, and see where the observed value falls.
Read-level shuffling is reported alongside purely to show how much it inflates.

This is Hewitt & Liang's control-task logic applied to gate 5, and it is
independent of every open labelling question: it says what the pipeline reads
when the labels carry no run-level signal.

    python scripts/gate5_permutation_test.py --labels models/v3_32b/labels.npz

Memory: labels.npz `h` is ~14 GB float32. It is streamed from the zip in row
blocks and randomly projected to --dim (Johnson-Lindenstrauss preserves the
distance ratios the statistic is built from; the noise floor is verified
dimension-independent).
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def stream_project(npz_path: Path, dim: int, seed: int = 0,
                   block: int = 4096) -> np.ndarray:
    """Random-project `h` to `dim` without ever holding it in memory."""
    with zipfile.ZipFile(npz_path) as zf:
        name = "h.npy" if "h.npy" in zf.namelist() else "h"
        with zf.open(name) as raw:
            buf = io.BufferedReader(raw, buffer_size=1 << 22)
            major, minor = np.lib.format.read_magic(buf)
            reader = (np.lib.format.read_array_header_1_0 if major == 1
                      else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = reader(buf)
            if fortran:
                raise ValueError("fortran-order h not supported")
            n, d = shape
            rng = np.random.default_rng(seed)
            P = rng.standard_normal((d, dim)).astype(np.float32) / np.sqrt(dim)
            out = np.empty((n, dim), dtype=np.float32)
            done = 0
            while done < n:
                k = min(block, n - done)
                chunk = np.frombuffer(buf.read(k * d * dtype.itemsize),
                                      dtype=dtype).reshape(k, d)
                out[done:done + k] = chunk.astype(np.float32) @ P
                done += k
    return out


def ratio(L: np.ndarray, c: np.ndarray) -> float:
    """gate5's statistic, verbatim from a4_gates.gate5_lean_separation."""
    cents = {g: L[c == g].mean(axis=0) for g in np.unique(c)}
    within = float(np.mean([np.linalg.norm(L[c == g] - cents[g], axis=1).mean()
                            for g in cents if (c == g).sum() > 1]))
    gs = sorted(cents)
    between = float(np.mean([np.linalg.norm(cents[a] - cents[b])
                             for i, a in enumerate(gs) for b in gs[i + 1:]]))
    return between / max(within, 1e-9)


def n_distinct_assignments(counts: list[int]) -> int:
    """Multinomial: distinct arrangements of the run-class multiset. Every
    distinct arrangement arises from exactly prod(counts!) raw permutations,
    so uniform-over-permutations == uniform-over-distinct-arrangements."""
    import math
    n = sum(counts)
    out = math.factorial(n)
    for c in counts:
        out //= math.factorial(c)
    return out


def multiset_perms(items: list):
    """All distinct arrangements of a multiset (lexicographic recursion)."""
    from collections import Counter
    counts = sorted(Counter(items).items())

    def rec(remaining, prefix, k):
        if k == 0:
            yield tuple(prefix)
            return
        for i, (g, c) in enumerate(remaining):
            if c:
                remaining[i] = (g, c - 1)
                prefix.append(g)
                yield from rec(remaining, prefix, k - 1)
                prefix.pop()
                remaining[i] = (g, c)
    yield from rec(list(counts), [], len(items))


def decision_null(Ld, cd, rd, rng, perms: int, exact_cap: int):
    """Run-level null for one decision: EXACT (all distinct run-class
    assignments) when the space is small enough, else Monte Carlo.

    Returns (null_ratios, p, p_method, n_assign, min_p). decisions/026 §5:
    the R2 report used MC-200 with the +1 correction, which pushed two
    decisions sitting at their exact-test floor (p=1/21=0.0476) above 0.05
    and reported '0/35' where the exact count was 2/35.
    """
    runs = np.unique(rd)
    run_cls = np.array([cd[rd == r][0] for r in runs])
    o = ratio(Ld, cd)
    n_assign = n_distinct_assignments(
        list(np.unique(run_cls, return_counts=True)[1]))

    read_of_run = {r: (rd == r) for r in runs}

    def ratio_of(assign) -> float:
        cp = np.empty_like(cd)
        for r, g in zip(runs, assign):
            cp[read_of_run[r]] = g
        return ratio(Ld, cp)

    if n_assign <= exact_cap:
        null = np.array([ratio_of(a) for a in multiset_perms(list(run_cls))])
        # exact test over the full group: the observed assignment is one of
        # the n_assign equally-likely arrangements, so no +1 correction
        p = float((null >= o - 1e-12).sum()) / n_assign
        return null, p, "exact", n_assign, 1.0 / n_assign
    null = np.array([ratio_of(rng.permutation(run_cls))
                     for _ in range(perms)])
    p = (1 + int((null >= o).sum())) / (1 + perms)
    return null, p, "mc", n_assign, 1.0 / (1 + perms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="models/v3_32b/labels.npz")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--perms", type=int, default=2000,
                    help="MC draws when the assignment space exceeds --exact-cap")
    ap.add_argument("--exact-cap", type=int, default=20000,
                    help="enumerate ALL distinct run-class assignments up to "
                         "this count (exact test, no MC noise)")
    ap.add_argument("--global-draws", type=int, default=1000,
                    help="joint permutations for the global Stouffer test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gate5_permutation_test.json")
    args = ap.parse_args()

    npz = Path(args.labels)
    z = np.load(npz, allow_pickle=False)
    dec, cls, run = z["decision"], z["cls"], z["run_idx"]
    print(f"projecting h -> {args.dim} dims (streaming) ...", flush=True)
    L = stream_project(npz, args.dim, seed=args.seed)
    print(f"projected: {L.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    obs, run_null, read_null, rows, nulls = [], [], [], [], []
    for d in np.unique(dec[dec >= 0]):
        m = (dec == d) & (cls >= 0)
        if m.sum() < 4 or len(np.unique(cls[m])) < 2:
            continue
        Ld, cd, rd = L[m], cls[m], run[m]
        o = ratio(Ld, cd)
        null, p, method, n_assign, min_p = decision_null(
            Ld, cd, rd, rng, args.perms, args.exact_cap)
        rl = [ratio(Ld, rng.permutation(cd)) for _ in range(200)]

        obs.append(o)
        run_null.append(float(null.mean()))
        read_null.append(float(np.mean(rl)))
        nulls.append(null)
        rows.append({"decision": int(d), "n_reads": int(m.sum()),
                     "n_runs": len(np.unique(rd)), "obs": o,
                     "run_null": run_null[-1], "read_null": read_null[-1],
                     "null_sd": float(null.std()),
                     "p_run": p, "p_method": method,
                     "n_assignments": n_assign, "min_p": min_p,
                     "testable": bool(min_p < 0.05)})

    obs = np.array(obs)
    rn_ = np.array(run_null, dtype=float)
    rd_ = np.array(read_null, dtype=float)
    ps = np.array([r["p_run"] for r in rows], dtype=float)
    testable = np.array([r["testable"] for r in rows], dtype=bool)

    # global test (decisions/026 §5): Stouffer sum of per-decision z against
    # the permutation distribution of the same sum. Decisions with a
    # degenerate null (sd == 0, e.g. a 1v1 run split where the statistic is
    # label-swap invariant) carry no information and are excluded from BOTH
    # the observed sum and the null draws.
    live = [i for i, r in enumerate(rows) if r["null_sd"] > 0]
    S_obs = float(sum((obs[i] - rows[i]["run_null"]) / rows[i]["null_sd"]
                      for i in live))
    grng = np.random.default_rng(args.seed + 1)
    S_null = np.array([
        sum((nulls[i][grng.integers(len(nulls[i]))]
             - rows[i]["run_null"]) / rows[i]["null_sd"] for i in live)
        for _ in range(args.global_draws)])
    p_global = (1 + int((S_null >= S_obs).sum())) / (1 + args.global_draws)

    print()
    print(f"gate5-eligible decisions: {len(obs)}")
    print(f"  observed ratio          mean {obs.mean():.3f}")
    print(f"  RUN-level null          mean {np.nanmean(rn_):.3f}   "
          f"<- the honest no-signal reference")
    print(f"  read-level null         mean {rd_.mean():.3f}   "
          f"<- too permissive (reads are not independent)")
    print(f"  observed / run-null     {obs.mean() / np.nanmean(rn_):.3f}x")
    print(f"  TESTABLE at alpha=0.05: {int(testable.sum())} / {len(rows)}   "
          f"(the rest cannot reach p<0.05 at ANY effect size)")
    print(f"  decisions with p < 0.05: {int((ps < 0.05).sum())} of "
          f"{int(testable.sum())} testable ({len(rows)} eligible)")
    print(f"  GLOBAL Stouffer test:   S_obs {S_obs:+.3f}  "
          f"p {p_global:.4f}   ({len(live)} informative decisions, "
          f"{args.global_draws} joint draws)   <- the headline")
    print(f"  median runs per decision: "
          f"{np.median([r['n_runs'] for r in rows]):.0f}")
    print()
    print("per-decision (first 15):")
    print(f"{'dec':>5} {'reads':>6} {'runs':>5} {'obs':>7} {'runnull':>8} "
          f"{'p':>7} {'method':>6} {'minp':>6} {'test':>4}")
    for r in rows[:15]:
        print(f"{r['decision']:>5} {r['n_reads']:>6} {r['n_runs']:>5} "
              f"{r['obs']:>7.3f} {r['run_null']:>8.3f} {r['p_run']:>7.4f} "
              f"{r['p_method']:>6} {r['min_p']:>6.3f} "
              f"{'y' if r['testable'] else '-':>4}")
    import json
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"n_decisions": len(obs),
                    "observed_mean": float(obs.mean()),
                    "run_null_mean": float(np.nanmean(rn_)),
                    "read_null_mean": float(rd_.mean()),
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
