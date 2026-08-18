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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="models/v3_32b/labels.npz")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    npz = Path(args.labels)
    z = np.load(npz, allow_pickle=False)
    dec, cls, run = z["decision"], z["cls"], z["run_idx"]
    print(f"projecting h -> {args.dim} dims (streaming) ...", flush=True)
    L = stream_project(npz, args.dim, seed=args.seed)
    print(f"projected: {L.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    obs, run_null, read_null, rows = [], [], [], []
    for d in np.unique(dec[dec >= 0]):
        m = (dec == d) & (cls >= 0)
        if m.sum() < 4 or len(np.unique(cls[m])) < 2:
            continue
        Ld, cd, rd = L[m], cls[m], run[m]
        o = ratio(Ld, cd)

        # a run commits to ONE class, so run-level permutation = shuffle the
        # per-run class assignment, then broadcast back to that run's reads
        runs = np.unique(rd)
        run_cls = np.array([cd[rd == r][0] for r in runs])
        rn = []
        for _ in range(args.perms):
            perm = rng.permutation(run_cls)
            cp = np.empty_like(cd)
            for r, g in zip(runs, perm):
                cp[rd == r] = g
            if len(np.unique(cp)) >= 2:
                rn.append(ratio(Ld, cp))
        rl = [ratio(Ld, rng.permutation(cd)) for _ in range(args.perms)]

        obs.append(o)
        run_null.append(np.mean(rn) if rn else np.nan)
        read_null.append(np.mean(rl))
        p = (1 + sum(x >= o for x in rn)) / (1 + len(rn)) if rn else np.nan
        rows.append({"decision": int(d), "n_reads": int(m.sum()),
                     "n_runs": len(runs), "obs": o,
                     "run_null": run_null[-1], "read_null": read_null[-1],
                     "p_run": p})

    obs = np.array(obs)
    rn_ = np.array(run_null, dtype=float)
    rd_ = np.array(read_null, dtype=float)
    ps = np.array([r["p_run"] for r in rows], dtype=float)
    print()
    print(f"gate5-eligible decisions: {len(obs)}")
    print(f"  observed ratio          mean {obs.mean():.3f}")
    print(f"  RUN-level null          mean {np.nanmean(rn_):.3f}   "
          f"<- the honest no-signal reference")
    print(f"  read-level null         mean {rd_.mean():.3f}   "
          f"<- too permissive (reads are not independent)")
    print(f"  observed / run-null     {obs.mean() / np.nanmean(rn_):.3f}x")
    print(f"  decisions with p_run < 0.05: {int((ps < 0.05).sum())} / {len(ps)}")
    print(f"  median runs per decision: {np.median([r['n_runs'] for r in rows]):.0f}")
    print()
    print("per-decision (first 15):")
    print(f"{'dec':>5} {'reads':>6} {'runs':>5} {'obs':>7} {'runnull':>8} "
          f"{'readnull':>9} {'p_run':>6}")
    for r in rows[:15]:
        print(f"{r['decision']:>5} {r['n_reads']:>6} {r['n_runs']:>5} "
              f"{r['obs']:>7.3f} {r['run_null']:>8.3f} {r['read_null']:>9.3f} "
              f"{r['p_run']:>6.3f}")
    import json
    Path("results").mkdir(exist_ok=True)
    Path("results/gate5_permutation_test.json").write_text(
        json.dumps({"n_decisions": len(obs),
                    "observed_mean": float(obs.mean()),
                    "run_null_mean": float(np.nanmean(rn_)),
                    "read_null_mean": float(rd_.mean()),
                    "n_sig_p05": int((ps < 0.05).sum()),
                    "rows": rows}, indent=1), encoding="utf-8")
    print("\nwrote results/gate5_permutation_test.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
