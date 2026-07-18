"""Do VALUE-TRIGGERED reads carry the interpretation lean? (decisions/016 -> v2)

    python scripts/v2_value_diag.py --a0 data/a0_v2

The v1 near/far analysis (value_read_analysis.py) showed the value lean is
transiently present at emission (near-reads 0.727 vs 0.50 chance) and 32-tok
cadence straddles it. The v2 collection fires a read AT every multi-digit
emission ('value' trigger, reads.py). This is the direct test:

Per forked decision, leave-one-run-out nearest-class-centroid accuracy on RAW
h (unit-normalized), per captured layer, computed on three read subsets --
ALL labeled reads, cadence-only, value-only -- and split by fork kind:
VALUE forks (>=half the class signatures contain multi-digit literals) vs
STRUCTURAL forks. Hypothesis: value-triggered reads recover the lean on value
forks where cadence reads sit at chance; structural forks separate either way.

Raw-h diagnostic only (no A2), the same protocol as decisions/015's decisive
check -- so numbers are comparable across collections.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.labeling import build_labels, load_class_artifact  # noqa: E402


def aligned_read_meta(ds, a0_dir: Path):
    """Per-row arrays rebuilt from the run JSONs in build_labels' row order
    (every read is appended, so run blocks align 1:1 with meta['reads'])."""
    trig = np.empty(len(ds.h), dtype=object)
    for r, (task, run_id) in enumerate(ds.runs):
        meta = json.loads((a0_dir / task / f"{run_id}.json")
                          .read_text(encoding="utf-8"))
        rows = np.where(ds.run_idx == r)[0]
        assert len(rows) == len(meta["reads"]), (run_id, len(rows),
                                                 len(meta["reads"]))
        for k, rd in zip(rows, meta["reads"]):
            trig[k] = rd["trigger"]
    return trig


def layer_h(ds, a0_dir: Path, pos: int) -> np.ndarray:
    """Raw h at stored layer position `pos`, rows aligned with ds."""
    mats = []
    for task, run_id in ds.runs:
        h = np.load(a0_dir / task / f"{run_id}.npz")["h"]
        mats.append((h[:, pos, :] if h.ndim == 3 else h).astype(np.float32))
    out = np.vstack(mats)
    assert len(out) == len(ds.h)
    return out


def fork_kind(ds, art) -> dict[int, str]:
    """decision id -> 'value' | 'structural' by signature content."""
    kind = {}
    for did, (task, blocker) in enumerate(ds.vocab.decisions):
        sigs = [s for c in art[task][blocker]["classes"] for s in c["signatures"]]
        n_num = sum(bool(re.search(r"\d\d", s)) for s in sigs)
        kind[did] = "value" if sigs and n_num >= len(sigs) / 2 else "structural"
    return kind


def loro(ds, h, mask) -> dict:
    """Leave-one-run-out nearest-class-centroid over forked decisions,
    restricted to labeled reads passing `mask`. Per-decision + pooled."""
    lab = (ds.cls >= 0) & mask
    per_dec = {}
    for dec in np.unique(ds.decision[lab]):
        m = lab & (ds.decision == dec)
        runs = np.unique(ds.run_idx[m])
        cls_of = {r: ds.cls[m & (ds.run_idx == r)][0] for r in runs}
        if len(set(cls_of.values())) < 2 or len(runs) < 4:
            continue
        cor = tot = 0
        for r_out in runs:
            tr, te = m & (ds.run_idx != r_out), m & (ds.run_idx == r_out)
            cls_tr = ds.cls[tr]
            if len(set(cls_tr.tolist())) < 2 or not te.any():
                continue
            cents = {}
            for c in set(cls_tr.tolist()):
                v = h[tr][cls_tr == c].mean(0)
                cents[c] = v / np.linalg.norm(v)
            for x, y in zip(h[te], ds.cls[te]):
                xn = x / np.linalg.norm(x)
                pred = max(cents, key=lambda c: float(xn @ cents[c]))
                cor += int(pred == y)
                tot += 1
        if tot:
            per_dec[int(dec)] = {"acc": cor / tot, "n": tot,
                                 "chance": 1 / len(set(cls_of.values()))}
    return per_dec


def pool(per_dec: dict) -> str:
    if not per_dec:
        return "acc   nan vs chance   nan (0 reads, 0 decisions)"
    n = sum(d["n"] for d in per_dec.values())
    acc = sum(d["acc"] * d["n"] for d in per_dec.values()) / n
    ch = float(np.mean([d["chance"] for d in per_dec.values()]))
    return f"acc {acc:.3f} vs chance {ch:.3f} ({n} reads, {len(per_dec)} decisions)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v2")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--layers", default="0,1,2,3")
    args = ap.parse_args()
    a0 = Path(args.a0)

    ds = build_labels(args.a0, args.classes)
    art = load_class_artifact(args.classes)
    trig = aligned_read_meta(ds, a0)
    kind = fork_kind(ds, art)

    subsets = {"ALL": np.ones(len(ds.h), bool),
               "cadence": trig == "cadence",
               "value": trig == "value"}

    for pos in [int(x) for x in args.layers.split(",")]:
        h = layer_h(ds, a0, pos)
        print(f"\n=== raw-h layer position {pos} ===")
        for sub, smask in subsets.items():
            per_dec = loro(ds, h, smask)
            groups = defaultdict(dict)
            for dec, d in per_dec.items():
                groups[kind[dec]][dec] = d
            print(f"  {sub:8s}: {pool(per_dec)}")
            for k in ("structural", "value"):
                print(f"    {k:11s}-forks: {pool(groups[k])}")
        # per-decision detail once per layer, ALL reads
        print("  per-decision (ALL reads):")
        for dec, d in sorted(loro(ds, h, subsets['ALL']).items(),
                             key=lambda kv: -kv[1]["acc"]):
            task, blocker = ds.vocab.decisions[dec]
            print(f"    {d['acc']:.3f} (chance {d['chance']:.2f}, n={d['n']:4d}) "
                  f"[{kind[dec][:6]}] {task}/{blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
