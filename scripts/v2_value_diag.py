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
    restricted to labeled reads passing `mask`. Per-decision + pooled.

    decisions/026 §5: eligibility now requires EVERY class to be carried by
    >= 2 runs. With a single-run minority, holding that run out left a
    one-class train set that was silently skipped, so minority reads were
    never tested while `chance` still claimed 1/n_classes — the defect
    behind the 14B '0.71-0.73 vs 0.50' scaling justification (its
    train-majority baseline was 0.916). Each decision now also reports the
    trivial train-majority baseline and balanced accuracy; decisions
    excluded by the new rule are counted in '_excluded_single_run_minority'
    so the census stays visible."""
    lab = (ds.cls >= 0) & mask
    per_dec = {}
    excluded = 0
    for dec in np.unique(ds.decision[lab]):
        m = lab & (ds.decision == dec)
        runs = np.unique(ds.run_idx[m])
        cls_of = {r: ds.cls[m & (ds.run_idx == r)][0] for r in runs}
        run_ct = {}
        for c in cls_of.values():
            run_ct[c] = run_ct.get(c, 0) + 1
        if len(run_ct) < 2 or len(runs) < 4:
            continue
        if min(run_ct.values()) < 2:
            excluded += 1
            continue
        cor = tot = 0
        base_cor = 0
        by_cls_cor, by_cls_tot = {}, {}
        for r_out in runs:
            tr, te = m & (ds.run_idx != r_out), m & (ds.run_idx == r_out)
            cls_tr = ds.cls[tr]
            if len(set(cls_tr.tolist())) < 2 or not te.any():
                continue
            cents = {}
            for c in set(cls_tr.tolist()):
                v = h[tr][cls_tr == c].mean(0)
                cents[c] = v / np.linalg.norm(v)
            maj = max(set(cls_tr.tolist()), key=cls_tr.tolist().count)
            for x, y in zip(h[te], ds.cls[te]):
                xn = x / np.linalg.norm(x)
                pred = max(cents, key=lambda c: float(xn @ cents[c]))
                cor += int(pred == y)
                base_cor += int(maj == y)
                by_cls_cor[y] = by_cls_cor.get(y, 0) + int(pred == y)
                by_cls_tot[y] = by_cls_tot.get(y, 0) + 1
                tot += 1
        if tot:
            bal = float(np.mean([by_cls_cor[c] / by_cls_tot[c]
                                 for c in by_cls_tot]))
            per_dec[int(dec)] = {"acc": cor / tot, "n": tot,
                                 "chance": 1 / len(run_ct),
                                 "majority_baseline": base_cor / tot,
                                 "balanced_acc": bal}
    per_dec["_excluded_single_run_minority"] = excluded
    return per_dec


def pool(per_dec: dict) -> str:
    excluded = per_dec.get("_excluded_single_run_minority", 0)
    decs = {k: v for k, v in per_dec.items() if isinstance(k, int)}
    if not decs:
        return ("acc   nan vs chance   nan (0 reads, 0 decisions; "
                f"{excluded} excluded single-run-minority)")
    n = sum(d["n"] for d in decs.values())
    acc = sum(d["acc"] * d["n"] for d in decs.values()) / n
    ch = float(np.mean([d["chance"] for d in decs.values()]))
    base = sum(d["majority_baseline"] * d["n"] for d in decs.values()) / n
    bal = float(np.mean([d["balanced_acc"] for d in decs.values()]))
    return (f"acc {acc:.3f} vs chance {ch:.3f} vs MAJORITY {base:.3f}; "
            f"balanced {bal:.3f} ({n} reads, {len(decs)} decisions; "
            f"{excluded} excluded single-run-minority)")


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
                if isinstance(dec, int):
                    groups[kind[dec]][dec] = d
            print(f"  {sub:8s}: {pool(per_dec)}")
            for k in ("structural", "value"):
                print(f"    {k:11s}-forks: {pool(groups[k])}")
        # per-decision detail once per layer, ALL reads
        print("  per-decision (ALL reads):")
        for dec, d in sorted(((k, v) for k, v in
                              loro(ds, h, subsets['ALL']).items()
                              if isinstance(k, int)),
                             key=lambda kv: -kv[1]["acc"]):
            task, blocker = ds.vocab.decisions[dec]
            print(f"    {d['acc']:.3f} (chance {d['chance']:.2f}, "
                  f"majority {d['majority_baseline']:.2f}, "
                  f"balanced {d['balanced_acc']:.2f}, n={d['n']:4d}) "
                  f"[{kind[dec][:6]}] {task}/{blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
