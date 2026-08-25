"""R6b uncertainty: task-clustered bootstrap CI for the LLM-comparison AUROC.

028 Amendment A item 7 requires EVERY T1/T5 separation AUROC to carry a
task-clustered bootstrap 95% CI. R6b is a T1 separation AUROC, so it gets
the same interval on the same terms as the R3/R4/R5 rows.

Additive: it does not rerun or vary the cell. The point estimate is
recomputed from the stored judgments and asserted to reproduce the as-run
number in results/r6_llm_cells.json.

Difference from t1_auroc_ci.py: the R3-R5 rows score a pair by the Euclidean
DISTANCE between two representation vectors; R6b scores it by the judge's
confidence signed by its same/different answer (r6_score.py's frozen rule,
higher = more different). Orientation and the exact-U estimator with ties at
0.5 are identical, so the bootstrap machinery is reused verbatim.

Clustering unit is the TASK, matching t1_auroc_ci.py: the 200 judged pairs
are not independent — they are drawn from 46 tasks and share runs within a
(task, blocker) group.

    python scripts/r6b_auroc_ci.py --out results/r6b_auroc_ci.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from r6_score import load_chunks  # noqa: E402
from t1_auroc_ci import BOOT, SEED, auroc_from  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", default="results/r6_items")
    ap.add_argument("--chunks-dir", default="results/r6_chunks")
    ap.add_argument("--cells", default="results/r6_llm_cells.json")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default="results/r6b_auroc_ci.json")
    args = ap.parse_args()

    items = json.loads((Path(args.items_dir) / "r6b_pairs.json")
                       .read_text(encoding="utf-8"))["items"]
    res = load_chunks(Path(args.chunks_dir), "r6b")

    rows = []
    for it in items:
        r = res.get(it["item_id"])
        if r is None:
            continue
        score = float(r["confidence"]) * (1 if bool(r["different"]) else -1)
        rows.append((it["task"], 1 if it["truth"] == "diff" else 0, score))
    if not rows:
        raise SystemExit("no judged R6b pairs found")

    tasks = np.array([r[0] for r in rows])
    labels = np.array([r[1] for r in rows], dtype=int)
    scores = np.array([r[2] for r in rows], dtype=float)
    point = auroc_from(labels, scores)

    # consistency guard: must reproduce the as-run cell exactly
    recorded = json.loads(Path(args.cells).read_text(
        encoding="utf-8"))["r6b"]["auroc"]
    if recorded is not None and abs(point - float(recorded)) > 1e-9:
        raise SystemExit(f"point estimate {point} != recorded {recorded}")

    uniq = sorted(set(tasks.tolist()))
    idx_by_task = {t: np.where(tasks == t)[0] for t in uniq}
    rng = np.random.default_rng(SEED)
    clustered, naive = [], []
    n = len(labels)
    for _ in range(args.boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_task[uniq[i]] for i in pick])
        a = auroc_from(labels[sel], scores[sel])
        if not np.isnan(a):
            clustered.append(a)
        j = rng.integers(0, n, size=n)
        a2 = auroc_from(labels[j], scores[j])
        if not np.isnan(a2):
            naive.append(a2)
    cl, nv = np.array(clustered), np.array(naive)

    out = {
        "tag": "32b", "rep": "r6b_llm_ensemble", "auroc": point,
        "n_pairs": int(n), "n_diff": int((labels == 1).sum()),
        "n_same": int((labels == 0).sum()), "n_tasks": len(uniq),
        "reproduces_recorded_cell": recorded,
        "task_clustered_bootstrap": {
            "ci95": [float(np.percentile(cl, 2.5)),
                     float(np.percentile(cl, 97.5))],
            "sd": float(cl.std()), "draws": len(cl)},
        "naive_pair_bootstrap_for_contrast": {
            "ci95": [float(np.percentile(nv, 2.5)),
                     float(np.percentile(nv, 97.5))],
            "sd": float(nv.std())},
        "matrix_context": {"r3_hashed": [0.555, [0.492, 0.631]],
                           "r4_minilm": [0.580, [0.535, 0.628]],
                           "r5_bge": [0.573, [0.525, 0.627]]},
    }
    c = out["task_clustered_bootstrap"]["ci95"]
    d = out["naive_pair_bootstrap_for_contrast"]["ci95"]
    print(f"32b/r6b: AUROC {point:.4f}  ({n} pairs, {len(uniq)} tasks)")
    print(f"  task-clustered 95% CI [{c[0]:.3f}, {c[1]:.3f}]  <- honest")
    print(f"  naive pair     95% CI [{d[0]:.3f}, {d[1]:.3f}]  <- understates")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
