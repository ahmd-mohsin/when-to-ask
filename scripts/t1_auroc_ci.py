"""T1 uncertainty: task-clustered bootstrap CIs for the separation AUROCs.

Additive analysis (028 Amendment A item 7) — it does NOT change or rerun any
pre-registered cell; it puts a sampling interval around the same statistic
each cell already reports.

Why clustered: the commitment-pair pool is NOT a sample of independent
pairs. Every pair shares runs with many others, and every run shares a task
and a blocker with many more. Treating pairs as independent (Hanley-McNeil
or a naive pair bootstrap) understates the SE for exactly the reason the
gate5 permutation test had to be run-level rather than read-level (026):
the independent unit is the cluster, not the comparison. Primary interval
here resamples TASKS with replacement (the outermost independent unit,
matching the k-fold grouping used everywhere else in this project); the
naive pair-level interval is reported alongside purely to show the inflation.

Pair rows are cached per representation so each embedder pass is paid once:
    results/t1_pair_rows_{rep}.json

    python scripts/t1_auroc_ci.py --rep hashed
    python scripts/t1_auroc_ci.py --rep minilm
    python scripts/t1_auroc_ci.py --rep bge
    python scripts/t1_auroc_ci.py --rep hashed --a0 data/a0_v2 \
        --labels-debug models/t5_v2_14b_fixed/labels_debug.jsonl --tag 14b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from offline_ask_headtohead import (commit_rounds, load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.divergence import behavior_features  # noqa: E402
from wta.labeling import load_class_artifact  # noqa: E402

BOOT = 2000
SEED = 0


def auroc_from(labels: np.ndarray, dists: np.ndarray) -> float:
    """AUROC with diff pairs as positives, ties counted 0.5 (exact U)."""
    pos, neg = dists[labels == 1], dists[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # chunked to keep the outer product bounded
    wins = ties = 0
    for i in range(0, len(pos), 4096):
        blk = pos[i:i + 4096][:, None]
        wins += int((blk > neg[None, :]).sum())
        ties += int((blk == neg[None, :]).sum())
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def warm_cache(actions, embedder) -> int:
    """Pre-embed every mutating turn's text in ONE batched pass.

    behavior_features calls the embedder one text at a time; for bge-large
    that is ~21h of batch-1 forward passes. The embedders cache by exact
    text, so warming the cache with the identical strings makes the later
    per-turn calls pure lookups. The text construction here MUST match
    wta.divergence.behavior_features exactly, and batching cannot change a
    CLS-pooled result given correct attention masking — verified by the
    AUROC reproducing the as-run point estimate.
    """
    from wta.labeling import _is_mutating
    texts = []
    for runs in actions.values():
        for acts in runs.values():
            for a in acts:
                if _is_mutating(a.action_text or ""):
                    obs = a.observables or {}
                    texts.append(" ".join([str(obs.get("subgoal") or ""),
                                           a.action_text or "",
                                           str(obs.get("error_signature") or "")]))
    uniq = list(dict.fromkeys(texts))
    for i in range(0, len(uniq), 256):
        embedder.embed(uniq[i:i + 256])
        print(f"  warmed {min(i + 256, len(uniq))}/{len(uniq)}", flush=True)
    return len(uniq)


def build_pair_rows(actions, art, committed, embedder):
    """[(task, blocker, label, distance)] over the frozen pool construction
    (feature_signal_gate.pair_distances), with task provenance kept."""
    rows = []
    for task, runs in actions.items():
        feats = {rid: behavior_features(a, r_embedder=embedder)
                 for rid, a in runs.items()}
        rounds = commit_rounds(runs, committed, art, task)
        for blocker in art[task]:
            vecs: dict[str, list] = {}
            for rid in runs:
                c = committed.get((rid, blocker))
                k = rounds.get((rid, blocker))
                if c is not None and k is not None and k < len(feats[rid]):
                    vecs.setdefault(c, []).append(feats[rid][k].r_vec)
            classes = sorted(vecs)
            for i, ca in enumerate(classes):
                va = vecs[ca]
                for x in range(len(va)):
                    for y in range(x + 1, len(va)):
                        rows.append((task, blocker, 0,
                                     float(np.linalg.norm(va[x] - va[y]))))
                for cb in classes[i + 1:]:
                    for u in vecs[ca]:
                        for v in vecs[cb]:
                            rows.append((task, blocker, 1,
                                         float(np.linalg.norm(u - v))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=("hashed", "minilm", "bge"),
                    default="hashed")
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--tag", default="32b")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    cache = Path(f"results/t1_pair_rows_{args.tag}_{args.rep}.json")
    if cache.exists():
        rows = [tuple(r) for r in json.loads(cache.read_text(encoding="utf-8"))]
        print(f"loaded {len(rows)} cached pair rows from {cache}")
    else:
        art = load_class_artifact(args.classes)
        committed = load_commitments(Path(args.labels_debug))
        actions = load_task_actions(Path(args.a0), None)
        actions = {t: r for t, r in actions.items() if t in art}
        emb = None
        if args.rep == "minilm":
            from wta.embed import MiniLMEmbedder
            emb = MiniLMEmbedder()
        elif args.rep == "bge":
            from wta.embed import BgeEmbedder
            emb = BgeEmbedder()
        if emb is not None:
            n = warm_cache(actions, emb)
            print(f"warmed {n} unique texts ({time.time() - t0:.0f}s)")
        rows = build_pair_rows(actions, art, committed, emb)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")
        print(f"built {len(rows)} pair rows -> {cache} "
              f"({time.time() - t0:.0f}s)")

    tasks = np.array([r[0] for r in rows])
    labels = np.array([r[2] for r in rows], dtype=int)
    dists = np.array([r[3] for r in rows], dtype=float)
    point = auroc_from(labels, dists)
    uniq = sorted(set(tasks.tolist()))
    idx_by_task = {t: np.where(tasks == t)[0] for t in uniq}

    rng = np.random.default_rng(SEED)
    clustered, naive = [], []
    n = len(labels)
    for _ in range(args.boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_task[uniq[i]] for i in pick])
        a = auroc_from(labels[sel], dists[sel])
        if not np.isnan(a):
            clustered.append(a)
        j = rng.integers(0, n, size=n)
        a2 = auroc_from(labels[j], dists[j])
        if not np.isnan(a2):
            naive.append(a2)
    cl = np.array(clustered)
    nv = np.array(naive)
    out = {
        "tag": args.tag, "rep": args.rep, "auroc": point,
        "n_pairs": int(n), "n_diff": int((labels == 1).sum()),
        "n_same": int((labels == 0).sum()), "n_tasks": len(uniq),
        "task_clustered_bootstrap": {
            "ci95": [float(np.percentile(cl, 2.5)),
                     float(np.percentile(cl, 97.5))],
            "sd": float(cl.std()), "draws": len(cl)},
        "naive_pair_bootstrap_for_contrast": {
            "ci95": [float(np.percentile(nv, 2.5)),
                     float(np.percentile(nv, 97.5))],
            "sd": float(nv.std())},
    }
    c = out["task_clustered_bootstrap"]["ci95"]
    d = out["naive_pair_bootstrap_for_contrast"]["ci95"]
    print(f"\n{args.tag}/{args.rep}: AUROC {point:.3f}")
    print(f"  task-clustered 95% CI [{c[0]:.3f}, {c[1]:.3f}]  <- honest")
    print(f"  naive pair     95% CI [{d[0]:.3f}, {d[1]:.3f}]  "
          f"<- understates (pairs share runs and tasks)")
    p = Path(args.out or f"results/t1_auroc_ci_{args.tag}_{args.rep}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
