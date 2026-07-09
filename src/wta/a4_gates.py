"""A4: the seven research validation gates -- OURS (spec A4).

Every gate is a pure function over FROZEN encodings + held-out labels; nothing
here trains or mutates. On real data the numbers go to the owner unfiltered
(build brief: hypotheses, not tests to pass). eta^2 follows ReDAct's
ANOVA-style leakage quantification (arXiv 2602.19396, implemented from the
paper).

Gate 7 note on the ground rule: this is OFFLINE analysis, where actions and
read-index alignment are legitimate teachers; the runtime trigger (wta/online)
never sees either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GateResult:
    name: str
    numbers: dict
    note: str = ""
    proxy: bool = False

    def __str__(self) -> str:
        nums = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in self.numbers.items())
        star = " [proxy]" if self.proxy else ""
        return f"{self.name}: {nums}{star}" + (f"  # {self.note}" if self.note else "")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def kfold_group_indices(group_ids: np.ndarray, k: int, seed: int = 0):
    """Yield (train_mask, test_mask) for k folds partitioned by GROUP so that
    all reads of one group (e.g. one (task, seed) run) land in the same fold --
    never split a run across train/test (decisions/014 power fix). Groups are
    shuffled deterministically then round-robin assigned to folds."""
    group_ids = np.asarray(group_ids)
    groups = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    fold_of_group = {groups[g]: i % k for i, g in enumerate(perm)}
    fold = np.array([fold_of_group[g] for g in group_ids])
    for f in range(k):
        test = fold == f
        if test.any() and (~test).any():
            yield ~test, test


def _probe_acc(z_tr, y_tr, z_he, y_he) -> float:
    from sklearn.linear_model import LogisticRegression

    probe = LogisticRegression(max_iter=1000).fit(z_tr, y_tr)
    return float(probe.score(z_he, y_he))


def eta_squared(z: np.ndarray, labels: np.ndarray) -> float:
    """ReDAct-style leakage: per-dimension variance explained by the label
    (SS_between / SS_total), averaged over dims. ~0 = blind."""
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels)
    grand = z.mean(axis=0)
    ss_tot = ((z - grand) ** 2).sum(axis=0)
    ss_b = np.zeros_like(grand)
    for g in np.unique(labels):
        zg = z[labels == g]
        ss_b += len(zg) * (zg.mean(axis=0) - grand) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        per_dim = np.where(ss_tot > 1e-12, ss_b / ss_tot, 0.0)
    return float(per_dim.mean())


def _unit_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def _sample_pair_cosines(t_vecs, mask_a, mask_b, n_pairs, rng) -> np.ndarray:
    """Cosines of random cross pairs (a from mask_a, b from mask_b, a != b)."""
    ia, ib = np.where(mask_a)[0], np.where(mask_b)[0]
    u = _unit_rows(t_vecs)
    out = []
    for _ in range(n_pairs):
        a, b = rng.choice(ia), rng.choice(ib)
        if a == b:
            continue
        out.append(float(u[a] @ u[b]))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# gates 1-6
# ---------------------------------------------------------------------------


def gate1_topic_leakage(t_tr, cls_tr, t_he, cls_he, dec_he=None) -> GateResult:
    """Hypothesis: T is blind to the resolution LEAN. Must be tested WITHIN a
    decision -- global class ids are nested inside decisions (each class
    belongs to one decision), so a global 'predict class from T' probe is
    confounded by decision identity (which gate 2 WANTS T to encode). Pass
    dec_he to measure within-decision leakage: per decision, cross-validated
    probe T->class over that decision's held reads, aggregated as accuracy vs
    that decision's chance and mean partial eta^2. Falls back to the (naive,
    confounded) global measure when dec_he is None -- used only by the
    fixture contract test, where classes are shared across decisions."""
    from sklearn.linear_model import LogisticRegression

    lab_he = cls_he >= 0
    if dec_he is None:
        lab_tr = cls_tr >= 0
        acc = _probe_acc(t_tr[lab_tr], cls_tr[lab_tr], t_he[lab_he], cls_he[lab_he])
        return GateResult("gate1_topic_leakage",
                          {"class_from_T_acc": acc,
                           "chance": 1.0 / len(np.unique(cls_he[lab_he])),
                           "eta2": eta_squared(t_he[lab_he], cls_he[lab_he])},
                          note="want acc ~ chance, eta2 ~ 0 (GLOBAL, fixture-only)")

    accs, chances, etas, weights, n_dec = [], [], [], [], 0
    T, C, D = t_he[lab_he], cls_he[lab_he], dec_he[lab_he]
    for d in np.unique(D):
        m = D == d
        cls_d, td = C[m], T[m]
        classes, counts = np.unique(cls_d, return_counts=True)
        if len(classes) < 2 or counts.min() < 2 or m.sum() < 6:
            continue
        n_dec += 1
        # 5-fold-ish: fit on 70%, score on 30%, stratified-ish by shuffling
        rng = np.random.default_rng(0)
        order = rng.permutation(m.sum())
        cut = int(0.7 * m.sum())
        tr_i, te_i = order[:cut], order[cut:]
        if len(np.unique(cls_d[tr_i])) < 2 or len(te_i) == 0:
            continue
        probe = LogisticRegression(max_iter=1000).fit(td[tr_i], cls_d[tr_i])
        accs.append(float(probe.score(td[te_i], cls_d[te_i])))
        chances.append(1.0 / len(classes))
        etas.append(eta_squared(td, cls_d))
        weights.append(int(m.sum()))
    if not accs:
        return GateResult("gate1_topic_leakage", {"n_decisions": 0},
                          note="INSUFFICIENT within-decision data (need decisions "
                               "with >=2 classes, >=2 reads/class in held-out)")
    w = np.array(weights, dtype=float)
    return GateResult("gate1_topic_leakage",
                      {"within_decision_class_from_T_acc": float(np.average(accs, weights=w)),
                       "within_decision_chance": float(np.average(chances, weights=w)),
                       "mean_partial_eta2": float(np.average(etas, weights=w)),
                       "n_decisions": n_dec},
                      note="want acc ~ chance, eta2 ~ 0 (WITHIN decision -- the real test)")


def gate2_decision_recovery(t_tr, top_tr, t_he, top_he) -> GateResult:
    acc = _probe_acc(t_tr, top_tr, t_he, top_he)
    return GateResult("gate2_decision_recovery",
                      {"topic_from_T_acc": acc,
                       "chance": 1.0 / len(np.unique(top_he))},
                      note="want acc high")


def gate3_fork_collocation(t_he, top_he, cls_he, n_pairs: int = 4000,
                           seed: int = 0) -> GateResult:
    """Same-decision/opposite-resolution cosine vs different-decision cosine;
    the crossover sets Part B's theta."""
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(seed)
    u = _unit_rows(t_he)
    same, diff = [], []
    idx_by_topic = {t: np.where(top_he == t)[0] for t in np.unique(top_he)}
    for _ in range(n_pairs):
        t = rng.choice(list(idx_by_topic))
        ids = idx_by_topic[t]
        if len(ids) >= 2:
            a, b = rng.choice(ids, 2, replace=False)
            if cls_he[a] >= 0 and cls_he[b] >= 0 and cls_he[a] != cls_he[b]:
                same.append(float(u[a] @ u[b]))
        t2 = rng.choice(list(idx_by_topic))
        if t2 != t:
            a, b = rng.choice(idx_by_topic[t]), rng.choice(idx_by_topic[t2])
            diff.append(float(u[a] @ u[b]))
    same, diff = np.asarray(same), np.asarray(diff)
    if len(same) < 10 or len(diff) < 10:
        return GateResult("gate3_fork_collocation",
                          {"n_same": len(same), "n_diff": len(diff)},
                          note="INSUFFICIENT PAIRS -- need held-out runs that "
                               "commit to different classes on shared decisions; "
                               "no theta produced")
    y = np.r_[np.ones(len(same)), np.zeros(len(diff))]
    fpr, tpr, thr = roc_curve(y, np.r_[same, diff])
    theta = float(thr[np.argmax(tpr - fpr)])
    return GateResult("gate3_fork_collocation",
                      {"mean_same_decision_cos": float(same.mean()),
                       "mean_diff_decision_cos": float(diff.mean()),
                       "theta": theta, "n_same": len(same), "n_diff": len(diff)},
                      note="want same >> diff; theta feeds Part B bucketing")


def gate4_conflation(t_he, pairs: np.ndarray, theta: float) -> GateResult:
    """pairs: (m, 2) indices of same-observable/different-decision reads
    (pairing built from observables by the caller)."""
    u = _unit_rows(t_he)
    cos = np.array([float(u[a] @ u[b]) for a, b in np.asarray(pairs)])
    return GateResult("gate4_conflation",
                      {"mean_cos": float(cos.mean()),
                       "frac_collocated": float((cos >= theta).mean()),
                       "n_pairs": len(cos)},
                      note="want frac_collocated ~ 0 (labels did not conflate decisions)")


def gate5_lean_separation(l_he, top_he, cls_he) -> GateResult:
    """Within each decision: between-class centroid distance vs within-class
    spread of committed L vectors."""
    ratios, sils = [], []
    from sklearn.metrics import silhouette_score

    for t in np.unique(top_he):
        m = (top_he == t) & (cls_he >= 0)
        if m.sum() < 4 or len(np.unique(cls_he[m])) < 2:
            continue
        L, c = l_he[m], cls_he[m]
        cents = {g: L[c == g].mean(axis=0) for g in np.unique(c)}
        within = float(np.mean([np.linalg.norm(L[c == g] - cents[g], axis=1).mean()
                                for g in cents if (c == g).sum() > 1]))
        gs = sorted(cents)
        between = float(np.mean([np.linalg.norm(cents[a] - cents[b])
                                 for i, a in enumerate(gs) for b in gs[i + 1:]]))
        ratios.append(between / max(within, 1e-9))
        sils.append(float(silhouette_score(L, c)))
    if not ratios:
        return GateResult("gate5_lean_separation", {"n_decisions": 0},
                          note="INSUFFICIENT DATA -- no held-out decision has "
                               ">= 2 classes with >= 4 labeled reads")
    return GateResult("gate5_lean_separation",
                      {"between_within_ratio": float(np.mean(ratios)),
                       "silhouette": float(np.mean(sils)),
                       "n_decisions": len(ratios)},
                      note="want ratio >> 1 (the trigger has something to fire on)")


def gate6_ood_transfer(t_unseen, top_unseen, theta: float) -> GateResult:
    """Bucket purity of leader-clustered T on decisions never seen in training."""
    from wta.bucketing import leader_cluster_points

    labels = leader_cluster_points(_unit_rows(t_unseen), theta)
    purity, n = 0.0, 0
    for b in np.unique(labels):
        m = labels == b
        _, counts = np.unique(top_unseen[m], return_counts=True)
        purity += counts.max()
        n += m.sum()
    return GateResult("gate6_ood_transfer",
                      {"bucket_purity": float(purity / max(n, 1)),
                       "n_buckets": int(len(np.unique(labels))),
                       "n_decisions": int(len(np.unique(top_unseen)))},
                      note="honest limitation number on unseen task families")


# ---------------------------------------------------------------------------
# gate 7 -- lead-time (offline analysis; read-index alignment is teacher-legal)
# ---------------------------------------------------------------------------


def gate7_lead_time(r_by_run: np.ndarray, weights: np.ndarray,
                    action_read: np.ndarray, class_by_run: np.ndarray,
                    onset_frac: float = 0.5) -> dict | None:
    """One decision. r_by_run (N, R, d_L); weights (N, R) commitment weights;
    action_read (N,) the read at which each run acted (-1 = never);
    class_by_run (N,) eventual interpretation.

    Returns {onset, action_divergence, K} or None if no differing pair acted.
    """
    n, reads, _ = r_by_run.shape
    disp = np.zeros(reads)
    for k in range(reads):
        num = den = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                w = weights[i, k] * weights[j, k]
                if w <= 0:
                    continue
                num += w * float(np.linalg.norm(r_by_run[i, k] - r_by_run[j, k]))
                den += w
        disp[k] = num / den if den > 0 else 0.0

    if disp.max() <= 0:
        return None
    onset = int(np.argmax(disp >= onset_frac * disp.max()))

    div_reads = [max(action_read[i], action_read[j])
                 for i in range(n) for j in range(i + 1, n)
                 if class_by_run[i] != class_by_run[j]
                 and action_read[i] >= 0 and action_read[j] >= 0]
    if not div_reads:
        return None
    action_div = int(min(div_reads))
    return {"onset": onset, "action_divergence": action_div,
            "K": action_div - onset}


def gate7_aggregate(per_decision: list, proxy: bool = False) -> GateResult:
    ks = np.array([d["K"] for d in per_decision if d is not None])
    return GateResult("gate7_lead_time",
                      {"median_K": float(np.median(ks)) if len(ks) else float("nan"),
                       "frac_positive": float((ks > 0).mean()) if len(ks) else float("nan"),
                       "n_decisions": int(len(ks))},
                      note="make-or-break: want K > 0 (internal signal leads behaviour)",
                      proxy=proxy)
