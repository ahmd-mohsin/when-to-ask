"""End-to-end pipeline smoke on synthetic fixtures (CPU laptop, no GPU).

    python scripts/run_pipeline_smoke.py

Exercises the WHOLE offline->online chain on planted-structure data:
fixtures -> A1 direction -> A2 training -> A3 calibration -> A4 gates ->
Part B online replay (fork + blip + loop decisions) -> summary table.

This validates plumbing, not science: the real-data run follows the AWS
runbook and stops at the gates for owner review (decisions/011).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.synthetic import FixtureConfig, generate  # noqa: E402
from wta.a1_direction import ambiguity_signal, auroc, build_direction  # noqa: E402
from wta.a2_autoencoder import A2Config, train_a2  # noqa: E402
from wta.a3_commitment import (  # noqa: E402
    CommitmentDetector, benign_spread_reference, calibrate_tau, s_reference,
)
from wta.a4_gates import (  # noqa: E402
    gate1_topic_leakage, gate2_decision_recovery, gate3_fork_collocation,
    gate5_lean_separation, gate7_aggregate, gate7_lead_time,
)
from wta.online import AskTrigger, TriggerConfig  # noqa: E402

WINDOW = 4


def flatten(fx, topics):
    rows = []
    for t in topics:
        for i in range(fx.cfg.n_runs):
            for k in range(fx.h.shape[2]):
                cls = int(fx.class_id[i, t]) if (fx.ambiguous[t] and fx.committed[i, t, k]) else -1
                rows.append((fx.h[i, t, k], t, cls))
    X = np.stack([r[0] for r in rows])
    return X, np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def main() -> int:
    print("=== fixtures ===")
    fx = generate(FixtureConfig(seed=33, n_runs=8, reads=24,
                                blip_topics=(1,), loop_runs=(7,)))
    print(f"topics={fx.cfg.n_topics} (ambiguous={fx.cfg.n_ambiguous}), "
          f"runs={fx.cfg.n_runs}, reads={fx.h.shape[2]}")

    print("\n=== A1: ambiguity direction ===")
    d = build_direction(fx.should_ask_states(), fx.settled_states())
    score = auroc(ambiguity_signal(fx.should_ask_states(), d),
                  ambiguity_signal(fx.proceed_states(), d))
    s_ref = s_reference(fx.should_ask_states() @ d, fx.proceed_states() @ d)
    print(f"cos(d, d*)={float(d @ fx.d_star):.3f}  AUROC={score:.3f}  s_ref={s_ref:.3f}")

    print("\n=== A2: disentangling autoencoder (CPU training) ===")
    X, top, cls = flatten(fx, range(fx.cfg.n_topics))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(0.7 * len(idx))
    tr, he = idx[:cut], idx[cut:]
    model = train_a2(X[tr], top[tr], cls[tr],
                     A2Config(in_dim=fx.cfg.hidden_dim, n_topics=fx.cfg.n_topics,
                              n_classes=fx.cfg.n_classes, epochs=120, seed=0))
    print("trained; frozen encoders ready")

    print("\n=== A3: conformal commitment ===")
    seqs = [model.encode_lean(fx.h[i, t]) for t in range(fx.cfg.n_ambiguous)
            for i in range(fx.cfg.n_runs)]
    EPS_SETTLE = 0.6  # in l_scale units (learned-L noise floor sits ~0.4)
    calib = calibrate_tau(seqs, window=WINDOW, eps_settle=EPS_SETTLE, delta=0.1)
    groups = [[(model.encode_lean(fx.h[i, t]), int(fx.class_id[i, t]))
               for i in range(fx.cfg.n_runs)] for t in range(fx.cfg.n_ambiguous)]
    ref, n_pairs = benign_spread_reference(groups, WINDOW, EPS_SETTLE)
    print(f"tau={calib.tau:.3f} (l_scale={calib.l_scale:.3f}) from {calib.n_points} "
          f"points ({calib.n_skipped_never_settled} never settled); "
          f"benign spread ref={ref:.3f} from {n_pairs} same-class pairs")

    print("\n=== A4 gates (machinery on fixtures) ===")
    t_tr, t_he = model.encode_topic(X[tr]), model.encode_topic(X[he])
    print(" ", gate1_topic_leakage(t_tr, cls[tr], t_he, cls[he]))
    print(" ", gate2_decision_recovery(t_tr, top[tr], t_he, top[he]))
    g3 = gate3_fork_collocation(t_he, top[he], cls[he])
    print(" ", g3)
    print(" ", gate5_lean_separation(model.encode_lean(X[he]), top[he], cls[he]))

    per_dec = []
    for t in range(fx.cfg.n_ambiguous):
        r_by_run = np.stack([model.encode_lean(fx.h[i, t]) for i in range(fx.cfg.n_runs)])
        weights = np.zeros(r_by_run.shape[:2])
        for i in range(fx.cfg.n_runs):
            det = CommitmentDetector(tau=calib.tau, s_ref=s_ref, window=WINDOW,
                                     l_scale=calib.l_scale)
            s_seq = fx.h[i, t] @ d
            for k in range(r_by_run.shape[1]):
                _, weights[i, k] = det.step(r_by_run[i, k], float(s_seq[k]))
        per_dec.append(gate7_lead_time(r_by_run, weights,
                                       fx.action_read[:, t], fx.class_id[:, t]))
    print(" ", gate7_aggregate(per_dec, proxy=True))

    print("\n=== Part B: online replay (theta from gate 3) ===")
    fired_at: dict[int, int | None] = {}
    for t in range(fx.cfg.n_topics):
        trig = AskTrigger(TriggerConfig(theta=g3.numbers["theta"],
                                        reference=ref, slack=0.1 * ref))
        dets = {i: CommitmentDetector(tau=calib.tau, s_ref=s_ref, window=WINDOW,
                                      l_scale=calib.l_scale)
                for i in range(fx.cfg.n_runs)}
        fired_at[t] = None
        for k in range(fx.h.shape[2]):
            for i in range(fx.cfg.n_runs):
                h = fx.h[i, t, k]
                # signal the stuck run's repeated env states on topic 2 only,
                # so topics 0/1 demonstrate the vote-dispersion path and
                # topic 2 demonstrates the loop channel
                if i in fx.cfg.loop_runs and t == 2:
                    trig.notify_env_state(i, "stuck")
                r = model.encode_lean(h[None])[0] / calib.l_scale  # scaled units
                _, w = dets[i].step(model.encode_lean(h[None])[0],
                                    float(ambiguity_signal(h, d)))
                decision = trig.observe(i, model.encode_topic(h[None])[0], r,
                                        float(ambiguity_signal(h, d)), w)
                if decision is not None and fired_at[t] is None:
                    fired_at[t] = k
        kind = ("LOOP+FORK" if t == 2
                else "FORK" if fx.ambiguous[t] and t not in fx.cfg.blip_topics
                else "BLIP+FORK" if t in fx.cfg.blip_topics else "CLEAR")
        want = "should fire" if fx.ambiguous[t] else "should NOT fire"
        print(f"  topic {t} [{kind}] ({want}): "
              f"{'fired at read ' + str(fired_at[t]) if fired_at[t] is not None else 'no fire'}")

    forks_fired = all(fired_at[t] is not None for t in range(fx.cfg.n_ambiguous))
    clears_quiet = all(fired_at[t] is None for t in range(fx.cfg.n_ambiguous, fx.cfg.n_topics))
    print(f"\nSMOKE: {'PASS' if forks_fired and clears_quiet else 'FAIL'} "
          f"(all ambiguous fired: {forks_fired}; all clear quiet: {clears_quiet})")
    return 0 if forks_fired and clears_quiet else 1


if __name__ == "__main__":
    raise SystemExit(main())
