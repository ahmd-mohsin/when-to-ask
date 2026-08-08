"""End-to-end Phase-4 eval smoke on synthetic fixtures (CPU laptop, no GPU).

    python scripts/run_eval_smoke.py

Exercises the WHOLE eval chain on planted-structure data: fixtures -> tiny
A1/A2/A3 (in-process, the run_pipeline_smoke recipe) -> DetectorRuntime ->
lockstep N-run orchestrator with fake session/env + MockJudge across three
arms (no_ask, model_initiated, detector) -> fake resolver -> scoring ->
miniature headline table. PASS iff the detector fires on the planted fork
task, stays quiet on the clear task, and the naive-ask arm's question is
judged and answered.

This validates plumbing, not science: the real eval follows AWS_RUNBOOK
step 5 and is gated on the A4 stop point (decisions/011, /022).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.synthetic import FixtureConfig, generate  # noqa: E402
from wta.a1_direction import auroc, ambiguity_signal, build_direction  # noqa: E402
from wta.a2_autoencoder import A2Config, train_a2  # noqa: E402
from wta.a3_commitment import benign_spread_reference, calibrate_tau, s_reference  # noqa: E402
from wta.a4_gates import gate3_fork_collocation  # noqa: E402
from wta.agent_loop import AgentLoopConfig  # noqa: E402
from wta.eval.artifacts import DetectorArtifacts, DetectorRuntime  # noqa: E402
from wta.eval.orchestrator import ArmSpec, run_task_pass  # noqa: E402
from wta.eval.policies import DetectorPolicy  # noqa: E402
from wta.eval.scoring import PassRecord, score  # noqa: E402
from wta.logging_schema import ReadRecord  # noqa: E402
from xtid.harness.ask_f1 import AskSession  # noqa: E402
from xtid.harness.judge import MockJudge  # noqa: E402
from xtid.harness.tasks import Blocker, Task  # noqa: E402

WINDOW = 4
EPS_SETTLE = 0.6
READS_PER_TURN = 6
N_RUNS = 4

PHRASED_QUESTION = "Which interpretation of the requirement should I implement?"


def flatten(fx, topics):
    rows = []
    for t in topics:
        for i in range(fx.cfg.n_runs):
            for k in range(fx.h.shape[2]):
                cls = int(fx.class_id[i, t]) if (fx.ambiguous[t] and fx.committed[i, t, k]) else -1
                rows.append((fx.h[i, t, k], t, cls))
    X = np.stack([r[0] for r in rows])
    return X, np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


class FakeEvalSession:
    """Streams one fixture topic's planted reads through the agent protocol.

    seed -> fixture run index. Phrasing calls (segment_idx >= 10000) return a
    canned judge-shaped question; trajectory turns emit READS_PER_TURN planted
    reads each until the topic's reads are exhausted, then TASK_DONE."""

    def __init__(self, fx, topic: int, ask_script: bool = False):
        self.fx, self.topic, self.ask_script = fx, topic, ask_script
        self._cursor: dict[int, int] = {}

    def generate_segment(self, messages, *, seed, temperature, max_new_tokens,
                         segment_idx):
        if segment_idx >= 10_000:                       # phrasing / elicitation
            return [], PHRASED_QUESTION
        i = seed % self.fx.cfg.n_runs
        if self.ask_script:
            turns = [
                "THOUGHT: explore.\n```bash\nls\n```",
                f'THOUGHT: unsure.\n```bash\nask_human "{PHRASED_QUESTION}"\n```',
                "THOUGHT: done.\n```bash\necho TASK_DONE\n```",
            ]
            return [], turns[min(segment_idx, len(turns) - 1)]

        cur = self._cursor.get(seed, 0)
        total = self.fx.h.shape[2]
        if cur >= total:
            return [], "THOUGHT: done.\n```bash\necho TASK_DONE\n```"
        chunk = self.fx.h[i, self.topic, cur:cur + READS_PER_TURN]
        self._cursor[seed] = cur + len(chunk)
        reads = [ReadRecord(token_idx=8 * (k + 1), trigger="cadence", cue=None,
                            h=np.asarray(h, dtype=np.float32),
                            segment_idx=segment_idx)
                 for k, h in enumerate(chunk)]
        cls = int(self.fx.class_id[i, self.topic])
        text = (f"THOUGHT: taking reading c{cls}.\n"
                f"```bash\nsed -i 's/x/c{cls}/' cfg.py\n```")
        return reads, text


class FakeEnv:
    """Varies its output per call: identical observations every turn would
    make every run look stuck to the loop channel (repeated env states) and
    false-fire the trigger -- real command outputs differ per step."""

    def __init__(self):
        self._n = 0

    def execute(self, cmd):
        if cmd == "git diff --cached HEAD":
            return 0, "diff --git a/cfg.py b/cfg.py\n+fixed\n"
        self._n += 1
        return 0, f"ok output {self._n} for: {cmd[:30]}"


def fork_task(tid: str) -> Task:
    return Task(
        instance_id=tid, domain="synthetic", statement=f"Synthetic task {tid}.",
        source="synthetic",
        blockers=[Blocker(
            id=f"{tid}_fork",
            description="The requirement is ambiguous: two valid "
                        "interpretations of the target behaviour.",
            resolution="Use interpretation 0 (the gold interpretation).",
            example_questions=[PHRASED_QUESTION],
            type="ambiguous_requirement")],
        meta={"statement_stub": False})


def clear_task(tid: str) -> Task:
    return Task(instance_id=tid, domain="synthetic",
                statement=f"Synthetic task {tid}.", source="synthetic",
                blockers=[], meta={"statement_stub": False})


def to_record(res, ask_session, task_name: str) -> PassRecord:
    slot = ask_session.logs.get(res.instance_key)
    return PassRecord(
        task=task_name, instance_id=res.task_id, arm=res.arm,
        pass_idx=res.pass_idx, prediction=res.prediction,
        questions=list(slot.questions) if slot else [],
        n_blockers=slot.n_blockers if slot else 0,
        ask_events=res.ask_events, stats=res.policy_stats, compute=res.compute)


def main() -> int:
    print("=== fixtures + offline artifacts (pipeline-smoke recipe) ===")
    # THE canonical pipeline-smoke fixture (run_pipeline_smoke.py): 8 runs for
    # calibration -- 4 runs leave no settled same-class pairs (benign ref NaN),
    # and dropping blip/loop deflates the reference to ~0.09, which false-fires
    # clear topics. The eval arms then drive N_RUNS=4 of those runs.
    fx = generate(FixtureConfig(seed=33, n_runs=8, reads=24,
                                blip_topics=(1,), loop_runs=(7,)))
    d = build_direction(fx.should_ask_states(), fx.settled_states())
    a1 = auroc(ambiguity_signal(fx.should_ask_states(), d),
               ambiguity_signal(fx.proceed_states(), d))
    s_ref = s_reference(fx.should_ask_states() @ d, fx.proceed_states() @ d)
    X, top, cls = flatten(fx, range(fx.cfg.n_topics))
    model = train_a2(X, top, cls,
                     A2Config(in_dim=fx.cfg.hidden_dim, n_topics=fx.cfg.n_topics,
                              n_classes=fx.cfg.n_classes, epochs=120, seed=0))
    seqs = [model.encode_lean(fx.h[i, t]) for t in range(fx.cfg.n_ambiguous)
            for i in range(fx.cfg.n_runs)]
    calib = calibrate_tau(seqs, window=WINDOW, eps_settle=EPS_SETTLE, delta=0.1)
    groups = [[(model.encode_lean(fx.h[i, t]), int(fx.class_id[i, t]))
               for i in range(fx.cfg.n_runs)] for t in range(fx.cfg.n_ambiguous)]
    ref, _ = benign_spread_reference(groups, WINDOW, EPS_SETTLE)
    if not np.isfinite(ref) or ref <= 0:
        print(f"SMOKE: FAIL (benign reference unusable: {ref!r})")
        return 1
    g3 = gate3_fork_collocation(model.encode_topic(X), top, cls)
    art = DetectorArtifacts(
        direction=d.astype(np.float64), a2=model, tau=calib.tau,
        l_scale=calib.l_scale, s_ref=s_ref, window=WINDOW,
        benign_reference=ref, theta=g3.numbers["theta"], meta={"smoke": True})
    runtime = DetectorRuntime(art)
    print(f"A1 AUROC={a1:.3f} tau={calib.tau:.3f} l_scale={calib.l_scale:.3f} "
          f"ref={ref:.3f} theta={art.theta:.3f}")

    print("\n=== arms on synthetic tasks (fake session/env + MockJudge) ===")
    judge = MockJudge()
    cfg = AgentLoopConfig(max_steps=8)
    ladder = (0.7, 0.9, 1.1, 1.3)
    records, resolved = [], {}

    jobs = [
        ("no_ask", ArmSpec("no_ask", 1, temperatures=(1.0,)),
         fork_task("syn_fork"), FakeEvalSession(fx, topic=0)),
        ("model_initiated", ArmSpec("model_initiated", 1, ask_affordance=True,
                                    temperatures=(1.0,)),
         fork_task("syn_fork"), FakeEvalSession(fx, topic=0, ask_script=True)),
        ("detector/fork", ArmSpec("detector", N_RUNS, temperatures=ladder,
                                  policy_factory=lambda: DetectorPolicy(runtime)),
         fork_task("syn_fork"), FakeEvalSession(fx, topic=0)),
        ("detector/clear", ArmSpec("detector", N_RUNS, temperatures=ladder,
                                   policy_factory=lambda: DetectorPolicy(runtime)),
         clear_task("syn_clear"),
         FakeEvalSession(fx, topic=fx.cfg.n_ambiguous)),   # first clear topic
    ]
    outcomes = {}
    for label, arm, task, session in jobs:
        ask_session = AskSession()
        res = run_task_pass(arm, task, session=session,
                            env_factory=lambda run_id: FakeEnv(), judge=judge,
                            ask_session=ask_session, cfg=cfg, pass_idx=0,
                            seed_base=0, model_id="fake")
        rec = to_record(res, ask_session, task.instance_id)
        records.append(rec)
        outcomes[label] = res
        resolved[rec.expanded_id] = res.runs[res.committed_run_id].finished
        print(f"  {label:16s}: {res.compute['turns']} turns, "
              f"{len(res.ask_events)} asks, "
              f"committed={res.committed_run_id}")

    print("\n=== checks ===")
    det_fork = outcomes["detector/fork"]
    det_clear = outcomes["detector/clear"]
    naive = outcomes["model_initiated"]
    checks = {
        "detector fires on planted fork": len(det_fork.ask_events) >= 1,
        "detector fork ask matched the blocker": any(
            e.get("blocker_id") for e in det_fork.ask_events),
        "answer injected into runs (dedup once per bucket)":
            det_fork.policy_stats.get("asks", 0) >= 1,
        "detector quiet on clear task": len(det_clear.ask_events) == 0,
        "naive-ask question judged and answered": bool(
            naive.ask_events
            and naive.ask_events[0]["response"].startswith("Use interpretation 0")),
        "no_ask never asks": len(outcomes["no_ask"].ask_events) == 0,
    }
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")

    print("\n=== miniature headline table (fake resolver) ===")
    blockers_by_task = {"syn_fork": fork_task("syn_fork").blockers,
                        "syn_clear": []}
    out = score(records, resolved, blockers_by_task,
                fork_annotations={"syn_fork_fork": "structural"},
                expected_passes=1, include_partial=True)
    print(out["markdown"])

    ok = all(checks.values())
    print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
