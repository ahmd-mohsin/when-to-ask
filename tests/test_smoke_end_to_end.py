"""End-to-end smoke test: the full de-risk pipeline on synthetic tasks.

This validates the whole stack (backbone -> N trajectories -> alignment -> signals ->
recording -> C1 analysis) and asserts the synthetic harness reproduces the brief's S4
structure, which is exactly what the analysis code must be able to surface:

  * internal cross-trajectory divergence beats output divergence (B1) on the fork slice;
  * internal divergence fires earlier than output (positive lead-time);
  * the correctness probe (B3) covers the confident-convergent blind spot that internal
    divergence misses.

It is a sanity harness on synthetic data, NOT evidence for C1 -- that needs the real GPU
run. But it proves the pipeline computes the intended quantities correctly.
"""

from xtid.agent.multi import run_tasks
from xtid.analysis.lead_time import lead_time_analysis
from xtid.analysis.separation import c1_verdict, separation_table
from xtid.backbone.model import build_model
from xtid.harness.executor import SyntheticExecutor
from xtid.harness.tasks import make_synthetic_tasks
from xtid.recording.recorder import build_records


def _run(n_tasks=12, n_traj=4, seed=0):
    model = build_model({"kind": "fake"})
    tasks = make_synthetic_tasks(n=n_tasks, seed=seed)
    trajs = run_tasks(model, tasks, n_trajectories=n_traj)
    return build_records(trajs, tasks, SyntheticExecutor(), seed=seed)


def test_pipeline_produces_one_record_per_decision_point():
    tasks = make_synthetic_tasks(n=6, seed=0)
    run = _run(n_tasks=6)
    assert len(run.records) == sum(len(t.decision_points) for t in tasks)
    assert all(r.n == 4 for r in run.records)


def test_internal_beats_output_on_fork_slice():
    table = separation_table(_run())
    internal = table["internal_mean_pairwise_cosine"]["fork_vs_clear"]
    output = table["output_divergence_b1"]["fork_vs_clear"]
    assert internal is not None and output is not None
    assert internal > output  # the make-or-break C1 comparison (on synthetic structure)


def test_internal_fires_earlier_than_output():
    lt = lead_time_analysis(_run())
    assert lt["n_both_cross"] >= 1
    assert lt["median_lead"] is not None and lt["median_lead"] >= 1


def test_probe_covers_confident_wrong_blind_spot():
    v = c1_verdict(separation_table(_run()))
    assert v["internal_beats_b1_on_fork"] is True
    assert v["probe_covers_blind_spot"] is True


def test_verdict_keys_present():
    v = c1_verdict(separation_table(_run()))
    assert {"internal_fork_auroc", "output_b1_fork_auroc", "probe_confident_wrong_auroc"} <= set(v)
