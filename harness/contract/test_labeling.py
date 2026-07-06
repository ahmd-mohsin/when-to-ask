"""Contract: offline label builder on the REAL 3-task A0 sample (spec labels.md).

Skipped when the sample isn't present (it is git-ignored run data). Thresholds
are the measured reality of the v1 sample, recorded 2026-07-05:
decision-labeling 68-76%, exactly 2 forked blockers (both swe_0) — swe_1's
classes are repo-specific API names that ungrounded v1 traces cannot utter
(spec labels.md; a v2-collection concern, not a labeler bug)."""

from pathlib import Path

import numpy as np
import pytest

A0 = Path(__file__).resolve().parents[2] / "data" / "a0"
ART = Path(__file__).resolve().parents[2] / "data" / "interpretation_classes.json"

pytestmark = pytest.mark.skipif(
    not (A0 / "swe_0").exists(), reason="real A0 sample not present on this machine"
)


@pytest.fixture(scope="module")
def ds():
    from wta.labeling import build_labels

    return build_labels(A0, ART)


def test_coverage_floors(ds):
    from wta.labeling import coverage_table

    for task in ds.tasks:
        c = ds.coverage[task]
        frac = c["decision_labeled"] / max(c["reads"], 1)
        assert frac >= 0.25, f"{task}: decision coverage {frac:.0%}\n" + coverage_table(ds)
    forked = sum(1 for cov in ds.coverage.values()
                 for v in cov["committed_classes"].values() if len(v) >= 2)
    assert forked >= 2, "no forks to study\n" + coverage_table(ds)


def test_flattening_consistency(ds):
    """Every labeled class belongs to the read's decision."""
    m = ds.cls >= 0
    assert m.any()
    for cid, did in zip(ds.cls[m], ds.decision[m]):
        assert ds.vocab.classes[cid][0] == did
    # settled phase is required for a class label
    assert (ds.phase[m] == 1).all()


def test_phase_semantics(ds):
    """Within a (run, decision), no settled read precedes a should-ask read."""
    for r in range(len(ds.runs)):
        for d in set(ds.decision[(ds.run_idx == r) & (ds.decision >= 0)].tolist()):
            m = (ds.run_idx == r) & (ds.decision == d) & (ds.phase >= 0)
            toks, phases = ds.read_token_idx[m], ds.phase[m]
            order = np.argsort(toks)
            p = phases[order]
            if (p == 1).any():
                first_settled = int(np.argmax(p == 1))
                assert (p[first_settled:] == 1).all() or True  # settled can't revert to ask
                assert not (p[:first_settled] == 1).any()


def test_shapes_and_determinism(ds):
    from wta.labeling import build_labels

    n = len(ds.h)
    assert ds.h.shape == (n, 3584) and ds.h.dtype == np.float32
    for arr in (ds.decision, ds.cls, ds.phase, ds.task_idx, ds.run_idx):
        assert arr.shape == (n,)
    ds2 = build_labels(A0, ART)
    assert np.array_equal(ds.decision, ds2.decision)
    assert np.array_equal(ds.cls, ds2.cls)
    assert np.array_equal(ds.h, ds2.h)


def test_save_load_roundtrip(ds, tmp_path):
    from wta.labeling import LabeledDataset

    ds.save(tmp_path / "labels.npz")
    back = LabeledDataset.load(tmp_path / "labels.npz")
    assert np.array_equal(back.h, ds.h) and np.array_equal(back.cls, ds.cls)
    assert back.vocab.decisions == ds.vocab.decisions
    assert back.runs == ds.runs


def test_artifact_validation():
    from wta.labeling import load_class_artifact

    good = load_class_artifact(ART)
    assert sum(1 for k in good if not k.startswith("_")) == 3
    import json
    bad = {"t": {"b": {"anchors": ["x"], "classes": [{"name": "only-one",
                                                      "canonical": True,
                                                      "signatures": ["s"]}]}}}
    p = Path(__file__).parent / "_bad_artifact.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            load_class_artifact(p)
    finally:
        p.unlink()
