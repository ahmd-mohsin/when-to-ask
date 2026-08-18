"""Contract: one coordinate system in the labeler (spec labels.md v3.1,
decisions/026).

The R2 audit found build_labels mixing three coordinate systems (raw segment
join / newline-translated .txt / whitespace-collapsed norm): windows sliced
from the wrong string, CRLF runs desynced wholesale, and phase compared
positions across systems. These tests are the proof of repair — each of the
fixture tests FAILS on the pre-026 labeler and passes after it.

House rules for fixtures here (decisions/026): every .txt is written with
write_bytes (Windows write_text would translate \n -> \r\n on disk and move
raw offsets), and token indices are COMPUTED from the tokenizer's own
token->char map, never hardcoded against a tokenizer version.
"""

import json

import numpy as np
import pytest

from wta.logging_schema import ReadRecord, RunLog, save_run_log

H = 16
DEF_TOK = "Qwen/Qwen2.5-Coder-7B-Instruct"  # build_labels' default

ANCHORS_TEXT = ("the retry policy question: for transient errors the "
                "backoff decision matters here. ")
ART = {"taskX": {"blockerA": {
    "anchors": ["retry policy", "transient errors", "backoff decision"],
    "classes": [
        {"name": "canonical", "canonical": True, "signatures": ["retry_all()"]},
        {"name": "alt", "signatures": ["retry_transient()"]},
    ]}}}
COMMIT = "I'll commit to retry_all(). "


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DEF_TOK)


def _tok_at(tok, seg_text: str, char_target: int) -> int:
    """Index of the last token whose char start is <= char_target."""
    from wta.labeling import token_char_positions
    starts = token_char_positions(seg_text, tok)
    return max(i for i, a in enumerate(starts) if a <= char_target)


def _write_run(root, segments, reads, run_id="taskX-s0"):
    """A taskX run on disk: run log + sidecar + raw-bytes .txt."""
    task_dir = root / "taskX"
    log = RunLog(run_id=run_id, task_id="taskX", seed=0, temperature=0.7,
                 model_id="fake", mid_layer=4, reads=reads)
    save_run_log(log, task_dir)
    (task_dir / f"{run_id}.segments.json").write_text(json.dumps(segments),
                                                      encoding="utf-8")
    (task_dir / f"{run_id}.txt").write_bytes(
        "\n\n".join(segments).encode("utf-8"))
    return root


def _art_file(tmp_path):
    p = tmp_path / "classes.json"
    p.write_text(json.dumps(ART), encoding="utf-8")
    return p


def _reads_at(tok, segments, targets):
    """ReadRecords at (segment_idx, char-in-segment) targets."""
    return [ReadRecord(token_idx=_tok_at(tok, segments[s], c),
                       trigger="cadence", cue=None,
                       h=np.zeros(H, dtype=np.float16), segment_idx=s)
            for s, c in targets]


def test_crlf_run_matches_lf_twin(tmp_path, tok):
    """decisions/026 defect (b): a run whose segments contain \\r\\n must get
    the SAME labels as its LF twin. Pre-fix, newline translation shortened the
    text by one char per CRLF and every downstream offset drifted."""
    from wta.labeling import build_labels

    art_f = _art_file(tmp_path)
    builds = {}
    for name, nl in [("lf", "x\n"), ("crlf", "x\r\n")]:
        filler = nl * 800
        seg0 = filler + ANCHORS_TEXT * 6
        seg1 = ANCHORS_TEXT + COMMIT + ANCHORS_TEXT
        segs = [seg0, seg1]
        reads = _reads_at(tok, segs, [
            (0, len(filler) + 250),                    # inside seg0 anchors
            (1, len(ANCHORS_TEXT) + len(COMMIT) + 40),  # after the commitment
        ])
        a0 = tmp_path / name / "a0"
        _write_run(a0, segs, reads)
        builds[name] = build_labels(a0, art_f, window_chars=150)

    lf, crlf = builds["lf"], builds["crlf"]
    assert (crlf.decision == 0).all(), \
        "CRLF twin lost its decision labels (window displaced by CR drift)"
    assert np.array_equal(lf.decision, crlf.decision)
    assert np.array_equal(lf.cls, crlf.cls)
    assert np.array_equal(lf.phase, crlf.phase)
    assert list(lf.phase) == [0, 1]


def test_whitespace_collapse_window_alignment(tmp_path, tok):
    """decisions/026 defect (a), the core fix: a long whitespace run BEFORE
    the anchors must not displace the scoring window. Pre-fix, the window was
    sliced from the collapsed string at raw offsets and missed the anchors
    entirely (or ran past EOF)."""
    from wta.labeling import build_labels

    art_f = _art_file(tmp_path)
    prefix = "filler.\n" + " " * 900          # collapses to "filler. "
    seg0 = prefix + ANCHORS_TEXT * 6
    segs = [seg0]
    reads = _reads_at(tok, segs, [(0, len(prefix) + 60),
                                  (0, len(prefix) + 250)])
    a0 = _write_run(tmp_path / "a0", segs, reads)
    ds = build_labels(a0, art_f, window_chars=150)

    assert (ds.decision == 0).all(), \
        "reads inside the anchor region must be labeled despite the " \
        "whitespace prefix (pre-026: window displaced ~900 chars)"


def test_trace_commit_phase_units(tmp_path, tok):
    """decisions/026 defect (c): trace-source commit_char must be a RAW
    offset. Pre-fix it was a normalized-text offset — systematically early —
    so a read BEFORE the commitment could read as settled."""
    from wta.labeling import build_labels

    art_f = _art_file(tmp_path)
    # collapse savings (~120) exceed the read->commitment distance (~60),
    # which is exactly the geometry where the old mixed-units comparison
    # flips a pre-commitment read to settled.
    w = "note:\n" + " " * 120
    seg0 = w + ANCHORS_TEXT + COMMIT + ANCHORS_TEXT
    segs = [seg0]
    reads = _reads_at(tok, segs, [
        (0, len(w) + 44),                                   # before COMMIT
        (0, len(w) + len(ANCHORS_TEXT) + len(COMMIT) + 44),  # after COMMIT
    ])
    a0 = _write_run(tmp_path / "a0", segs, reads)
    dbg = tmp_path / "labels_debug.jsonl"
    ds = build_labels(a0, art_f, window_chars=150, debug_path=dbg)

    assert (ds.decision == 0).all()
    assert list(ds.phase) == [0, 1], \
        "read before the commitment sentence must stay should_ask"
    text = "\n\n".join(segs)
    commits = [json.loads(l) for l in dbg.read_text(encoding="utf-8").splitlines()
               if json.loads(l).get("kind") == "commitment"]
    (c,) = commits
    assert c["label_source"] == "trace"
    assert c["commit_char"] == text.find("retry_all()"), \
        "trace commit_char must be the raw offset of the signature"


def test_norm_map_roundtrip():
    """_norm_map: exact _norm output plus a norm->raw index that survives
    leading/trailing whitespace, CRLF, unicode spaces, and 1->2 lowercase
    expansions (the failure modes of the old judge _norm_with_map)."""
    from wta.labeling import _norm, _norm_map

    cases = ["", "   ", "  leading", "trailing  ", "a\r\nb", "a b",
             "a b", "İstanbul İİ", "ΣΟΦΟΣ ΟΔΥΣΣΕΥΣ",
             "x\r\n\t mixed   runs\r\n", "plain text no tricks"]
    for s in cases:
        norm, idx = _norm_map(s)
        assert norm == _norm(s), f"norm mismatch for {s!r}"
        assert len(idx) == len(norm), f"map length mismatch for {s!r}"
        assert all(b >= a for a, b in zip(idx, idx[1:])), \
            f"map not non-decreasing for {s!r}"

    # a term behind messy whitespace maps back to its exact raw offset
    raw = "start\r\n\r\n   \t needle here"
    norm, idx = _norm_map(raw)
    q = norm.find("needle")
    assert q >= 0 and idx[q] == raw.find("needle")

    # 1->2 lowercase expansion earlier in the string must not shift later
    # mapped offsets (old _norm_with_map desynced here)
    raw2 = "İ marker"
    norm2, idx2 = _norm_map(raw2)
    q2 = norm2.find("marker")
    assert q2 >= 0 and idx2[q2] == raw2.find("marker")

    # locate_evidence round-trips whitespace/case-mangled evidence to the
    # raw offset of the span
    from wta.judge_labels import locate_evidence
    trace = "prefix İ text\r\n  the CHOSEN   fix\r\nsuffix"
    assert locate_evidence(trace, "the chosen fix") == trace.find("the CHOSEN")


def test_windows_nonempty_monotone_in_bounds(tmp_path, tok):
    """spec labels.md observable 4, mechanized: every read char is in-bounds
    on the RAW text, every window is non-empty, chars are monotone per run.
    (Pre-026, 78,584 real reads indexed past the translated EOF.)"""
    from wta.labeling import build_labels

    art_f = _art_file(tmp_path)
    filler = "x\r\n" * 800
    seg0 = filler + ANCHORS_TEXT * 6
    seg1 = ANCHORS_TEXT + COMMIT + ANCHORS_TEXT
    segs = [seg0, seg1]
    reads = _reads_at(tok, segs, [(0, 40), (0, len(filler) + 250),
                                  (1, 10),
                                  (1, len(ANCHORS_TEXT) + len(COMMIT) + 40)])
    a0 = _write_run(tmp_path / "a0", segs, reads)
    dbg = tmp_path / "labels_debug.jsonl"
    build_labels(a0, art_f, window_chars=150, debug_path=dbg)

    text = "\n\n".join(segs)
    chars = []
    for line in dbg.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") != "read":
            continue
        c = row["char"]
        assert 0 <= c <= len(text), f"read char {c} out of raw bounds"
        assert text[max(0, c - 150):c + 150], "empty scoring window"
        chars.append(c)
    assert chars == sorted(chars), "read chars must be monotone in row order"


def test_debug_trail_reproducible(tmp_path, tok):
    """spec labels.md observable 5 (v3.1): snippets and anchor scores must be
    RECOMPUTABLE from the raw trace + the row's char alone. The pre-026 trail
    sliced snippets from a different string than the scorer used, which is
    how the displaced-window bug survived human audits."""
    from wta.labeling import _hits, _norm, build_labels

    art_f = _art_file(tmp_path)
    prefix = "filler.\n" + " " * 300
    seg0 = prefix + ANCHORS_TEXT * 6
    seg1 = ANCHORS_TEXT + COMMIT + ANCHORS_TEXT
    segs = [seg0, seg1]
    reads = _reads_at(tok, segs, [(0, len(prefix) + 60),
                                  (1, len(ANCHORS_TEXT) + len(COMMIT) + 40)])
    a0 = _write_run(tmp_path / "a0", segs, reads)
    dbg = tmp_path / "labels_debug.jsonl"
    build_labels(a0, art_f, window_chars=150, debug_path=dbg)

    text = "\n\n".join(segs)
    anchors = ART["taskX"]["blockerA"]["anchors"]
    n_reads = 0
    for line in dbg.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") == "read":
            n_reads += 1
            c = row["char"]
            assert row["window_snippet"] == text[max(0, c - 80):c + 80]
            want = _hits(_norm(text[max(0, c - 150):c + 150]), anchors)
            assert row["anchor_scores"].get("blockerA", 0) == want, \
                "stored anchor score must equal a recomputation from the " \
                "raw window (the scorer's actual input)"
        elif row.get("kind") == "commitment" and row.get("chosen"):
            cc = row["commit_char"]
            assert row["snippet"] == text[max(0, cc - 60):cc + 120]
    assert n_reads == 2
