"""Contract: composite decision-identity observables (spec A0, decisions/021 §4).

The method doc: "Do not label a decision by the file alone -- that conflates
two decisions that touch the same file. Use a composite label: file + code
region/span + stated sub-goal + error signature." The v2 loop recorded only
{files, step}, which is the conflation gate 4 regressed on at 14B (0.257 ->
0.785). None of these fields can be reconstructed after collection, so they are
recorded at collection time and pinned here."""

from wta.agent_loop import (
    error_signature, extract_region, extract_subgoal,
)


class _Env:
    """Executor stub: returns a scripted (exit_code, output) per call."""

    def __init__(self, results):
        self.results = list(results)
        self.seen = []

    def execute(self, cmd):
        self.seen.append(cmd)
        return self.results.pop(0) if self.results else (0, "")


def test_region_from_sed_ranges():
    assert extract_region("sed -i '12s/a/b/' f.py") == ["12-12"]
    assert extract_region("sed -i '10,20d' f.py") == ["10-20"]


def test_region_from_patch_hunk():
    assert extract_region("patch <<'EOF'\n@@ -12,7 +12,9 @@\nEOF") == ["12-18"]


def test_region_empty_without_line_info():
    assert extract_region("cat > f.py <<'EOF'\nx = 1\nEOF") == []


def test_two_edits_same_file_get_different_regions():
    """The conflation case the composite label exists to separate."""
    a = extract_region("sed -i '20,25s/x/y/' app.py")
    b = extract_region("sed -i '300,310s/p/q/' app.py")
    assert a and b and a != b


def test_subgoal_is_the_thought_before_the_block():
    text = "THOUGHT: I will widen the timeout.\n\n```bash\nls\n```"
    sub = extract_subgoal(text)
    assert "widen the timeout" in sub
    assert "```" not in sub and "ls" not in sub


def test_subgoal_truncates_and_collapses_whitespace():
    assert extract_subgoal("a\n\n   b\n```bash\nx\n```") == "a b"
    assert len(extract_subgoal("y " * 500)) <= 400


def test_error_signature_captures_code_and_first_error():
    sig = error_signature(1, "loading\nModuleNotFoundError: No module named 'x'\n")
    assert sig.startswith("exit 1")
    assert "ModuleNotFoundError" in sig


def test_error_signature_on_success_is_just_the_code():
    assert error_signature(0, "all good\n") == "exit 0"


def test_loop_records_all_composite_fields(monkeypatch):
    """End-to-end through run_agent: every field lands on the ActionEvent, and
    error_signature is filled in AFTER execution."""
    from wta.agent_loop import AgentLoopConfig, run_agent

    class Session:
        def generate_segment(self, messages, *, seed, temperature,
                             max_new_tokens, segment_idx):
            if segment_idx == 0:
                return [], "THOUGHT: fix the guard.\n```bash\nsed -i '5,9d' a.py\n```"
            return [], "THOUGHT: done.\n```bash\necho TASK_DONE\n```"

    env = _Env([(1, "Traceback\nValueError: bad guard\n")])
    res = run_agent(Session(), env, "task", run_id="r0", task_id="t0", seed=0,
                    cfg=AgentLoopConfig(max_steps=3))

    first = res.log.actions[0].observables
    assert first["files"] == ["a.py"]
    assert first["region"] == ["5-9"]
    assert "fix the guard" in first["subgoal"]
    assert "ValueError" in first["error_signature"]
    # the submit action never executes, so it carries no signature
    assert "error_signature" not in res.log.actions[-1].observables
    assert res.finished
