"""Patch extraction after an our-loop run (spec eval, decisions/022).

The evaluator (`calculate_pass_at_1` -> custom_eval) consumes SWE-Agent-shaped
predictions: ``{instance_id: {model_name_or_path, instance_id, model_patch}}``.
Our loop mutates the repo inside the task container; the patch is the staged
diff against HEAD. Prototype-grade: an empty patch is a recorded outcome
(model changed nothing), not an error; a git failure returns "" with the
stderr recorded by the caller's manifest.
"""

from __future__ import annotations


def extract_patch(env, workdir: str = "/app") -> str:
    """`env` is duck-typed on execute(cmd) -> (exit_code, output) — the
    DockerTaskEnv contract (its execute already cd's to the workdir)."""
    env.execute("git config core.fileMode false")
    code, _ = env.execute("git add -A")
    if code != 0:
        return ""
    code, out = env.execute("git diff --cached HEAD")
    if code != 0:
        return ""
    return out


def prediction_row(instance_id: str, model_patch: str,
                   model_name: str) -> dict:
    """One preds.json row in the exact shape custom_eval expects."""
    return {"instance_id": instance_id, "model_name_or_path": model_name,
            "model_patch": model_patch}
