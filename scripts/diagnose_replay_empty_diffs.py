"""DIAGNOSTIC on the replay-and-diff pilot: the no-op runs run the arm-A number.

Not a pre-registered cell, and not a rerun of the pilot -- it re-scores the
diffs the pilot already cached in --work-dir. `scripts/replay_diff_pilot.py`
stands as-run; this asks WHY its positive control landed where it did.

The pilot's arm A (`A_cross_task`) is the sanity check the whole approach has
to pass: two runs on DIFFERENT REPOSITORIES must be separable. It scored
AUROC 0.581, barely above chance, which reads as "replay-and-diff is not an
instrument either". Two numbers in the same artifact say that reading is
wrong:

    diff_sizes.n_empty_diffs        26 of 115
    A_cross_task.exact_match_rate_diff  0.0437

4.4% of CROSS-TASK pairs scored an exact match. Two runs on different
repositories cannot produce the same changed-line set unless both sets are
EMPTY -- and `jaccard_distance` returns 0.0 (the "identical" end of the
scale) for two empty sets. So every pair of no-op runs is scored as a perfect
match, including the ~231 cross-task ones, and they sit exactly on top of the
"same" distribution the AUROC is trying to separate.

The 26 empty diffs are REAL, not a replay defect. Verified two ways:

  1. 12 of 26 ran no write-like command at all.
  2. The other 14 wrote, but wrote nothing with content -- the dominant shape
     is `mkdir -p <dir> && touch <file>`, which creates EMPTY files. An empty
     new file is a `new file mode` header and zero +/- content lines, which
     `normalize_diff` drops by design. swe_12-s2 is the clean example: 15
     write commands, all `touch`, final diff empty.

Also verified, and NOT a replay defect: `cd` does not persist across actions,
so `swe_12-s2`'s opening `cd applications/drive` does not scope the paths that
follow it. That is not something replay introduced -- `wta.agent_env.execute`
prefixes every command with `cd /app`, and `scripts/collect_v2.py` drove the
ORIGINAL collection through that same `DockerTaskEnv`. The recorded run had no
`cd` persistence either, so replay reproduces it faithfully. It does mean
exit-code fidelity is blind here: `mkdir -p`/`touch` exit 0 wherever they land,
which is why these runs sit at fidelity 1.00 with an empty diff.

Read-out: arm A recomputed over runs with a non-empty diff, alongside the
as-run number. The exclusion is stated, not silent -- an empty diff is a run
that produced no measurable artifact, so it carries no evidence about whether
two runs decided the same thing, in either direction.

    python scripts/diagnose_replay_empty_diffs.py --work-dir /ssd3/wta-replay
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from feature_signal_gate import auroc  # noqa: E402
from offline_ask_headtohead import load_task_actions  # noqa: E402
from replay_diff_pilot import jaccard_distance, normalize_diff  # noqa: E402

# write-like shapes, used only to split "ran nothing" from "wrote nothing"
_WRITE = re.compile(r"(>>?\s*\S|sed -i|tee\b|patch\b|cp \b|mv \b|touch\b|"
                    r"cat\s*<<|apply_patch|git apply|rm \b)")


def cross_task_arm(keys, norm, task_of):
    """Arm A verbatim: same-task pairs vs cross-task pairs, Jaccard distance."""
    same, diff = [], []
    ks = sorted(keys)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            d = jaccard_distance(norm[a], norm[b])
            (same if task_of[a] == task_of[b] else diff).append(d)
    s, d = np.array(same), np.array(diff)
    if not len(s) or not len(d):
        return None
    return {"auroc_jaccard": auroc(s, d), "n_same": len(s), "n_diff": len(d),
            "same_mean": float(s.mean()), "diff_mean": float(d.mean()),
            "exact_match_rate_same": float((s == 0).mean()),
            "exact_match_rate_diff": float((d == 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="/ssd3/wta-replay")
    ap.add_argument("--a0", default="/ssd/wta_data/a0_v3_32b")
    ap.add_argument("--pilot", default="results/replay_diff_pilot.json")
    ap.add_argument("--out", default="results/diag_replay_empty_diffs.json")
    args = ap.parse_args()

    pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
    tasks = pilot["pilot_tasks"]

    norm, task_of, fid = {}, {}, {}
    for f in sorted(Path(args.work_dir).glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "diff" not in rec:
            continue
        rid = f.stem
        norm[rid] = normalize_diff(rec.get("diff") or "")
        task_of[rid] = rid.rsplit("-s", 1)[0]
        fid[rid] = rec.get("fidelity")

    actions = load_task_actions(Path(args.a0), None)
    n_write, n_exec = {}, {}
    for t in tasks:
        for rid, acts in actions.get(t, {}).items():
            n_exec[rid] = len(acts)
            n_write[rid] = sum(1 for a in acts
                               if _WRITE.search(a.action_text or ""))

    empty = sorted(r for r, v in norm.items() if not v)
    nonempty = sorted(r for r, v in norm.items() if v)

    as_run = cross_task_arm(list(norm), norm, task_of)
    cleaned = cross_task_arm(nonempty, norm, task_of)

    by_task = defaultdict(int)
    for r in empty:
        by_task[task_of[r]] += 1

    res = {
        "note": "DIAGNOSTIC on the replay-and-diff PILOT, not a pre-registered "
                "cell and not a rerun. Re-scores the pilot's own cached diffs; "
                "results/replay_diff_pilot.json stands as-run.",
        "question": "Arm A (cross-task) must be ~1.0 for the approach to be an "
                    "instrument at all. It landed at 0.581. Is that the "
                    "representation failing, or the metric?",
        "finding": "The metric. jaccard_distance(empty, empty) = 0.0 scores two "
                   "runs that changed nothing as an EXACT match, so every pair "
                   "of no-op runs lands on the 'same' end of the scale -- "
                   "including the cross-task ones, which is why "
                   "exact_match_rate_diff is non-zero at all.",
        "n_runs": len(norm),
        "n_empty_diffs": len(empty),
        "empty_by_task": dict(sorted(by_task.items())),
        "empty_diffs_are_real": {
            "ran_no_write_command": sum(1 for r in empty if n_write.get(r, 0) == 0),
            "wrote_but_no_content": sum(1 for r in empty if n_write.get(r, 0) > 0),
            "median_fidelity_of_empty_runs": float(np.median(
                [fid[r] for r in empty if fid.get(r) is not None])),
            "mechanism": "dominant shape is `mkdir -p <dir> && touch <file>`; "
                         "an empty new file contributes a `new file mode` "
                         "header and zero +/- content lines, which "
                         "normalize_diff drops by design",
            "worked_example": "swe_12-s2: 15 write commands, all touch, "
                              "fidelity 1.00, diff empty",
            "cd_persistence": "`cd` does not carry between actions "
                              "(wta.agent_env.execute prefixes every command "
                              "with `cd /app`). NOT a replay artifact: "
                              "collect_v2.py drove the original collection "
                              "through the same DockerTaskEnv, so the recorded "
                              "run had no cd persistence either. It does make "
                              "exit-code fidelity blind here -- mkdir -p/touch "
                              "exit 0 wherever they land.",
        },
        "arm_A_cross_task": {
            "as_run_all_runs": as_run,
            "excluding_empty_diffs": cleaned,
            "n_runs_scored_cleaned": len(nonempty),
        },
        "reading": (
            "Arm A goes 0.581 -> {:.3f} once no-op runs are excluded, and "
            "cross-task exact matches go {:.4f} -> {:.4f}, i.e. to zero: with "
            "the degenerate pairs gone, runs on different repositories are "
            "essentially disjoint (mean distance {:.3f}), which is what the "
            "positive control was asserting. The cleaned number is still below "
            "the pilot's 0.95 ceiling bar, so this does NOT convert the pilot "
            "into a GO -- it relocates the failure. What limits arm A is that "
            "same-task pairs are themselves far apart (mean {:.3f}): two runs "
            "on the SAME repo routinely touch disjoint line sets."
        ).format(cleaned["auroc_jaccard"], as_run["exact_match_rate_diff"],
                 cleaned["exact_match_rate_diff"], cleaned["diff_mean"],
                 cleaned["same_mean"]),
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"empty diffs: {len(empty)}/{len(norm)}  "
          f"({res['empty_diffs_are_real']['ran_no_write_command']} ran no write "
          f"command, {res['empty_diffs_are_real']['wrote_but_no_content']} wrote "
          f"nothing with content)")
    print(f"arm A as-run   : {as_run['auroc_jaccard']:.4f} "
          f"(cross-task exact matches {as_run['exact_match_rate_diff']:.4f})")
    print(f"arm A cleaned  : {cleaned['auroc_jaccard']:.4f} "
          f"(cross-task exact matches {cleaned['exact_match_rate_diff']:.4f})")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
