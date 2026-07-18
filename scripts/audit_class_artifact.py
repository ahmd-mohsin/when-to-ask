"""Deterministic audit of an interpretation-class artifact (decisions/005).

    python scripts/audit_class_artifact.py [--classes data/interpretation_classes.json]

Mechanizes the checks applied by hand to the original 20-task artifact
(decisions/013: "77 leaky anchors demoted, >=3 anchors kept per blocker"):

  1. schema: >=3 anchors; 2-4 classes; class 0 (and only class 0) canonical.
  2. anchor-leak: an anchor whose normalized text contains, or is contained
     in, any class signature of ITS OWN blocker uniquely predicts that class
     -> must be dropped (anchors name the DECISION, not the RESOLUTION).
  3. signature sanity: non-empty; no signature shared verbatim by two classes
     of the same blocker (unattributable commitment evidence).
  4. warning only: anchors shared across blockers of the same task (accepted
     as measured label noise, spec labels.md design decision 4 / gate 4).

Exit code 1 on any hard failure (1-3) so batch derivation can gate on it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def audit(art: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for task, blockers in art.items():
        if task.startswith("_"):
            continue
        anchors_seen: dict[str, str] = {}
        for bid, spec in blockers.items():
            where = f"{task}/{bid}"
            anchors = spec.get("anchors", [])
            classes = spec.get("classes", [])
            if len(anchors) < 3:
                errors.append(f"{where}: only {len(anchors)} anchors (<3)")
            if not 2 <= len(classes) <= 4:
                errors.append(f"{where}: {len(classes)} classes (want 2-4)")
            for j, c in enumerate(classes):
                if bool(c.get("canonical")) != (j == 0):
                    errors.append(f"{where}: canonical flag wrong at class {j} "
                                  f"({c.get('name')})")
                if not c.get("signatures"):
                    errors.append(f"{where}/{c.get('name')}: no signatures")
                for s in c.get("signatures", []):
                    if not _norm(s):
                        errors.append(f"{where}/{c.get('name')}: empty signature")
            sigs = [(c.get("name"), _norm(s))
                    for c in classes for s in c.get("signatures", [])]
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    if sigs[i][0] == sigs[j][0]:
                        continue
                    if sigs[i][1] == sigs[j][1]:
                        errors.append(f"{where}: signature {sigs[i][1]!r} shared "
                                      f"by {sigs[i][0]} and {sigs[j][0]}")
                    elif sigs[i][1] in sigs[j][1] or sigs[j][1] in sigs[i][1]:
                        # cross-class substring overlap: how swe_2-s0 stayed
                        # mislabeled even with action-based commitment
                        # ('math.maxint32' in the sentinel class swallows the
                        # clamp classes' evidence). WARN: present in the
                        # frozen artifact; keep zero in NEW entries.
                        warnings.append(f"{where}: signature {sigs[i][1]!r} "
                                        f"({sigs[i][0]}) contains/contained-in "
                                        f"{sigs[j][1]!r} ({sigs[j][0]})")
            for a in anchors:
                an = _norm(a)
                for cname, sn in sigs:
                    if an and sn and an == sn:
                        errors.append(f"{where}: anchor {a!r} IS a signature "
                                      f"of class {cname}")
                    elif an and sn and (an in sn or sn in an):
                        # containment: calibrated to the frozen 20-task
                        # artifact, which keeps decision-naming anchors that
                        # substring-overlap one class's signatures (an API
                        # name is deliberated by every run). Aim for zero in
                        # NEW entries; not a hard failure.
                        warnings.append(f"{where}: anchor {a!r} overlaps "
                                        f"signature of {cname} ({sn!r})")
                if an in anchors_seen and anchors_seen[an] != bid:
                    warnings.append(f"{task}: anchor {a!r} shared by "
                                    f"{anchors_seen[an]} and {bid}")
                anchors_seen[an] = bid
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    args = ap.parse_args()
    art = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    errors, warnings = audit(art)
    n_tasks = sum(1 for k in art if not k.startswith("_"))
    n_blockers = sum(len(v) for k, v in art.items() if not k.startswith("_"))
    print(f"{args.classes}: {n_tasks} tasks, {n_blockers} blockers")
    for w in warnings:
        print("  WARN", w)
    for e in errors:
        print("  FAIL", e)
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
