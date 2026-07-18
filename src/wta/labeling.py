"""Offline label builder: A0 logs + registries -> training labels (spec labels.md).

OURS. Observables (trace text, registries, the frozen class artifact) are the
offline teacher; nothing here runs at trigger time. Unlabeled (-1) always
beats mislabeled; coverage is reported, never forced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wta.logging_schema import load_run_log


# ---------------------------------------------------------------------------
# artifact + registry loading
# ---------------------------------------------------------------------------


def load_class_artifact(path: str | Path) -> dict:
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    for task, blockers in art.items():
        if task.startswith("_"):
            continue
        for bid, spec in blockers.items():
            if not spec.get("anchors"):
                raise ValueError(f"{task}/{bid}: empty anchors")
            classes = spec.get("classes", [])
            if len(classes) < 2:
                raise ValueError(f"{task}/{bid}: need >= 2 interpretation classes")
            if not classes[0].get("canonical"):
                raise ValueError(f"{task}/{bid}: class 0 must be the canonical resolution")
            for c in classes:
                if not c.get("signatures"):
                    raise ValueError(f"{task}/{bid}/{c.get('name')}: empty signatures")
    return art


@dataclass
class Vocab:
    """Global ids: decisions are (task, blocker); classes are (decision, local)."""

    decisions: list = field(default_factory=list)        # [(task, blocker_id)]
    classes: list = field(default_factory=list)          # [(decision_id, local_idx, name)]
    class_of_decision: dict = field(default_factory=dict)  # decision_id -> [class ids]

    def add_decision(self, task: str, blocker: str, n_classes: int, names: list) -> int:
        did = len(self.decisions)
        self.decisions.append((task, blocker))
        ids = []
        for j in range(n_classes):
            cid = len(self.classes)
            self.classes.append((did, j, names[j]))
            ids.append(cid)
        self.class_of_decision[did] = ids
        return did


# ---------------------------------------------------------------------------
# scoring primitives (case-insensitive substring hits)
# ---------------------------------------------------------------------------


_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower())


def _hits(text_norm: str, terms: list[str]) -> int:
    return sum(text_norm.count(_norm(t)) for t in terms)


# spec labels.md "v2: action-based commitment": writing to files is the
# behavioural commitment; read-only exploration must not count.
_MUTATING_TOKENS = ("sed -i", ">", ">>", "tee ", "patch ", "git apply",
                    "perl -i")


def _is_mutating(cmd: str) -> bool:
    return any(t in cmd for t in _MUTATING_TOKENS)


def token_char_positions(text: str, tokenizer) -> list[int]:
    """Char start offset per token of the re-tokenized trace. Approximates
    generation-time positions (spec labels.md caveat 5)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return [a for a, _ in enc["offset_mapping"]]


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------


@dataclass
class LabeledDataset:
    h: np.ndarray            # (n, H) float32
    decision: np.ndarray     # (n,) global decision id, -1 background
    cls: np.ndarray          # (n,) global class id, -1 unlabeled
    phase: np.ndarray        # (n,) 0 = should_ask, 1 = settled, -1 = n/a
    task_idx: np.ndarray     # (n,)
    run_idx: np.ndarray      # (n,)
    read_token_idx: np.ndarray
    tasks: list
    runs: list               # [(task, run_id)]
    vocab: Vocab
    coverage: dict

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, h=self.h, decision=self.decision, cls=self.cls,
            phase=self.phase, task_idx=self.task_idx, run_idx=self.run_idx,
            read_token_idx=self.read_token_idx,
            meta=json.dumps({
                "tasks": self.tasks, "runs": self.runs,
                "decisions": self.vocab.decisions,
                "classes": self.vocab.classes,
                "class_of_decision": {str(k): v for k, v in
                                      self.vocab.class_of_decision.items()},
                "coverage": self.coverage,
            }))

    @classmethod
    def load(cls, path: str | Path) -> "LabeledDataset":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        vocab = Vocab(decisions=[tuple(d) for d in meta["decisions"]],
                      classes=[tuple(c) for c in meta["classes"]],
                      class_of_decision={int(k): v for k, v in
                                         meta["class_of_decision"].items()})
        return cls(h=z["h"], decision=z["decision"], cls=z["cls"], phase=z["phase"],
                   task_idx=z["task_idx"], run_idx=z["run_idx"],
                   read_token_idx=z["read_token_idx"], tasks=meta["tasks"],
                   runs=[tuple(r) for r in meta["runs"]], vocab=vocab,
                   coverage=meta["coverage"])


def build_labels(a0_dir: str | Path, classes_path: str | Path,
                 tokenizer_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                 window_chars: int = 400, min_anchor_hits: int = 1,
                 min_sig_hits: int = 1,
                 debug_path: str | Path | None = None, layer=None) -> LabeledDataset:
    """See spec labels.md. With debug_path, every labeling decision is written
    as JSONL (per read: window snippet + per-blocker anchor scores + outcome
    reason; per (run, decision): per-class signature scores + commit position)
    so 'a number looks wrong' is always traceable to the text that caused it."""
    from transformers import AutoTokenizer

    a0_dir = Path(a0_dir)
    art = load_class_artifact(classes_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dbg = open(debug_path, "w", encoding="utf-8") if debug_path else None

    def dwrite(**kw):
        if dbg:
            dbg.write(json.dumps(kw, ensure_ascii=False) + "\n")

    vocab = Vocab()
    task_specs: dict[str, list] = {}  # task -> [(decision_id, spec)]
    for task in sorted(k for k in art if not k.startswith("_")):
        entries = []
        for bid, spec in art[task].items():
            did = vocab.add_decision(task, bid, len(spec["classes"]),
                                     [c["name"] for c in spec["classes"]])
            entries.append((did, spec))
        task_specs[task] = entries

    rows_h, rows = [], {k: [] for k in
                        ("decision", "cls", "phase", "task_idx", "run_idx", "tok")}
    tasks, runs = [], []
    coverage: dict[str, dict] = {}

    for task_dir in sorted(p for p in a0_dir.iterdir() if p.is_dir()):
        task = task_dir.name
        if task not in task_specs:
            continue
        tasks.append(task)
        t_i = tasks.index(task)
        cov = coverage.setdefault(task, {"reads": 0, "decision_labeled": 0,
                                         "class_labeled": 0, "anchor_ties": 0,
                                         "committed_classes": {}})
        for jf in sorted(task_dir.glob("*.json")):
            run_id = jf.stem
            if run_id.endswith(".segments") or not (task_dir / f"{run_id}.npz").exists():
                continue  # sidecar/metadata json, not a run log
            log = load_run_log(task_dir, run_id, layer=layer)  # multi-layer: select-at-load
            text = (task_dir / f"{run_id}.txt").read_text(encoding="utf-8",
                                                          errors="replace")
            text_norm = _norm(text)
            # v2 multi-segment runs (decisions/017): token_idx restarts per
            # turn, so the token->char map is per segment; the run's .txt is
            # "\n\n".join(segments), so segment k's chars start at offs[k].
            seg_file = task_dir / f"{run_id}.segments.json"
            if seg_file.exists():
                segments = json.loads(seg_file.read_text(encoding="utf-8"))
                seg_starts, offs, pos = [], [], 0
                for s in segments:
                    seg_starts.append(token_char_positions(s, tokenizer))
                    offs.append(pos)
                    pos += len(s) + 2  # the join separator
            else:
                seg_starts, offs = [token_char_positions(text, tokenizer)], [0]
            runs.append((task, run_id))
            r_i = len(runs) - 1

            # per (run, decision): committed class + behavioural commitment char.
            # v2 (spec labels.md "v2: action-based commitment"): mutating
            # actions are scored FIRST — deliberation mentions must not commit;
            # trace scoring is the v1 fallback. label_source records which won.
            mut_actions = [a for a in log.actions if _is_mutating(a.action_text)]
            mut_norm = _norm("\n".join(a.action_text for a in mut_actions))
            committed: dict[int, tuple[int, int]] = {}  # did -> (global cls, commit_char)
            for did, spec in task_specs[task]:
                blocker = vocab.decisions[did][1]
                local, pos, source = -1, -1, None
                if mut_norm:
                    a_scores = [_hits(mut_norm, c["signatures"])
                                for c in spec["classes"]]
                    a_order = np.argsort(a_scores)[::-1]
                    if (a_scores[a_order[0]] >= min_sig_hits
                            and a_scores[a_order[0]] > a_scores[a_order[1]]):
                        cand = int(a_order[0])
                        sig_norms = [_norm(t) for t in
                                     spec["classes"][cand]["signatures"]]
                        for a in mut_actions:
                            if any(s in _norm(a.action_text) for s in sig_norms):
                                seg = min(a.segment_idx, len(seg_starts) - 1)
                                s_st = seg_starts[seg]
                                pos = offs[seg] + (s_st[min(a.token_idx,
                                                            len(s_st) - 1)]
                                                   if s_st else 0)
                                local, source, scores = cand, "actions", a_scores
                                break
                if source is None:
                    scores = [_hits(text_norm, c["signatures"])
                              for c in spec["classes"]]
                    order = np.argsort(scores)[::-1]
                    if (scores[order[0]] >= min_sig_hits
                            and scores[order[0]] > scores[order[1]]):
                        local = int(order[0])
                        sig_terms = spec["classes"][local]["signatures"]
                        pos = min((p for t in sig_terms
                                   if (p := text_norm.find(_norm(t))) >= 0),
                                  default=-1)
                        source = "trace"
                if source is not None:
                    gcls = vocab.class_of_decision[did][local]
                    committed[did] = (gcls, pos)
                    name = spec["classes"][local]["name"]
                    cov["committed_classes"].setdefault(blocker, set()).add(name)
                    dwrite(kind="commitment", run=run_id, blocker=blocker,
                           chosen=name, commit_char=pos, label_source=source,
                           scores={c["name"]: s for c, s in
                                   zip(spec["classes"], scores)},
                           snippet=text[max(0, pos - 60):pos + 120] if pos >= 0 else "")
                else:
                    reason = ("no signature hits" if scores[order[0]] < min_sig_hits
                              else "tie between top classes")
                    dwrite(kind="commitment", run=run_id, blocker=blocker,
                           chosen=None, reason=reason,
                           scores={c["name"]: s for c, s in
                                   zip(spec["classes"], scores)})

            h = log.read_matrix().astype(np.float32)
            for k, read in enumerate(log.reads):
                tok = read.token_idx
                seg = min(read.segment_idx, len(seg_starts) - 1)
                s_starts = seg_starts[seg]
                local = s_starts[min(tok, len(s_starts) - 1)] if s_starts else 0
                char = offs[seg] + local
                lo, hi = max(0, char - window_chars), char + window_chars
                win = text_norm[lo:hi]

                d_scores = [(did, _hits(win, spec["anchors"]))
                            for did, spec in task_specs[task]]
                d_scores.sort(key=lambda x: -x[1])
                did, why = -1, "labeled"
                if d_scores[0][1] < min_anchor_hits:
                    why = "no anchor hits in window"
                elif len(d_scores) > 1 and d_scores[0][1] == d_scores[1][1]:
                    why = "anchor tie between blockers"
                    cov["anchor_ties"] += 1
                else:
                    did = d_scores[0][0]

                gcls, phase = -1, -1
                if did >= 0 and did in committed:
                    c, commit_char = committed[did]
                    if commit_char >= 0:
                        phase = 1 if char >= commit_char else 0
                        if phase == 1:
                            gcls = c

                dwrite(kind="read", run=run_id, read_idx=k, token_idx=tok,
                       char=char, decision=(vocab.decisions[did][1] if did >= 0 else None),
                       outcome=why, phase=phase,
                       anchor_scores={vocab.decisions[d][1]: s
                                      for d, s in d_scores if s > 0},
                       window_snippet=text[max(0, char - 80):char + 80])

                rows_h.append(h[k])
                rows["decision"].append(did)
                rows["cls"].append(gcls)
                rows["phase"].append(phase)
                rows["task_idx"].append(t_i)
                rows["run_idx"].append(r_i)
                rows["tok"].append(tok)
                cov["reads"] += 1
                cov["decision_labeled"] += int(did >= 0)
                cov["class_labeled"] += int(gcls >= 0)

        cov["committed_classes"] = {k: sorted(v) for k, v in
                                    cov["committed_classes"].items()}

    if dbg:
        dbg.close()
    return LabeledDataset(
        h=np.stack(rows_h).astype(np.float32),
        decision=np.array(rows["decision"], dtype=np.int64),
        cls=np.array(rows["cls"], dtype=np.int64),
        phase=np.array(rows["phase"], dtype=np.int64),
        task_idx=np.array(rows["task_idx"], dtype=np.int64),
        run_idx=np.array(rows["run_idx"], dtype=np.int64),
        read_token_idx=np.array(rows["tok"], dtype=np.int64),
        tasks=tasks, runs=runs, vocab=vocab, coverage=coverage,
    )


def coverage_table(ds: LabeledDataset) -> str:
    lines = ["task        reads  dec%   cls%  ties   forked-blockers"]
    for task in ds.tasks:
        c = ds.coverage[task]
        forked = sum(1 for v in c["committed_classes"].values() if len(v) >= 2)
        lines.append(f"{task:<11} {c['reads']:>5}  {c['decision_labeled']/max(c['reads'],1):>5.0%}"
                     f"  {c['class_labeled']/max(c['reads'],1):>5.0%}"
                     f"  {c.get('anchor_ties', 0):>4}   "
                     f"{forked}/{len(c['committed_classes'])} with >=2 classes")
    return "\n".join(lines)
