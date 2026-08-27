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

# collect_v2 writes <run>.txt as _SEG_SEP.join(segments); this join IS the
# canonical coordinate system for every char offset (spec labels.md v3.1,
# decisions/026).
_SEG_SEP = "\n\n"


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower())


def _norm_map(text: str) -> tuple[str, list[int]]:
    """_norm(text) plus, per normalized char, the raw index that produced it
    (spec labels.md v3.1) -- so a position found in normalized text can be
    mapped back to the raw trace instead of being compared across coordinate
    systems (the decisions/026 defect).

    Whitespace runs collapse to one " " mapped to the run's FIRST raw char --
    leading/trailing runs included. Non-whitespace chars contribute one entry
    per char of ch.lower(): some Unicode lowercases 1->2 chars ('İ'), and the
    map must stay in lockstep with _norm's whole-string lower(). lower()'s
    only context-dependent rule (final sigma) is 1->1, so the per-char walk
    stays length-synchronized; the assert turns any exception into a loud
    failure instead of a silent mislabel. idx is non-decreasing, so argmin
    over normalized positions maps to argmin over raw positions."""
    norm = _norm(text)
    idx: list[int] = []
    pos = 0
    for m in _WS.finditer(text):
        s, e = m.span()
        for i in range(pos, s):
            idx.extend([i] * len(text[i].lower()))
        idx.append(s)  # the whole whitespace run -> one " "
        pos = e
    for i in range(pos, len(text)):
        idx.extend([i] * len(text[i].lower()))
    assert len(idx) == len(norm), \
        f"_norm_map desync: {len(idx)} map entries vs {len(norm)} norm chars"
    return norm, idx


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


def resolve_tokenizer(a0_dir: str | Path, name: str = "auto") -> str:
    """Token->char maps must be built with the COLLECTION model's tokenizer
    (token_idx in the logs is in its units). 'auto' reads model_id from the
    collection manifest; an explicit name is passed through unchanged.

    Data-parallel collections (collect_v2 --num-shards N) suffix the manifest
    PER SHARD -- `collection_manifest.s0.json` -- so the unsuffixed name only
    exists for single-shard runs. Without the glob, 'auto' found nothing on
    every sharded collection and silently fell back to the 7B default, which
    is precisely the Qwen3-labelled-with-Qwen2.5 drift this function exists to
    prevent (observed on the R1 pilot, 2026-08-08)."""
    if name != "auto":
        return name
    d = Path(a0_dir)
    found: dict[str, str] = {}
    for manifest in [d / "collection_manifest.json",
                     *sorted(d.glob("collection_manifest.*.json"))]:
        if not manifest.exists():
            continue
        model_id = (json.loads(manifest.read_text(encoding="utf-8"))
                    .get("args", {}).get("model_id"))
        if model_id:
            found.setdefault(model_id, manifest.name)
    if len(found) > 1:
        raise ValueError(
            "collection manifests disagree on model_id -- one tokenizer "
            f"cannot label a mixed collection: {found}")
    if found:
        return next(iter(found))
    return "Qwen/Qwen2.5-Coder-7B-Instruct"


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


def _run_files(task_dir: Path):
    """Run-log selection predicate for one task dir. Shared by build_labels'
    counting pass and labeling pass so the two cannot diverge
    (decisions/026); the final `assert ptr == n_rows` backstops it."""
    for jf in sorted(task_dir.glob("*.json")):
        run_id = jf.stem
        if run_id.endswith(".segments") or not (task_dir / f"{run_id}.npz").exists():
            continue  # sidecar/metadata json, not a run log
        yield jf, run_id


def _iter_run_files(a0_dir: Path, task_specs: dict):
    for task_dir in sorted(p for p in a0_dir.iterdir() if p.is_dir()):
        if task_dir.name not in task_specs:
            continue
        for jf, run_id in _run_files(task_dir):
            yield task_dir, task_dir.name, jf, run_id


def _npz_h_dim(npz_path: Path) -> int:
    """Last-axis size of the stored h from the npy header alone (no data
    read) -- same technique as gate5_permutation_test.stream_project."""
    import zipfile
    with zipfile.ZipFile(npz_path) as zf:
        name = "h.npy" if "h.npy" in zf.namelist() else "h"
        with zf.open(name) as raw:
            major, _minor = np.lib.format.read_magic(raw)
            reader = (np.lib.format.read_array_header_1_0 if major == 1
                      else np.lib.format.read_array_header_2_0)
            shape, _fortran, _dtype = reader(raw)
    return int(shape[-1]) if shape else 0


def build_labels(a0_dir: str | Path, classes_path: str | Path,
                 tokenizer_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                 window_chars: int = 400, min_anchor_hits: int = 1,
                 min_sig_hits: int = 1,
                 debug_path: str | Path | None = None, layer=None,
                 judge_labels: dict[tuple[str, str],
                                    tuple[str, int, float]] | None = None,
                 ) -> LabeledDataset:
    """See spec labels.md. With debug_path, every labeling decision is written
    as JSONL (per read: window snippet + per-blocker anchor scores + outcome
    reason; per (run, decision): per-class signature scores + commit position)
    so 'a number looks wrong' is always traceable to the text that caused it.

    judge_labels (spec labels.md v3 / judge_labels.md): optional FROZEN
    (run_id, blocker) -> (class_name, commit_char, confidence) map from
    wta.judge_labels.load_judge_labels. Consulted only where both the actions
    and trace stages abstained; never overrides a lexicon label. The builder
    stays deterministic given its inputs -- no model call happens here.
    commit_char in the artifact is a RAW-join offset (spec judge_labels.md §4,
    labels.md v3.1) -- the same coordinate system as every other offset here.

    Coordinates (v3.1, decisions/026): text IS the raw segment join; all char
    offsets are raw; windows are sliced raw and their CONTENT normalized."""
    # decisions/028 Amendment F: load through the SAME path the collector
    # used. token_idx in the logs is in the collector's tokenizer units
    # (see the invariant in resolve_tokenizer above), so a bare
    # AutoTokenizer.from_pretrained here silently drifts the token->char map
    # whenever the collector passed a flag that changes tokenization --
    # measured at 23/40 real Mistral transcripts, 0/40 real Qwen ones.
    from wta.hf_reader import _load_tokenizer

    a0_dir = Path(a0_dir)
    art = load_class_artifact(classes_path)
    tokenizer = _load_tokenizer(tokenizer_name)
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

    # pass 1 (decisions/026): count rows + find H so h can be PREALLOCATED.
    # The old list-of-views + np.stack path kept every per-run matrix alive
    # and peaked at ~2x the final array (~32GB on the 1415-run collection).
    n_rows, H = 0, 0
    for task_dir, _task, jf, run_id in _iter_run_files(a0_dir, task_specs):
        n_reads = len(json.loads(jf.read_text(encoding="utf-8"))["reads"])
        n_rows += n_reads
        if H == 0 and n_reads:
            H = _npz_h_dim(task_dir / f"{run_id}.npz")
    if n_rows == 0:
        raise ValueError(f"no labelable runs with reads under {a0_dir}")
    h_all = np.empty((n_rows, H), dtype=np.float32)
    ptr = 0

    rows = {k: [] for k in
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
                                         "txt_join_mismatch": 0,
                                         "segment_clamped": 0,
                                         "token_clamped": 0,
                                         "committed_classes": {}})
        for jf, run_id in _run_files(task_dir):
            log = load_run_log(task_dir, run_id, layer=layer)  # multi-layer: select-at-load
            # v3.1 (decisions/026): ONE coordinate system. text IS the raw
            # segment join (CR preserved) -- the string every offset is born
            # in. The old universal-newline .read_text() shortened it on
            # CRLF runs and desynced every offset downstream.
            txt_file = task_dir / f"{run_id}.txt"
            seg_file = task_dir / f"{run_id}.segments.json"
            if seg_file.exists():
                segments = json.loads(seg_file.read_text(encoding="utf-8"))
                text = _SEG_SEP.join(segments)
                # v2 multi-segment runs (decisions/017): token_idx restarts
                # per turn, so the token->char map is per segment; segment
                # k's chars start at offs[k].
                seg_starts, offs, pos = [], [], 0
                for s in segments:
                    seg_starts.append(token_char_positions(s, tokenizer))
                    offs.append(pos)
                    pos += len(s) + len(_SEG_SEP)
                # diagnostics only: the join must equal the on-disk .txt
                # (raw, or newline-translated when a Windows writer CRLF'd
                # the whole file). A mismatch means the sidecar and txt
                # disagree about the trace -- count it, never guess.
                with open(txt_file, encoding="utf-8", errors="replace",
                          newline="") as f:
                    disk_txt = f.read()
                if text not in (disk_txt, disk_txt.replace("\r\n", "\n")):
                    cov["txt_join_mismatch"] += 1
            else:
                with open(txt_file, encoding="utf-8", errors="replace",
                          newline="") as f:
                    text = f.read()
                seg_starts, offs = [token_char_positions(text, tokenizer)], [0]
            text_norm, norm_raw = _norm_map(text)
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
                                cov["segment_clamped"] += int(
                                    a.segment_idx > len(seg_starts) - 1)
                                seg = min(a.segment_idx, len(seg_starts) - 1)
                                s_st = seg_starts[seg]
                                cov["token_clamped"] += int(
                                    bool(s_st) and a.token_idx > len(s_st) - 1)
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
                        # earliest signature mention, found in normalized
                        # text, mapped BACK to raw coords (v3.1): the map is
                        # non-decreasing, so the norm-side min is the raw-side
                        # min and the earliest-occurrence semantics are exact.
                        pos_norm = min((p for t in sig_terms
                                        if (p := text_norm.find(_norm(t))) >= 0),
                                       default=-1)
                        pos = norm_raw[pos_norm] if pos_norm >= 0 else -1
                        source = "trace"
                # v3 (spec labels.md): frozen judge labels fill in ONLY where
                # both lexicon stages abstained -- never override.
                judge_conf = None
                if source is None and judge_labels:
                    jl = judge_labels.get((run_id, blocker))
                    if jl is not None:
                        jname, jpos, judge_conf = jl
                        names = [c["name"] for c in spec["classes"]]
                        if jname in names:
                            local, pos = names.index(jname), int(jpos)
                            source = "judge"
                if source is not None:
                    gcls = vocab.class_of_decision[did][local]
                    committed[did] = (gcls, pos)
                    name = spec["classes"][local]["name"]
                    cov["committed_classes"].setdefault(blocker, set()).add(name)
                    extra = ({"judge_conf": judge_conf}
                             if source == "judge" else {})
                    dwrite(kind="commitment", run=run_id, blocker=blocker,
                           chosen=name, commit_char=pos, label_source=source,
                           scores={c["name"]: s for c, s in
                                   zip(spec["classes"], scores)},
                           snippet=text[max(0, pos - 60):pos + 120] if pos >= 0 else "",
                           **extra)
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
                cov["segment_clamped"] += int(
                    read.segment_idx > len(seg_starts) - 1)
                seg = min(read.segment_idx, len(seg_starts) - 1)
                s_starts = seg_starts[seg]
                cov["token_clamped"] += int(
                    bool(s_starts) and tok > len(s_starts) - 1)
                local = s_starts[min(tok, len(s_starts) - 1)] if s_starts else 0
                char = offs[seg] + local
                # v3.1 core fix: slice the RAW text at the raw char, then
                # normalize the window CONTENT. Slicing text_norm at raw
                # offsets displaced the window (decisions/026 defect (a)).
                lo, hi = max(0, char - window_chars), char + window_chars
                win = _norm(text[lo:hi])

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

                h_all[ptr] = h[k]
                ptr += 1
                rows["decision"].append(did)
                rows["cls"].append(gcls)
                rows["phase"].append(phase)
                rows["task_idx"].append(t_i)
                rows["run_idx"].append(r_i)
                rows["tok"].append(tok)
                cov["reads"] += 1
                cov["decision_labeled"] += int(did >= 0)
                cov["class_labeled"] += int(gcls >= 0)
            del h, log  # free this run before loading the next (026 memory)

        cov["committed_classes"] = {k: sorted(v) for k, v in
                                    cov["committed_classes"].items()}

    if dbg:
        dbg.close()
    assert ptr == n_rows, \
        f"pass-1/pass-2 divergence: counted {n_rows} reads, filled {ptr}"
    return LabeledDataset(
        h=h_all,
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
