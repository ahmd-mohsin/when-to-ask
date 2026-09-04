"""CIS scoring driver (decisions/029) -- the box side: docker + GPU.

For every run: rebuild the contexts by replaying the recorded commands
(wta.cis_context, driving run_agent itself), then teacher-force each recorded
segment under its baseline context and under one injected variant per
resolution (own blockers, 2 matched foreign controls, and on the pilot the
owner-approved rival texts), on a branched KV cache (wta.cis_scorer).
Pre-action features are captured on the BASELINE branch only.

Per-run outputs (resumable by <run>.json existing):
  <work>/<run>.json  -- per-unit table: S_k, per-variant block/segment sums and
                        per-token minimum shift, block span, prompt length,
                        fidelity, flags; plus run-level fidelity
  <work>/<run>.npz   -- feats (n_units, 2 positions, L layers, H) float16 at
                        P_k-1 and P_bash-1; feat_tok7 (n_units, H) at the
                        state that emits generated token 7, layer 32 (G0.5)

Universe guard (029 §4): aborts unless the loaded collection matches the
frozen counts for --scope (pilot: 67 runs / 1,845 segments over swe_0,
swe_10, swe_11; full: 1,415 / 41,538) -- the gate2_probe_robustness idiom.

    python scripts/cis_score.py --dry-run --only-tasks swe_0,swe_10,swe_11
    python scripts/cis_score.py --a0 /ssd/wta_data/a0_v3_32b --only-tasks swe_0,swe_10,swe_11 \
        --controls foreign=2,rival --work-dir /ssd3/wta-cis/pilot --scope pilot
    python scripts/cis_score.py --a0 /ssd3/wta_data/a0_full_info_32b --mode full_info \
        --only-tasks swe_0,swe_10,swe_11 --controls none --work-dir /ssd3/wta-cis/pilot_fi --scope pilot_fullinfo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from restore_hilbench_images import eligible_tasks  # noqa: E402
from wta.cis_context import (ObservationEnv, build_instruction,  # noqa: E402
                             history_extension_ids, inject_into_last_user,
                             rebuild_contexts, render_turn)
from wta.cis_registry import (foreign_controls, load_resolutions,  # noqa: E402
                              load_rival_fixture)
from wta.labeling import _is_mutating  # noqa: E402

LAYERS_FRACTIONS = "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85"   # R2's --layers
MID_POS = 3                                              # layer 32 in the 8-list
CONTEXT_CAP = 65_536
UNIVERSE = {"pilot": {"tasks": ["swe_0", "swe_10", "swe_11"], "runs": 67, "segments": 1845},
            # the full_info arm (028 Am.H.1): 68 of 72 seeds succeeded; its
            # segment total was never recorded on the laptop, so it is logged
            # as-run rather than asserted
            "pilot_fullinfo": {"tasks": ["swe_0", "swe_10", "swe_11"], "runs": 68, "segments": None},
            "full": {"tasks": None, "runs": 1415, "segments": 41538}}
_EXIT = re.compile(r"exit\s+(-?\d+)")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RecordedExitEnv(ObservationEnv):
    """029 §4: the model saw the RECORDED exit code with the replayed text.
    Returns recorded N, records the replayed one for the fidelity read-out."""

    def bind(self, recorded_codes: list[int | None]) -> None:
        self._queue = list(recorded_codes)
        self.fidelity = {"matched": 0, "compared": 0, "mismatches": []}
        self.captures: list[dict] = []

    def execute(self, command: str) -> tuple[int, str]:
        rc, text = super().execute(command)
        self.captures.append(dict(self.last_capture))
        rec = self._queue.pop(0) if self._queue else None
        if rec is None:
            return rc, text
        self.fidelity["compared"] += 1
        if rec == rc:
            self.fidelity["matched"] += 1
        else:
            self.fidelity["mismatches"].append({"recorded": rec, "replayed": rc,
                                                "cmd": command[:80]})
        return rec, text


def recorded_codes(log_json: dict) -> list[int | None]:
    """Recorded exit code per EXECUTED action, in order (the submit action is
    logged but never executed, so it is skipped)."""
    out = []
    for a in sorted(log_json["actions"], key=lambda x: x["segment_idx"]):
        if "TASK_DONE" in (a.get("action_text") or ""):
            continue
        m = _EXIT.search(str((a.get("observables") or {}).get("error_signature") or ""))
        out.append(int(m.group(1)) if m else None)
    return out


def unit_seed(run_id: str, k: int, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{run_id}:{k}".encode()).hexdigest()[:8], 16)


def score_run(scorer, tok, run_id: str, task: str, segs: list[str],
              ctxs: list[list[dict]], variants_for, layers: list[int],
              cap: int, check_from_scratch: int) -> tuple[dict, dict]:
    """Teacher-force every segment of one run. Returns (units, npz arrays)."""
    units, feats, feat7, checks = [], [], [], []
    n_checked = 0
    for k, (m, seg) in enumerate(zip(ctxs, segs)):
        r = render_turn(tok, m, seg)
        u = {"k": k, "seg_tokens": len(r.seg_ids), "block_span": r.block_span,
             "no_block": r.block_span is None, "prompt_len": r.prompt_len,
             "is_mutating": False, "unscored_context_cap": False, "variants": {}}
        cmd_block = seg.split("```bash")[-1] if "```bash" in seg else ""
        u["is_mutating"] = _is_mutating(cmd_block)
        if r.prompt_len + len(r.seg_ids) > cap:
            u["unscored_context_cap"] = True
            units.append(u)
            continue                      # monotone: later turns exceed too; still recorded
        if k == 0:
            scorer.prefill(r.prefix_ids)
        elif scorer.prefix_len != len(r.prefix_ids):
            raise RuntimeError(f"{run_id} turn {k}: cache holds {scorer.prefix_len} "
                               f"tokens, G_k has {len(r.prefix_ids)}")
        t0, t1 = r.block_span if r.block_span else (0, len(r.seg_ids))
        off = len(r.user_suffix_ids) + len(r.header_ids)
        pos = [off - 1, off + t0 - 1]                 # P_k-1, P_bash-1
        pos7 = off - 1 + 7 if len(r.seg_ids) > 7 else None
        branch_ids = r.user_suffix_ids + r.header_ids + r.seg_ids
        lp, f = scorer.branch(branch_ids, off, len(r.seg_ids),
                              positions=pos + ([pos7] if pos7 is not None else []))
        u["S_k"] = float(lp[t0:t1].sum()); u["S_k_seg"] = float(lp.sum())
        u["block_tokens"] = int(t1 - t0)
        feats.append(f[:, :2, :].astype(np.float16))
        feat7.append(f[MID_POS, 2, :].astype(np.float16) if pos7 is not None
                     else np.full(f.shape[-1], np.nan, dtype=np.float16))
        if n_checked < check_from_scratch:
            lp_s, _ = scorer.from_scratch(r.prefix_ids + branch_ids,
                                          len(r.prefix_ids) + off, len(r.seg_ids))
            checks.append({"k": k, "max_abs_tok": float(np.abs(lp - lp_s).max()),
                           "block_sum_branch": float(lp[t0:t1].sum()),
                           "block_sum_scratch": float(lp_s[t0:t1].sum())})
            n_checked += 1
        for name, text in variants_for(task, run_id, k):
            m_inj = inject_into_last_user(m, text)
            ri = render_turn(tok, m_inj, seg)
            if ri.prefix_ids != r.prefix_ids:
                raise RuntimeError(f"{run_id} turn {k}: injection changed the prefix")
            off_i = len(ri.user_suffix_ids) + len(ri.header_ids)
            lpi, _ = scorer.branch(ri.user_suffix_ids + ri.header_ids + ri.seg_ids,
                                   off_i, len(ri.seg_ids))
            d = lpi - lp
            u["variants"][name] = {
                "block_sum": float(lpi[t0:t1].sum()), "seg_sum": float(lpi.sum()),
                "cis_block": float(d[t0:t1].sum()), "cis_seg": float(d.sum()),
                "min_tok_shift": float(d[t0:t1].min()) if t1 > t0 else None,
                "inject_tokens": len(ri.user_suffix_ids) - len(r.user_suffix_ids)}
        units.append(u)
        if k + 1 < len(ctxs):
            scorer.extend(history_extension_ids(tok, m, seg, ctxs[k + 1][-1]["content"]))
    # feats: (n_units, 2 positions, L layers, H) -- transposed from the hook's (L, P, H)
    arrays = {"feats": (np.stack(feats).transpose(0, 2, 1, 3) if feats
                        else np.zeros((0, 2, len(layers), 0), np.float16)),
              "feat_tok7": np.stack(feat7) if feat7 else np.zeros((0, 0), np.float16),
              "unit_k": np.array([u["k"] for u in units if not u["unscored_context_cap"]])}
    return {"units": units, "from_scratch_checks": checks}, arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="/ssd/wta_data/a0_v3_32b")
    ap.add_argument("--mode", default=None, help="baseline|full_info; default from manifest")
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--only-tasks", default="")
    ap.add_argument("--scope", choices=tuple(UNIVERSE), default="pilot")
    ap.add_argument("--controls", default="foreign=2,rival",
                    help="comma list of: foreign=N, rival, none")
    ap.add_argument("--rival-fixture", default="data/cis_rival_resolutions_pilot.json")
    ap.add_argument("--model-id", default="Qwen/Qwen3-32B")
    ap.add_argument("--layers", default=LAYERS_FRACTIONS)
    ap.add_argument("--work-dir", default="/ssd3/wta-cis/pilot")
    ap.add_argument("--exec-timeout", type=int, default=120)
    ap.add_argument("--context-cap", type=int, default=CONTEXT_CAP)
    ap.add_argument("--check-from-scratch", type=int, default=0,
                    help="per run, verify this many units against a no-cache forward (G0.4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    tasks_dir = Path(args.tasks_dir)
    eligible = eligible_tasks(tasks_dir, Path(args.classes), 60)
    if args.only_tasks:
        want = {t.strip() for t in args.only_tasks.split(",") if t.strip()}
        missing = want - {d.name for d in eligible}
        if missing:
            log(f"FATAL: --only-tasks names tasks not in the eligible set: {sorted(missing)}")
            return 2
        eligible = [d for d in eligible if d.name in want]
    task_ids = [d.name for d in eligible]
    uni = UNIVERSE[args.scope]
    if uni["tasks"] is not None and sorted(task_ids) != sorted(uni["tasks"]):
        log(f"FATAL: scope {args.scope} expects tasks {uni['tasks']}, got {task_ids}")
        return 2

    a0 = Path(args.a0)
    manifest = next(iter(sorted(a0.glob("collection_manifest*.json"))), None)
    mode = args.mode or (json.loads(manifest.read_text(encoding="utf-8"))["args"].get("mode", "baseline")
                         if manifest else "baseline")
    runs = []
    for t in task_ids:
        for f in sorted((a0 / t).glob(f"{t}-s*.segments.json")):
            rid = f.name.replace(".segments.json", "")
            if (a0 / t / f"{rid}.json").exists():
                runs.append((t, rid))
    n_seg = sum(len(json.loads((a0 / t / f"{r}.segments.json").read_text(encoding="utf-8")))
                for t, r in runs)
    log(f"mode={mode} tasks={len(task_ids)} runs={len(runs)} segments={n_seg}")
    if len(runs) != uni["runs"] or (uni["segments"] is not None and n_seg != uni["segments"]):
        log(f"!! universe mismatch: scope {args.scope} expects {uni['runs']} runs / "
            f"{uni['segments'] or 'as-run'} segments. Aborting (029 §4 guard).")
        return 1

    res = load_resolutions(tasks_dir, Path(args.classes))
    instr_by_task = {t: (tasks_dir / t / "baseline" / "instruction.md").read_text(encoding="utf-8")
                     for t in sorted({k[0] for k in res})}
    controls = {c.split("=")[0]: (int(c.split("=")[1]) if "=" in c else True)
                for c in args.controls.split(",") if c and c != "none"}
    rivals = load_rival_fixture(args.rival_fixture) if controls.get("rival") else {}
    n_rival = sum(1 for k in rivals if k[0] in task_ids)
    log(f"controls={controls} approved rivals in scope={n_rival}")

    def variants_for(task: str, run_id: str, k: int):
        out = []
        keys = [key for key in res if key[0] == task]
        for key in keys:
            out.append((f"own:{key[1]}", res[key].resolution))
        nf = controls.get("foreign", 0)
        if nf:
            for fk in foreign_controls(keys[0], res, instr_by_task, n=nf,
                                       seed=unit_seed(run_id, k, args.seed)):
                out.append((f"foreign:{fk[0]}/{fk[1]}", res[fk].resolution))
        for (t, bid, cls), text in rivals.items():
            if t == task:
                out.append((f"rival:{bid}/{cls}", text))
        return out

    if args.dry_run:
        for t, r in runs[:3]:
            log(f"  {r}: {len(variants_for(t, r, 0))} variants at k=0")
        log("dry-run OK")
        return 0

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    from wta.cis_scorer import TeacherForcedScorer, load_model
    from wta.hf_reader import resolve_layers
    model, tok = load_model(args.model_id)
    n_layers = model.config.num_hidden_layers
    layers = resolve_layers(n_layers, [float(x) for x in args.layers.split(",")])
    log(f"model loaded: {n_layers} layers, capture {layers} ({time.time() - t0:.0f}s)")
    scorer = TeacherForcedScorer(model, layers)

    for t, rid in runs:
        out_json = work / f"{rid}.json"
        if out_json.exists():
            continue
        segs = json.loads((a0 / t / f"{rid}.segments.json").read_text(encoding="utf-8"))
        lj = json.loads((a0 / t / f"{rid}.json").read_text(encoding="utf-8"))
        image = (tasks_dir / t / "shared" / "image_ref.txt").read_text(encoding="utf-8").strip()
        instruction = build_instruction(tasks_dir / t, mode, nudge=True)
        tr = time.time()
        env = RecordedExitEnv(image, name=f"wta-cis-{rid}", exec_timeout=args.exec_timeout)
        env.bind(recorded_codes(lj))
        try:
            with env:
                ctxs, _ = rebuild_contexts(segs, instruction, env, run_id=rid, task_id=t,
                                           seed=lj["seed"], temperature=lj["temperature"])
        except Exception as e:
            out_json.write_text(json.dumps({"run": rid, "task": t, "error": f"{type(e).__name__}: {e}"}),
                                encoding="utf-8")
            log(f"  {rid}: REPLAY ERROR {e}")
            continue
        t_replay = time.time() - tr
        try:
            rec, arrays = score_run(scorer, tok, rid, t, segs, ctxs, variants_for, layers,
                                    args.context_cap, args.check_from_scratch)
        except Exception as e:
            out_json.write_text(json.dumps({"run": rid, "task": t, "error": f"{type(e).__name__}: {e}"}),
                                encoding="utf-8")
            log(f"  {rid}: SCORE ERROR {e}")
            continue
        fid = env.fidelity
        rec.update({"run": rid, "task": t, "mode": mode, "seed": lj["seed"],
                    "temperature": lj["temperature"], "n_segments": len(segs),
                    "layers": layers, "mid_pos": MID_POS,
                    "fidelity": {**fid, "rate": (fid["matched"] / fid["compared"]) if fid["compared"] else None},
                    "captures_capped": sum(1 for c in env.captures if c.get("capped")),
                    "wall_replay_s": round(t_replay, 1),
                    "wall_score_s": round(time.time() - tr - t_replay, 1)})
        np.savez_compressed(work / f"{rid}.npz", **arrays)
        out_json.write_text(json.dumps(rec), encoding="utf-8")
        n_scored = sum(1 for u in rec["units"] if not u["unscored_context_cap"])
        log(f"  {rid}: {n_scored}/{len(segs)} units, fidelity "
            f"{rec['fidelity']['rate']}, replay {t_replay:.0f}s score "
            f"{rec['wall_score_s']:.0f}s ({time.time() - t0:.0f}s)")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
