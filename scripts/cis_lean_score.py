"""029.3 lean-readout driver -- the box side (docker + GPU).

Per run: rebuild the contexts exactly as scripts/cis_score.py does
(wta.cis_context driving run_agent; recorded [exit N], replayed text), then at
every turn's pre-action position teacher-force each statement of the task's
blockers (data/cis_lean_statements_pilot.json) plus the canonical statements
of 2 matched foreign blockers, all as assistant continuations after the
generation header, on the branched KV cache. Statement null log-probs (PMI
denominators) are computed once per statement under the frozen null context.

Per-run outputs (resumable by <run>.json existing):
  <work>/<run>.json  -- per unit: k, is_mutating, has_block, prompt_len, per
                        blocker {classes, lp_ctx, n_tok, pmi, p, p_canonical,
                        entropy, argmax}, foreign canonical PMIs, relevance;
                        plus fidelity and from-scratch checks
  <work>/<run>.npz   -- feats (n_units, L, H) float16 at P_k-1 (baseline branch)
  <work>/__null__.json -- statement -> lp_null

    python scripts/cis_lean_score.py --dry-run --only-tasks swe_0,swe_10,swe_11 --scope pilot
    python scripts/cis_lean_score.py --a0 /ssd/wta_data/a0_v3_32b --only-tasks swe_0,swe_10,swe_11 \
        --scope pilot --work-dir /ssd3/wta-lean/pilot --check-from-scratch 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cis_score import (CONTEXT_CAP, LAYERS_FRACTIONS, MID_POS,  # noqa: E402
                       UNIVERSE, RecordedExitEnv, log, recorded_codes,
                       unit_seed)
from restore_hilbench_images import eligible_tasks  # noqa: E402
from wta.cis_context import (GEN_HEADER, build_instruction,  # noqa: E402
                             history_extension_ids, rebuild_contexts,
                             render_turn)
from wta.cis_lean import (null_messages, readout,  # noqa: E402
                          length_normalised_p, statement)
from wta.cis_registry import foreign_controls, load_resolutions  # noqa: E402
from wta.labeling import _is_mutating  # noqa: E402


def stmt_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def null_scores(scorer, tok, texts: list[str], cache_path: Path) -> dict[str, float]:
    """log p(stmt | null context), once per statement, cached on disk."""
    done = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [t for t in texts if t not in done]
    if todo:
        m = null_messages()
        prompt = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        assert prompt.endswith(GEN_HEADER)
        pid = tok(prompt)["input_ids"]
        # The logit that predicts a statement's FIRST token is emitted at the
        # prompt's last position, which a branch never recomputes. So hold the
        # last prompt token out of the cache and branch with it at offset 1 --
        # the same geometry as the main path (head = user turn + header).
        scorer.prefill(pid[:-1])
        for t in todo:
            s = stmt_ids(tok, t)
            lp, _ = scorer.branch([pid[-1]] + s, 1, len(s))
            done[t] = float(lp.sum())
        cache_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    return done


def score_run(scorer, tok, run_id, task, segs, ctxs, menu, foreign_for, lp_null,
              cap, check_from_scratch):
    units, feats, checks, n_checked = [], [], [], 0
    for k, (m, seg) in enumerate(zip(ctxs, segs)):
        r = render_turn(tok, m, seg)
        u = {"k": k, "seg_tokens": len(r.seg_ids), "has_block": r.block_span is not None,
             "prompt_len": r.prompt_len, "unscored_context_cap": False, "blockers": {},
             "foreign": {}}
        u["is_mutating"] = _is_mutating(seg.split("```bash")[-1] if "```bash" in seg else "")
        if r.prompt_len + 256 > cap:
            u["unscored_context_cap"] = True
            units.append(u)
            continue
        if k == 0:
            scorer.prefill(r.prefix_ids)
        elif scorer.prefix_len != len(r.prefix_ids):
            raise RuntimeError(f"{run_id} turn {k}: cache {scorer.prefix_len} != G_k {len(r.prefix_ids)}")
        head = r.user_suffix_ids + r.header_ids
        off = len(head)
        first = True
        scratch_rows = []
        for b in menu:                                   # own blockers, complete menus only
            lps, ntoks = [], []
            for text in b["statements"]:
                s = stmt_ids(tok, text)
                lp, f = scorer.branch(head + s, off, len(s),
                                      positions=[off - 1] if first else None)
                if first:
                    feats.append(f[:, 0, :].astype(np.float16)); first = False
                lps.append(float(lp.sum())); ntoks.append(len(s))
                if n_checked < check_from_scratch:
                    lp_s, _ = scorer.from_scratch(r.prefix_ids + head + s,
                                                  len(r.prefix_ids) + off, len(s))
                    scratch_rows.append((b["blocker_id"], text, float(lp.sum()), float(lp_s.sum())))
            ro = readout(b["blocker_id"], b["classes"], lps,
                         [lp_null[t] for t in b["statements"]], ntoks)
            u["blockers"][b["blocker_id"]] = {
                "classes": ro.classes, "lp_ctx": ro.lp_ctx, "n_tok": ro.n_tok,
                "pmi": ro.pmi, "p": ro.p, "p_canonical": ro.p_canonical,
                "entropy": ro.entropy, "argmax": ro.argmax,
                "p_lennorm": length_normalised_p(ro.lp_ctx, ro.n_tok)}
        if scratch_rows:
            checks.append({"k": k, "rows": scratch_rows}); n_checked += 1
        for fk, text in foreign_for(task, run_id, k):
            s = stmt_ids(tok, text)
            lp, _ = scorer.branch(head + s, off, len(s))
            u["foreign"][f"{fk[0]}/{fk[1]}"] = {"lp_ctx": float(lp.sum()),
                                                "pmi": float(lp.sum() - lp_null[text])}
        own_canon = [v["pmi"][0] for v in u["blockers"].values()]
        frn = [v["pmi"] for v in u["foreign"].values()]
        u["relevance"] = (float(np.mean(own_canon) - np.mean(frn))
                          if own_canon and frn else None)
        if first:                                        # no complete menu: still capture
            lp, f = scorer.branch(head + [tok.eos_token_id or 0], off, 1, positions=[off - 1])
            feats.append(f[:, 0, :].astype(np.float16))
        units.append(u)
        if k + 1 < len(ctxs):
            scorer.extend(history_extension_ids(tok, m, seg, ctxs[k + 1][-1]["content"]))
    arrays = {"feats": np.stack(feats) if feats else np.zeros((0, 0, 0), np.float16),
              "unit_k": np.array([u["k"] for u in units if not u["unscored_context_cap"]])}
    return {"units": units, "from_scratch_checks": checks}, arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="/ssd/wta_data/a0_v3_32b")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--statements", default="data/cis_lean_statements_pilot.json")
    ap.add_argument("--only-tasks", default="")
    ap.add_argument("--scope", choices=tuple(UNIVERSE), default="pilot")
    ap.add_argument("--n-foreign", type=int, default=2)
    ap.add_argument("--model-id", default="Qwen/Qwen3-32B")
    ap.add_argument("--layers", default=LAYERS_FRACTIONS)
    ap.add_argument("--work-dir", default="/ssd3/wta-lean/pilot")
    ap.add_argument("--exec-timeout", type=int, default=120)
    ap.add_argument("--context-cap", type=int, default=CONTEXT_CAP)
    ap.add_argument("--check-from-scratch", type=int, default=0)
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
    runs = [(t, f.name.replace(".segments.json", ""))
            for t in task_ids for f in sorted((a0 / t).glob(f"{t}-s*.segments.json"))
            if (a0 / t / f"{f.name.replace('.segments.json', '')}.json").exists()]
    n_seg = sum(len(json.loads((a0 / t / f"{r}.segments.json").read_text(encoding="utf-8")))
                for t, r in runs)
    log(f"mode={mode} tasks={len(task_ids)} runs={len(runs)} segments={n_seg}")
    if len(runs) != uni["runs"] or (uni["segments"] is not None and n_seg != uni["segments"]):
        log(f"!! universe mismatch: scope {args.scope} expects {uni['runs']} runs / "
            f"{uni['segments'] or 'as-run'} segments. Aborting (029 par.4 guard).")
        return 1

    st = json.loads(Path(args.statements).read_text(encoding="utf-8"))
    menus = {}
    for b in st["blockers"]:
        if b["complete"] and b["task"] in task_ids:
            menus.setdefault(b["task"], []).append(b)
    res = load_resolutions(tasks_dir, Path(args.classes))
    instr = {t: (tasks_dir / t / "baseline" / "instruction.md").read_text(encoding="utf-8")
             for t in sorted({k[0] for k in res})}
    all_texts = sorted({s for b in st["blockers"] if b["complete"] for s in b["statements"]}
                       | {statement(r.resolution) for r in res.values()})

    def foreign_for(task, run_id, k):
        keys = [key for key in res if key[0] == task]
        out = []
        for fk in foreign_controls(keys[0], res, instr, n=args.n_foreign,
                                   seed=unit_seed(run_id, k, args.seed)):
            out.append((fk, statement(res[fk].resolution)))
        return out

    n_menu = {t: sum(len(b["statements"]) for b in menus.get(t, [])) for t in task_ids}
    log(f"menus: {n_menu} statements per unit (+{args.n_foreign} foreign canonical)")
    if args.dry_run:
        log("dry-run OK")
        return 0

    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    from wta.cis_scorer import TeacherForcedScorer, load_model
    from wta.hf_reader import resolve_layers
    model, tok = load_model(args.model_id)
    layers = resolve_layers(model.config.num_hidden_layers, [float(x) for x in args.layers.split(",")])
    scorer = TeacherForcedScorer(model, layers)
    log(f"model loaded; capture layers {layers} ({time.time() - t0:.0f}s)")
    lp_null = null_scores(scorer, tok, all_texts, work / "__null__.json")
    log(f"null scores for {len(lp_null)} statements")

    for t, rid in runs:
        out_json = work / f"{rid}.json"
        if out_json.exists():
            continue
        segs = json.loads((a0 / t / f"{rid}.segments.json").read_text(encoding="utf-8"))
        lj = json.loads((a0 / t / f"{rid}.json").read_text(encoding="utf-8"))
        image = (tasks_dir / t / "shared" / "image_ref.txt").read_text(encoding="utf-8").strip()
        instruction = build_instruction(tasks_dir / t, mode, nudge=True)
        tr = time.time()
        env = RecordedExitEnv(image, name=f"wta-lean-{rid}", exec_timeout=args.exec_timeout)
        env.bind(recorded_codes(lj))
        try:
            with env:
                ctxs, _ = rebuild_contexts(segs, instruction, env, run_id=rid, task_id=t,
                                           seed=lj["seed"], temperature=lj["temperature"])
            t_replay = time.time() - tr
            rec, arrays = score_run(scorer, tok, rid, t, segs, ctxs, menus.get(t, []),
                                    foreign_for, lp_null, args.context_cap,
                                    args.check_from_scratch)
        except Exception as e:
            out_json.write_text(json.dumps({"run": rid, "task": t,
                                            "error": f"{type(e).__name__}: {e}"}), encoding="utf-8")
            log(f"  {rid}: ERROR {e}")
            continue
        fid = env.fidelity
        rec.update({"run": rid, "task": t, "mode": mode, "seed": lj["seed"],
                    "temperature": lj["temperature"], "n_segments": len(segs),
                    "layers": layers, "mid_pos": MID_POS,
                    "fidelity": {**fid, "rate": (fid["matched"] / fid["compared"]) if fid["compared"] else None},
                    "wall_replay_s": round(t_replay, 1),
                    "wall_score_s": round(time.time() - tr - t_replay, 1)})
        np.savez_compressed(work / f"{rid}.npz", **arrays)
        out_json.write_text(json.dumps(rec), encoding="utf-8")
        log(f"  {rid}: {len(segs)} units, fidelity {rec['fidelity']['rate']}, "
            f"score {rec['wall_score_s']:.0f}s ({time.time() - t0:.0f}s)")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
