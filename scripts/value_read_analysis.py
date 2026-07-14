"""Do reads NEAR value-emission moments carry the lean? (decisions/016)

    python scripts/value_read_analysis.py

Approximate test on the EXISTING multi-layer collection (no GPU): cadence
reads land every 32 tokens, so some happen to fall close to the moment a
run's trace mentions a class signature (e.g. writes "timeout = 30") and some
far. If value information is transiently present in the residual stream at
emission, NEAR reads should separate interpretations better than FAR reads,
on exactly the value-fork decisions where the average was chance.

Per forked decision: leave-one-run-out nearest-class-centroid on RAW layer-14
h, computed twice -- using only reads within +-NEAR tokens of any signature
mention of the run's committed class family, and using only reads >= FAR
tokens away. Honest caveats printed with the numbers: n shrinks sharply on
the near side; this cannot REPLACE the value-triggered collection (the true
emission-moment read), only justify or kill it cheaply.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wta.labeling import build_labels, load_class_artifact, token_char_positions  # noqa: E402

NEAR, FAR = 12, 24  # tokens; reads <=NEAR from a mention vs >=FAR from all


def mention_tokens(text: str, sigs: list[str], tokenizer) -> list[int]:
    """Token indices where any signature occurs in the trace (case-insensitive,
    raw text -- no whitespace collapse so char offsets stay valid)."""
    low = text.lower()
    starts = token_char_positions(text, tokenizer)
    if not starts:
        return []
    starts_arr = np.array(starts)
    out = []
    for sig in sigs:
        s = sig.lower()
        pos = low.find(s)
        while pos >= 0:
            out.append(int(np.searchsorted(starts_arr, pos, side="right") - 1))
            pos = low.find(s, pos + 1)
    return sorted(set(out))


def main() -> int:
    from transformers import AutoTokenizer

    ds = build_labels("data/a0", "data/interpretation_classes.json", layer=1)
    art = load_class_artifact("data/interpretation_classes.json")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    # per (run, decision): distance of each read to the nearest mention of the
    # decision's signatures in that run's trace
    sig_by_dec = {}
    for did, (task, blocker) in enumerate(ds.vocab.decisions):
        sig_by_dec[did] = [s for c in art[task][blocker]["classes"]
                           for s in c["signatures"]]
    dist = np.full(len(ds.h), np.inf)
    for r, (task, run_id) in enumerate(ds.runs):
        text = Path(f"data/a0/{task}/{run_id}.txt").read_text(
            encoding="utf-8", errors="replace")
        toks_cache: dict[int, list[int]] = {}
        m_run = ds.run_idx == r
        for dec in set(ds.decision[m_run & (ds.decision >= 0)].tolist()):
            if dec not in toks_cache:
                toks_cache[dec] = mention_tokens(text, sig_by_dec[dec], tokenizer)
            ments = toks_cache[dec]
            if not ments:
                continue
            m = m_run & (ds.decision == dec)
            for i in np.where(m)[0]:
                dist[i] = min(abs(ds.read_token_idx[i] - t) for t in ments)

    def loro_acc(mask_extra) -> tuple[float, float, int, int]:
        """leave-one-run-out centroid acc over forked decisions, on reads
        passing mask_extra; returns (acc, chance, n_reads, n_decisions)."""
        lab = (ds.cls >= 0) & mask_extra
        cor = tot = 0
        chances, n_dec = [], 0
        for dec in np.unique(ds.decision[lab]):
            m = lab & (ds.decision == dec)
            runs = np.unique(ds.run_idx[m])
            cls_of = {r: ds.cls[m & (ds.run_idx == r)][0] for r in runs}
            if len(set(cls_of.values())) < 2 or len(runs) < 4:
                continue
            dec_cor = dec_tot = 0
            for r_out in runs:
                tr, te = m & (ds.run_idx != r_out), m & (ds.run_idx == r_out)
                cls_tr = ds.cls[tr]
                if len(set(cls_tr.tolist())) < 2 or not te.any():
                    continue
                cents = {}
                for c in set(cls_tr.tolist()):
                    v = ds.h[tr][cls_tr == c].mean(0)
                    cents[c] = v / np.linalg.norm(v)
                for x, y in zip(ds.h[te], ds.cls[te]):
                    xn = x / np.linalg.norm(x)
                    pred = max(cents, key=lambda c: float(xn @ cents[c]))
                    dec_cor += int(pred == y)
                    dec_tot += 1
            if dec_tot:
                n_dec += 1
                chances.append(1 / len(set(cls_of.values())))
                cor += dec_cor
                tot += dec_tot
        return (cor / tot if tot else float("nan"),
                float(np.mean(chances)) if chances else float("nan"), tot, n_dec)

    print(f"reads with a finite mention-distance: {(np.isfinite(dist)).sum()} "
          f"of {len(dist)}")
    for name, mask in [("NEAR (<= %d tok of a signature mention)" % NEAR,
                        dist <= NEAR),
                       ("FAR  (>= %d tok from every mention)" % FAR,
                        np.isfinite(dist) & (dist >= FAR)),
                       ("ALL  (any labeled read)", np.isfinite(dist))]:
        acc, chance, n, nd = loro_acc(mask)
        print(f"{name}: acc {acc:.3f} vs chance {chance:.3f} "
              f"({n} reads, {nd} decisions)")
    print("\nInterpretation: if NEAR >> FAR, value info is transiently present "
          "at emission -> the --value-reads collection (and 70B run) should "
          "capture it; if NEAR ~ FAR ~ chance, the value-fork negative hardens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
