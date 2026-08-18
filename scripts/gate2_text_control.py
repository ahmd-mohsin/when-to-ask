"""Is gate 2 measuring the model's internal state, or just the words on screen?

gate2_decision_recovery classifies WHICH decision a read belongs to from the A2
topic vector T: 0.4102 vs 0.0101 chance (~40x), the strongest positive in the
R2 run. But activations are computed FROM the surrounding text, and different
decisions live in different files with different identifiers. If a bag-of-words
classifier on the same text window does as well, the result is lexical and says
nothing about internal states.

Same reads, same held-out seed split, two feature sets:
    (a) activations -- h from labels.npz, JL-projected
    (b) text        -- TF-IDF over the +-window_chars of raw trace text
The comparison that matters is (a) vs (b), not (a) vs chance.

    python scripts/gate2_text_control.py
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def stream_project(npz_path: Path, dim: int, rows: np.ndarray,
                   seed: int = 0, block: int = 8192) -> np.ndarray:
    """JL-project only the rows we need, streaming h out of the zip."""
    with zipfile.ZipFile(npz_path) as zf:
        name = "h.npy" if "h.npy" in zf.namelist() else "h"
        with zf.open(name) as rawf:
            buf = io.BufferedReader(rawf, buffer_size=1 << 22)
            major, _ = np.lib.format.read_magic(buf)
            reader = (np.lib.format.read_array_header_1_0 if major == 1
                      else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = reader(buf)
            if fortran:
                raise ValueError("fortran-order h unsupported")
            n, d = shape
            want = np.zeros(n, dtype=bool)
            want[rows] = True
            pos = np.full(n, -1, dtype=np.int64)
            pos[rows] = np.arange(len(rows))
            rng = np.random.default_rng(seed)
            identity = dim >= d          # full-dim: no projection at all
            P = (None if identity else
                 rng.standard_normal((d, dim)).astype(np.float32) / np.sqrt(dim))
            out = np.empty((len(rows), d if identity else dim),
                           dtype=np.float32)
            done = 0
            while done < n:
                k = min(block, n - done)
                raw = buf.read(k * d * dtype.itemsize)
                sel = want[done:done + k]
                if sel.any():
                    chunk = np.frombuffer(raw, dtype=dtype).reshape(k, d)[sel]
                    chunk = chunk.astype(np.float32)
                    out[pos[done:done + k][sel]] = (chunk if identity
                                                    else chunk @ P)
                done += k
                if done % (block * 20) == 0:
                    log(f"    h streamed {done}/{n}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="models/v3_32b/labels.npz")
    ap.add_argument("--debug", default="models/v3_32b/labels_debug.jsonl")
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--held-seeds", default="6,7")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--max-test", type=int, default=20000)
    ap.add_argument("--causal", action="store_true",
                    help="use ONLY text BEFORE the read position. The "
                         "default +-window includes text written AFTER "
                         "the read, which did not exist when the model "
                         "produced that activation -- lookahead the online "
                         "detector could never have. This is the fair "
                         "baseline for a lead-time claim.")
    ap.add_argument("--mask-anchors", action="store_true",
                    help="blank every blocker anchor string out of the "
                         "text window first. The decision label IS argmax "
                         "of anchor hits in this window (labeling.py:315), "
                         "so unmasked text partly reverse-engineers the "
                         "labeler. Masked = the honest lexical control.")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    npz = Path(args.labels)
    log("loading label arrays ...")
    z = np.load(npz, allow_pickle=False)
    dec = z["decision"]
    run_idx = z["run_idx"]
    tok = z["read_token_idx"]
    meta = json.loads(str(z["meta"]))
    runs = [tuple(r) for r in meta["runs"]]
    log(f"reads={len(dec)} decisions={len(meta['decisions'])} runs={len(runs)}")

    log("scanning debug trail (compact: run/char/token_idx only) ...")
    d_run: list[str] = []
    d_char: list[int] = []
    d_tok: list[int] = []
    marker_a, marker_b = '"kind": "read"', '"kind":"read"'
    for line in Path(args.debug).open(encoding="utf-8"):
        if marker_a not in line and marker_b not in line:
            continue
        row = json.loads(line)
        d_run.append(row["run"])
        d_char.append(int(row.get("char", -1)))
        d_tok.append(int(row.get("token_idx", -1)))
    d_char_a = np.asarray(d_char)
    d_tok_a = np.asarray(d_tok)
    log(f"debug read records={len(d_char_a)}")
    if len(d_char_a) != len(dec):
        log("!! debug/npz read counts differ -- cannot align")
        return 1
    align = float(np.mean(d_tok_a == tok))
    log(f"token_idx alignment = {align:.4f}")
    if align < 0.99:
        log("!! rows do not align -- aborting")
        return 1

    held = {int(s) for s in args.held_seeds.split(",")}
    seed_of_row = np.array([int(runs[i][1].rsplit("-s", 1)[1]) for i in run_idx])
    m = dec >= 0
    is_held = np.isin(seed_of_row, list(held))
    tr = np.where(m & ~is_held)[0]
    te = np.where(m & is_held)[0]
    tr = tr[np.isin(dec[tr], np.unique(dec[te]))]
    rng = np.random.default_rng(0)
    if len(tr) > args.max_train:
        tr = np.sort(rng.choice(tr, args.max_train, replace=False))
    if len(te) > args.max_test:
        te = np.sort(rng.choice(te, args.max_test, replace=False))
    n_cls = len(np.unique(dec[te]))
    log(f"train={len(tr)} test={len(te)} classes={n_cls} chance={1 / n_cls:.4f}")

    log("materializing text windows for selected rows only ...")
    sel = np.concatenate([tr, te])
    cache: dict[str, str] = {}
    texts: dict[int, str] = {}
    for i in sel:
        run = d_run[i]
        if run not in cache:
            task = run.rsplit("-s", 1)[0]
            # v3.1 (decisions/026): same text acquisition as the labeler --
            # raw segment join, else untranslated .txt -- so the raw `char`
            # offsets from the debug trail slice the string they were born
            # in. The old read_text() slice was displaced on CRLF runs.
            seg = Path(args.a0) / task / f"{run}.segments.json"
            p = Path(args.a0) / task / f"{run}.txt"
            if seg.exists():
                cache[run] = "\n\n".join(
                    json.loads(seg.read_text(encoding="utf-8")))
            elif p.exists():
                with open(p, encoding="utf-8", errors="replace",
                          newline="") as fh:
                    cache[run] = fh.read()
            else:
                cache[run] = ""
        t = cache[run]
        c = int(d_char_a[i])
        if c < 0:
            texts[int(i)] = ""
        elif args.causal:
            texts[int(i)] = t[max(0, c - 2 * args.window):c]
        else:
            texts[int(i)] = t[max(0, c - args.window):c + args.window]
    log(f"windows built for {len(texts)} rows over {len(cache)} traces")

    if args.mask_anchors:
        from wta.labeling import load_class_artifact
        art = load_class_artifact("data/interpretation_classes.json")
        terms = sorted({a for t in art if not t.startswith("_")
                        for spec in art[t].values()
                        for a in spec["anchors"]}, key=len, reverse=True)
        pat = re.compile("|".join(re.escape(a) for a in terms), re.I)
        n_before = sum(len(v) for v in texts.values())
        texts = {k: pat.sub(" ", v) for k, v in texts.items()}
        n_after = sum(len(v) for v in texts.values())
        log(f"    masked {len(terms)} anchor strings; chars "
            f"{n_before}->{n_after} ({100*(n_before-n_after)/max(n_before,1):.1f}% removed)")

    log("(b) fitting TEXT model (word tfidf 1-2gram) ...")
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                          max_features=120000, min_df=3, sublinear_tf=True,
                          token_pattern=r"(?u)\b\w[\w./_-]+\b")
    Xtr = vec.fit_transform([texts[int(i)] for i in tr])
    Xte = vec.transform([texts[int(i)] for i in te])
    log(f"    tfidf features={Xtr.shape[1]}")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, dec[tr])
    acc_text = float(clf.score(Xte, dec[te]))
    log(f"(b) TEXT tfidf {'prefix-only ' if args.causal else '+-'}"
        f"{args.window}ch : acc={acc_text:.4f}")

    log("(a) streaming + projecting activations ...")
    H = stream_project(npz, args.dim, sel, seed=0)
    idx = {int(r): k for k, r in enumerate(sel)}
    Htr = H[[idx[int(i)] for i in tr]]
    Hte = H[[idx[int(i)] for i in te]]
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    clf2 = LogisticRegression(max_iter=2000)
    clf2.fit((Htr - mu) / sd, dec[tr])
    acc_act = float(clf2.score((Hte - mu) / sd, dec[te]))
    log(f"(a) ACTIV h->{args.dim}d (JL)   : acc={acc_act:.4f}")

    print()
    print(f"chance {1 / n_cls:.4f} | TEXT {acc_text:.4f} | "
          f"ACTIVATIONS {acc_act:.4f} | "
          f"act/text {acc_act / max(acc_text, 1e-9):.2f}x")
    print("gate2 as reported (A2 topic space T): 0.4102 vs 0.0101 chance")
    Path("results").mkdir(exist_ok=True)
    Path("results/gate2_text_control%s.json" % (("_masked" if args.mask_anchors else "")
          + ("_causal" if args.causal else ""))).write_text(json.dumps(
        {"n_train": int(len(tr)), "n_test": int(len(te)),
         "n_classes": int(n_cls), "chance": 1 / n_cls,
         "acc_text": acc_text, "acc_activations": acc_act,
         "window_chars": args.window, "proj_dim": args.dim,
         "mask_anchors": bool(args.mask_anchors),
         "causal_prefix_only": bool(args.causal),
         "gate2_reported_T": 0.4102}, indent=1), encoding="utf-8")
    log("wrote results json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
