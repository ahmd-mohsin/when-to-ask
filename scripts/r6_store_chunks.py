"""R6 chunk store (decisions/028 T1): the persistence + resume glue between
the runner workflow and r6_score.py.

Two modes:

  --pending          list payload chunk files with no stored result yet
                     (feeds the runner workflow's args on each launch)
  --store OUT.json   take a saved runner-workflow output ({"r6a": [{file,
                     judgments}, ...], "r6b": [...]}) and write one result
                     file per VALID chunk: results/r6_chunks/r6a_chunk_NN.json
                     (payload_ prefix stripped), judgments translated from
                     blind ids back to build ids via the private map.

Chunk validation before storing (pre-run review findings: judge-authored ids
must not be trusted): a chunk is stored only if its judgment ids are exactly
the payload's id set (no dupes, no unknowns, no missing). Invalid chunks are
reported and left unstored — they show up in --pending again.

    python scripts/r6_store_chunks.py --pending
    python scripts/r6_store_chunks.py --store scratch/runner_output.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def result_name(payload_file: str) -> str:
    return payload_file.removeprefix("payload_")


def pending(chunks_dir: Path) -> list[str]:
    out = []
    for p in sorted(chunks_dir.glob("payload_*_chunk_*.json")):
        if not (chunks_dir / result_name(p.name)).exists():
            out.append(p.name)
    return out


def store(chunks_dir: Path, items_dir: Path, out_json: Path) -> int:
    id_map = json.loads((items_dir / "blind_id_map.json")
                        .read_text(encoding="utf-8"))
    doc = json.loads(out_json.read_text(encoding="utf-8"))
    n_ok = n_bad = 0
    for cell in ("r6a", "r6b"):
        for chunk in doc.get(cell) or []:
            if not chunk or not chunk.get("judgments"):
                continue
            f = chunk["file"]
            payload = json.loads((chunks_dir / f).read_text(encoding="utf-8"))
            want = [it["item_id"] for it in payload["items"]]
            got = [j["item_id"] for j in chunk["judgments"]]
            if sorted(got) != sorted(want):
                print(f"INVALID {f}: judgment ids != payload ids "
                      f"(missing {sorted(set(want) - set(got))[:3]}..., "
                      f"unknown {sorted(set(got) - set(want))[:3]}...) — "
                      f"NOT stored")
                n_bad += 1
                continue
            rows = []
            for j in chunk["judgments"]:
                r = dict(j)
                r["item_id"] = id_map[r["item_id"]]
                rows.append(r)
            dest = chunks_dir / result_name(f)
            if dest.exists():
                print(f"SKIP {f}: {dest.name} already stored")
                continue
            dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            n_ok += 1
            print(f"stored {dest.name} ({len(rows)} judgments)")
    print(f"\nstored {n_ok} chunks, rejected {n_bad}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default="results/r6_chunks")
    ap.add_argument("--items-dir", default="results/r6_items")
    ap.add_argument("--pending", action="store_true")
    ap.add_argument("--store", default=None,
                    help="path to a saved runner-workflow output JSON")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if args.pending:
        p = pending(chunks_dir)
        print(json.dumps({
            "r6a": [f for f in p if "_r6a_" in f],
            "r6b": [f for f in p if "_r6b_" in f]}, indent=1))
        return 0
    if args.store:
        return store(chunks_dir, Path(args.items_dir), Path(args.store))
    print("pass --pending or --store OUT.json")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
