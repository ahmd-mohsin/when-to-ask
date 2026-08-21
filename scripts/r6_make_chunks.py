"""R6 chunk payload files (decisions/028 T1, 025 Amendment A transport).

Splits the frozen item sets into 20-item chunks and writes JUDGE-VISIBLE
payload files: each row keeps EXACTLY the whitelisted payload keys
(R6A_PAYLOAD_KEYS / R6B_PAYLOAD_KEYS — item_id + instruction/prefix or
excerpts). Truth fields (truth_needs_ask, truth, task, run, blocker, ...)
never enter a payload file; the in-session runner hands one payload file +
the prompt template to each Fable subagent and stores its judgments as
results/r6_chunks/{r6a|r6b}_chunk_NN.json (resumable: a chunk with an
existing result file is skipped).

Items are SHUFFLED deterministically (np.random.default_rng(0)) before
chunking: the builders emit items grouped by truth (R6a forked-first, R6b
same-then-diff), and truth-homogeneous chunks would let a judge's implicit
base-rate prior act as a per-chunk bias. Transport-level choice, frozen
before any judgment.

item_ids are RE-KEYED to opaque salted-hash ids in the payloads (pre-run
review finding: build-order ids monotonically encode ground truth — an
id-threshold rule scores 100% with zero excerpt use). The private map
results/r6_items/blind_id_map.json translates back at store/score time and
must never be sent to a judge.

    python scripts/r6_make_chunks.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from r6_build_items import R6A_PAYLOAD_KEYS, R6B_PAYLOAD_KEYS  # noqa: E402

CHUNK = 20
SEED = 0
BLIND_SALT = "r6-blind-v1"


def blind_id(item_id: str) -> str:
    return hashlib.sha1(f"{BLIND_SALT}|{item_id}".encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", default="results/r6_items")
    ap.add_argument("--out-dir", default="results/r6_chunks")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    id_map: dict[str, str] = {}    # blind -> build id, both cells pooled
    for name, keys in (("r6a", R6A_PAYLOAD_KEYS), ("r6b", R6B_PAYLOAD_KEYS)):
        fn = ("r6a_items.json" if name == "r6a" else "r6b_pairs.json")
        doc = json.loads((Path(args.items_dir) / fn)
                         .read_text(encoding="utf-8"))
        items = doc["items"]
        rng = np.random.default_rng(SEED)
        items = [items[i] for i in rng.permutation(len(items))]
        n_chunks = 0
        for c0 in range(0, len(items), CHUNK):
            chunk = []
            for it in items[c0:c0 + CHUNK]:
                row = {k: it[k] for k in keys}
                b = blind_id(row["item_id"])
                assert b not in id_map or id_map[b] == row["item_id"], \
                    "blind id collision"
                id_map[b] = row["item_id"]
                row["item_id"] = b
                chunk.append(row)
            p = out_dir / f"payload_{name}_chunk_{c0 // CHUNK:02d}.json"
            p.write_text(json.dumps(
                {"prompt_template": doc["prompt_template"],
                 "items": chunk}, indent=1), encoding="utf-8")
            n_chunks += 1
        print(f"{name}: {len(items)} items -> {n_chunks} payload chunks")
    map_p = Path(args.items_dir) / "blind_id_map.json"
    map_p.write_text(json.dumps(id_map, indent=1), encoding="utf-8")
    print(f"blind id map: {len(id_map)} entries -> {map_p} "
          f"(PRIVATE — never send to a judge)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
