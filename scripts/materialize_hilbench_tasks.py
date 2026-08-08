"""Materialize hil-bench harbor tasks into the FLAT layout the `hil swe`
CLI needs (spec eval-bridge, decisions/022).

    python scripts/materialize_hilbench_tasks.py --tasks swe_60..swe_99 \
        --out data/hilbench_flat_sealed [--extract-scripts]

The vendored harbor_swe/swe_i/ keeps problem_statement.txt / metadata.json /
blocker_registry.json under shared/, but `hil swe <dir>` (resolve_swe_input_path
+ ask_human_server --tasks-dir) requires them at each task dir's ROOT --
run_hil_bench.py::prepare_swe_task materializes exactly this layout from HF
rows; this script materializes it from the LOCAL clone instead (no HF
download, and it can target the sealed pool, which run_hil_bench's first-N
selection cannot).

Conventions mirrored from prepare_swe_task @352d14c: instance_id = the task
dir name; log_parser=sweap_json + the SWEAP test_cmd defaulted when absent.
--extract-scripts (docker; GPU box) copies /root/run_script.sh + /root/parser.py
out of the task image so calculate_pass_at_1 can score.

Never touches the source tree; refuses to mix train and sealed pools in one
output dir (seal hygiene, decisions/018 §4).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# verbatim constants (run_hil_bench.py @352d14c)
SWEAP_TEST_CMD = (
    "bash /root/run_script.sh > /tmp/stdout.log 2> /tmp/stderr.log; "
    "python /root/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json; "
    "python -c \"print('SWEAP_JSON_START'); print(open('/tmp/output.json')."
    "read()); print('SWEAP_JSON_END')\""
)
SWEAP_LOG_PARSER = "sweap_json"

_NUM = re.compile(r"(\d+)$")


def task_number(task_id: str) -> int | None:
    m = _NUM.search(task_id)
    return int(m.group(1)) if m else None


def check_pool_hygiene(task_ids: list[str], sealed_boundary: int) -> str:
    """-> 'train' | 'sealed'; raises on a mixed selection."""
    nums = [task_number(t) for t in task_ids]
    if any(n is None for n in nums):
        raise ValueError(f"unnumbered task ids in {task_ids}")
    train = [t for t, n in zip(task_ids, nums) if n < sealed_boundary]
    sealed = [t for t, n in zip(task_ids, nums) if n >= sealed_boundary]
    if train and sealed:
        raise ValueError(
            f"selection mixes train ({train[:3]}...) and sealed "
            f"({sealed[:3]}...) pools -- materialize them into separate "
            f"output dirs (decisions/018 §4 seal rule)")
    return "sealed" if sealed else "train"


def materialize_task(src_task_dir: Path, out_dir: Path,
                     extract_scripts: bool = False,
                     scratch_dir: str | None = None) -> dict:
    shared = src_task_dir / "shared"
    statement = shared / "problem_statement.txt"
    metadata_f = shared / "metadata.json"
    registry = shared / "ask-human-data" / "blocker_registry.json"
    for f in (statement, metadata_f, registry):
        if not f.exists():
            raise FileNotFoundError(f"{src_task_dir.name}: missing {f}")

    task_id = src_task_dir.name
    dest = out_dir / task_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(statement, dest / "problem_statement.txt")
    shutil.copyfile(registry, dest / "blocker_registry.json")

    meta = json.loads(metadata_f.read_text(encoding="utf-8"))
    meta["instance_id"] = task_id            # prepare_swe_task convention
    meta.setdefault("repo_name", "app")
    meta.setdefault("base_commit", "HEAD")
    meta.setdefault("log_parser", SWEAP_LOG_PARSER)
    meta.setdefault("test_cmd", SWEAP_TEST_CMD)
    if not meta.get("image_name"):
        ref = shared / "image_ref.txt"
        if ref.exists():
            meta["image_name"] = ref.read_text(encoding="utf-8").strip()
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2),
                                        encoding="utf-8")

    status = {"task": task_id, "image": meta.get("image_name", ""),
              "scripts": "skipped"}
    if extract_scripts:
        from extract_task_context import image_available, try_load_archive

        image = meta.get("image_name", "")
        if not image_available(image):
            load_log: list[str] = []
            try_load_archive(src_task_dir, load_log, scratch_dir=scratch_dir)
        if not image_available(image):
            status["scripts"] = "image unavailable"
        else:
            got = []
            for script in ("run_script.sh", "parser.py"):
                target = dest / script
                if target.exists():
                    got.append(script)
                    continue
                r = subprocess.run(
                    ["docker", "run", "--rm", "--entrypoint", "", image,
                     "cat", f"/root/{script}"],
                    capture_output=True, text=True, check=False)
                if r.returncode == 0 and r.stdout:
                    target.write_text(r.stdout, encoding="utf-8")
                    got.append(script)
            status["scripts"] = ",".join(got) or "none extracted"
    return status


def main() -> int:
    from run_eval import parse_task_selector

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--tasks", required=True,
                    help="'swe_60..swe_99' or comma list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sealed-boundary", type=int, default=60,
                    help="swe_<n> with n >= boundary is the sealed pool")
    ap.add_argument("--extract-scripts", action="store_true",
                    help="docker-cat /root/run_script.sh + parser.py from the "
                         "task image (GPU box)")
    ap.add_argument("--scratch-dir", default=None)
    args = ap.parse_args()

    tasks_dir = Path(args.tasks_dir)
    wanted = parse_task_selector(args.tasks)
    pool = check_pool_hygiene(wanted, args.sealed_boundary)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_f = out_dir / "materialize_manifest.json"
    if manifest_f.exists():
        prior = json.loads(manifest_f.read_text(encoding="utf-8"))
        if prior.get("pool") != pool:
            raise SystemExit(
                f"{out_dir} already holds the {prior.get('pool')!r} pool; "
                f"refusing to add {pool!r} tasks (decisions/018 §4)")

    statuses = []
    for task_id in wanted:
        src = tasks_dir / task_id
        if not src.is_dir():
            print(f"{task_id}: MISSING under {tasks_dir}")
            continue
        st = materialize_task(src, out_dir, extract_scripts=args.extract_scripts,
                              scratch_dir=args.scratch_dir)
        statuses.append(st)
        print(f"{task_id}: ok (scripts: {st['scripts']})")

    manifest_f.write_text(json.dumps(
        {"pool": pool, "source": str(tasks_dir), "tasks": statuses},
        indent=1), encoding="utf-8")
    print(f"\n{len(statuses)} tasks -> {out_dir} (pool={pool})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
