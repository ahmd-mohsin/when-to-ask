"""Restore the hil-bench task docker images after an ephemeral-NVMe wipe.

Infrastructure restoration, not an experiment -- it produces no paper number.
The box's instance store is rebuilt empty on every stop/start, which destroys
the docker/containerd roots and every task image with them. `docker images`
then lists the old tags at 0B and `docker run` fails with `blob not found`,
which reads like an image problem and is not.

The per-task Dockerfiles are `FROM hilbench-swe:<attempt_id>` -- they wrap a
prebuilt base image rather than building one, so the images cannot be rebuilt
from source. They are restored from the PUBLIC HF bucket named in each task's
`shared/image_archive.json` (no token required; `hf buckets` reads it
anonymously).

Task set is derived EXACTLY as scripts/collect_v2.py derives it -- sorted dir
order, filtered through the frozen class artifact, truncated to --n-tasks --
so the sealed pool (swe_60+) is excluded by construction, not by a filter
that could drift.

    python scripts/restore_hilbench_images.py                 # 60 tasks
    python scripts/restore_hilbench_images.py --dry-run

Resumable: an image already present in docker is skipped, and each archive is
deleted after a successful load, so peak extra disk is one archive.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def image_present(ref: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", ref],
                          capture_output=True).returncode == 0


def eligible_tasks(tasks_dir: Path, classes: Path, n_tasks: int) -> list[Path]:
    """Verbatim the eligibility walk in collect_v2.main()."""
    from collect_v2 import artifact_task_ids
    class_tasks = artifact_task_ids(classes)
    out: list[Path] = []
    for d in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        if d.name not in class_tasks:
            continue
        if not (d / "baseline" / "instruction.md").exists():
            continue
        if not (d / "shared" / "image_ref.txt").exists():
            continue
        out.append(d)
        if len(out) >= n_tasks:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--work-dir", default="/opt/dlami/nvme/hilbench_images")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = eligible_tasks(Path(args.tasks_dir), Path(args.classes),
                           args.n_tasks)
    log(f"{len(tasks)} eligible tasks: {tasks[0].name} ... {tasks[-1].name}")

    # sealed-pool assertion: cheap, and the one mistake that must never happen
    sealed = [d.name for d in tasks
              if d.name.startswith("swe_")
              and d.name.split("_")[-1].isdigit()
              and int(d.name.split("_")[-1]) >= 60]
    if sealed:
        log(f"ABORT: sealed-pool tasks in list: {sealed}")
        return 2

    # distinct images (several tasks can share one base image)
    need: dict[str, dict] = {}
    for d in tasks:
        a = json.loads((d / "shared" / "image_archive.json").read_text())
        need.setdefault(a["local_image_ref"], {
            "artifact_path": a["artifact_path"],
            "bytes": a["artifact_bytes"],
            "bucket": a.get("hf_bucket", ""),
            "tasks": [],
        })["tasks"].append(d.name)

    todo = {r: v for r, v in need.items() if not image_present(r)}
    have = len(need) - len(todo)
    gb = sum(v["bytes"] for v in todo.values()) / 1e9
    log(f"{len(need)} distinct images | {have} already present | "
        f"{len(todo)} to fetch ({gb:.1f} GB)")
    if args.dry_run or not todo:
        return 0

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for i, (ref, v) in enumerate(sorted(todo.items()), 1):
        src = f"hf://buckets/{v['bucket']}/{v['artifact_path']}"
        dst = work / Path(v["artifact_path"]).name
        log(f"[{i}/{len(todo)}] {ref}  ({v['bytes']/1e9:.2f} GB)  "
            f"tasks={','.join(v['tasks'])}")
        try:
            if not dst.exists():
                subprocess.run(["hf", "buckets", "cp", src, str(dst)],
                               check=True)
            subprocess.run(f"zstd -dc {dst} | docker load", shell=True,
                           check=True)
            if not image_present(ref):
                raise RuntimeError(f"loaded but not present: {ref}")
            ok += 1
        except Exception as exc:
            log(f"    FAILED: {type(exc).__name__}: {str(exc)[:200]}")
            failed.append(ref)
        finally:
            dst.unlink(missing_ok=True)

    log(f"done: {ok} loaded, {len(failed)} failed")
    if failed:
        log("failed refs: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
