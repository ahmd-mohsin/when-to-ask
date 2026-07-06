"""Extract PRE-PATCH source context from hil-bench task docker images (AWS box).

    python scripts/extract_task_context.py --tasks-dir third_party/hil-bench/harbor_swe \
        --n-tasks 20 --out data/task_context

For each task: read the gold patch ONLY for its touched file paths (tests/
changelogs/docs excluded -- test names encode expected behaviour), then copy
the CURRENT (pre-patch) content of those files out of the task's docker image.
Pre-patch files cannot contain the patch's added lines by construction, so
this grounding is leak-free w.r.t. the gold change; residual risk (a resolution
token that already existed pre-patch) is accepted and noted in ADR 012.

Per task writes: data/task_context/<task>/CONTEXT_MANIFEST.txt (+ files).
Graceful degradation, always logged in the manifest header:
  mode=docker      pre-patch file contents extracted
  mode=paths-only  docker/image unavailable -> only file paths + function names
                   (from hunk headers) are provided as context
  mode=none        no usable patch -> instruction-only collection
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.collect_utils import patch_touched_files  # noqa: E402


def sh(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # docker missing, timeout, ...
        return 1, f"{type(e).__name__}: {e}"


def image_available(image: str) -> bool:
    code, _ = sh(["docker", "image", "inspect", image], timeout=30)
    return code == 0


def try_load_archive(task_dir: Path, log: list[str]) -> str | None:
    """Some hil-bench releases ship an image archive pointer; try it."""
    arch = task_dir / "shared" / "image_archive.json"
    if not arch.exists():
        return None
    try:
        info = json.loads(arch.read_text(encoding="utf-8"))
    except Exception as e:
        log.append(f"image_archive.json unreadable: {e}")
        return None
    url = info.get("url") or info.get("archive_url")
    if not url:
        log.append(f"image_archive.json has no url field (keys: {list(info)})")
        return None
    log.append(f"pulling image archive: {url}")
    code, out = sh(["bash", "-c", f"curl -fsSL '{url}' | docker load"], timeout=1800)
    log.append(out.strip()[-500:])
    return None if code != 0 else (info.get("image") or None)


def hunk_functions(patch_text: str, path: str) -> list[str]:
    """Function names from @@ hunk headers of one file (paths-only fallback)."""
    out, in_file = [], False
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            in_file = line.endswith("b/" + path)
        elif in_file and line.startswith("@@"):
            tail = line.split("@@")[-1].strip()
            if tail and tail not in out:
                out.append(tail)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--out", default="data/task_context")
    ap.add_argument("--max-files", type=int, default=3)
    args = ap.parse_args()

    tasks_dir, out_root = Path(args.tasks_dir), Path(args.out)
    done = 0
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        if done >= args.n_tasks:
            break
        patch_file = task_dir / "baseline" / "solution" / "ground_truth.patch"
        instr = task_dir / "baseline" / "instruction.md"
        if not instr.exists():
            continue
        done += 1
        task = task_dir.name
        out_dir = out_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        log: list[str] = []

        patch_text = patch_file.read_text(encoding="utf-8", errors="replace") \
            if patch_file.exists() else ""
        paths = patch_touched_files(patch_text)[: args.max_files]
        image = (task_dir / "shared" / "image_ref.txt").read_text(encoding="utf-8").strip() \
            if (task_dir / "shared" / "image_ref.txt").exists() else ""

        mode = "none"
        entries: list[tuple[str, str]] = []
        if paths:
            mode = "paths-only"
            if image and not image_available(image):
                try_load_archive(task_dir, log)
            if image and image_available(image):
                mode = "docker"
                for i, path in enumerate(paths):
                    code, out = sh(["docker", "run", "--rm", "--entrypoint", "cat",
                                    image, f"/app/{path}"], timeout=180)
                    if code != 0:  # repo root may differ; try a find
                        code2, found = sh(["docker", "run", "--rm", "--entrypoint",
                                           "sh", image, "-c",
                                           f"find / -path '*/{path}' -not -path '*/node_modules/*' 2>/dev/null | head -1"],
                                          timeout=180)
                        found = found.strip().splitlines()[0].strip() if found.strip() else ""
                        if code2 == 0 and found:
                            code, out = sh(["docker", "run", "--rm", "--entrypoint",
                                            "cat", image, found], timeout=180)
                    if code == 0 and out.strip():
                        local = f"ctx_{i}_{Path(path).name}"
                        (out_dir / local).write_text(out, encoding="utf-8")
                        entries.append((path, local))
                        log.append(f"extracted {path} ({len(out)} chars)")
                    else:
                        log.append(f"FAILED to extract {path}: {out.strip()[:200]}")
                if not entries:
                    mode = "paths-only"
            if mode == "paths-only":
                lines = [f"# touched file: {p}" for p in paths]
                for p in paths:
                    for fn in hunk_functions(patch_text, p):
                        lines.append(f"#   function: {fn}")
                (out_dir / "ctx_paths.txt").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
                entries.append(("(file/function map from task metadata)", "ctx_paths.txt"))

        manifest = [f"# task={task} mode={mode} image={image}",
                    *(f"# {line}" for line in log),
                    *(f"{repo}\t{local}" for repo, local in entries)]
        (out_dir / "CONTEXT_MANIFEST.txt").write_text("\n".join(manifest) + "\n",
                                                      encoding="utf-8")
        print(f"{task}: mode={mode}, {len(entries)} context entries")
    print(f"\n{done} tasks processed -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
