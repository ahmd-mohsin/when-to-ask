"""Docker execution environment for v2 agent runs -- OURS (decisions/017).

One persistent container per run (started from the task's hil-bench image),
commands exec'd inside it, torn down at the end. Pure subprocess + stdlib so
the interface is trivially fakeable in tests; nothing here touches the model.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"[timeout after {timeout}s]"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


@dataclass
class DockerTaskEnv:
    """`with DockerTaskEnv(image, name) as env: env.execute("ls")`"""

    image: str
    name: str
    workdir: str = "/app"
    exec_timeout: int = 120
    started: bool = field(default=False, init=False)

    def start(self) -> None:
        _run(["docker", "rm", "-f", self.name], timeout=30)  # stale container
        code, out = _run(["docker", "run", "-d", "--rm", "--name", self.name,
                          "--entrypoint", "sh", self.image, "-c", "sleep infinity"],
                         timeout=120)
        if code != 0:
            raise RuntimeError(f"container start failed for {self.image}: {out[-400:]}")
        self.started = True

    def execute(self, command: str) -> tuple[int, str]:
        """Run one shell command in the container's workdir."""
        if not self.started:
            raise RuntimeError("env not started")
        return _run(["docker", "exec", self.name, "sh", "-lc",
                     f"cd {self.workdir} 2>/dev/null; {command}"],
                    timeout=self.exec_timeout)

    def stop(self) -> None:
        if self.started:
            _run(["docker", "rm", "-f", self.name], timeout=60)
            self.started = False

    def __enter__(self) -> "DockerTaskEnv":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
