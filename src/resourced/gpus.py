"""GPU discovery and compute-process probing through nvidia-smi."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

log = logging.getLogger("resourced")


@dataclass(frozen=True)
class GpuProc:
    uuid: str
    pid: int
    name: str
    memory_mib: int | None


def parse_gpus(text: str) -> list[dict]:
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",", 2)]
        if len(parts) == 3 and parts[1]:
            index = int(parts[0]) if parts[0].isdigit() else None
            gpus.append({"index": index, "uuid": parts[1], "name": parts[2]})
    return gpus


def parse_processes(text: str) -> list[GpuProc]:
    procs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            head, memory = line.rsplit(",", 1)
            uuid, pid, name = (p.strip() for p in head.split(",", 2))
            procs.append(
                GpuProc(uuid, int(pid), name, int(memory) if memory.strip().isdigit() else None)
            )
        except ValueError:
            continue
    return procs


class NvidiaSmiProbe:
    """Every method returns None when the driver cannot be queried."""

    def __init__(self, executable: str = "nvidia-smi", timeout: float = 10.0):
        self.executable = executable
        self.timeout = timeout

    def _run(self, *args: str) -> str | None:
        try:
            result = subprocess.run(
                [self.executable, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("nvidia-smi probe failed: %s", exc)
            return None
        if result.returncode != 0:
            log.warning("nvidia-smi exited with %s: %s", result.returncode, result.stderr.strip())
            return None
        return result.stdout

    def gpus(self) -> list[dict] | None:
        out = self._run("--query-gpu=index,uuid,name", "--format=csv,noheader")
        return None if out is None else parse_gpus(out)

    def processes(self) -> list[GpuProc] | None:
        out = self._run(
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
        return None if out is None else parse_processes(out)
