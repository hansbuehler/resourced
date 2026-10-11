"""GPU discovery and compute-process probing through nvidia-smi."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger("resourced")

# Services get a minimal PATH, so look in the places where drivers install nvidia-smi
# (WSL keeps it in /usr/lib/wsl/lib, which only login shells have on PATH).
KNOWN_NVIDIA_SMI_PATHS = (
    "/usr/lib/wsl/lib/nvidia-smi",
    "/usr/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
)


def find_nvidia_smi(configured: str = "nvidia-smi") -> str | None:
    """Absolute path of a usable nvidia-smi, or None. An explicit path is never second-guessed."""
    if os.sep in configured:
        return configured if os.path.isfile(configured) and os.access(configured, os.X_OK) else None
    found = shutil.which(configured)
    if found:
        return found
    if configured == "nvidia-smi":
        for candidate in KNOWN_NVIDIA_SMI_PATHS:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def nvidia_smi_missing_message(configured: str = "nvidia-smi") -> str:
    if os.sep in configured:
        return f"nvidia-smi not found or not executable at the configured path {configured}"
    return (
        f"{configured} not found on PATH or in {', '.join(KNOWN_NVIDIA_SMI_PATHS)}. "
        "Install the NVIDIA driver, or set nvidia_smi = \"/full/path/to/nvidia-smi\" under [daemon] in the config"
    )


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
        self.last_error: str | None = None

    def _fail(self, message: str) -> None:
        # Log on every change, not every poll, so a permanent failure does not flood the journal.
        if message != self.last_error:
            log.warning("%s", message)
        self.last_error = message

    def _run(self, *args: str) -> str | None:
        path = find_nvidia_smi(self.executable)
        if path is None:
            self._fail(nvidia_smi_missing_message(self.executable))
            return None
        try:
            result = subprocess.run(
                [path, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._fail(f"nvidia-smi probe failed: {exc}")
            return None
        if result.returncode != 0:
            self._fail(f"nvidia-smi exited with {result.returncode}: {result.stderr.strip()}")
            return None
        self.last_error = None
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
