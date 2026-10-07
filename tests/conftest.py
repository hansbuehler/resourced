import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from resourced.config import Config, parse_config
from resourced.core import Core
from resourced.daemon import Daemon
from resourced.gpus import GpuProc

UUIDS = [f"GPU-0000000{i}-aaaa-bbbb-cccc-ddddeeeeffff" for i in range(4)]


class FakeProbe:
    """Stands in for nvidia-smi."""

    def __init__(self, uuids=UUIDS):
        self.uuids = list(uuids)
        self.procs: list[GpuProc] = []
        self.up = True
        self.missing: set[str] = set()

    def gpus(self):
        if not self.up:
            return None
        return [{"index": i, "uuid": u, "name": "Fake GPU"} for i, u in enumerate(self.uuids) if u not in self.missing]

    def processes(self):
        return list(self.procs) if self.up else None

    def add(self, uuid, pid, name="python"):
        self.procs.append(GpuProc(uuid, pid, name, 100))

    def clear(self):
        self.procs.clear()


def make_config(runtime_dir: Path, n: int = 4, checks: list[dict] | None = None, **daemon) -> Config:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "daemon": {"runtime_dir": str(runtime_dir), "probe_timeout": 2, **daemon},
        "gpu": [{"uuid": UUIDS[i], "alias": f"gpu{i}", "index": i} for i in range(n)],
    }
    if checks:
        data["check"] = checks
    return parse_config(data)


PEER = {"pid": os.getpid(), "uid": os.getuid(), "user": "tester"}


@pytest.fixture
def short_tmp():
    # unix socket paths are limited to ~100 characters
    path = Path(tempfile.mkdtemp(prefix="rd"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def probe():
    return FakeProbe()


@pytest.fixture
def config(short_tmp):
    return make_config(short_tmp / "run")


@pytest.fixture
def core(config, probe):
    return Core(config, probe, boot_id="boot-1")


@pytest.fixture
def daemon(config, probe):
    d = Daemon(config, probe)
    d.start()
    yield d
    d.stop()


@pytest.fixture
def sock(daemon, config):
    return str(config.socket_path)


def claim_dev(core, gpus=1, **kw):
    params = {"mode": "dev", "gpus": gpus, "task_name": "t", "task_id": "t1", "owner_pid": os.getpid(), **kw}
    return core.claim(params, PEER)["allocation"]


def claim_adhoc(core, gpus=1, **kw):
    params = {"mode": "adhoc", "gpus": gpus, "task_name": "t", "task_id": "t1", **kw}
    return core.claim(params, PEER)["allocation"]


def cli_env():
    env = dict(os.environ)
    root = Path(__file__).resolve().parent.parent / "src"
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def cli_command(*args):
    return [sys.executable, "-m", "resourced.cli", *args]
