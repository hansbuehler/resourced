"""NvidiaSmiProbe against a fake nvidia-smi executable."""

import os
import stat
import time

import pytest

from conftest import UUIDS, make_config
from resourced.client import Client
from resourced.daemon import Daemon
from resourced.gpus import NvidiaSmiProbe

SCRIPT = """#!/bin/sh
case "$1" in
  --query-gpu*) {gpus} ;;
  --query-compute-apps*) {apps} ;;
esac
{extra}
"""


def fake_smi(path, gpus='printf "0, GPU-a, Fake\\n"', apps="true", extra="", exit_code=0):
    path.write_text(SCRIPT.format(gpus=gpus, apps=apps, extra=extra + f"\nexit {exit_code}"))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_probe_parses_output(short_tmp):
    exe = fake_smi(short_tmp / "smi", apps='printf "GPU-a, 77, python, 100\\n"')
    probe = NvidiaSmiProbe(exe, 5)
    assert probe.gpus() == [{"index": 0, "uuid": "GPU-a", "name": "Fake"}]
    assert [(p.uuid, p.pid) for p in probe.processes()] == [("GPU-a", 77)]


def test_probe_failures_return_none(short_tmp):
    assert NvidiaSmiProbe(str(short_tmp / "missing"), 1).processes() is None
    assert NvidiaSmiProbe(fake_smi(short_tmp / "bad", exit_code=3), 1).processes() is None


def test_probe_timeout_returns_none_quickly(short_tmp):
    exe = fake_smi(short_tmp / "slow", extra="sleep 5")
    start = time.monotonic()
    assert NvidiaSmiProbe(exe, 0.3).processes() is None
    assert time.monotonic() - start < 3


def test_daemon_with_fake_nvidia_smi_end_to_end(short_tmp):
    apps = short_tmp / "apps"
    apps.write_text("")
    exe = fake_smi(
        short_tmp / "smi",
        gpus="printf '" + "".join(f"{i}, {u}, Fake\\n" for i, u in enumerate(UUIDS)) + "'",
        apps=f"cat {apps}",
    )
    config = make_config(short_tmp / "run", nvidia_smi=exe)
    daemon = Daemon(config)
    daemon.start()
    try:
        client = Client(config.socket_path)
        alloc = client.claim(mode="dev", gpus=1, task_name="t", owner_pid=os.getpid())["allocation"]
        apps.write_text(f"{alloc['gpus'][0]}, {os.getpid()}, python, 10\n")
        assert client.verify()["healthy"]
        with pytest.raises(Exception, match="still in use"):
            client.release(alloc["ref"])
        apps.write_text("")
        assert client.release(alloc["ref"])["released"] == alloc["gpus"]
    finally:
        daemon.stop()
