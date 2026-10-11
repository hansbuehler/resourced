"""End-to-end tests: a real daemon on a temporary socket, driven through the CLI."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from conftest import UUIDS, FakeProbe, cli_command, cli_env, make_config
from resourced.cli import main
from resourced.client import Client
from resourced.daemon import Daemon


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


def test_claim_status_run_release_cycle(sock, capfd):
    assert main(["claim", "--gpus", "2", "--owner-pid", str(os.getpid()), "--task-name", "agent", "--label", "dev", "--json", "--socket", sock]) == 0
    claim = json.loads(capfd.readouterr().out)
    ref = claim["allocation"]["ref"]
    assert claim["env"]["CUDA_VISIBLE_DEVICES"] == ",".join(claim["allocation"]["gpus"])

    assert main(["status", "--json", "--socket", sock]) == 0
    status = json.loads(capfd.readouterr().out)
    assert (status["used"], status["available"]) == (2, 2)

    code = main(["run", "--allocation", ref, "-q", "--socket", sock, "--", sys.executable, "-c", "import os,sys;print(os.environ['CUDA_VISIBLE_DEVICES']);sys.exit(5)"])
    assert code == 5
    assert capfd.readouterr().out.strip() == claim["env"]["CUDA_VISIBLE_DEVICES"]
    assert Client(sock).status()["used"] == 2  # run --allocation never releases

    assert main(["env", "--allocation", ref, "--socket", sock]) == 0
    assert "export CUDA_VISIBLE_DEVICES=" in capfd.readouterr().out
    assert main(["release", "--allocation", ref, "--socket", sock]) == 0
    assert Client(sock).status()["used"] == 0
    assert main(["release", "--allocation", ref, "--socket", sock]) == 6  # unknown allocation


def test_claim_without_owner_pid_is_rejected(sock, capfd, monkeypatch):
    monkeypatch.delenv("RESOURCED_OWNER_PID", raising=False)
    assert main(["claim", "--gpus", "1", "--socket", sock]) == 2
    assert "owner" in capfd.readouterr().err
    monkeypatch.setenv("RESOURCED_OWNER_PID", str(os.getpid()))
    assert main(["claim", "--gpus", "1", "--socket", sock]) == 0


def test_adhoc_run_releases_on_exit_and_failure(sock, capfd):
    ok = main(["run", "--gpus", "1", "-q", "--socket", sock, "--", sys.executable, "-c", "import os;print(os.environ['CUDA_VISIBLE_DEVICES'])"])
    assert ok == 0
    assert capfd.readouterr().out.strip() in UUIDS
    assert Client(sock).status()["used"] == 0
    assert main(["run", "--gpus", "1", "-q", "--socket", sock, "--", sys.executable, "-c", "import sys;sys.exit(3)"]) == 3
    assert main(["run", "--gpus", "1", "-q", "--socket", sock, "--", "definitely-not-a-command"]) == 127
    assert Client(sock).status()["used"] == 0


def test_adhoc_run_never_takes_development_gpus(sock):
    assert main(["claim", "--gpus", "4", "--owner-pid", str(os.getpid()), "--socket", sock]) == 0
    assert main(["run", "--gpus", "1", "-q", "--socket", sock, "--", "true"]) == 3
    assert Client(sock).status()["used"] == 4


def test_run_requires_exactly_one_mode_and_a_command(sock):
    assert main(["run", "--socket", sock, "--", "true"]) == 2
    assert main(["run", "--gpus", "1", "--socket", sock]) == 2


def test_adhoc_run_forwards_sigterm_and_releases(sock):
    wrapper = subprocess.Popen(cli_command("run", "--gpus", "1", "-q", "--socket", sock, "--", "sleep", "60"), env=cli_env())
    try:
        wait_for(lambda: Client(sock).status()["used"] == 1)
        wait_for(lambda: Client(sock).status()["allocations"][0]["owner"])  # command PID registered
        wrapper.send_signal(signal.SIGTERM)
        assert wrapper.wait(timeout=15) == 128 + signal.SIGTERM
        assert Client(sock).status()["used"] == 0
    finally:
        if wrapper.poll() is None:
            wrapper.kill()


def test_adhoc_command_pid_is_tracked_for_verify(sock, probe):
    wrapper = subprocess.Popen(cli_command("run", "--gpus", "1", "-q", "--socket", sock, "--", "sleep", "60"), env=cli_env())
    try:
        alloc = wait_for(lambda: (Client(sock).status()["allocations"] or [{}])[0].get("owner") and Client(sock).status()["allocations"][0])
        command_pid, gpu = alloc["owner"]["pid"], alloc["gpus"][0]
        assert alloc["owner"]["kind"] == "command"
        probe.add(gpu, command_pid)
        assert Client(sock).verify()["healthy"]
        probe.add(gpu, 1)
        report = Client(sock).verify()
        assert not report["healthy"]
        assert [r["state"] for r in report["gpus"] if r["uuid"] == gpu] == ["FOREIGN PROCESS"]
    finally:
        probe.clear()  # otherwise the wrapper waits for these fake processes to leave
        wrapper.terminate()
        wrapper.wait(timeout=15)


def test_killed_wrapper_leaves_claim_until_force_release(sock, capfd):
    wrapper = subprocess.Popen(cli_command("run", "--gpus", "1", "-q", "--socket", sock, "--", "sleep", "60"), env=cli_env())
    alloc = wait_for(lambda: (Client(sock).status()["allocations"] or [{}])[0].get("owner") and Client(sock).status()["allocations"][0])
    wrapper.kill()
    wrapper.wait()
    os.kill(alloc["owner"]["pid"], signal.SIGKILL)  # the orphaned command
    held = wait_for(lambda: Client(sock).status()["allocations"][0]["holder_alive"] is False)
    assert held
    capfd.readouterr()
    assert main(["status", "--socket", sock]) == 0
    assert "DEAD" in capfd.readouterr().out
    assert main(["release", "--force", "--gpu", alloc["aliases"][0], "--reason", "test", "--socket", sock]) == 0
    assert Client(sock).status()["used"] == 0


def test_wrapper_tolerates_force_release_of_its_allocation(sock):
    wrapper = subprocess.Popen(cli_command("run", "--gpus", "1", "-q", "--socket", sock, "--", "sleep", "60"), env=cli_env())
    try:
        wait_for(lambda: (Client(sock).status()["allocations"] or [{}])[0].get("owner"))
        assert main(["release", "--force", "--gpu", "0", "--reason", "t", "--socket", sock]) == 0
        wrapper.terminate()
        assert wrapper.wait(timeout=15) == 128 + signal.SIGTERM  # no crash on the stale release
    finally:
        if wrapper.poll() is None:
            wrapper.kill()


def test_adhoc_release_warns_about_leftover_processes(sock, probe, capfd, monkeypatch):
    monkeypatch.setenv("RESOURCED_RELEASE_WAIT", "0.6")

    def leave_process():
        # a process shows up on every GPU while the command runs and never leaves
        time.sleep(0.3)
        for uuid in UUIDS:
            probe.add(uuid, 4242)

    thread = threading.Thread(target=leave_process)
    thread.start()
    code = main(["run", "--gpus", "1", "-q", "--socket", sock, "--", sys.executable, "-c", "import time;time.sleep(0.8)"])
    thread.join()
    assert code == 0
    assert "still use the GPU" in capfd.readouterr().err
    assert Client(sock).status()["used"] == 0


def test_force_release_requires_gpu_flag_semantics(sock, capfd):
    assert main(["release", "--gpu", "gpu0", "--socket", sock]) == 2
    assert main(["release", "--socket", sock]) == 2
    assert main(["release", "--force", "--gpu", "gpu0", "--socket", sock]) == 1  # not claimed


def test_verify_exit_code_and_output(sock, probe, capfd):
    assert main(["verify", "--socket", sock]) == 0
    probe.add(UUIDS[2], 4242)
    assert main(["verify", "--socket", sock]) == 1
    out = capfd.readouterr().out
    assert "UNCLAIMED USE" in out and "4242" in out


def test_free_and_status_text(sock, capfd):
    assert main(["claim", "--gpus", "1", "--owner-pid", str(os.getpid()), "--task-name", "agent-x", "--socket", sock]) == 0
    capfd.readouterr()
    assert main(["free", "--socket", sock]) == 0
    assert capfd.readouterr().out.startswith("3 of 4 GPUs available")
    assert main(["status", "--task-id", "agent-x", "--socket", sock]) == 0
    out = capfd.readouterr().out
    assert "agent-x" in out and "owner pid" in out and "allocations of task agent-x" in out


def test_daemon_restart_keeps_claims_fresh_runtime_dir_does_not(config, probe, short_tmp):
    first = Daemon(config, probe)
    first.start()
    ref = Client(config.socket_path).claim(mode="dev", gpus=1, task_name="t", owner_pid=os.getpid())["allocation"]["ref"]
    first.stop()
    assert not config.socket_path.exists()  # clean shutdown removes the socket

    second = Daemon(config, FakeProbe())
    second.start()
    try:
        assert Client(config.socket_path).get(ref)["allocation"]["ref"] == ref
    finally:
        second.stop()

    fresh = Daemon(make_config(short_tmp / "other"), FakeProbe())
    fresh.start()
    try:
        assert Client(fresh.config.socket_path).status()["used"] == 0
    finally:
        fresh.stop()


def test_second_daemon_on_same_runtime_dir_refuses(daemon, config):
    other = Daemon(config, FakeProbe())
    with pytest.raises(RuntimeError, match="already uses"):
        other.start()
    assert Client(config.socket_path).ping()  # the first one is unharmed


def test_symlinked_runtime_dir_is_refused(short_tmp):
    target = short_tmp / "real"
    target.mkdir()
    link = short_tmp / "link"
    link.symlink_to(target)
    cfg = make_config(short_tmp / "unused")
    cfg.runtime_dir = link
    with pytest.raises(RuntimeError, match="symlink"):
        Daemon(cfg, FakeProbe()).start()


def test_socket_is_private_by_default(sock):
    assert oct(os.stat(sock).st_mode & 0o777) == "0o600"


def test_malformed_and_oversized_requests(sock):
    def exchange(payload: bytes) -> dict:
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(5)
            s.connect(sock)
            s.sendall(payload)
            return json.loads(s.makefile("rb").readline())

    assert exchange(b"not json\n")["error"]["code"] == "bad_request"
    assert exchange(b"[1,2]\n")["error"]["code"] == "bad_request"
    assert exchange(b'{"v":1,"op":"ping"}\n')["ok"] is True
    assert exchange(b"x" * 70000 + b"\n")["error"]["code"] == "request_too_large"


def test_unreachable_daemon_has_helpful_message(short_tmp, capfd, monkeypatch):
    monkeypatch.delenv("RESOURCED_SOCKET", raising=False)
    assert main(["status", "--socket", str(short_tmp / "nothing.sock")]) == 5
    assert "Is the daemon running" in capfd.readouterr().err


def test_resource_stop_never_blocks_claims(short_tmp, probe, capfd):
    cfg = make_config(
        short_tmp / "run",
        checks=[
            {"name": "always-stop", "type": "disk", "path": str(short_tmp), "warn_used_pct": 0, "stop_used_pct": 0},
            {"name": "broken", "type": "disk", "path": str(short_tmp / "missing"), "warn_free_pct": 10},
        ],
    )
    d = Daemon(cfg, probe)
    d.start()
    try:
        d.monitor.sample_once()
        assert main(["resources", "--json", "--socket", str(cfg.socket_path)]) == 0
        checks = {c["name"]: c for c in json.loads(capfd.readouterr().out)["checks"]}
        assert checks["always-stop"]["health"] == "STOP" and checks["broken"]["health"] == "UNAVAILABLE"
        assert main(["claim", "--gpus", "4", "--owner-pid", str(os.getpid()), "--socket", str(cfg.socket_path)]) == 0
    finally:
        d.stop()


def test_resources_fall_back_to_local_measurement(short_tmp, capfd, monkeypatch):
    cfg_file = short_tmp / "c.toml"
    cfg_file.write_text(
        f'[daemon]\nruntime_dir = "{short_tmp}/norun"\n[[gpu]]\nuuid = "GPU-1"\n'
        '[[check]]\nname = "ram"\ntype = "ram"\nwarn_free_gib = 0\n'
    )
    monkeypatch.delenv("RESOURCED_SOCKET", raising=False)
    assert main(["resources", "--config", str(cfg_file)]) == 0
    captured = capfd.readouterr()
    assert "measuring locally" in captured.err and "ram" in captured.out


def test_init_writes_config_and_refuses_overwrite(short_tmp, monkeypatch, capfd):
    from resourced import cli

    monkeypatch.setattr(cli, "find_nvidia_smi", lambda *a: "/fake/bin/nvidia-smi")
    monkeypatch.setattr(cli.NvidiaSmiProbe, "gpus", lambda self: [{"index": 0, "uuid": "GPU-a", "name": "Fake"}])
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_tmp))
    monkeypatch.delenv("RESOURCED_RUNTIME_DIR", raising=False)
    target = short_tmp / "c.toml"
    assert main(["init", "--output", str(target)]) == 0
    text = target.read_text()
    assert 'uuid = "GPU-a"' in text
    assert 'nvidia_smi = "/fake/bin/nvidia-smi"' in text
    out = capfd.readouterr().out
    assert "sandboxed coding agent" in out and str(short_tmp / "resourced") in out and "network_access = true" in out
    assert main(["init", "--output", str(target)]) == 1
    assert main(["init", "--output", str(target), "--force"]) == 0


def test_init_sandbox_hint_skipped_without_runtime_dir(short_tmp, monkeypatch, capfd):
    from resourced import cli

    monkeypatch.setattr(cli, "find_nvidia_smi", lambda *a: "/fake/bin/nvidia-smi")
    monkeypatch.setattr(cli.NvidiaSmiProbe, "gpus", lambda self: [{"index": 0, "uuid": "GPU-a", "name": "Fake"}])
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("RESOURCED_RUNTIME_DIR", raising=False)
    assert main(["init", "--output", str(short_tmp / "c.toml")]) == 0
    assert "sandboxed coding agent" not in capfd.readouterr().out


def test_agent_instructions_mention_key_commands(capfd):
    assert main(["agent-instructions"]) == 0
    out = capfd.readouterr().out
    for text in ("gpuctl run --gpus", "--owner-pid", "gpuctl release", "gpuctl resources"):
        assert text in out


def test_init_explains_missing_nvidia_smi(short_tmp, monkeypatch, capfd):
    from resourced import cli

    monkeypatch.setattr(cli, "find_nvidia_smi", lambda *a: None)
    target = short_tmp / "c.toml"
    assert main(["init", "--output", str(target)]) == 1
    err = capfd.readouterr().err
    assert "nvidia-smi not found" in err and "nvidia_smi =" in err
    assert not target.exists()
