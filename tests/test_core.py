import os
import subprocess
import threading
from dataclasses import replace

import pytest

from conftest import PEER, UUIDS, claim_adhoc, claim_dev
from resourced.core import Core, ServiceError


def error_code(fn, *args, **kwargs):
    with pytest.raises(ServiceError) as info:
        fn(*args, **kwargs)
    return info.value.code


def test_concurrent_single_claims_get_distinct_gpus(core):
    results, errors = [], []

    def worker():
        try:
            results.append(claim_adhoc(core)["gpus"][0])
        except ServiceError as exc:
            errors.append(exc.code)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(results) == 4 and len(set(results)) == 4
    assert errors == ["not_enough_gpus", "not_enough_gpus"]


def test_multi_gpu_claim_is_all_or_nothing(core):
    claim_adhoc(core, gpus=3)
    assert error_code(claim_adhoc, core, gpus=2) == "not_enough_gpus"
    assert core.status({}, PEER)["available"] == 1
    assert len(claim_adhoc(core, gpus=1)["gpus"]) == 1


def test_same_task_parallel_claims_are_independent(core):
    a, b = claim_adhoc(core), claim_adhoc(core)
    assert a["ref"] != b["ref"] and set(a["gpus"]).isdisjoint(b["gpus"])


def test_stale_allocation_cannot_release_newer_claim(core):
    first = claim_dev(core)
    gpu = first["aliases"][0]
    core.release({"gpu": gpu, "force": True, "expect_ref": first["ref"]}, PEER)
    second = claim_dev(core)
    assert second["gpus"] == first["gpus"]
    assert error_code(core.release, {"ref": first["ref"]}, PEER) == "unknown_allocation"
    assert error_code(core.release, {"gpu": gpu, "force": True, "expect_ref": first["ref"]}, PEER) == "stale_allocation"
    assert core.status({}, PEER)["used"] == 1


def test_dev_claim_requires_live_owner_pid(core):
    base = {"mode": "dev", "gpus": 1, "task_name": "t"}
    assert error_code(core.claim, base, PEER) == "owner_pid_required"
    assert error_code(core.claim, {**base, "owner_pid": 2**22 + 12345}, PEER) == "owner_pid_dead"


def test_claim_skips_gpu_with_unknown_process_and_reports_error(core, probe, caplog):
    probe.add(UUIDS[0], 4242)
    alloc = claim_adhoc(core)
    assert alloc["gpus"] == [UUIDS[1]]
    assert any("not claimed but has running processes" in r.message for r in caplog.records)


def test_claim_fails_listing_pids_when_too_few_clean_gpus(core, probe):
    for uuid in UUIDS[:3]:
        probe.add(uuid, 4242)
    with pytest.raises(ServiceError) as info:
        claim_adhoc(core, gpus=2)
    assert info.value.code == "gpu_busy"
    assert "4242" in info.value.message
    assert core.status({}, PEER)["used"] == 0


def test_probe_unavailable_never_blocks_claim_or_release(core, probe):
    probe.up = False
    result = core.claim({"mode": "adhoc", "gpus": 1, "task_name": "t"}, PEER)
    assert result["warnings"]
    released = core.release({"ref": result["allocation"]["ref"]}, PEER)
    assert released["probe"] == "unknown"


def test_release_refused_while_processes_remain(core, probe):
    alloc = claim_dev(core)
    probe.add(alloc["gpus"][0], 4242)
    assert error_code(core.release, {"ref": alloc["ref"]}, PEER) == "gpu_busy"
    assert core.status({}, PEER)["used"] == 1
    forced = core.release({"ref": alloc["ref"], "force": True}, PEER)
    assert forced["released"] == alloc["gpus"]


def test_adhoc_release_warns_about_leftovers_but_releases(core, probe):
    alloc = claim_adhoc(core)
    probe.add(alloc["gpus"][0], 4242)
    result = core.release({"ref": alloc["ref"], "on_busy": "warn", "wait": 0}, PEER)
    assert [p["pid"] for p in result["leftover"]] == [4242]
    assert core.status({}, PEER)["used"] == 0


def test_release_waits_for_processes_to_exit(core, probe):
    alloc = claim_adhoc(core)
    probe.add(alloc["gpus"][0], 4242)
    threading.Timer(0.3, probe.clear).start()
    result = core.release({"ref": alloc["ref"], "on_busy": "warn", "wait": 5}, PEER)
    assert result["leftover"] == []


def test_released_gpu_has_no_process_check_passes(core, probe):
    alloc = claim_dev(core)
    core.release({"ref": alloc["ref"]}, PEER)
    assert core.verify({}, PEER)["healthy"]


def test_claims_survive_restart_but_not_reboot(config, probe):
    first = Core(config, probe, boot_id="boot-1")
    claim_dev(first)
    assert len(Core(config, probe, boot_id="boot-1").allocations) == 1
    assert Core(config, probe, boot_id="boot-2").allocations == {}


def test_corrupt_claims_file_is_set_aside(config, probe):
    config.claims_path.write_text("{not json")
    assert Core(config, probe, boot_id="b").allocations == {}
    assert config.claims_path.with_suffix(".json.corrupt").exists()


def test_write_failure_surfaces_and_keeps_state(core, monkeypatch):
    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    assert error_code(claim_adhoc, core) == "state_write_failed"
    assert core.allocations == {}


def test_status_lists_counts_and_callers_allocations(core):
    claim_dev(core, task_id="mine")
    claim_adhoc(core, task_id="other")
    status = core.status({"task_id": "mine"}, PEER)
    assert (status["total"], status["used"], status["available"]) == (4, 2, 2)
    assert [a["task_id"] for a in status["mine"]] == ["mine"]


def test_missing_gpu_is_reported_not_fatal(core, probe):
    probe.missing.add(UUIDS[3])
    status = core.status({}, PEER)
    assert [r["present"] for r in status["gpus"]] == [True, True, True, False]
    assert status["available"] == 3


def test_verify_classifies_states(core, probe):
    child = subprocess.Popen(["sleep", "30"])
    try:
        alloc = claim_dev(core, gpus=2)  # owner is this test process; child descends from it
        mine, idle = alloc["gpus"]
        free = [u for u in UUIDS if u not in alloc["gpus"]]
        probe.add(mine, child.pid)
        probe.add(free[0], 4242)
        states = {r["uuid"]: r["state"] for r in core.verify({}, PEER)["gpus"]}
        assert states[mine] == "OK"
        assert states[idle] == "IDLE CLAIM"
        assert states[free[0]] == "UNCLAIMED USE"
        assert states[free[1]] == "OK"
        assert not core.verify({}, PEER)["healthy"]

        probe.clear()
        probe.add(mine, 1)  # PID 1 is not a descendant of the owner
        report = core.verify({}, PEER)
        assert {r["uuid"]: r["state"] for r in report["gpus"]}[mine] == "FOREIGN PROCESS"
        assert not report["healthy"]
    finally:
        child.kill()
        child.wait()


def test_owner_pid_matching_includes_owner_and_guards_pid_reuse(core, probe):
    alloc = claim_dev(core)
    gpu = alloc["gpus"][0]
    probe.add(gpu, os.getpid())
    assert core.verify({}, PEER)["gpus"][0]["state"] == "OK"
    # a different start time means the PID was reused by another process
    stored = core.allocations[alloc["ref"]]
    core.allocations[alloc["ref"]] = replace(stored, pid_start=stored.pid_start + 1)
    assert core.verify({}, PEER)["gpus"][0]["state"] == "FOREIGN PROCESS"


def test_status_flags_dead_owner(core):
    child = subprocess.Popen(["sleep", "30"])
    alloc = core.claim({"mode": "dev", "gpus": 1, "task_name": "t", "owner_pid": child.pid}, PEER)["allocation"]
    assert alloc["owner"]["alive"] is True
    child.kill()
    child.wait()
    described = core.status({}, PEER)["allocations"][0]
    assert described["owner"]["alive"] is False and described["holder_alive"] is False


def test_force_release_by_gpu_alias_index_uuid(core):
    a = claim_dev(core)
    result = core.release({"gpu": "0", "force": True, "reason": "test"}, PEER)
    assert result["released"] == a["gpus"]
    assert error_code(core.release, {"gpu": "gpu0", "force": True}, PEER) == "not_claimed"
    assert error_code(core.release, {"gpu": "nope", "force": True}, PEER) == "unknown_gpu"


def test_partial_force_release_keeps_rest_of_allocation(core):
    alloc = claim_adhoc(core, gpus=2)
    core.release({"gpu": alloc["aliases"][0], "force": True}, PEER)
    remaining = core.status({}, PEER)["allocations"][0]
    assert remaining["gpus"] == [alloc["gpus"][1]]
    core.release({"ref": alloc["ref"]}, PEER)
    assert core.status({}, PEER)["used"] == 0


def test_handle_rejects_bad_requests(core):
    assert core.handle({"v": 99, "op": "ping"}, PEER)["error"]["code"] == "version_mismatch"
    assert core.handle({"v": 1, "op": "nope"}, PEER)["error"]["code"] == "bad_request"
    assert core.handle({"v": 1, "op": "claim", "params": []}, PEER)["error"]["code"] == "bad_request"
    assert core.handle({"v": 1, "op": "claim", "params": {"mode": "adhoc", "gpus": "x"}}, PEER)["error"]["code"] == "bad_request"
    assert core.handle({"v": 1, "op": "ping"}, PEER)["ok"] is True
