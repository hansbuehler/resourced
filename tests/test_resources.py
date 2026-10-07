import time

import pytest

from resourced import resources
from resourced.config import CheckConfig
from resourced.resources import GIB, evaluate, sample_check

RAM = CheckConfig("ram", "ram", None, {"warn_free_gib": 200, "stop_free_gib": 100})
USED = CheckConfig("tmp", "disk", "/tmp", {"warn_used_pct": 80, "stop_used_pct": 90})
FREE_PCT = CheckConfig("data", "disk", "/data", {"warn_free_pct": 20, "stop_free_pct": 10})


def usage(free_gib=None, free_pct=None, used_pct=None):
    cur = {"free_gib": free_gib or 0.0, "free_pct": free_pct or 0.0, "used_pct": used_pct or 0.0, "total_gib": 1000.0}
    return cur


@pytest.mark.parametrize(
    "free, expected",
    [(200.0001, "OK"), (200.0, "WARN"), (150, "WARN"), (100.0001, "WARN"), (100.0, "STOP"), (0, "STOP")],
)
def test_free_gib_limits_are_inclusive(free, expected):
    assert evaluate(RAM, usage(free_gib=free))[0] == expected


@pytest.mark.parametrize("used, expected", [(79.99, "OK"), (80, "WARN"), (89.99, "WARN"), (90, "STOP"), (100, "STOP")])
def test_used_pct_limits_are_inclusive(used, expected):
    assert evaluate(USED, usage(used_pct=used))[0] == expected


@pytest.mark.parametrize("free, expected", [(20.01, "OK"), (20, "WARN"), (10.01, "WARN"), (10, "STOP")])
def test_free_pct_limits_are_inclusive(free, expected):
    assert evaluate(FREE_PCT, usage(free_pct=free))[0] == expected


def test_evaluate_without_measurement_triggers_nothing():
    health, limits = evaluate(RAM, None)
    assert health == "OK" and all(lim["triggered"] is None for lim in limits)
    assert {lim["level"] for lim in limits} == {"WARN", "STOP"}


def test_gib_conversion():
    assert resources._usage(2 * GIB, 8 * GIB)["free_gib"] == 2.0
    assert resources._usage(2 * GIB, 8 * GIB)["free_pct"] == 25.0


def test_real_ram_and_disk_measurements(tmp_path):
    ram = resources.measure_ram()
    assert 0 < ram["free_gib"] <= ram["total_gib"]
    disk = resources.measure_disk(str(tmp_path))
    assert disk["free_pct"] + disk["used_pct"] == pytest.approx(100.0)


def test_missing_path_is_unavailable_not_healthy(tmp_path):
    check = CheckConfig("gone", "disk", str(tmp_path / "nope"), {"warn_free_pct": 20})
    result = sample_check(check, 2)
    assert result["health"] == "UNAVAILABLE" and result["current"] is None
    assert result["limits"][0]["value"] == 20  # limits stay visible


def test_probe_timeout_reports_error(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "measure_disk", lambda path: time.sleep(2))
    check = CheckConfig("slow", "disk", str(tmp_path), {"warn_free_pct": 20})
    start = time.monotonic()
    result = sample_check(check, 0.2)
    assert time.monotonic() - start < 1.5
    assert result["health"] == "ERROR" and "did not finish" in result["error"]


def test_monitor_tracks_previous_and_writes_snapshot(short_tmp):
    from conftest import make_config

    config = make_config(short_tmp / "run", checks=[{"name": "ram", "type": "ram", "warn_free_gib": 0}])
    monitor = resources.ResourceMonitor(config)
    monitor.sample_once()
    second = monitor.sample_once()
    assert second["checks"][0]["previous"] is not None
    assert config.resources_path.exists()
