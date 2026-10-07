from pathlib import Path

import pytest

from resourced.config import ConfigError, find_config, load_config, parse_config
from resourced.gpus import parse_gpus, parse_processes
from resourced.template import render_config

GPU = {"uuid": "GPU-1", "alias": "a", "index": 0}


def parse(**extra):
    return parse_config({"daemon": {"runtime_dir": "/tmp/x"}, "gpu": [GPU], **extra})


def test_valid_config_defaults():
    cfg = parse()
    assert cfg.gpus[0].alias == "a"
    assert cfg.probe_timeout == 10 and cfg.sample_interval == 30
    assert cfg.socket_path == Path("/tmp/x/resourced.sock")
    assert cfg.find_gpu("0").uuid == "GPU-1" and cfg.find_gpu("a").uuid == "GPU-1" and cfg.find_gpu("GPU-1")


@pytest.mark.parametrize(
    "data, message",
    [
        ({"gpu": []}, "at least one"),
        ({"gpu": [GPU], "bogus": 1}, "unknown top-level"),
        ({"gpu": [GPU, GPU]}, "duplicate GPU uuid"),
        ({"gpu": [GPU], "daemon": {"oops": 1}}, r"unknown \[daemon\]"),
        ({"gpu": [GPU], "daemon": {"probe_timeout": "x"}}, "must be a number"),
        ({"gpu": [GPU], "check": [{"name": "r", "type": "ram"}]}, "at least one"),
        ({"gpu": [GPU], "check": [{"name": "r", "type": "cpu", "warn_free_gib": 1}]}, "type"),
        ({"gpu": [GPU], "check": [{"name": "d", "type": "disk", "warn_free_gib": 1}]}, "path"),
        ({"gpu": [GPU], "check": [{"name": "r", "type": "ram", "warn_free_gib": 5, "stop_free_gib": 10}]}, "stop_free_gib must be <="),
        ({"gpu": [GPU], "check": [{"name": "r", "type": "ram", "warn_used_pct": 90, "stop_used_pct": 80}]}, "stop_used_pct must be >="),
        ({"gpu": [GPU], "check": [{"name": "r", "type": "ram", "warn_used_pct": 120}]}, "<= 100"),
    ],
)
def test_invalid_configs_give_clear_errors(data, message):
    data = {**data, "daemon": {"runtime_dir": "/tmp/x", **data.get("daemon", {})}}
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_load_config_reports_bad_toml(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[[gpu\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_lookup_precedence(tmp_path, monkeypatch):
    explicit, env_file, home = tmp_path / "a.toml", tmp_path / "b.toml", tmp_path / "cfg"
    explicit.write_text("")
    env_file.write_text("")
    (home / "resourced").mkdir(parents=True)
    (home / "resourced" / "config.toml").write_text("")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.setenv("RESOURCED_CONFIG", str(env_file))
    assert find_config(explicit) == explicit
    assert find_config() == env_file
    monkeypatch.delenv("RESOURCED_CONFIG")
    assert find_config() == home / "resourced" / "config.toml"
    with pytest.raises(ConfigError, match="not found"):
        find_config(tmp_path / "missing.toml")


def test_runtime_dir_default_uses_xdg(monkeypatch):
    monkeypatch.delenv("RESOURCED_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1")
    assert parse_config({"gpu": [GPU]}).runtime_dir == Path("/run/user/1/resourced")
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    with pytest.raises(ConfigError, match="XDG_RUNTIME_DIR"):
        parse_config({"gpu": [GPU]})


def test_example_config_and_generated_config_parse(tmp_path):
    example = Path(__file__).resolve().parent.parent / "examples" / "config.toml"
    cfg = load_config(example)
    assert len(cfg.gpus) >= 2 and cfg.checks
    generated = tmp_path / "g.toml"
    generated.write_text(render_config([{"index": 0, "uuid": "GPU-x", "name": "Test"}]).replace("[daemon]", '[daemon]\nruntime_dir = "/tmp/r"'))
    assert load_config(generated).gpus[0].alias == "gpu0"


def test_nvidia_smi_parsing():
    assert parse_gpus("0, GPU-a, NVIDIA RTX, 6000\n1, GPU-b, Other\n")[0]["name"] == "NVIDIA RTX, 6000"
    procs = parse_processes("GPU-a, 12, /usr/bin/python, 2142\nGPU-b, 7, my,odd name, [N/A]\n\njunk\n")
    assert (procs[0].uuid, procs[0].pid, procs[0].memory_mib) == ("GPU-a", 12, 2142)
    assert procs[1].name == "my,odd name" and procs[1].memory_mib is None
    assert parse_processes("") == []
