import shutil
import subprocess
from pathlib import Path

import pytest

from resourced import service
from resourced.cli import main

CONFIG = Path("/etc/resourced/config.toml")


def test_user_unit_rendering(tmp_path):
    unit = service.render_unit("user", "/opt/venv/bin/python", tmp_path / "c.toml")
    assert f"ExecStart=/opt/venv/bin/python -m resourced.daemon --config {tmp_path}/c.toml" in unit
    assert "WantedBy=default.target" in unit and "Restart=on-failure" in unit
    assert "PrivateTmp" not in unit


def test_system_unit_rendering(tmp_path):
    unit = service.render_unit("system", "/opt/venv/bin/python", CONFIG, "gpuusers")
    for expected in (
        "User=resourced",
        "SupplementaryGroups=gpuusers",
        "RuntimeDirectory=resourced",
        "RuntimeDirectoryPreserve=yes",
        "Environment=RESOURCED_RUNTIME_DIR=/run/resourced",
        "WantedBy=multi-user.target",
    ):
        assert expected in unit
    assert "PrivateTmp" not in unit
    assert "SupplementaryGroups" not in service.render_unit("system", "/x/python", CONFIG, None)


def test_paths_with_spaces_are_quoted(tmp_path):
    unit = service.render_unit("user", "/my env/bin/python", tmp_path / "a b.toml")
    assert '"/my env/bin/python"' in unit and f'"{tmp_path}/a b.toml"' in unit


@pytest.mark.skipif(not shutil.which("systemd-analyze"), reason="systemd-analyze not available")
@pytest.mark.parametrize("mode", ["user", "system"])
def test_units_pass_systemd_verify(mode, tmp_path):
    path = tmp_path / "resourced.service"
    import sys

    path.write_text(service.render_unit(mode, sys.executable, tmp_path / "c.toml"))
    command = ["systemd-analyze", *(["--user"] if mode == "user" else []), "verify", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    problems = [
        line for line in (result.stdout + result.stderr).splitlines()
        if line.strip() and "resourced" in line and "Failed to resolve user" not in line and "c.toml" not in line
    ]
    assert not problems, problems


def test_service_enable_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    config = tmp_path / "c.toml"
    config.write_text("")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert main(["service", "enable", "--user", "--config", str(config), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "systemctl --user enable --now resourced.service" in out
    assert not (tmp_path / "xdg").exists()


def test_service_enable_needs_a_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "none"))
    monkeypatch.delenv("RESOURCED_CONFIG", raising=False)
    assert main(["service", "enable", "--user", "--dry-run"]) == 2
    assert "gpuctl init" in capsys.readouterr().err
