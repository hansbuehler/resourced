"""systemd unit rendering and the `gpuctl service` helper."""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from .config import SYSTEM_CONFIG, ConfigError, find_config

UNIT_NAME = "resourced.service"
SYSTEM_USER = "resourced"

USER_UNIT = """\
[Unit]
Description=resourced: GPU allocation and resource limits

[Service]
Type=simple
ExecStart={exec_start}
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
"""

SYSTEM_UNIT = """\
[Unit]
Description=resourced: GPU allocation and resource limits
After=local-fs.target

[Service]
Type=simple
User={user}
{supplementary}Environment=RESOURCED_RUNTIME_DIR=/run/resourced
ExecStart={exec_start}
RuntimeDirectory=resourced
RuntimeDirectoryMode=0755
RuntimeDirectoryPreserve=yes
Restart=on-failure
RestartSec=2
NoNewPrivileges=yes
ProtectSystem=full

[Install]
WantedBy=multi-user.target
"""


def _quote(value: str) -> str:
    return f'"{value}"' if any(c.isspace() for c in value) else value


def socket_group_of(config_path: Path) -> str | None:
    try:
        group = tomllib.loads(config_path.read_text()).get("daemon", {}).get("socket_group")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return group if isinstance(group, str) and group else None


def render_unit(mode: str, python: str, config_path: Path, socket_group: str | None = None) -> str:
    exec_start = f"{_quote(python)} -m resourced.daemon --config {_quote(str(config_path))}"
    if mode == "user":
        return USER_UNIT.format(exec_start=exec_start)
    supplementary = f"SupplementaryGroups={socket_group}\n" if socket_group else ""
    return SYSTEM_UNIT.format(user=SYSTEM_USER, supplementary=supplementary, exec_start=exec_start)


def unit_path(mode: str) -> Path:
    if mode == "system":
        return Path("/etc/systemd/system") / UNIT_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user" / UNIT_NAME


def _systemctl(mode: str, *args: str) -> list[str]:
    return ["systemctl", *(["--user"] if mode == "user" else []), *args]


def _run(command: list[str], dry_run: bool, out) -> int:
    out("+ " + " ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command).returncode


def _resolve_config(mode: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path
    if mode == "system":
        if not SYSTEM_CONFIG.is_file():
            raise ConfigError(f"{SYSTEM_CONFIG} not found. Create it with `sudo gpuctl init --system`.")
        return SYSTEM_CONFIG
    return find_config().resolve()


def enable(mode: str, config: str | None, dry_run: bool, out=print) -> int:
    config_path = _resolve_config(mode, config)
    python = sys.executable
    if mode == "system":
        if os.geteuid() != 0 and not dry_run:
            out("system installation needs root: re-run with sudo (and the same Python), or use --user.")
            return 1
        if python.startswith("/home/") or python.startswith("/root/"):
            out(
                f"warning: {python} lives in a home directory; the '{SYSTEM_USER}' account probably "
                "cannot run it. Install the package where all users can read it (for example "
                "`uv tool install` or a virtualenv under /opt) and run this command from there."
            )
    group = socket_group_of(config_path) if mode == "system" else None
    unit = render_unit(mode, python, config_path, group)
    path = unit_path(mode)
    out(f"unit file: {path}")
    out(unit)
    if mode == "system" and shutil.which("getent") and subprocess.run(["getent", "passwd", SYSTEM_USER], capture_output=True).returncode != 0:
        rc = _run(["useradd", "--system", "--no-create-home", "--shell", "/usr/sbin/nologin", SYSTEM_USER], dry_run, out)
        if rc != 0:
            return rc
    if group and shutil.which("getent") and subprocess.run(["getent", "group", group], capture_output=True).returncode != 0:
        out(f"warning: group '{group}' does not exist; create it (`sudo groupadd {group}`) and add users to it.")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(unit)
    for command in (_systemctl(mode, "daemon-reload"), _systemctl(mode, "enable", "--now", UNIT_NAME)):
        rc = _run(command, dry_run, out)
        if rc != 0:
            out("systemctl failed; see the message above and `journalctl" + (" --user" if mode == "user" else "") + " -u resourced`.")
            return rc
    if mode == "user" and shutil.which("loginctl"):
        result = subprocess.run(
            ["loginctl", "show-user", getpass.getuser(), "--property=Linger"], capture_output=True, text=True
        )
        if "Linger=no" in result.stdout:
            out(
                "note: lingering is off, so the service stops when you log out (and claims are lost). "
                f"Run once: sudo loginctl enable-linger {getpass.getuser()}"
            )
    return 0


def disable(mode: str, dry_run: bool, out=print) -> int:
    if mode == "system" and os.geteuid() != 0 and not dry_run:
        out("needs root: re-run with sudo.")
        return 1
    return _run(_systemctl(mode, "disable", "--now", UNIT_NAME), dry_run, out)
