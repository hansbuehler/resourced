"""Configuration loading and runtime paths."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SYSTEM_CONFIG = Path("/etc/resourced/config.toml")
SYSTEM_RUNTIME_DIR = Path("/run/resourced")
SOCKET_NAME = "resourced.sock"

LIMIT_KEYS = (
    "warn_free_gib",
    "stop_free_gib",
    "warn_free_pct",
    "stop_free_pct",
    "warn_used_pct",
    "stop_used_pct",
)


class ConfigError(Exception):
    """Invalid or missing configuration."""


@dataclass(frozen=True)
class GpuConfig:
    uuid: str
    alias: str
    index: int | None = None


@dataclass(frozen=True)
class CheckConfig:
    name: str
    kind: str  # "ram" or "disk"
    path: str | None
    limits: dict[str, float]


@dataclass
class Config:
    gpus: list[GpuConfig]
    checks: list[CheckConfig] = field(default_factory=list)
    runtime_dir: Path = SYSTEM_RUNTIME_DIR
    socket_group: str | None = None
    sample_interval: float = 30.0
    probe_timeout: float = 10.0
    nvidia_smi: str = "nvidia-smi"
    path: Path | None = None

    @property
    def socket_path(self) -> Path:
        return self.runtime_dir / SOCKET_NAME

    @property
    def claims_path(self) -> Path:
        return self.runtime_dir / "claims.json"

    @property
    def lock_path(self) -> Path:
        return self.runtime_dir / "daemon.lock"

    @property
    def resources_path(self) -> Path:
        return self.runtime_dir / "resources.json"

    def find_gpu(self, selector: str) -> GpuConfig | None:
        """Resolve an alias, index or UUID."""
        for gpu in self.gpus:
            if selector in (gpu.uuid, gpu.alias):
                return gpu
        if selector.isdigit():
            for gpu in self.gpus:
                if gpu.index == int(selector):
                    return gpu
        return None


def user_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "resourced" / "config.toml"


def default_runtime_dir() -> Path:
    env = os.environ.get("RESOURCED_RUNTIME_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "resourced"
    raise ConfigError(
        "XDG_RUNTIME_DIR is not set, so there is no per-user runtime directory. "
        "Set runtime_dir in the config or RESOURCED_RUNTIME_DIR."
    )


def find_config(explicit: str | os.PathLike | None = None) -> Path:
    """Return the config file to use: explicit, RESOURCED_CONFIG, user, system."""
    chosen = explicit or os.environ.get("RESOURCED_CONFIG")
    if chosen:
        path = Path(chosen).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path
    for path in (user_config_path(), SYSTEM_CONFIG):
        if path.is_file():
            return path
    raise ConfigError(
        f"no config found (looked at {user_config_path()} and {SYSTEM_CONFIG}). "
        "Create one with `gpuctl init`."
    )


def runtime_dir_from_file(path: Path) -> Path | None:
    """Read only runtime_dir from a config file; ignore all problems."""
    try:
        data = tomllib.loads(path.read_text())
        value = data.get("daemon", {}).get("runtime_dir")
        return Path(value) if isinstance(value, str) and value else None
    except (OSError, tomllib.TOMLDecodeError):
        return None


def _number(value, where: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, got {value}")
    return float(value)


def _string(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value.strip()


def parse_config(data: dict, path: Path | None = None) -> Config:
    known = {"daemon", "gpu", "check"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")

    daemon = data.get("daemon", {})
    if not isinstance(daemon, dict):
        raise ConfigError("[daemon] must be a table")
    daemon_keys = {"runtime_dir", "socket_group", "sample_interval", "probe_timeout", "nvidia_smi"}
    if set(daemon) - daemon_keys:
        raise ConfigError(f"unknown [daemon] keys: {sorted(set(daemon) - daemon_keys)}")

    runtime_dir = (
        Path(_string(daemon["runtime_dir"], "daemon.runtime_dir")) if "runtime_dir" in daemon else default_runtime_dir()
    )

    gpus: list[GpuConfig] = []
    raw_gpus = data.get("gpu", [])
    if not isinstance(raw_gpus, list) or not raw_gpus:
        raise ConfigError("at least one [[gpu]] entry is required (run `gpuctl init`)")
    for i, entry in enumerate(raw_gpus):
        where = f"gpu[{i}]"
        if not isinstance(entry, dict) or set(entry) - {"uuid", "alias", "index"}:
            raise ConfigError(f"{where}: expected a table with uuid, alias, index")
        uuid = _string(entry.get("uuid"), f"{where}.uuid")
        index = entry.get("index")
        if index is not None:
            index = int(_number(index, f"{where}.index"))
        alias = _string(entry["alias"], f"{where}.alias") if "alias" in entry else f"gpu{index if index is not None else i}"
        gpus.append(GpuConfig(uuid, alias, index))
    for attr in ("uuid", "alias"):
        values = [getattr(g, attr) for g in gpus]
        if len(set(values)) != len(values):
            raise ConfigError(f"duplicate GPU {attr} in config")

    checks: list[CheckConfig] = []
    raw_checks = data.get("check", [])
    if not isinstance(raw_checks, list):
        raise ConfigError("[[check]] must be an array of tables")
    for i, entry in enumerate(raw_checks):
        where = f"check[{i}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: expected a table")
        name = _string(entry.get("name"), f"{where}.name")
        kind = entry.get("type")
        if kind not in ("ram", "disk"):
            raise ConfigError(f"{where}.type must be 'ram' or 'disk'")
        path_value = None
        if kind == "disk":
            path_value = _string(entry.get("path"), f"{where}.path")
        extra = set(entry) - {"name", "type", "path", *LIMIT_KEYS}
        if extra:
            raise ConfigError(f"{where}: unknown keys {sorted(extra)}")
        limits = {key: _number(entry[key], f"{where}.{key}") for key in LIMIT_KEYS if key in entry}
        for key in limits:
            if key.endswith("_pct") and limits[key] > 100:
                raise ConfigError(f"{where}.{key} must be <= 100")
        for unit in ("free_gib", "free_pct"):
            warn, stop = limits.get(f"warn_{unit}"), limits.get(f"stop_{unit}")
            if warn is not None and stop is not None and stop > warn:
                raise ConfigError(f"{where}: stop_{unit} must be <= warn_{unit}")
        warn, stop = limits.get("warn_used_pct"), limits.get("stop_used_pct")
        if warn is not None and stop is not None and stop < warn:
            raise ConfigError(f"{where}: stop_used_pct must be >= warn_used_pct")
        if not limits:
            raise ConfigError(f"{where}: at least one warn_*/stop_* limit is required")
        checks.append(CheckConfig(name, kind, path_value, limits))
    names = [c.name for c in checks]
    if len(set(names)) != len(names):
        raise ConfigError("duplicate check names in config")

    group = daemon.get("socket_group")
    return Config(
        gpus=gpus,
        checks=checks,
        runtime_dir=runtime_dir,
        socket_group=_string(group, "daemon.socket_group") if group is not None else None,
        sample_interval=_number(daemon.get("sample_interval", 30), "daemon.sample_interval", minimum=1),
        probe_timeout=_number(daemon.get("probe_timeout", 10), "daemon.probe_timeout", minimum=0.1),
        nvidia_smi=_string(daemon.get("nvidia_smi", "nvidia-smi"), "daemon.nvidia_smi"),
        path=path,
    )


def load_config(explicit: str | os.PathLike | None = None) -> Config:
    path = find_config(explicit)
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read: {exc}") from exc
    try:
        return parse_config(data, path)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
