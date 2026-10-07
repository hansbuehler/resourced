"""Advisory RAM and disk limits. Nothing here blocks or kills anything."""

from __future__ import annotations

import json
import logging
import os
import threading
import time

from .config import CheckConfig, Config

log = logging.getLogger("resourced")

GIB = 2**30

# metric -> comparison operator that reaches a level
_METRICS = {
    "free_gib": "<=",
    "free_pct": "<=",
    "used_pct": ">=",
}
LEVELS = ("OK", "WARN", "STOP")


def call_with_timeout(fn, timeout: float):
    """Run fn in a helper thread; raise TimeoutError if it does not finish."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # reported to the caller below
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"probe did not finish within {timeout:g}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def measure_ram() -> dict:
    values = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in ("MemAvailable", "MemTotal"):
                values[key] = int(rest.split()[0]) * 1024
    if "MemAvailable" not in values or "MemTotal" not in values:
        raise RuntimeError("MemAvailable/MemTotal not found in /proc/meminfo")
    return _usage(values["MemAvailable"], values["MemTotal"])


def measure_disk(path: str) -> dict:
    st = os.statvfs(path)
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    available = st.f_bavail * st.f_frsize
    # same convention as df: used + available is the capacity usable by users
    result = _usage(available, used + available)
    result["device"] = os.stat(path).st_dev
    return result


def _usage(free_bytes: int, total_bytes: int) -> dict:
    pct_free = 100.0 * free_bytes / total_bytes if total_bytes else 0.0
    return {
        "free_gib": free_bytes / GIB,
        "total_gib": total_bytes / GIB,
        "free_pct": pct_free,
        "used_pct": 100.0 - pct_free,
    }


def evaluate(check: CheckConfig, current: dict | None) -> tuple[str, list[dict]]:
    """Return (health, limits). Limits are inclusive: a level is reached AT its value."""
    limits = []
    health = "OK"
    for key, value in check.limits.items():
        level, _, metric = key.partition("_")
        level = level.upper()  # "WARN" or "STOP"
        op = _METRICS[metric]
        triggered = None
        if current is not None:
            measured = current[metric]
            triggered = measured <= value if op == "<=" else measured >= value
            if triggered and LEVELS.index(level) > LEVELS.index(health):
                health = level
        limits.append({"key": key, "level": level, "metric": metric, "op": op, "value": value, "triggered": triggered})
    return health, limits


def sample_check(check: CheckConfig, timeout: float) -> dict:
    result = {"name": check.name, "type": check.kind, "path": check.path, "timestamp": time.time(), "error": None}
    current = None
    try:
        if check.kind == "ram":
            current = call_with_timeout(measure_ram, timeout)
        else:
            current = call_with_timeout(lambda: measure_disk(check.path), timeout)
        health, limits = evaluate(check, current)
    except FileNotFoundError:
        health, limits = "UNAVAILABLE", evaluate(check, None)[1]
        result["error"] = f"path not found: {check.path}"
    except Exception as exc:
        health, limits = "ERROR", evaluate(check, None)[1]
        result["error"] = f"{type(exc).__name__}: {exc}"
    result.update(health=health, current=current, limits=limits)
    return result


class ResourceMonitor:
    """Samples all configured checks in the background."""

    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.Lock()
        self._snapshot: dict = self._empty()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _empty(self) -> dict:
        checks = []
        for check in self.config.checks:
            checks.append(
                {
                    "name": check.name,
                    "type": check.kind,
                    "path": check.path,
                    "health": "UNAVAILABLE",
                    "error": "not sampled yet",
                    "current": None,
                    "previous": None,
                    "timestamp": None,
                    "limits": evaluate(check, None)[1],
                }
            )
        return {"timestamp": None, "interval": self.config.sample_interval, "checks": checks}

    def sample_once(self) -> dict:
        with self._lock:
            previous = {c["name"]: c for c in self._snapshot["checks"]}
        checks = []
        for check in self.config.checks:
            result = sample_check(check, self.config.probe_timeout)
            old = previous.get(check.name)
            result["previous"] = old["current"] if old else None
            if old and old["health"] != result["health"] and old["timestamp"] is not None:
                level = logging.INFO if result["health"] == "OK" else logging.WARNING
                log.log(level, "resource check %s: %s -> %s %s", check.name, old["health"], result["health"], result["error"] or "")
            checks.append(result)
        snapshot = {"timestamp": time.time(), "interval": self.config.sample_interval, "checks": checks}
        with self._lock:
            self._snapshot = snapshot
        self._write(snapshot)
        return snapshot

    def _write(self, snapshot: dict) -> None:
        try:
            path = self.config.resources_path
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snapshot))
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("cannot write resource snapshot: %s", exc)

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot

    def start(self) -> None:
        if not self.config.checks:
            return

        def loop():
            while not self._stop.is_set():
                try:
                    self.sample_once()
                except Exception:
                    log.exception("resource sampling failed")
                self._stop.wait(self.config.sample_interval)

        self._thread = threading.Thread(target=loop, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def local_snapshot(config: Config) -> dict:
    """One-off sampling without a daemon (used by the CLI as a fallback)."""
    checks = []
    for check in config.checks:
        item = sample_check(check, config.probe_timeout)
        item["previous"] = None
        checks.append(item)
    return {"timestamp": time.time(), "interval": config.sample_interval, "checks": checks}
