"""Allocation logic. Transport-independent so it can be tested without sockets."""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, replace

from . import procs as proc
from .config import Config
from .gpus import GpuProc
from .protocol import PROTOCOL_VERSION
from .resources import local_snapshot

log = logging.getLogger("resourced")

MAX_WAIT = 60.0


class ServiceError(Exception):
    def __init__(self, code: str, message: str, data: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}


@dataclass(frozen=True)
class Allocation:
    ref: str
    mode: str  # "adhoc" or "dev"
    gpus: tuple[str, ...]  # GPU UUIDs
    task_name: str
    task_id: str
    user: str
    label: str
    created: float
    client_pid: int
    client_start: int | None
    # dev: owner PID given by the caller; adhoc: PID of the launched command
    pid: int | None = None
    pid_start: int | None = None
    pid_comm: str | None = None

    def to_json(self) -> dict:
        data = asdict(self)
        data["gpus"] = list(self.gpus)
        return data

    @classmethod
    def from_json(cls, data: dict) -> Allocation:
        data = dict(data)
        data["gpus"] = tuple(data["gpus"])
        return cls(**data)


def read_boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id") as handle:
            return handle.read().strip()
    except OSError:
        return "unknown"


def env_for(alloc: Allocation) -> dict:
    return {"CUDA_VISIBLE_DEVICES": ",".join(alloc.gpus), "RESOURCED_ALLOCATION": alloc.ref}


def _text(params: dict, key: str, default: str = "") -> str:
    value = params.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise ServiceError("bad_request", f"{key} must be a string")
    return value.strip()[:200]


def _int(params: dict, key: str) -> int | None:
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServiceError("bad_request", f"{key} must be an integer")
    return value


class Core:
    def __init__(self, config: Config, probe, monitor=None, boot_id: str | None = None):
        self.config = config
        self.probe = probe
        self.monitor = monitor
        self.boot_id = boot_id or read_boot_id()
        self.started = time.time()
        self.lock = threading.RLock()
        self.allocations: dict[str, Allocation] = {}
        self._load()

    # ---------------------------------------------------------------- state
    def _load(self) -> None:
        path = self.config.claims_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            if data.get("boot_id") != self.boot_id:
                log.warning("claims file is from another boot; discarding all claims")
                return
            self.allocations = {a["ref"]: Allocation.from_json(a) for a in data["allocations"]}
            log.info("restored %d allocation(s)", len(self.allocations))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            broken = path.with_suffix(".json.corrupt")
            log.error("claims file unreadable (%s); moved to %s and starting empty", exc, broken)
            try:
                os.replace(path, broken)
            except OSError:
                pass

    def _commit(self, allocations: dict[str, Allocation]) -> None:
        path = self.config.claims_path
        tmp = path.with_suffix(".json.tmp")
        payload = {"boot_id": self.boot_id, "allocations": [a.to_json() for a in allocations.values()]}
        try:
            with open(tmp, "w") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            raise ServiceError("state_write_failed", f"cannot save claims to {path}: {exc}") from exc
        self.allocations = allocations

    def _holder(self, uuid: str) -> Allocation | None:
        for alloc in self.allocations.values():
            if uuid in alloc.gpus:
                return alloc
        return None

    # --------------------------------------------------------------- probes
    def _procs(self) -> dict[str, list[GpuProc]] | None:
        found = self.probe.processes()
        if found is None:
            return None
        by_uuid: dict[str, list[GpuProc]] = {}
        for item in found:
            by_uuid.setdefault(item.uuid, []).append(item)
        return by_uuid

    @staticmethod
    def _proc_dict(item: GpuProc) -> dict:
        return {
            "gpu": item.uuid,
            "pid": item.pid,
            "name": item.name,
            "command": proc.command_line(item.pid) or item.name,
            "user": proc.user_of(item.pid),
            "memory_mib": item.memory_mib,
        }

    # ------------------------------------------------------------ describing
    def alias_of(self, uuid: str) -> str:
        gpu = self.config.find_gpu(uuid)
        return gpu.alias if gpu else uuid

    def describe(self, alloc: Allocation) -> dict:
        now = time.time()
        pid_alive = proc.is_alive(alloc.pid, alloc.pid_start)
        owner = None
        if alloc.pid is not None:
            owner = {
                "kind": "owner" if alloc.mode == "dev" else "command",
                "pid": alloc.pid,
                "command": (proc.comm_of(alloc.pid) if pid_alive else None) or alloc.pid_comm,
                "alive": pid_alive,
            }
        # the process whose disappearance suggests the claim was lost
        holder_alive = pid_alive if alloc.mode == "dev" else proc.is_alive(alloc.client_pid, alloc.client_start)
        return {
            "ref": alloc.ref,
            "mode": alloc.mode,
            "gpus": list(alloc.gpus),
            "aliases": [self.alias_of(u) for u in alloc.gpus],
            "task_name": alloc.task_name,
            "task_id": alloc.task_id,
            "user": alloc.user,
            "label": alloc.label,
            "created": alloc.created,
            "age_s": max(0.0, now - alloc.created),
            "owner": owner,
            "wrapper_pid": alloc.client_pid if alloc.mode == "adhoc" else None,
            "holder_alive": holder_alive,
        }

    # ------------------------------------------------------------------ ops
    def claim(self, params: dict, peer: dict) -> dict:
        mode = params.get("mode")
        if mode not in ("adhoc", "dev"):
            raise ServiceError("bad_request", "mode must be 'adhoc' or 'dev'")
        count = _int(params, "gpus")
        if count is None or count < 1:
            raise ServiceError("bad_request", "gpus must be an integer >= 1")
        task_name = _text(params, "task_name") or proc.comm_of(peer["pid"]) or "unknown"
        task_id = _text(params, "task_id") or task_name
        label = _text(params, "label")

        owner_pid = owner_start = owner_comm = None
        if mode == "dev":
            owner_pid = _int(params, "owner_pid")
            if owner_pid is None:
                raise ServiceError(
                    "owner_pid_required",
                    "a development claim needs the PID of the long-lived process that owns it "
                    "(for example the Codex/Claude process): pass --owner-pid or set RESOURCED_OWNER_PID",
                )
            owner_start = proc.start_time(owner_pid)
            if owner_start is None or not proc.is_alive(owner_pid):
                raise ServiceError("owner_pid_dead", f"owner PID {owner_pid} is not a running process")
            owner_comm = proc.comm_of(owner_pid)

        warnings: list[str] = []
        with self.lock:
            held = {u for a in self.allocations.values() for u in a.gpus}
            free = [g for g in self.config.gpus if g.uuid not in held]
            procs = self._procs()
            clean, busy = [], []
            if procs is None:
                warnings.append("GPU process check unavailable (nvidia-smi failed); claim granted without it")
                clean = free
            else:
                for gpu in free:
                    found = procs.get(gpu.uuid, [])
                    if found:
                        busy.append((gpu, found))
                    else:
                        clean.append(gpu)
            for gpu, found in busy:
                listing = ", ".join(f"pid {p.pid} ({p.name})" for p in found)
                log.error(
                    "GPU %s is not claimed but has running processes: %s. "
                    "Something bypassed resourced or a release was missed.",
                    gpu.alias,
                    listing,
                )
            if len(clean) < count:
                busy_data = [
                    {"gpu": g.alias, "uuid": g.uuid, "processes": [self._proc_dict(p) for p in found]}
                    for g, found in busy
                ]
                data = {"requested": count, "managed": len(self.config.gpus), "free": len(free), "usable": len(clean), "busy": busy_data}
                if busy_data:
                    details = "; ".join(
                        f"{b['gpu']}: " + ", ".join(f"pid {p['pid']} {p['command']} ({p['user']})" for p in b["processes"])
                        for b in busy_data
                    )
                    raise ServiceError(
                        "gpu_busy",
                        f"requested {count} GPU(s) but only {len(clean)} usable: unclaimed GPU(s) have running "
                        f"processes, which should not happen ({details}). Stop them or investigate.",
                        data,
                    )
                raise ServiceError(
                    "not_enough_gpus",
                    f"requested {count} GPU(s) but only {len(free)} available "
                    f"({len(self.config.gpus) - len(free)} of {len(self.config.gpus)} in use)",
                    data,
                )

            chosen = tuple(g.uuid for g in clean[:count])
            ref = secrets.token_hex(6)
            while ref in self.allocations:
                ref = secrets.token_hex(6)
            alloc = Allocation(
                ref=ref,
                mode=mode,
                gpus=chosen,
                task_name=task_name,
                task_id=task_id,
                user=peer["user"],
                label=label,
                created=time.time(),
                client_pid=peer["pid"],
                client_start=proc.start_time(peer["pid"]),
                pid=owner_pid,
                pid_start=owner_start,
                pid_comm=owner_comm,
            )
            self._commit({**self.allocations, ref: alloc})
        log.info("claim %s: %s x%d to %s [%s] by %s", ref, mode, count, task_name, task_id, peer["user"])
        return {"allocation": self.describe(alloc), "env": env_for(alloc), "warnings": warnings}

    def attach(self, params: dict, peer: dict) -> dict:
        ref = _text(params, "ref")
        pid = _int(params, "pid")
        if pid is None:
            raise ServiceError("bad_request", "pid is required")
        with self.lock:
            alloc = self.allocations.get(ref)
            if alloc is None:
                raise ServiceError("unknown_allocation", f"no allocation {ref}")
            start = proc.start_time(pid)
            updated = replace(alloc, pid=pid, pid_start=start, pid_comm=proc.comm_of(pid))
            self._commit({**self.allocations, ref: updated})
        return {"allocation": self.describe(updated)}

    def get(self, params: dict, peer: dict) -> dict:
        ref = _text(params, "ref")
        with self.lock:
            alloc = self.allocations.get(ref)
            if alloc is None:
                raise ServiceError(
                    "unknown_allocation",
                    f"no allocation {ref} (already released, force-released, or lost in a reboot)",
                )
            return {"allocation": self.describe(alloc), "env": env_for(alloc)}

    def release(self, params: dict, peer: dict) -> dict:
        force = bool(params.get("force"))
        on_busy = params.get("on_busy", "error")
        if on_busy not in ("error", "warn"):
            raise ServiceError("bad_request", "on_busy must be 'error' or 'warn'")
        try:
            wait = min(max(float(params.get("wait", 0)), 0.0), MAX_WAIT)
        except (TypeError, ValueError):
            raise ServiceError("bad_request", "wait must be a number") from None
        ref, gpu_sel, expect = _text(params, "ref"), _text(params, "gpu"), _text(params, "expect_ref")
        reason = _text(params, "reason")

        with self.lock:
            if ref:
                alloc = self.allocations.get(ref)
                if alloc is None:
                    raise ServiceError(
                        "unknown_allocation",
                        f"no allocation {ref} (already released, force-released, or lost in a reboot)",
                    )
                uuids = list(alloc.gpus)
            elif gpu_sel:
                gpu = self.config.find_gpu(gpu_sel)
                if gpu is None:
                    raise ServiceError("unknown_gpu", f"no managed GPU matches {gpu_sel!r}")
                alloc = self._holder(gpu.uuid)
                if alloc is None:
                    raise ServiceError("not_claimed", f"GPU {gpu.alias} is not claimed")
                if expect and expect != alloc.ref:
                    raise ServiceError(
                        "stale_allocation",
                        f"GPU {gpu.alias} is now held by allocation {alloc.ref}, not {expect}; nothing released",
                    )
                uuids = [gpu.uuid]
            else:
                raise ServiceError("bad_request", "release needs ref or gpu")
            ref = alloc.ref
            previous = self.describe(alloc)

        leftover: list[dict] = []
        probe_state = "skipped" if force else "ok"
        if not force:
            deadline = time.monotonic() + wait
            while True:
                procs = self._procs()
                if procs is None:
                    probe_state, leftover = "unknown", []
                    break
                leftover = [self._proc_dict(p) for u in uuids for p in procs.get(u, [])]
                if not leftover or time.monotonic() >= deadline:
                    break
                time.sleep(0.5)
            if leftover and on_busy == "error":
                listing = "; ".join(f"pid {p['pid']} {p['command']} ({p['user']})" for p in leftover)
                raise ServiceError(
                    "gpu_busy",
                    f"GPU(s) still in use: {listing}. Stop them first, or use --force.",
                    {"processes": leftover},
                )

        with self.lock:
            current = self.allocations.get(ref)
            if current is None:
                raise ServiceError("unknown_allocation", f"allocation {ref} was released concurrently")
            remaining = tuple(u for u in current.gpus if u not in uuids)
            updated = {k: v for k, v in self.allocations.items() if k != ref}
            if remaining:
                updated[ref] = replace(current, gpus=remaining)
            self._commit(updated)
        if force:
            log.warning(
                "FORCE release of %s (%s) by %s pid %s; reason: %s; was held by %s [%s]",
                ",".join(self.alias_of(u) for u in uuids),
                ref,
                peer["user"],
                peer["pid"],
                reason or "-",
                previous["task_name"],
                previous["user"],
            )
        else:
            log.info("release %s: %s", ref, ",".join(self.alias_of(u) for u in uuids))
        if leftover:
            log.warning("released %s while processes remain: %s", ref, [p["pid"] for p in leftover])
        return {
            "released": uuids,
            "aliases": [self.alias_of(u) for u in uuids],
            "previous": previous,
            "leftover": leftover,
            "probe": probe_state,
        }

    def status(self, params: dict, peer: dict) -> dict:
        driver = self.probe.gpus()
        by_uuid = None if driver is None else {g["uuid"]: g for g in driver}
        rows = []
        with self.lock:
            allocations = dict(self.allocations)
            for gpu in self.config.gpus:
                info = by_uuid.get(gpu.uuid) if by_uuid is not None else None
                present = None if by_uuid is None else info is not None
                holder = next((a for a in allocations.values() if gpu.uuid in a.gpus), None)
                rows.append(
                    {
                        "uuid": gpu.uuid,
                        "alias": gpu.alias,
                        "index": gpu.index if gpu.index is not None else (info or {}).get("index"),
                        "name": (info or {}).get("name"),
                        "present": present,
                        "state": "USED" if holder else "AVAILABLE",
                        "allocation": self.describe(holder) if holder else None,
                    }
                )
            managed = {g.uuid for g in self.config.gpus}
            unmanaged = sorted({u for a in allocations.values() for u in a.gpus} - managed)
            task_id = _text(params, "task_id")
            mine = [self.describe(a) for a in allocations.values() if task_id and a.task_id == task_id]
            allocations_info = [self.describe(a) for a in allocations.values()]
        used = sum(1 for r in rows if r["state"] == "USED")
        available = sum(1 for r in rows if r["state"] == "AVAILABLE" and r["present"] is not False)
        return {
            "version": PROTOCOL_VERSION,
            "pid": os.getpid(),
            "boot_id": self.boot_id,
            "uptime_s": time.time() - self.started,
            "total": len(rows),
            "available": available,
            "used": used,
            "gpus": rows,
            "allocations": allocations_info,
            "mine": mine,
            "unmanaged_claims": unmanaged,
            "driver": "ok" if driver is not None else "unavailable",
        }

    def verify(self, params: dict, peer: dict) -> dict:
        driver = self.probe.gpus()
        by_uuid = None if driver is None else {g["uuid"]: g for g in driver}
        with self.lock:
            procs = self._procs()
            rows = []
            for gpu in self.config.gpus:
                holder = self._holder(gpu.uuid)
                found = [] if procs is None else procs.get(gpu.uuid, [])
                entries = []
                foreign = False
                for item in found:
                    entry = self._proc_dict(item)
                    belongs = None
                    if holder is not None and holder.pid is not None:
                        if proc.read_stat(item.pid) is None:
                            belongs = None  # PID not visible here (container?)
                        else:
                            belongs = proc.descends_from(item.pid, holder.pid, holder.pid_start)
                            foreign = foreign or not belongs
                    elif holder is not None:
                        foreign = True
                        belongs = False
                    entry["belongs"] = belongs
                    entries.append(entry)
                if by_uuid is not None and gpu.uuid not in by_uuid:
                    state = "MISSING"
                elif procs is None:
                    state = "UNKNOWN"
                elif holder is None:
                    state = "UNCLAIMED USE" if found else "OK"
                elif not found:
                    state = "IDLE CLAIM"
                else:
                    state = "FOREIGN PROCESS" if foreign else "OK"
                rows.append(
                    {
                        "uuid": gpu.uuid,
                        "alias": gpu.alias,
                        "state": state,
                        "allocation": self.describe(holder) if holder else None,
                        "processes": entries,
                    }
                )
        problems = [r for r in rows if r["state"] in ("UNCLAIMED USE", "FOREIGN PROCESS")]
        return {"healthy": not problems, "probe": "unknown" if procs is None else "ok", "gpus": rows}

    def resources(self, params: dict, peer: dict) -> dict:
        if self.monitor is not None:
            return self.monitor.snapshot()
        return local_snapshot(self.config)

    def ping(self, params: dict, peer: dict) -> dict:
        return {"version": PROTOCOL_VERSION, "pid": os.getpid()}

    def handle(self, request: dict, peer: dict) -> dict:
        from .protocol import error, ok

        try:
            if request.get("v") != PROTOCOL_VERSION:
                raise ServiceError("version_mismatch", f"server speaks protocol {PROTOCOL_VERSION}, request used {request.get('v')!r}")
            op = request.get("op")
            if op not in ("claim", "release", "attach", "get", "status", "verify", "resources", "ping"):
                raise ServiceError("bad_request", f"unknown operation {op!r}")
            params = request.get("params", {})
            if not isinstance(params, dict):
                raise ServiceError("bad_request", "params must be an object")
            return ok(**getattr(self, op)(params, peer))
        except ServiceError as exc:
            return error(exc.code, exc.message, exc.data)
        except Exception:
            log.exception("internal error handling %r", request.get("op"))
            return error("internal", "internal error; see the daemon log")
