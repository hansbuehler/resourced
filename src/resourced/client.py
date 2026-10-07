"""Thin client for the resourced socket protocol."""

from __future__ import annotations

import os
import socket
from pathlib import Path

from .config import (
    SOCKET_NAME,
    SYSTEM_RUNTIME_DIR,
    ConfigError,
    find_config,
    runtime_dir_from_file,
)
from .protocol import MAX_RESPONSE, PROTOCOL_VERSION, ProtocolError, decode, encode


class ResourcedError(Exception):
    """An error reported by the daemon, or a failure to reach it."""

    def __init__(self, code: str, message: str, data: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}


def find_socket(explicit: str | os.PathLike | None = None, config_path: str | os.PathLike | None = None) -> Path:
    """--socket, RESOURCED_SOCKET, config runtime_dir, user runtime dir, system runtime dir."""
    chosen = explicit or os.environ.get("RESOURCED_SOCKET")
    if chosen:
        return Path(chosen)
    try:
        configured = runtime_dir_from_file(find_config(config_path))
    except ConfigError:
        configured = None
    if configured:
        return configured / SOCKET_NAME
    env_dir = os.environ.get("RESOURCED_RUNTIME_DIR")
    if env_dir:
        return Path(env_dir) / SOCKET_NAME
    candidates = []
    if os.environ.get("XDG_RUNTIME_DIR"):
        candidates.append(Path(os.environ["XDG_RUNTIME_DIR"]) / "resourced" / SOCKET_NAME)
    candidates.append(SYSTEM_RUNTIME_DIR / SOCKET_NAME)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class Client:
    def __init__(
        self,
        socket_path: str | os.PathLike | None = None,
        config_path: str | os.PathLike | None = None,
        timeout: float = 120.0,
    ):
        self.socket_path = find_socket(socket_path, config_path)
        self.timeout = timeout

    def request(self, op: str, **params) -> dict:
        message = encode({"v": PROTOCOL_VERSION, "op": op, "params": params})
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(str(self.socket_path))
                sock.sendall(message)
                with sock.makefile("rb") as stream:
                    line = stream.readline(MAX_RESPONSE)
        except FileNotFoundError:
            raise ResourcedError(
                "daemon_unreachable",
                f"cannot find the resourced socket at {self.socket_path}. Is the daemon running? "
                "Start it with `gpuctl service enable --user` or `resourced`.",
            ) from None
        except PermissionError:
            raise ResourcedError(
                "daemon_unreachable",
                f"no permission to use {self.socket_path}; are you in the socket group of the shared install?",
            ) from None
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise ResourcedError("daemon_unreachable", f"cannot talk to resourced at {self.socket_path}: {exc}") from None
        if not line:
            raise ResourcedError("daemon_unreachable", "resourced closed the connection without answering")
        try:
            response = decode(line)
        except ProtocolError as exc:
            raise ResourcedError("protocol", str(exc)) from None
        if not response.get("ok"):
            err = response.get("error", {})
            raise ResourcedError(err.get("code", "error"), err.get("message", "unknown error"), err.get("data"))
        response.pop("ok")
        return response

    # Convenience wrappers ------------------------------------------------
    def ping(self) -> dict:
        return self.request("ping")

    def status(self, task_id: str | None = None) -> dict:
        return self.request("status", task_id=task_id)

    def claim(self, *, mode: str, gpus: int, task_name: str = "", task_id: str = "", label: str = "", owner_pid: int | None = None) -> dict:
        return self.request("claim", mode=mode, gpus=gpus, task_name=task_name, task_id=task_id, label=label, owner_pid=owner_pid)

    def release(self, ref: str, **options) -> dict:
        return self.request("release", ref=ref, **options)

    def get(self, ref: str) -> dict:
        return self.request("get", ref=ref)

    def verify(self) -> dict:
        return self.request("verify")

    def resources(self) -> dict:
        return self.request("resources")
