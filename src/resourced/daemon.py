"""The resourced daemon: Unix socket server around Core."""

from __future__ import annotations

import argparse
import fcntl
import grp
import logging
import os
import signal
import socket
import socketserver
import stat
import struct
import sys
import threading

from . import __version__, procs as proc
from .config import Config, ConfigError, load_config
from .core import Core
from .gpus import NvidiaSmiProbe
from .protocol import MAX_REQUEST, ProtocolError, decode, encode, error
from .resources import ResourceMonitor

log = logging.getLogger("resourced")

REQUEST_TIMEOUT = 10.0


def peer_credentials(sock: socket.socket) -> dict:
    pid, uid, _gid = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    return {"pid": pid, "uid": uid, "user": proc.user_name(uid)}


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        core: Core = self.server.core  # type: ignore[attr-defined]
        try:
            peer = peer_credentials(self.request)
        except OSError:
            return
        self.request.settimeout(REQUEST_TIMEOUT)
        while True:
            try:
                line = self.rfile.readline(MAX_REQUEST + 1)
            except (TimeoutError, OSError):
                return
            if not line:
                return
            if len(line) > MAX_REQUEST and not line.endswith(b"\n"):
                self._send(error("request_too_large", f"request exceeds {MAX_REQUEST} bytes"))
                return
            try:
                response = core.handle(decode(line), peer)
            except ProtocolError as exc:
                response = error("bad_request", str(exc))
            if not self._send(response):
                return

    def _send(self, response: dict) -> bool:
        try:
            self.wfile.write(encode(response))
            self.wfile.flush()
            return True
        except OSError:
            return False


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class Daemon:
    """Owns the runtime directory, lock file, socket, and background sampler."""

    def __init__(self, config: Config, probe=None):
        self.config = config
        self.probe = probe or NvidiaSmiProbe(config.nvidia_smi, config.probe_timeout)
        self.monitor = ResourceMonitor(config)
        self.core: Core | None = None
        self.server: Server | None = None
        self._lock_file = None
        self._thread: threading.Thread | None = None

    def _prepare_runtime_dir(self) -> None:
        directory = self.config.runtime_dir
        if directory.is_symlink():
            raise RuntimeError(f"runtime directory {directory} is a symlink; refusing to use it")
        if not directory.exists():
            directory.mkdir(parents=True, mode=0o700)
        info = directory.stat()
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"{directory} is not a directory")
        if info.st_uid != os.geteuid():
            raise RuntimeError(f"runtime directory {directory} is owned by another user (uid {info.st_uid})")

    def _acquire_lock(self) -> None:
        handle = open(self.config.lock_path, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError(
                f"another resourced already uses {self.config.runtime_dir} (lock held); "
                "stop it first or use a different runtime_dir"
            ) from None
        self._lock_file = handle

    def _bind(self) -> Server:
        path = self.config.socket_path
        if path.exists() or path.is_symlink():
            path.unlink()
        old_umask = os.umask(0o177)
        try:
            server = Server(str(path), Handler)
        finally:
            os.umask(old_umask)
        mode = 0o600
        if self.config.socket_group:
            try:
                gid = grp.getgrnam(self.config.socket_group).gr_gid
                os.chown(path, -1, gid)
                mode = 0o660
            except (KeyError, PermissionError) as exc:
                server.server_close()
                raise RuntimeError(
                    f"cannot use socket_group {self.config.socket_group!r}: {exc!r}. "
                    "The group must exist and the service account must be a member."
                ) from None
        os.chmod(path, mode)
        return server

    def start(self) -> None:
        self._prepare_runtime_dir()
        self._acquire_lock()
        try:
            self.core = Core(self.config, self.probe, self.monitor)
            self.server = self._bind()
            self.server.core = self.core  # type: ignore[attr-defined]
        except BaseException:
            self._release_lock()
            raise
        if isinstance(self.probe, NvidiaSmiProbe) and self.probe.gpus() is None:
            log.error(
                "GPU presence and process checks are off until nvidia-smi works (claims still work; "
                "the daemon recovers by itself once it does)"
            )
        self.monitor.start()
        self._thread = threading.Thread(target=self.server.serve_forever, name="server", daemon=True)
        self._thread.start()
        log.info(
            "resourced %s listening on %s (%d GPUs, %d checks)",
            __version__, self.config.socket_path, len(self.config.gpus), len(self.config.checks),
        )

    def serve(self) -> None:
        assert self._thread is not None
        self._thread.join()

    def stop(self) -> None:
        self.monitor.stop()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            try:
                self.config.socket_path.unlink()
            except OSError:
                pass
            self.server = None
        self._release_lock()

    def _release_lock(self) -> None:
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resourced", description="GPU allocation and resource limits daemon")
    parser.add_argument("--config", help="config file (default: RESOURCED_CONFIG, user config, /etc/resourced/config.toml)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"resourced: {exc}", file=sys.stderr)
        return 2
    daemon = Daemon(config)
    try:
        daemon.start()
    except (RuntimeError, OSError) as exc:
        print(f"resourced: {exc}", file=sys.stderr)
        return 1

    def shutdown(signum, _frame):
        log.info("received signal %d; shutting down (claims are kept)", signum)
        threading.Thread(target=daemon.server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    daemon.serve()
    daemon.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
