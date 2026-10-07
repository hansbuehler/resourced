"""Inspect processes through /proc."""

from __future__ import annotations

import os
import pwd
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stat:
    pid: int
    ppid: int
    start: int  # start time in clock ticks since boot; distinguishes reused PIDs
    comm: str
    state: str


def read_stat(pid: int) -> Stat | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # comm may contain spaces and parentheses, so split around the last ")"
    right = data.rfind(")")
    comm = data[data.find("(") + 1 : right]
    fields = data[right + 2 :].split()
    try:
        return Stat(pid, int(fields[1]), int(fields[19]), comm, fields[0])
    except (IndexError, ValueError):
        return None


def start_time(pid: int) -> int | None:
    st = read_stat(pid)
    return st.start if st else None


def is_alive(pid: int | None, start: int | None = None) -> bool:
    """True if the process exists, is not a zombie, and is the same process."""
    if not pid:
        return False
    st = read_stat(pid)
    if st is None or st.state in ("Z", "X"):
        return False
    return start is None or st.start == start


def descends_from(pid: int, ancestor_pid: int, ancestor_start: int | None) -> bool:
    """True if pid is ancestor_pid or one of its descendants (same start time)."""
    current = pid
    for _ in range(4096):
        st = read_stat(current)
        if st is None:
            return False
        if current == ancestor_pid:
            return ancestor_start is None or st.start == ancestor_start
        if st.ppid <= 0:
            return False
        current = st.ppid
    return False


def user_of(pid: int) -> str | None:
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def user_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def command_line(pid: int, limit: int = 120) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    text = raw.replace(b"\0", b" ").decode(errors="replace").strip()
    if not text:
        st = read_stat(pid)
        return f"[{st.comm}]" if st else None
    return text if len(text) <= limit else text[: limit - 3] + "..."


def comm_of(pid: int) -> str | None:
    st = read_stat(pid)
    return st.comm if st else None
