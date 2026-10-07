"""Wire format: one JSON object per line over a Unix domain socket."""

from __future__ import annotations

import json

PROTOCOL_VERSION = 1
MAX_REQUEST = 64 * 1024
MAX_RESPONSE = 8 * 1024 * 1024


class ProtocolError(Exception):
    pass


def encode(message: dict) -> bytes:
    return json.dumps(message, separators=(",", ":")).encode() + b"\n"


def decode(line: bytes) -> dict:
    try:
        message = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"malformed JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("message must be a JSON object")
    return message


def ok(**fields) -> dict:
    return {"ok": True, **fields}


def error(code: str, message: str, data: dict | None = None) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, "data": data or {}}}
