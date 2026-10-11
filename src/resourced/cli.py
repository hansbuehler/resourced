"""gpuctl: command line interface."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from . import __version__, procs as proc, service
from .client import Client, ResourcedError
from .config import SYSTEM_CONFIG, SYSTEM_RUNTIME_DIR, ConfigError, default_runtime_dir, load_config, user_config_path
from .gpus import NvidiaSmiProbe, find_nvidia_smi, nvidia_smi_missing_message
from .resources import local_snapshot
from .template import render_config

EXIT_CODES = {
    "bad_request": 2,
    "owner_pid_required": 2,
    "owner_pid_dead": 2,
    "not_enough_gpus": 3,
    "gpu_busy": 4,
    "daemon_unreachable": 5,
    "unknown_allocation": 6,
    "stale_allocation": 6,
}

AGENT_INSTRUCTIONS = """\
## GPU usage (resourced)

GPUs on this machine are shared. Use `gpuctl` instead of picking a GPU yourself.

- One-off job: `gpuctl run --gpus N -- <command>`. It gets N free GPUs (pinned through
  CUDA_VISIBLE_DEVICES) and releases them when the command exits. If no GPU is free it
  fails immediately: retry later, fall back to CPU, or tell the user.
- Development session (compiling, testing, tuning): claim once, reuse, release when done.
    gpuctl claim --gpus 1 --label <what-for> --task-name "<your task name>" --owner-pid <PID of your own process>
    gpuctl run --allocation <ref> -- <command>        # as often as needed
    gpuctl release --allocation <ref>                  # release when finished
  Always release a development claim when you are done.
- `gpuctl status` shows who holds which GPU; `gpuctl verify` checks claims against running processes.
- Never use a GPU that is not allocated to you. If an allocation is lost, tell the user
  (`gpuctl release --force --gpu <gpu> --reason "..."` frees it).
- Resource limits (RAM, disk): `gpuctl resources`. Stay above the STOP limits; at WARN, clean up.
"""


# ------------------------------------------------------------------ helpers
def make_client(args) -> Client:
    return Client(args.socket, args.config)


def fmt_age(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 120:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h{minutes % 60:02d}m"
    return f"{hours // 24}d{hours % 24}h"


def short_uuid(uuid: str) -> str:
    return uuid if len(uuid) <= 12 else uuid[:12]


def task_identity(args) -> tuple[str, str]:
    name = getattr(args, "task_name", None) or os.environ.get("RESOURCED_TASK_NAME") or ""
    if not name:
        name = proc.comm_of(os.getppid()) or ""
    task_id = getattr(args, "task_id", None) or os.environ.get("RESOURCED_TASK_ID") or name
    return name, task_id


def owner_pid(args) -> int | None:
    value = getattr(args, "owner_pid", None)
    if value is None and os.environ.get("RESOURCED_OWNER_PID"):
        try:
            value = int(os.environ["RESOURCED_OWNER_PID"])
        except ValueError:
            raise ResourcedError("bad_request", "RESOURCED_OWNER_PID must be an integer") from None
    return value


def warn(message: str) -> None:
    print(f"gpuctl: {message}", file=sys.stderr)


def describe_allocation(a: dict) -> str:
    owner = a.get("owner")
    if a["mode"] == "dev":
        who = f"owner pid {owner['pid']} {owner.get('command') or '?'} ({'alive' if owner['alive'] else 'DEAD'})" if owner else "no owner"
    else:
        who = f"wrapper pid {a['wrapper_pid']} ({'alive' if a['holder_alive'] else 'DEAD'})"
    label = f' "{a["label"]}"' if a["label"] else ""
    return (
        f"{a['task_name']} [{a['task_id']}] {a['user']} {a['mode']}{label} {fmt_age(a['age_s'])} "
        f"ref {a['ref']}, {who}"
    )


def print_json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


# ----------------------------------------------------------------- commands
def cmd_claim(args) -> int:
    name, task_id = task_identity(args)
    client = make_client(args)
    result = client.claim(
        mode="dev", gpus=args.gpus, task_name=name, task_id=task_id, label=args.label or "", owner_pid=owner_pid(args)
    )
    for message in result["warnings"]:
        warn(message)
    if args.json:
        print_json(result)
        return 0
    alloc = result["allocation"]
    print(f"allocation: {alloc['ref']}")
    print("gpus: " + ", ".join(f"{a} ({short_uuid(u)})" for a, u in zip(alloc["aliases"], alloc["gpus"])))
    print(f"task: {describe_allocation(alloc)}")
    print(f"use:  gpuctl run --allocation {alloc['ref']} -- <command>")
    print(f"      eval \"$(gpuctl env --allocation {alloc['ref']})\"")
    print(f"free: gpuctl release --allocation {alloc['ref']}")
    return 0


def cmd_release(args) -> int:
    client = make_client(args)
    if args.gpu and not args.force:
        raise ResourcedError("bad_request", "--gpu is only for --force release; use --allocation otherwise")
    if not args.gpu and not args.allocation:
        raise ResourcedError("bad_request", "give --allocation <ref>, or --force --gpu <gpu>")
    options = {"force": args.force, "reason": args.reason or "", "wait": args.wait}
    if args.gpu:
        status = client.status()
        match = None
        for row in status["gpus"]:
            if args.gpu in (row["uuid"], row["alias"]) or (args.gpu.isdigit() and row["index"] == int(args.gpu)):
                match = row
        if match is None:
            raise ResourcedError("unknown_gpu", f"no managed GPU matches {args.gpu!r}")
        if match["allocation"] is None:
            raise ResourcedError("not_claimed", f"GPU {match['alias']} is not claimed")
        print(f"force releasing {match['alias']}, held by {describe_allocation(match['allocation'])}", file=sys.stderr)
        result = client.request("release", gpu=args.gpu, expect_ref=match["allocation"]["ref"], **options)
    else:
        result = client.request("release", ref=args.allocation, **options)
    if args.json:
        print_json(result)
    else:
        print("released: " + ", ".join(result["aliases"]))
    if result["leftover"]:
        warn("released while processes remain: " + ", ".join(f"pid {p['pid']} ({p['command']})" for p in result["leftover"]))
    return 0


def run_command(command: list[str], env_extra: dict, on_start=None) -> int:
    """Run the command, forward TERM/HUP, and return its exit code (128+n for signals)."""
    child: subprocess.Popen | None = None

    def forward(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    # A terminal sends Ctrl-C to the whole foreground group, so the wrapper only needs to
    # survive it. (A handler, not SIG_IGN: ignored signals would be inherited by the command.)
    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, lambda *_: None),
        signal.SIGTERM: signal.signal(signal.SIGTERM, forward),
        signal.SIGHUP: signal.signal(signal.SIGHUP, forward),
    }
    try:
        try:
            child = subprocess.Popen(command, env={**os.environ, **env_extra})
        except FileNotFoundError:
            warn(f"command not found: {command[0]}")
            return 127
        except PermissionError:
            warn(f"cannot execute: {command[0]}")
            return 126
        if on_start:
            on_start(child.pid)
        code = child.wait()
        return 128 - code if code < 0 else code
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def cmd_run(args, command: list[str]) -> int:
    if not command:
        raise ResourcedError("bad_request", "no command given: gpuctl run --gpus N -- <command>")
    if bool(args.gpus) == bool(args.allocation):
        raise ResourcedError("bad_request", "give exactly one of --gpus N (one-off) or --allocation <ref> (development)")
    client = make_client(args)

    if args.allocation:
        info = client.get(args.allocation)
        if not args.quiet:
            warn(f"running on {','.join(info['allocation']['aliases'])} (allocation {args.allocation})")
        return run_command(command, info["env"])

    name, task_id = task_identity(args)
    claim = client.claim(mode="adhoc", gpus=args.gpus, task_name=name, task_id=task_id, label=args.label or "")
    for message in claim["warnings"]:
        warn(message)
    alloc = claim["allocation"]
    ref = alloc["ref"]
    if not args.quiet:
        warn(f"allocated {','.join(alloc['aliases'])} (allocation {ref})")
    try:
        code = run_command(command, claim["env"], on_start=lambda pid: _attach(client, ref, pid))
    finally:
        wait = float(os.environ.get("RESOURCED_RELEASE_WAIT", "10"))
        try:
            result = client.release(ref, wait=wait, on_busy="warn")
            if result["leftover"]:
                warn(
                    "released while processes still use the GPU(s): "
                    + ", ".join(f"pid {p['pid']} ({p['command']})" for p in result["leftover"])
                )
        except ResourcedError as exc:
            if exc.code != "unknown_allocation":  # already force-released: nothing to do
                warn(f"could not release allocation {ref}: {exc.message}. Free it with `gpuctl release --force`.")
    return code


def _attach(client: Client, ref: str, pid: int) -> None:
    try:
        client.request("attach", ref=ref, pid=pid)
    except ResourcedError as exc:
        warn(f"could not register the command PID: {exc.message}")


def cmd_env(args) -> int:
    info = make_client(args).get(args.allocation)
    for key, value in info["env"].items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


def cmd_status(args) -> int:
    _, task_id = task_identity(args)
    status = make_client(args).status(task_id or None)
    if args.json:
        print_json(status)
        return 0
    print(
        f"{status['total']} GPUs: {status['available']} available, {status['used']} in use "
        f"(resourced {__version__}, pid {status['pid']}, up {fmt_age(status['uptime_s'])})"
    )
    if status["driver"] != "ok":
        print("  warning: GPU presence and process checks are unknown")
        print(f"    {status.get('driver_error') or 'nvidia-smi is not answering'}")
    for row in status["gpus"]:
        index = "?" if row["index"] is None else row["index"]
        flag = "  (MISSING from driver)" if row["present"] is False else ""
        line = f"  {row['alias']:<8} idx {index!s:<3} {short_uuid(row['uuid']):<13} {row['state']:<10}"
        if row["allocation"]:
            line += " " + describe_allocation(row["allocation"])
        print(line + flag)
    if status["unmanaged_claims"]:
        print("  claims on GPUs not in the config: " + ", ".join(status["unmanaged_claims"]))
    if status["mine"]:
        print(f"allocations of task {task_id}:")
        for a in status["mine"]:
            print(f"  {a['ref']}  {','.join(a['aliases'])}  {a['mode']}")
    return 0


def cmd_free(args) -> int:
    status = make_client(args).status()
    free = [r for r in status["gpus"] if r["state"] == "AVAILABLE" and r["present"] is not False]
    if args.json:
        print_json({"available": len(free), "total": status["total"], "gpus": [r["alias"] for r in free]})
    else:
        print(f"{len(free)} of {status['total']} GPUs available" + (": " + ", ".join(r["alias"] for r in free) if free else ""))
    return 0


def cmd_verify(args) -> int:
    result = make_client(args).verify()
    if args.json:
        print_json(result)
        return 0 if result["healthy"] else 1
    if result["probe"] == "unknown":
        print("warning: process checks are unknown")
        print(f"  {result.get('probe_error') or 'nvidia-smi is not answering'}")
    for row in result["gpus"]:
        line = f"{row['alias']:<8} {row['state']}"
        if row["allocation"]:
            line += "  claim: " + describe_allocation(row["allocation"])
        print(line)
        for p in row["processes"]:
            mark = {True: "belongs to claim", False: "NOT part of the claim", None: ""}[p["belongs"]]
            mem = f" {p['memory_mib']} MiB" if p["memory_mib"] is not None else ""
            print(f"    pid {p['pid']} {p['user'] or '?'}{mem} {p['command']}  {mark}".rstrip())
    print("OK" if result["healthy"] else "PROBLEMS FOUND")
    return 0 if result["healthy"] else 1


_LIMIT_UNITS = {"free_gib": " GiB free", "free_pct": "% free", "used_pct": "% used"}


def format_check(check: dict) -> str:
    cur = check["current"]
    if check["health"] in ("ERROR", "UNAVAILABLE"):
        measure = check["error"] or "no measurement"
    else:
        measure = f"{cur['free_gib']:.1f} GiB free of {cur['total_gib']:.1f} GiB ({cur['free_pct']:.1f}% free)"
    limits = "  ".join(
        f"{lim['level'].lower()}: {lim['op']} {lim['value']:g}{_LIMIT_UNITS[lim['metric']]}" for lim in check["limits"]
    )
    where = check["path"] or "RAM (MemAvailable)"
    return f"{check['name']:<10} {check['health']:<11} {where}: {measure}\n{'':<22}limits {limits}"


def cmd_resources(args) -> int:
    try:
        snapshot = make_client(args).resources()
        source = "daemon"
    except ResourcedError as exc:
        if exc.code != "daemon_unreachable":
            raise
        warn("daemon not reachable; measuring locally")
        snapshot = local_snapshot(load_config(args.config))
        source = "local"
    if args.json:
        print_json({**snapshot, "source": source})
    elif not snapshot["checks"]:
        print("no resource checks configured")
    else:
        for check in snapshot["checks"]:
            print(format_check(check))
    return 0


def cmd_init(args) -> int:
    nvidia_smi = find_nvidia_smi()
    if nvidia_smi is None:
        print(f"gpuctl: cannot create a config: {nvidia_smi_missing_message()}", file=sys.stderr)
        return 1
    probe = NvidiaSmiProbe(nvidia_smi)
    gpus = probe.gpus()
    if not gpus:
        reason = probe.last_error or "nvidia-smi did not list any GPUs"
        print(f"gpuctl: cannot create a config: {reason}", file=sys.stderr)
        return 1
    target = Path(args.output) if args.output else (SYSTEM_CONFIG if args.system else user_config_path())
    if target.exists() and not args.force:
        print(f"gpuctl: {target} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_config(gpus, system=args.system, nvidia_smi=nvidia_smi))
    print(f"wrote {target} with {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  gpu{gpu['index']}  {gpu['uuid']}  {gpu['name']}")
    mode = "--system" if args.system else "--user"
    print(f"next: gpuctl service enable {mode}")
    try:
        socket_dir = SYSTEM_RUNTIME_DIR if args.system else default_runtime_dir()
    except ConfigError:
        socket_dir = None
    if socket_dir:
        print(
            f"\nIf gpuctl runs inside a sandboxed coding agent (e.g. Codex CLI's workspace-write mode), it needs\n"
            f"{socket_dir} reachable (plus any disk-check path you add to this config). For Codex CLI, add under\n"
            f'[sandbox_workspace_write] in ~/.codex/config.toml: writable_roots = ["{socket_dir}/"]'
        )
    return 0


def cmd_service(args) -> int:
    mode = "system" if args.system else "user"
    if args.action == "enable":
        return service.enable(mode, args.config, args.dry_run)
    return service.disable(mode, args.dry_run)


def cmd_agent_instructions(args) -> int:
    print(AGENT_INSTRUCTIONS, end="")
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuctl", description="Cooperative GPU allocation and resource limits")
    parser.add_argument("--version", action="version", version=__version__)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--socket", help="daemon socket (default: found automatically)")
    common.add_argument("--config", help="config file (used to find the runtime directory)")

    identity = argparse.ArgumentParser(add_help=False)
    identity.add_argument("--task-name", help="readable name of your task, e.g. the Codex/Claude task (or RESOURCED_TASK_NAME)")
    identity.add_argument("--task-id", help="task ID (or RESOURCED_TASK_ID; default: the task name)")
    identity.add_argument("--label", help="what the GPUs are used for")

    sub = parser.add_subparsers(dest="command_name", required=True)

    p = sub.add_parser("claim", parents=[common, identity], help="claim GPUs for a development session")
    p.add_argument("--gpus", type=int, required=True, help="number of GPUs")
    p.add_argument("--owner-pid", type=int, help="PID of the long-lived owner, e.g. the Codex/Claude process (or RESOURCED_OWNER_PID)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_claim)

    p = sub.add_parser("release", parents=[common], help="release an allocation (or --force a GPU)")
    p.add_argument("--allocation", help="allocation reference printed by claim")
    p.add_argument("--gpu", help="alias, index or UUID (only with --force)")
    p.add_argument("--force", action="store_true", help="release even if processes are running")
    p.add_argument("--reason", help="why (logged)")
    p.add_argument("--wait", type=float, default=0, help="seconds to wait for GPU processes to exit")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("run", parents=[common, identity], help="run a command on GPUs: run --gpus N -- cmd | run --allocation REF -- cmd")
    p.add_argument("--gpus", type=int, help="one-off: allocate N GPUs for this command and release afterwards")
    p.add_argument("--allocation", help="development: run on an existing claim (never released)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("env", parents=[common], help="print shell exports for an allocation")
    p.add_argument("--allocation", required=True)
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("status", parents=[common, identity], help="show GPUs and who holds them")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("free", parents=[common], help="show how many GPUs are available")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_free)

    p = sub.add_parser("verify", parents=[common], help="compare claims with processes on the GPUs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("resources", parents=[common], help="show RAM/disk limits and measurements")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_resources)

    p = sub.add_parser("init", help="write a starter config from nvidia-smi")
    p.add_argument("--system", action="store_true", help=f"write {SYSTEM_CONFIG}")
    p.add_argument("--output", help="config file to write")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("service", help="enable or disable start at boot (systemd)")
    p.add_argument("action", choices=["enable", "disable"])
    group = p.add_mutually_exclusive_group()
    group.add_argument("--user", action="store_true", help="per-user service (default)")
    group.add_argument("--system", action="store_true", help="system-wide service (root)")
    p.add_argument("--config", help="config file the service should use")
    p.add_argument("--dry-run", action="store_true", help="print the unit and commands only")
    p.set_defaults(func=cmd_service)

    p = sub.add_parser("agent-instructions", help="print a snippet for CLAUDE.md / AGENTS.md")
    p.set_defaults(func=cmd_agent_instructions)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1 :]
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command_name == "run":
            return cmd_run(args, command)
        if command:
            parser.error("unexpected arguments after --")
        return args.func(args)
    except ResourcedError as exc:
        print(f"gpuctl: {exc.message}", file=sys.stderr)
        return EXIT_CODES.get(exc.code, 1)
    except ConfigError as exc:
        print(f"gpuctl: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
