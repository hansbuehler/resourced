# resourced implementation plan

## Handoff and scope

This is the plan for the agent that implements resourced. It is a plan, not an
implementation. `/home/hans/python/resourced` is the git clone of
`git@github.com:hansbuehler/resourced.git` (initial commit with LICENSE,
.gitignore, README). Implement here; do not create a second directory.

resourced is a standalone Python project with no dependency on AAE or any
agent harness.

## Design principle: keep it simple

resourced is a **lightweight tool for one user or a few users on one machine**,
not an enterprise service. Users can talk to each other if something goes
wrong. Priorities, in order: ease of use, speed, low complexity.

- Prefer the simplest mechanism that works. Do not add features, roles,
  protocol layers, or safeguards beyond this document.
- Failures should be visible (clear error message, status output) and
  recoverable with one command (`gpuctl release --force`, restart the daemon).
- Standard library only (plus pytest for tests). Linux only. Python 3.13.

## Goals

One small daemon with two independent functions:

1. **GPU allocation.** Cooperative allocation of GPUs between agents, scripts,
   and a few users.
2. **Resource limits.** Publish configured RAM and disk warning/stop limits and
   live measurements, so tasks can follow them.

Resource STOP levels never block GPU claims and never kill anything. Tasks are
responsible for following the limits. No agent needs to supervise the service.

## Non-goals

- No GPU ownership, reservation, or reclaim.
- No waiting queue: a claim fails immediately if GPUs are unavailable; the caller
  retries, uses the CPU, or tells the user.
- No automatic expiry of claims; lost claims are fixed with force release.
- No automatic resource-driven kill or cleanup, no notification hook.
- No MCP server, no OS-level enforcement, no AAE integration.
- No roles or per-user permissions: anyone who can reach the socket may do
  everything.

## Architecture

- Daemon `resourced`, CLI `gpuctl`, and a thin Python client using the same
  protocol.
- Newline-delimited JSON over a Unix domain socket. Keep messages small (cap the
  size), time out slow clients, and return clear error messages. A simple
  `"version"` field is enough.
- The daemon serializes all mutations; multi-GPU claims are all-or-nothing.
- Claims are saved to a file in the runtime directory by atomic replace (write
  temp file, rename) before replying.
- The peer pid (and user name, for display) from the socket is recorded with each
  claim; see Process check.

Layout:

```text
pyproject.toml   (requires-python >=3.13)
README.md        (user documentation)
BUILD.md         (this file: record of agent instructions)
src/resourced/   (systemd unit templates are rendered in service.py)
tests/
examples/config.toml
```

## GPU semantics

- GPU UUID is authoritative; the index is a display/selection hint. Only GPUs
  listed in the configuration are managed.
- A GPU is `AVAILABLE` or `USED`. `USED` records a readable **task name** (for
  example the Codex or Claude task, given by `--task-name` or
  `RESOURCED_TASK_NAME`; if missing, fall back to the calling process's command
  name), the task ID, the user name (display only), the optional **owner PID**
  (see Process check), label, timestamp, and a fresh unique `allocation_id`.
  Identity is by PID and task name, not by Unix uid.
- Claim all GPUs for a stage in one step, and pin every process to those UUIDs
  (`CUDA_VISIBLE_DEVICES` by UUID). Claim only what the current stage needs.
- Release requires the expected `allocation_id`, so a stale release can never
  clear a newer claim. Timestamps and liveness information are for display only.
- Record the root task's name and ID, even if a subagent launches the job.
- The same task may hold several allocations at once (parallel jobs, different
  models). Each has its own ID and disjoint GPUs.
- A multi-GPU allocation has one reference that stands for its per-GPU
  allocation IDs.

### Ad-hoc mode: `gpuctl run --gpus N -- command ...`

The service picks N available GPUs (in configured order) atomically and starts
the command with them pinned. When the command exits, normally or not, the GPUs
are released and the command's exit code is returned. Signals (INT/TERM/HUP) are
forwarded to the command.

Ad-hoc and development allocations never mix: `run --gpus N` always takes new
GPUs and never uses a development claim of the same task.

Known limit (document in README): a process the command deliberately detaches
is not tracked and may keep using a GPU after release. If the wrapper itself is
SIGKILLed, the GPUs stay allocated until force release.

### Development mode: explicit claim and release

```sh
gpuctl claim --gpus 1 --label jax-dev --owner-pid <pid-of-agent>   # or RESOURCED_OWNER_PID
gpuctl run --allocation <ref> -- python smoke.py     # may be repeated
gpuctl release --allocation <ref>
```

The claim stays until explicitly released; it survives the CLI exit.
`run --allocation` never releases it. `gpuctl env --allocation <ref>` prints the
pinning variables for shell use (environment variables are not authority).

### Force release

```sh
gpuctl release --force --gpu <alias|index|uuid> [--reason "<text>"]
```

For lost claims (crashed wrapper, forgotten development claim). It releases
regardless of what is running, prints who held it, and is logged. It needs no
TTY. A wrapper whose allocation was force-released ignores the stale ID when it
exits.

### Process check (claim, release, verify)

The daemon cross-checks claims against the processes actually using the GPUs.
The probe is
`nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits`
(compute processes; graphics-only processes are ignored). Put it behind a small
probe interface so tests use fake GPUs, and run it with the probe timeout.

Most of the check needs no knowledge of who owns a process, only of whether a
GPU has any process at all:
- an available GPU should have none (claim time, `verify`),
- a released GPU should have none (release time).

Only one check needs attribution: whether the processes on a *used* GPU are the
claimant's. The service does **not** guess this. Attribution is known only here:
- **Ad-hoc `run`:** the wrapper knows the PID of the command it launched. A GPU
  process belongs to the claim if it is that PID or its descendant (parent chain
  via `/proc/<pid>/stat`).
- **Development `claim`:** the caller must pass `--owner-pid <pid>` (or set
  `RESOURCED_OWNER_PID`), typically the PID of the Codex/Claude process; a claim
  without one is rejected with a clear error explaining this. It is
  stored with its start time (so a reused PID is not mistaken for it), printed by
  `claim`, and shown in `status` (PID, command, alive or dead). A GPU process
  belongs to the claim if it is the owner PID or a descendant. There is no
  auto-detection.
  `run --allocation` does not register anything.

- **On claim:** an available GPU that has any compute process means something
  went wrong (a process bypassed the tool or a release was missed). It is
  reported as an **error** (message in the reply, error line in the daemon log,
  and `UNCLAIMED USE` in `status`/`verify`) and the GPU is not granted.
  Auto-selection skips such GPUs; if too few clean GPUs remain, the claim fails and lists the
  offending PIDs (pid, user, command, memory). There is no override; stop the
  process or investigate.
- **On release:** the GPU should have no compute process.
  - Explicit `release` fails and lists the remaining PIDs ("stop them first");
    `--force` releases anyway.
  - Ad-hoc `run`: after the command exits, wait up to 10 seconds for the driver
    to clear the GPU, then release anyway and print a warning with the leftover
    PIDs on stderr (never block forever, never change the command's exit code).
- **If the probe is unavailable** (no `nvidia-smi`, timeout, error), the check
  is reported as unknown: claim and release proceed with a warning. A broken
  probe never blocks GPU allocation.
- **`gpuctl verify`** (read-only) compares every managed GPU with its claim:
  - `OK`: used GPU whose processes all belong to the claim, or an idle available GPU
  - `IDLE CLAIM`: used GPU with no process (informational)
  - `UNCLAIMED USE`: available GPU with processes
  - `FOREIGN PROCESS`: used GPU with a process that does not belong to the claim

  For every process it prints the PID, command, memory, and user, and for the
  claim the readable task name (for example "process 4711 python on GPU 2 is
  not part of claim 'codex: jax-dev'"). It supports `--json` and
  exits 0 only if no `UNCLAIMED USE` or `FOREIGN PROCESS` is found (so it can be
  used in scripts). It never changes any state.

Known limits (document in README): a process that detaches (daemonizes, is reparented to init)
loses its ancestry and shows up as `FOREIGN PROCESS`, as do all processes of a
claim whose owner PID has exited (for example the agent was restarted); `status`
shows the owner as dead, and the fix is release or force release. The check is a point-in-time snapshot and
the check-then-grant step is not atomic with respect to a program that starts
using a GPU right afterwards; inside containers PIDs may be namespaced, in which case the check is
reported as unknown for those entries; MIG partitions are not handled.

### Status

`gpuctl status` (human text, `--json` for machines; see `gpuctl verify` for the
process check) shows: number of GPUs total,
available, and used; per GPU the UUID, index, alias, and claimant (task name,
task ID, user, label, mode, age, and the owner PID with its command and
whether it is still alive); the
allocations of the caller's task; and daemon health. Also `gpuctl free`.

## Configuration

Lookup: `--config`, then `RESOURCED_CONFIG`, then
`$XDG_CONFIG_HOME/resourced/config.toml` (default `~/.config/...`), then
`/etc/resourced/config.toml`.

TOML with: managed GPUs (UUID, alias, optional index), resource checks, sample
interval, optional `nvidia-smi` path, probe timeout (default 10 s), and optional socket group. To reload,
restart the daemon; claims survive restarts. An invalid config makes the daemon
refuse to start, with a clear message. Ship `examples/config.toml`.

## Runtime state and install modes

There is no durable state, no `/tmp` use, and no ownership file.

| Mode | Runtime directory |
| --- | --- |
| Per-user (default) | `$XDG_RUNTIME_DIR/resourced/` |
| System-wide | `/run/resourced/` (`RuntimeDirectory=resourced`, `RuntimeDirectoryPreserve=yes`) |

The directory holds the socket, `claims.json`, and the resource snapshot. It is
tmpfs: claims survive a daemon restart but are gone after a reboot, so
everything asks for GPUs again after a reboot. If `$XDG_RUNTIME_DIR` is missing,
fail with a clear error. Per-user installs need lingering to keep running after
logout (document it). A second daemon on the same directory must refuse to
start (simple lock file).

Access: socket mode 0600 by default (single user). For a few users, a
configured group and mode 0660. Never world-writable. A system install runs as a
dedicated `resourced` account.

## Resource monitoring (advisory only)

`gpuctl resources` (and the client) returns limits and measurements for each
check: metric/path, units, warning/stop limits, current and previous value,
timestamp, and health. Limits remain visible even if a measurement fails.

- RAM: `MemAvailable` (not `MemFree`), limits in GiB (2^30 bytes).
- Disk: available space (`f_bavail`) of the filesystem holding each configured
  path. Keep each path's own thresholds, even when two paths share a filesystem.
- A probe that fails, times out (10 s), or targets a missing path is reported
  as `error`/`unavailable`, never as healthy.
- Sampling runs in a background thread and never delays GPU requests.
- Log level changes (OK to WARN to STOP). No latch, acknowledgment, kill, or
  cleanup.

Thresholds are inclusive ("at"):

| Metric | Level | Triggered when |
| --- | --- | --- |
| Free RAM / free disk (GiB or %) | warning | value <= warning limit |
| Free RAM / free disk (GiB or %) | stop | value <= stop limit |
| Used disk (%) | warning, stop | value >= limit |

Health is the highest triggered level. Example values (not defaults): RAM warn
200 GiB, stop 100 GiB; a storage path warn 20% free, stop 10%; temporary
filesystem warn 80% used, stop 90%.

## Service operation

- Ship systemd user and system unit files in `systemd/`
  (`Restart=on-failure`, logs to the journal). Do not use `PrivateTmp` or other
  settings that hide the socket or monitored filesystems.
- **Auto-start at boot** (opt-in, through systemd):

  ```sh
  gpuctl service enable  [--user|--system]   # install unit, enable on boot, start now
  gpuctl service disable [--user|--system]
  ```

  `enable` writes the unit from the template (with the resolved config path and
  Python), runs `systemctl [--user] enable --now resourced.service`, and, for
  `--user`, prints a hint if lingering is off. `--system` needs root and the
  command prints what to run otherwise. README also documents the unit files
  for manual installation.
- A managed GPU missing from the driver is reported in status, not fatal.

## Deployment

The project is an ordinary installable package with the console scripts
`resourced` (daemon) and `gpuctl` (CLI), standard library only. A new machine
needs three steps:

1. Install: `pipx install git+https://github.com/hansbuehler/resourced` (or
   `uv tool install ...`). Requires Python 3.13 and `nvidia-smi`.
2. `gpuctl init` reads `nvidia-smi` and writes a starter `config.toml` (GPUs
   with UUID, index, alias `gpuN`; commented example resource checks). It does
   not overwrite an existing file without `--force`; `--system` targets
   `/etc/resourced/config.toml`; `--output` selects a path.
3. `gpuctl service enable --user` (or `--system`), see Service operation.
   `--dry-run` prints the unit and commands without doing anything.

Rules:
- Exactly one daemon per set of GPUs. Several users on one machine use one
  system install with `socket_group`, not one daemon each (separate daemons
  would not see each other's claims). The CLI finds the socket by itself:
  `--socket`, `RESOURCED_SOCKET`, the config's `runtime_dir`,
  `$XDG_RUNTIME_DIR/resourced/`, then `/run/resourced/`.
- System mode needs an installation readable by the `resourced` account (not
  inside a private home directory); `service enable --system` warns about this.
- Update with `pipx upgrade resourced` plus a service restart; claims survive.
- `gpuctl agent-instructions` prints a short snippet for `CLAUDE.md` or
  `AGENTS.md`: use `gpuctl run --gpus N -- <cmd>` for jobs, `claim` with
  `--owner-pid` and an explicit `release` for development sessions, and
  `gpuctl resources` for limits.
- Not in scope: Docker images, distro packages, PyPI release.

## Implementation sequence

1. Package metadata, test layout.
2. State machine and claims file.
3. Socket protocol, daemon, client.
4. CLI: `claim`, `release`, `release --force`, `run`, `env`, `status`, `free`,
   `verify`, including the process check described above.
5. Resource monitoring and `gpuctl resources`.
6. systemd units, `gpuctl service`, README.
7. Test with fake GPUs and an isolated temporary runtime directory. Do not
   install or enable any real service on the host without the user's say-so.

Workflow: commit directly to `main` in small commits; do not push unless asked.
`README.md` is user documentation and replaces the one-line README; keep
`BUILD.md` as is.

## Tests

Fast, offline, CPU-only (fake GPUs, temporary runtime dir):

- Concurrent single-GPU claims from one task get distinct GPUs.
- Multi-GPU claims are all-or-nothing.
- A stale allocation ID cannot release a newer claim.
- `run --gpus N` releases on exit, failure, and signal, passes through the exit
  code, and never takes development-claimed GPUs.
- Development claims survive client exit and a daemon restart; they are gone in a
  fresh runtime directory.
- Force release works non-interactively; a wrapper tolerates it.
- Owner PID: a development `claim` without `--owner-pid` or
  `RESOURCED_OWNER_PID` is rejected; the owner is printed and shown in `status`
  with a dead/alive flag; descendant matching and reused-PID protection.
- Claim skips available GPUs that have any process, logs and reports an error for
  them, and fails with the PIDs listed if too few clean GPUs remain.
- Release is refused while processes remain (explicit release), `--force`
  overrides, and an ad-hoc `run` warns after the 10-second wait and still
  releases.
- `verify` classifies OK, IDLE CLAIM, UNCLAIMED USE, FOREIGN PROCESS and sets
  its exit code accordingly; an unavailable probe is reported as unknown and
  never blocks claim or release.
- Status shows counts, claimants, and the caller's allocations.
- Bad config, malformed or oversized requests, and failed probes give clear
  errors.
- Resource threshold boundaries and GiB conversion; probe timeout.
- Resource STOP or probe failure never blocks claims.
- A second daemon on the same runtime directory refuses to start.
- Unit files render (and pass `systemd-analyze verify` if available).

## README

Explain: purpose and cooperative nature, install modes (user and system),
config lookup and schema, runtime directory and reboot behaviour, CLI and Python
usage, ad-hoc versus development mode, parallel jobs, force release, auto-start,
resource limits and their semantics, known limits, and troubleshooting.

## Left to the implementer

Exact JSON schema, allocation reference format, exit codes, and group
provisioning for shared use. Keep them simple and document them.
