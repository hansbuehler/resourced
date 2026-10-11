# resourced

Lightweight, cooperative GPU allocation for one machine shared by a few people, scripts and AI agents,
plus advisory RAM/disk limits that tasks can follow.

- **`resourced`**: a small daemon (standard library only, Linux, Python 3.13).
- **`gpuctl`**: the command line tool. A Python client (`resourced.Client`) speaks the same protocol.

It is **cooperative**: nothing stops a program that ignores it. `gpuctl verify` and the checks at claim and
release time make such programs visible. It is meant for a single user or a few users who can talk to each
other when something goes wrong, not for an enterprise setup.

## Install

```sh
pipx install git+https://github.com/hansbuehler/resourced     # or: uv tool install git+https://github.com/hansbuehler/resourced
gpuctl init                       # reads nvidia-smi, writes ~/.config/resourced/config.toml
gpuctl service enable --user      # install the systemd unit, start at boot, start now
gpuctl status
```

Needs Python 3.13 and `nvidia-smi`. Both `pipx` and `uv tool` put `gpuctl` in `~/.local/bin`; if the shell
cannot find it, run `pipx ensurepath` or `uv tool update-shell` and open a new shell. Coding agents that run
commands in a sandbox need one extra setting, see [Sandboxed coding agents](#sandboxed-coding-agents).
`gpuctl service enable --dry-run` shows the unit and commands without
doing anything. `gpuctl service disable` stops the autostart.

Per-user installs need *lingering* so the service keeps running after you log out
(`sudo loginctl enable-linger $USER`); `gpuctl service enable --user` tells you when it is off.

Update with `pipx upgrade resourced` (or `uv tool upgrade resourced`) and restart the service
(`systemctl --user restart resourced`); claims survive a restart.

### Install modes

| Mode | Runtime directory | Socket | Use for |
| --- | --- | --- | --- |
| Per-user (default) | `$XDG_RUNTIME_DIR/resourced/` | private (0600) | one person |
| System-wide | `/run/resourced/` | group (0660) via `socket_group` | several users |

There must be **exactly one daemon per set of GPUs**. Several users on one machine share one system install;
separate per-user daemons would not see each other's claims and could hand out the same GPU twice.

### System-wide install (several users)

The service runs as a dedicated `resourced` account, which cannot read home directories. Install the package
in a location every user can read, such as a virtualenv under `/opt` (not `pipx` in your home directory), and run
every `gpuctl` command below from that environment: the unit records the Python that runs `service enable`.

```sh
# 1. Install where the 'resourced' account can read it (needs Python 3.13)
sudo python3.13 -m venv /opt/resourced
sudo /opt/resourced/bin/pip install git+https://github.com/hansbuehler/resourced
sudo ln -s /opt/resourced/bin/gpuctl /usr/local/bin/gpuctl        # optional: gpuctl for all users

# 2. Write /etc/resourced/config.toml from nvidia-smi
sudo /opt/resourced/bin/gpuctl init --system

# 3. Create the access group and add users (log in again for membership to apply)
sudo groupadd gpuusers
sudo usermod -aG gpuusers alice

# 4. Edit /etc/resourced/config.toml: uncomment/set  socket_group = "gpuusers"  (and any resource checks)

# 5. Preview, then install and start
sudo /opt/resourced/bin/gpuctl service enable --system --dry-run
sudo /opt/resourced/bin/gpuctl service enable --system     # creates the account, writes the unit, enables and starts it
gpuctl status
```

Notes:

- **Set `socket_group` before step 5.** Without it the socket is private (0600, owned by `resourced`) and ordinary
  users get `no permission to use ... socket`. The unit adds the group to the service account automatically.
- The CLI finds the socket by itself (`--socket`, `RESOURCED_SOCKET`, the config's `runtime_dir`,
  `$XDG_RUNTIME_DIR/resourced/`, then `/run/resourced/`), so users of the shared install need no configuration.
- Run only one daemon per set of GPUs: `gpuctl service disable --user` any per-user service first.
- System services start at boot; lingering is not needed.
- Update: `sudo /opt/resourced/bin/pip install --upgrade git+https://github.com/hansbuehler/resourced`, then
  `sudo systemctl restart resourced` (claims survive).
- Remove: `sudo /opt/resourced/bin/gpuctl service disable --system`.
- Logs: `journalctl -u resourced`; state: `systemctl status resourced`.

### Manual unit files

`gpuctl service enable` writes these for you (with the real interpreter and config path). To install by hand, save
the user unit as `~/.config/systemd/user/resourced.service` and run
`systemctl --user daemon-reload && systemctl --user enable --now resourced`:

```ini
[Unit]
Description=resourced: GPU allocation and resource limits

[Service]
Type=simple
ExecStart=/path/to/python -m resourced.daemon --config /path/to/config.toml
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

For the system unit, create the account first
(`sudo useradd --system --no-create-home --shell /usr/sbin/nologin resourced`), save it as
`/etc/systemd/system/resourced.service`, and run `sudo systemctl daemon-reload && sudo systemctl enable --now resourced`:

```ini
[Unit]
Description=resourced: GPU allocation and resource limits
After=local-fs.target

[Service]
Type=simple
User=resourced
SupplementaryGroups=gpuusers
Environment=RESOURCED_RUNTIME_DIR=/run/resourced
ExecStart=/opt/resourced/bin/python -m resourced.daemon --config /etc/resourced/config.toml
RuntimeDirectory=resourced
RuntimeDirectoryMode=0755
RuntimeDirectoryPreserve=yes
Restart=on-failure
RestartSec=2
NoNewPrivileges=yes
ProtectSystem=full

[Install]
WantedBy=multi-user.target
```

Do not add `PrivateTmp` or similar settings that hide the socket or the monitored filesystems.

## Using GPUs

### One-off jobs (ad-hoc)

```sh
gpuctl run --gpus 2 -- python train.py
```

The service picks 2 free GPUs, starts the command with `CUDA_VISIBLE_DEVICES` set to their UUIDs, and releases
them when the command exits (normally, on failure, or on a signal). The command's exit code is returned. If not
enough GPUs are free it fails immediately with exit code 3; retry later, use the CPU, or ask the user.
Run several jobs at once, each gets its own GPUs.

### Development sessions

For compiling, testing and tuning where the GPU must stay yours between commands:

```sh
gpuctl claim --gpus 1 --label jax-dev --task-name "my task" --owner-pid <PID of your long-lived process>
gpuctl run --allocation <ref> -- python smoke.py        # as often as you like
eval "$(gpuctl env --allocation <ref>)"                  # or pin a shell yourself
gpuctl release --allocation <ref>                        # always release when done
```

The claim stays until explicitly released. The **owner PID** (for an agent: its own Codex/Claude process) is
required, is shown in `gpuctl status`, and is what `gpuctl verify` uses to tell your processes from others.
One-off runs never use a development claim's GPUs.

### Status, checks, recovery

```sh
gpuctl status [--json]     # who holds which GPU: task name, user, mode, age, owner PID and whether it is alive
gpuctl free                # how many are available
gpuctl verify              # compare claims with the processes actually on the GPUs (exit 1 on problems)
gpuctl release --force --gpu gpu1 --reason "agent crashed"      # recover a lost claim, no TTY needed
```

`verify` reports `OK`, `IDLE CLAIM` (claimed, nothing running), `UNCLAIMED USE` (a process on an available GPU),
`FOREIGN PROCESS` (a process on a claimed GPU that is not the claimant's), and `MISSING` (GPU not seen by the
driver). `UNKNOWN` means `nvidia-smi` did not answer.

What is checked, and when:

- **Claim:** an available GPU that has any compute process is an *error* (something bypassed the tool or a release
  was missed). It is logged, not granted, and listed with its PIDs; another GPU is used if possible.
- **Release:** an explicit release is refused while processes still use the GPU (stop them, or `--force`).
  For one-off runs, `gpuctl` waits up to 10 seconds (`RESOURCED_RELEASE_WAIT`) for the driver to clear the GPU, then
  releases anyway with a warning, and never changes the command's exit code.
- If `nvidia-smi` is unavailable the checks are reported as unknown and claims and releases proceed with a warning.

A process belongs to a claim if it is the owner PID (development) or the launched command (one-off), or a
descendant of it. PIDs are stored with their start time, so a reused PID is not mistaken for the original.

### Python

```python
import os

from resourced import Client

client = Client()                      # finds the socket automatically
claim = client.claim(mode="dev", gpus=1, task_name="my task", owner_pid=os.getpid())
env = claim["env"]                     # {"CUDA_VISIBLE_DEVICES": ..., "RESOURCED_ALLOCATION": ...}
client.release(claim["allocation"]["ref"])
```

Errors raise `resourced.ResourcedError` with `.code` and `.message`.

### Telling agents about it

`gpuctl agent-instructions` prints a short snippet for `CLAUDE.md` / `AGENTS.md`. For a machine-wide baseline
(project-independent resource guardrails: output directories, progress logging, detached-job environment
variables, `gpuctl resources` thresholds and STOP handling, treating Git trees as source), merge
[examples/HOST_AGENTS.md](examples/HOST_AGENTS.md) and the `gpuctl agent-instructions` output into the agent's
user-level instructions file, which applies to every project of that user:

| Agent | User-level file | Machine-wide (all users) |
| --- | --- | --- |
| Codex CLI | `~/.codex/AGENTS.md` | — |
| Claude Code | `~/.claude/CLAUDE.md` | `/etc/claude-code/CLAUDE.md` |

[examples/PROJECT_AGENTS.md](examples/PROJECT_AGENTS.md) is a starting template for a single project's own
`AGENTS.md` / `CLAUDE.md` on top of that baseline. Remove any older GPU or resource policy from project files, so
agents do not follow two sets of rules. These files are instructions, not enforcement: `gpuctl verify` is what
detects a program that bypassed `gpuctl`.

### Sandboxed coding agents

`gpuctl` talks to the daemon over a Unix socket. Agents that run commands in a sandbox must be allowed to connect
to it, otherwise `gpuctl` reports `no permission to use ... socket` and `gpuctl resources` falls back to measuring
locally.

- **Codex CLI** (tested with 0.149.1 on Linux) treats a Unix socket connection as network access. Adding the
  runtime directory to `writable_roots` does not help. In `~/.codex/config.toml`:

  ```toml
  sandbox_mode = "workspace-write"

  [sandbox_workspace_write]
  network_access = true
  ```

  This also gives sandboxed commands general network access. With the `network_proxy` feature enabled, Unix
  socket destinations additionally need an explicit allow rule (not tested here).
- Check it from inside the sandbox:
  `codex sandbox -c 'sandbox_mode="workspace-write"' gpuctl status` (use the full path to `gpuctl` if
  `~/.local/bin` is not on the sandbox's `PATH`).
- **GPUs are invisible inside the Codex sandbox.** It gives commands a fresh `/dev` without the NVIDIA device
  files, and neither `writable_roots` nor permission profiles can add them (tested with 0.149.1). A claim made
  in the sandbox succeeds, but the program then finds no GPU. Since GPU work is Python, let only
  `gpuctl run --gpus N -- python ...` start outside the sandbox, with a Codex rule in
  `~/.codex/rules/default.rules` (adjust the paths to your install and interpreters):

  ```
  prefix_rule(
      pattern=[["gpuctl", "/home/hans/.local/bin/gpuctl"], "run", "--gpus", ["1", "2", "3", "4"], "--",
               ["python", "python3", "/home/hans/vaae/bin/python", "/home/hans/vaae/bin/python3"]],
      decision="allow",
  )
  ```

  Agents must then launch GPU processes exactly in that form; a command started after
  `eval "$(gpuctl env ...)"`, with other options before `--gpus`, or through `gpuctl run --allocation REF`
  (the reference changes every time) still runs in the sandbox unless you approve it. Anything after
  `-- python` runs unsandboxed without a prompt. `nvidia-smi` needs no rule: the daemon runs it outside the
  sandbox. Check the rule with `codex execpolicy check --rules ~/.codex/rules/default.rules gpuctl run --gpus 1 -- python x.py`.
- Other agents: allow connecting to `$XDG_RUNTIME_DIR/resourced/resourced.sock` (per-user install) or
  `/run/resourced/resourced.sock` (system install), and give GPU commands access to `/dev/nvidia*`.

### Exit codes

`0` ok, `1` other error, `2` bad usage or config, `3` not enough GPUs, `4` GPUs busy with unknown processes,
`5` daemon unreachable, `6` unknown or stale allocation. `gpuctl run` returns the command's own exit code
(`128+n` if it died from signal `n`); errors from gpuctl itself are prefixed with `gpuctl:`.

## Resource limits (advisory)

```sh
gpuctl resources [--json]
```

Shows the configured limits and live measurements (RAM from `MemAvailable`, disk space from `f_bavail` of the
filesystem holding each path). Limits stay visible even if a measurement fails; a failed, timed-out (10 s) or missing
path is reported as `ERROR`/`UNAVAILABLE`, never as healthy. If the daemon is down it measures locally.

Limits are **inclusive**: a level is reached *at* its value.

| Key | Reached when |
| --- | --- |
| `warn_free_gib`, `stop_free_gib`, `warn_free_pct`, `stop_free_pct` | free value <= limit |
| `warn_used_pct`, `stop_used_pct` | used % >= limit |

The highest level reached is the health (`OK`, `WARN`, `STOP`). Resource levels **never block GPU claims and never
kill or clean up anything**; tasks are responsible for following them. Level changes are logged.

## Configuration

Lookup: `--config`, `RESOURCED_CONFIG`, `$XDG_CONFIG_HOME/resourced/config.toml` (default
`~/.config/resourced/config.toml`), `/etc/resourced/config.toml`. See [examples/config.toml](examples/config.toml).
Restart the daemon to apply changes; an invalid file makes it refuse to start with a clear message.

```toml
[daemon]
runtime_dir = "..."        # default: $XDG_RUNTIME_DIR/resourced (system unit: /run/resourced)
socket_group = "gpuusers"  # shared installs: group allowed to use the socket
sample_interval = 30       # seconds between resource samples
probe_timeout = 10         # seconds before nvidia-smi / a disk probe counts as failed
nvidia_smi = "nvidia-smi"

[[gpu]]                    # only listed GPUs are managed
uuid = "GPU-..."           # authoritative
alias = "gpu0"             # optional, default gpuN
index = 0                  # optional display hint

[[check]]
name = "ram"
type = "ram"               # or "disk" (then also: path = "/some/mount")
warn_free_pct = 15         # any of warn_/stop_ free_gib, free_pct, used_pct
stop_free_pct = 10
```

## Runtime state, reboot and restart

There is no database and nothing is stored in `/tmp`. The runtime directory (tmpfs) holds the socket,
`claims.json`, and the latest resource snapshot.

- **Daemon restart or crash:** claims are kept and restored (the systemd unit restarts it automatically).
- **Reboot:** all claims are gone (and a stored boot ID discards stale files); everything asks for GPUs again.
- A second daemon on the same runtime directory refuses to start; a symlinked runtime directory is refused.
- Logs and audit lines (claims, releases, force releases, resource level changes) go to the journal:
  `journalctl --user -u resourced` (or without `--user` for the system unit).

## Known limits

- Cooperative only; a program that ignores `gpuctl` is detected by `verify` and at claim time, not stopped.
- A process that detaches (daemonizes, reparented to init) loses its ancestry and shows up as `FOREIGN PROCESS`
  and is not tracked at release; the same happens to a claim whose owner PID has exited (for example an agent that
  restarted). `gpuctl status` marks a dead owner/wrapper `DEAD`; release the claim or `--force`.
- If a `gpuctl run` wrapper is SIGKILLed, its GPUs stay allocated until `gpuctl release --force`.
- `gpuctl run` forwards SIGTERM and SIGHUP to the command; Ctrl-C reaches it through the terminal. A SIGINT sent
  only to the wrapper is not forwarded.
- Process checks are point-in-time snapshots and not atomic with a program that starts right after a grant.
  Inside containers PIDs may be namespaced, so such processes cannot be attributed. Only compute processes count
  (graphics-only processes and MIG partitions are ignored).
- Anyone who can reach the socket can claim, release and force release; access is controlled by the socket mode
  and group only.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `cannot find the resourced socket` | Is the daemon running? `systemctl --user status resourced`, `gpuctl service enable --user` |
| `no permission to use ... socket` | Inside a coding agent's sandbox: allow socket access, see [Sandboxed coding agents](#sandboxed-coding-agents). Otherwise join the socket group of the shared install and log in again; if no `socket_group` is set in the config, set it and restart the service |
| System service fails to start (`status=203/EXEC`) | The Python in `ExecStart` is not readable by `resourced` (home directory?); reinstall under `/opt` and re-run `service enable --system` |
| `nvidia-smi not found ...` in `gpuctl status` or the log | `gpuctl init` records the path it found; otherwise the daemon searches `PATH`, `/usr/lib/wsl/lib`, `/usr/bin`, `/usr/local/bin`. Install the driver or set `nvidia_smi = "/full/path"` under `[daemon]` |
| Daemon exits right away | Run `resourced --config <file>` in a terminal; messages are printed (bad config, duplicate daemon, ...) |
| `XDG_RUNTIME_DIR is not set` | Set `runtime_dir` in the config, or use a login session / lingering |
| Claims vanish after logout | Per-user service without lingering: `sudo loginctl enable-linger $USER` |
| `unclaimed GPU(s) have running processes` | Something is using a GPU outside resourced; `gpuctl verify` lists the PIDs |
| A GPU stays claimed | `gpuctl status` shows the owner; if it is `DEAD`, `gpuctl release --force --gpu <gpu>` |

## Development

```sh
pip install -e '.[test]'
pytest
```

Tests are fast, offline and CPU-only (fake GPUs and temporary runtime directories). `BUILD.md` is the
record of the agent instructions this project was built from.

License: Apache-2.0.
