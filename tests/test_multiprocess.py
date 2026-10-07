"""Many separate gpuctl processes competing for the same GPUs."""

import subprocess
import sys

from conftest import UUIDS, cli_command, cli_env

# prints the GPU it was given, holds it briefly, then exits
HOLD = "import os,time;print(os.environ['CUDA_VISIBLE_DEVICES'], flush=True);time.sleep(1.5)"


def start(sock, *command):
    return subprocess.Popen(
        cli_command("run", "--gpus", "1", "-q", "--socket", sock, "--", *command),
        env=cli_env(),
        stdout=subprocess.PIPE,
        text=True,
    )


def test_simultaneous_processes_get_distinct_gpus_and_extras_fail_fast(sock):
    procs = [start(sock, sys.executable, "-c", HOLD) for _ in range(7)]
    results = [(p.wait(timeout=60), p.stdout.read().strip()) for p in procs]
    granted = [out for code, out in results if code == 0]
    refused = [code for code, _ in results if code != 0]
    assert sorted(granted) == sorted(UUIDS)  # 4 GPUs, 4 distinct winners
    assert refused == [3] * 3  # the others fail promptly with "not enough GPUs"


def test_no_gpu_is_ever_held_by_two_processes_at_once(sock, short_tmp):
    log = short_tmp / "log"
    script = (
        "import os,time,fcntl\n"
        "def note(kind):\n"
        f"    with open({str(log)!r}, 'a') as f:\n"
        "        fcntl.flock(f, fcntl.LOCK_EX)\n"
        "        f.write(f\"{time.monotonic():.6f} {kind} {os.environ['CUDA_VISIBLE_DEVICES']}\\n\")\n"
        "note('start'); time.sleep(0.05); note('end')\n"
    )
    workers = [
        subprocess.Popen(
            [
                sys.executable, "-c",
                "import subprocess,sys\n"
                "done=0\n"
                "while done<6:\n"
                f"    code=subprocess.call({cli_command('run', '--gpus', '1', '-q', '--socket', sock, '--', sys.executable, '-c', script)!r})\n"
                "    done+= code==0\n",
            ],
            env=cli_env(),
        )
        for _ in range(8)
    ]
    assert all(w.wait(timeout=120) == 0 for w in workers)

    holders: dict[str, int] = {}
    runs = 0
    for line in sorted(log.read_text().splitlines()):
        _, kind, uuid = line.split()
        if kind == "start":
            holders[uuid] = holders.get(uuid, 0) + 1
            runs += 1
            assert holders[uuid] == 1, f"{uuid} held by two processes at once"
        else:
            holders[uuid] -= 1
    assert runs == 8 * 6
    from resourced.client import Client

    assert Client(sock).status()["used"] == 0  # everything was released
