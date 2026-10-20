"""Persistent job queue for long experiment campaigns.

Start once (it keeps running when the terminal or the session closes):

    nohup python scripts/queue_runner.py > runs/queue/runner.log 2>&1 &

Queue files (all under runs/queue/, plain text, safe to edit at any time):

    todo.txt      one job per line:  <tag> | <estimated minutes> | <shell command>
                  jobs are started from the top; edit, reorder or delete lines freely
    parallel      number of jobs run at the same time (default 5); re-read every loop
    done.txt      finished jobs with exit code and duration
    state.json    jobs running right now (read by scripts/status.py)
    STOP          create this file to stop the runner after the current jobs

The runner never exits on an empty queue: it waits for new lines, so jobs
added later start automatically.  ``python scripts/queue.py`` adds jobs.
"""
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "runs", "queue")
TODO = os.path.join(QDIR, "todo.txt")
LOCK = os.path.join(QDIR, ".lock")


def locked(fn):
    def wrapper(*a, **k):
        os.makedirs(QDIR, exist_ok=True)
        with open(LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                return fn(*a, **k)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
    return wrapper


def parse_line(line):
    parts = [p.strip() for p in line.split("|", 2)]
    if len(parts) != 3 or not parts[2] or line.lstrip().startswith("#"):
        return None
    try:
        est = float(parts[1])
    except ValueError:
        est = 30.0
    return dict(tag=parts[0], est=est, cmd=parts[2])


@locked
def pop_job():
    if not os.path.exists(TODO):
        return None
    with open(TODO) as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        job = parse_line(line)
        if job:
            rest = lines[:i] + lines[i + 1:]
            tmp = TODO + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(rest) + ("\n" if rest else ""))
            os.replace(tmp, TODO)
            return job
    return None


def parallel():
    try:
        with open(os.path.join(QDIR, "parallel")) as f:
            return max(1, int(f.read().strip()))
    except (OSError, ValueError):
        return 5


def write_state(running):
    state = [dict(tag=j["tag"], est=j["est"], pid=p.pid, start=j["start"], cmd=j["cmd"])
             for p, j in running]
    tmp = os.path.join(QDIR, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(dict(updated=time.time(), parallel=parallel(), running=state), f, indent=1)
    os.replace(tmp, os.path.join(QDIR, "state.json"))


def main():
    os.makedirs(os.path.join(ROOT, "runs", "logs"), exist_ok=True)
    os.chdir(ROOT)
    running = []
    print(f"[{time.ctime()}] queue runner started (pid {os.getpid()})", flush=True)
    while True:
        still = []
        for p, j in running:
            rc = p.poll()
            if rc is None:
                still.append((p, j))
                continue
            mins = (time.time() - j["start"]) / 60
            with open(os.path.join(QDIR, "done.txt"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M')} | {j['tag']} | exit={rc} | "
                        f"{mins:.1f} min (est {j['est']:.0f}) | {j['cmd']}\n")
            print(f"[{time.ctime()}] finished {j['tag']} exit={rc} after {mins:.1f} min", flush=True)
        running = still
        stop = os.path.exists(os.path.join(QDIR, "STOP"))
        while not stop and len(running) < parallel():
            job = pop_job()
            if job is None:
                break
            log = open(os.path.join(ROOT, "runs", "logs", f"{job['tag']}.log"), "w")
            p = subprocess.Popen(job["cmd"], shell=True, cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            job["start"] = time.time()
            running.append((p, job))
            print(f"[{time.ctime()}] started {job['tag']} (pid {p.pid})", flush=True)
        write_state(running)
        if stop and not running:
            print(f"[{time.ctime()}] STOP file found, exiting", flush=True)
            return
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
