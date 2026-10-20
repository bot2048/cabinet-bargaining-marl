"""Persistent job queue for long experiment campaigns.

Start once (it keeps running when the terminal or the session closes):

    nohup python scripts/queue_runner.py > runs/queue/runner.log 2>&1 &

Queue files (all under runs/queue/, plain text, safe to edit at any time):

    todo.txt      one job per line:  <tag> | <estimated minutes> | <shell command>
                  jobs are started from the top; edit, reorder or delete lines freely
    parallel      maximum number of jobs at the same time (default 5); re-read every loop
    reserve_mb    RAM that must stay free (default 1024): a job starts only if, after
                  it starts, at least this much memory remains available; if free
                  memory falls below 600 MB the newest job is stopped and re-queued
    done.txt      finished jobs with exit code and duration
    state.json    jobs running right now (read by scripts/status.py)
    STOP          create this file to stop the runner after the current jobs

The runner never exits on an empty queue: it waits for new lines, so jobs
added later start automatically.  ``python scripts/jobs.py`` adds jobs.

After a crash or reboot just start the runner again: jobs that were running
and never finished are put back at the front of the queue, and training
continues from its last checkpoint.
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


def write_state(running, avail_mb=None, job_mb=None):
    state = [dict(tag=j["tag"], est=j["est"], pid=p.pid, start=j["start"], cmd=j["cmd"])
             for p, j in running]
    tmp = os.path.join(QDIR, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(dict(updated=time.time(), parallel=parallel(), running=state,
                       avail_mb=avail_mb, job_mb=job_mb, reserve_mb=reserve_mb()), f, indent=1)
    os.replace(tmp, os.path.join(QDIR, "state.json"))


def finished(tag):
    try:
        with open(os.path.join(ROOT, "runs", "logs", f"{tag}.log"), errors="ignore") as f:
            return any(l.startswith("DONE") for l in f)
    except OSError:
        return False


@locked
def recover():
    """Put back at the front of the queue every job that was running when the
    previous runner stopped (power cut, reboot, kill) and never finished.
    Training resumes from its last checkpoint (runs/ckpt/)."""
    state_p = os.path.join(QDIR, "state.json")
    if not os.path.exists(state_p):
        return []
    with open(state_p) as f:
        state = json.load(f)
    jobs = state.get("running", [])
    adopt = [j for j in jobs if alive(j["pid"])]          # runner restarted, job still going
    lost = [j for j in jobs if not alive(j["pid"]) and not finished(j["tag"])]
    if not lost:
        return [], adopt
    lines = []
    if os.path.exists(TODO):
        with open(TODO) as f:
            lines = f.read().splitlines()
    queued = {(parse_line(l) or {}).get("tag") for l in lines}
    head = [f"{j['tag']} | {j['est']:g} | {j['cmd']}" for j in lost if j["tag"] not in queued]
    with open(TODO + ".tmp", "w") as f:
        f.write("\n".join(head + lines) + "\n")
    os.replace(TODO + ".tmp", TODO)
    return [j["tag"] for j in lost], adopt


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


class Adopted:
    """A job started by a previous runner; polled through its pid."""

    def __init__(self, pid, tag):
        self.pid, self.tag = pid, tag

    def poll(self):
        if alive(self.pid):
            return None
        return 0 if finished(self.tag) else 1


# ------------------------------------------------------------------ memory
RESERVE_MB = 1024          # RAM that must stay free after starting a job
EMERGENCY_MB = 600         # below this, the newest job is stopped and re-queued
DEFAULT_JOB_MB = 1500      # assumed size of a job before any is measured


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def session_rss_mb(sid):
    """Resident memory of every process in a job's session (shell + python)."""
    total = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if int(fields[3]) != sid:          # field 6 of stat: session id
                continue
            with open(f"/proc/{d}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) // 1024
        except (OSError, IndexError, ValueError):
            continue
    return total


def reserve_mb():
    try:
        with open(os.path.join(QDIR, "reserve_mb")) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return RESERVE_MB


@locked
def requeue_front(job):
    lines = []
    if os.path.exists(TODO):
        with open(TODO) as f:
            lines = f.read().splitlines()
    with open(TODO + ".tmp", "w") as f:
        f.write("\n".join([f"{job['tag']} | {job['est']:g} | {job['cmd']}"] + lines) + "\n")
    os.replace(TODO + ".tmp", TODO)


def main():
    os.makedirs(os.path.join(ROOT, "runs", "logs"), exist_ok=True)
    os.chdir(ROOT)
    running = []
    print(f"[{time.ctime()}] queue runner started (pid {os.getpid()})", flush=True)
    lost, adopt = recover()
    for tag in lost:
        print(f"[{time.ctime()}] re-queued interrupted job {tag}", flush=True)
    for j in adopt:
        running.append((Adopted(j["pid"], j["tag"]), j))
        print(f"[{time.ctime()}] adopted running job {j['tag']} (pid {j['pid']})", flush=True)
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
        # memory: measure jobs, stop the newest one if RAM is nearly exhausted
        sizes = [session_rss_mb(p.pid) for p, _ in running]
        job_mb = max(sizes + [0]) or DEFAULT_JOB_MB
        avail = mem_available_mb()
        if avail < EMERGENCY_MB and len(running) > 1:
            p, j = max(running, key=lambda pj: pj[1]["start"])
            try:
                os.killpg(p.pid, 15)
            except OSError:
                pass
            time.sleep(5)
            running = [(q, k) for q, k in running if q is not p]
            requeue_front(j)
            print(f"[{time.ctime()}] low memory ({avail} MB free): stopped {j['tag']} "
                  f"and re-queued it (it resumes from its checkpoint)", flush=True)
            avail = mem_available_mb()
        stop = os.path.exists(os.path.join(QDIR, "STOP"))
        while not stop and len(running) < parallel():
            if avail - job_mb < reserve_mb():
                break                       # starting another job would leave < reserve free
            job = pop_job()
            if job is None:
                break
            log = open(os.path.join(ROOT, "runs", "logs", f"{job['tag']}.log"), "a")
            p = subprocess.Popen(job["cmd"], shell=True, cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            job["start"] = time.time()
            running.append((p, job))
            print(f"[{time.ctime()}] started {job['tag']} (pid {p.pid}, {avail} MB free)", flush=True)
            avail -= job_mb                 # count it before its memory shows up
        write_state(running, avail_mb=mem_available_mb(), job_mb=job_mb)
        if stop and not running:
            print(f"[{time.ctime()}] STOP file found, exiting", flush=True)
            return
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
