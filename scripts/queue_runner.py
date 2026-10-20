"""Persistent job queue for long experiment campaigns.

Start once (it keeps running when the terminal or the session closes):

    nohup python scripts/queue_runner.py > runs/queue/runner.log 2>&1 &

Queue files (all under runs/queue/, plain text, safe to edit at any time):

    todo.txt      one job per line:  <tag> | <estimated minutes> | <shell command>
                  jobs are started from the top; edit, reorder or delete lines freely
    parallel      upper bound on simultaneous jobs (default 16); re-read every loop.
                  Below it, a job starts whenever resources allow: on the GPU if RAM
                  (1.9 GB per job) and GPU memory permit, otherwise CPU-only
                  (0.8 GB), and only if the CPU is not saturated
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
    """Upper bound on simultaneous jobs; memory, GPU and CPU decide below it."""
    try:
        with open(os.path.join(QDIR, "parallel")) as f:
            return max(1, int(f.read().strip()))
    except (OSError, ValueError):
        return 16


def write_state(running, avail_mb=None, job_mb=None):
    state = [dict(tag=j["tag"], est=j["est"], pid=p.pid, start=j["start"], cmd=j["cmd"],
                  device=j.get("device", "cuda" if "--device cuda" in j["cmd"] else "cpu"))
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
DEFAULT_MB = {"cuda": 1900, "cpu": 800}   # job sizes before any is measured
GPU_JOB_MB = 600           # GPU memory a job needs
THREADS = 2                # PyTorch threads per job (CPU budget)
LOAD_MAX = 24              # never start a job above this load (physical cores)
CPU_LOAD_MAX = 20          # CPU-only jobs (whose update also runs on the CPU) only below this
CPU_JOBS_MAX = 6           # ... and at most this many at once
# Measured on this laptop: 14 mostly CPU-only jobs finished 0.18 jobs/min,
# 6 GPU jobs 0.27 jobs/min; past the core count, more jobs means less work done.


def gpu_free_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0


def as_device(cmd, dev):
    """Rewrite a queued command to run its PPO update on ``dev``.  CPU jobs use
    the CPU-only environment, which needs ~1 GB less RAM than CUDA."""
    gpu_py = os.path.join(ROOT, ".venv-gpu", "bin", "python")
    cpu_py = os.path.join(ROOT, ".venv", "bin", "python")
    if dev == "cpu":
        cmd = cmd.replace(gpu_py, cpu_py).replace("--device cuda", "--device cpu")
    elif os.path.exists(gpu_py):
        cmd = cmd.replace(cpu_py, gpu_py).replace("--device cpu", "--device cuda")
    return cmd


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
        # memory: measure jobs by device, stop the newest job if RAM is nearly exhausted
        sizes = {"cuda": [], "cpu": []}
        young = []                           # (device, rss) of jobs still growing
        for p, j in running:
            dev_j = j.get("device", "cuda" if "--device cuda" in j["cmd"] else "cpu")
            rss = session_rss_mb(p.pid)
            if time.time() - j["start"] < 180:
                young.append((dev_j, rss))
            else:
                sizes[dev_j].append(rss)
        job_mb = {d: max(max(v) if v else 0, DEFAULT_MB[d] // 2) or DEFAULT_MB[d]
                  for d, v in sizes.items()}
        for d in job_mb:
            if not sizes[d]:
                job_mb[d] = DEFAULT_MB[d]
        avail = mem_available_mb()
        # memory that recently started jobs will still take
        growth = sum(max(0, job_mb[d] - rss) for d, rss in young)
        avail_plan = avail - growth
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
            avail_plan = avail - growth
        stop = os.path.exists(os.path.join(QDIR, "STOP"))
        gpu_free = gpu_free_mb()
        started_now = 0
        while not stop and len(running) < parallel():
            # as many jobs as resources allow: GPU if RAM and GPU memory permit,
            # else a lighter CPU-only job, else wait for memory to free up
            load = os.getloadavg()[0] + THREADS * started_now
            if load + THREADS > LOAD_MAX:
                break
            n_cpu = sum(1 for _, j in running if j.get("device") == "cpu")
            if avail_plan - job_mb["cuda"] >= reserve_mb() and gpu_free - GPU_JOB_MB >= 512:
                dev = "cuda"
            elif (avail_plan - job_mb["cpu"] >= reserve_mb() and load + THREADS <= CPU_LOAD_MAX
                  and n_cpu < CPU_JOBS_MAX):
                dev = "cpu"
            else:
                break                       # no room: memory reserve, CPU or GPU limit
            job = pop_job()
            if job is None:
                break
            if "run_experiment.py" in job["cmd"] or "parlgov_eval.py" in job["cmd"]:
                job["cmd"] = as_device(job["cmd"], dev)
            job["device"] = dev
            log = open(os.path.join(ROOT, "runs", "logs", f"{job['tag']}.log"), "a")
            p = subprocess.Popen(job["cmd"], shell=True, cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            job["start"] = time.time()
            running.append((p, job))
            print(f"[{time.ctime()}] started {job['tag']} on {dev} (pid {p.pid}, {avail} MB free)",
                  flush=True)
            avail_plan -= job_mb[dev]       # count it before its memory shows up
            started_now += 1
            if dev == "cuda":
                gpu_free -= GPU_JOB_MB
        write_state(running, avail_mb=mem_available_mb(), job_mb=job_mb)
        if stop and not running:
            print(f"[{time.ctime()}] STOP file found, exiting", flush=True)
            return
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
