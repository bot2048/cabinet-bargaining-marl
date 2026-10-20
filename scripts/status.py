"""Progress of background experiments, with estimated completion times.

    python scripts/status.py            # once
    watch -n 30 python scripts/status.py

Each training log line has the form  "it= k/total ... t=<elapsed>s eta=<s>s".
``scripts/run_all.sh`` writes runs/logs/_queue.txt (start time, parallelism,
one command per job); from it we estimate when the whole queue finishes.
"""
import glob
import os
import re
import subprocess
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
LINE = re.compile(r"it=\s*(\d+)/(\d+).*?t=\s*([\d.]+)s eta=\s*([\d.]+)s(.*)")


def clock(seconds_from_now):
    return time.strftime("%a %H:%M", time.localtime(time.time() + seconds_from_now))


def hm(s):
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def parse(path):
    """(done, it, total, elapsed, eta, tail) for a log file."""
    try:
        with open(path, errors="ignore") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    done = any(l.startswith("DONE") for l in lines)
    for l in reversed(lines):
        m = LINE.search(l)
        if m:
            it, tot, t, eta, rest = m.groups()
            keep = " ".join(x for x in rest.split() if x.split("=")[0] in ("phi~", "dg~", "gapBF", "fail"))
            return done, int(it), int(tot), float(t), float(eta), keep
    return done, None, None, None, None, ""


def main():
    os.chdir(ROOT)
    running = subprocess.run(["pgrep", "-af", "run_experiment.py|probe_e0.py"],
                             capture_output=True, text=True).stdout.splitlines()
    running = [r for r in running if "pgrep" not in r and "bash -c" not in r]
    print(f"== {len(running)} training processes running")
    rows = []
    for path in sorted(glob.glob("runs/logs/*.log") + glob.glob("runs/*.log")):
        p = parse(path)
        if not p or p[1] is None:
            continue
        done, it, tot, t, eta, keep = p
        name = os.path.basename(path)[:-4]
        if done or it + 1 >= tot:
            continue
        rows.append((eta, name, it, tot, t, keep))
    if rows:
        print(f"{'job':34s} {'progress':>10s} {'elapsed':>8s} {'left':>8s} {'done at':>10s}  latest")
        for eta, name, it, tot, t, keep in sorted(rows):
            print(f"{name:34s} {it + 1:>4d}/{tot:<5d} {hm(t):>8s} {hm(eta):>8s} {clock(eta):>10s}  {keep}")
    q = os.path.join("runs", "logs", "_queue.txt")
    if os.path.exists(q):
        with open(q) as f:
            lines = f.read().splitlines()
        start = int(lines[0].split()[1])
        par = int(lines[1].split()[1])
        jobs = [l for l in lines[2:] if l.strip()]
        logs = [re.search(r"> (\S+\.log)", j).group(1) for j in jobs]
        done, active, durations, remaining_active = 0, 0, [], []
        for lg in logs:
            p = parse(lg)
            if p is None:
                continue
            if p[0]:
                done += 1
                if p[3] is not None:
                    durations.append(p[3] + p[4])
            elif p[1] is not None:
                active += 1
                remaining_active.append(p[4])
                durations.append(p[3] + p[4])
        pending = len(jobs) - done - active
        avg = sum(durations) / len(durations) if durations else None
        print(f"\n== queue: {done} done, {active} running, {pending} waiting "
              f"(of {len(jobs)}; started {hm(time.time() - start)} ago, {par} in parallel)")
        if avg:
            left = (sum(remaining_active) + pending * avg) / par
            print(f"   average job ~{hm(avg)}; queue expected to finish in ~{hm(left)} "
                  f"(around {clock(left)})")
    res = sorted(glob.glob("results/*/"))
    if res:
        print("\n== results on disk: " + ", ".join(
            f"{os.path.basename(d[:-1])}({len(os.listdir(d))})" for d in res))
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print(f"\n== GPU util/mem: {g}")
    except FileNotFoundError:
        pass
    with open("/proc/loadavg") as f:
        print(f"== CPU load: {' '.join(f.read().split()[:3])}")


if __name__ == "__main__":
    main()
