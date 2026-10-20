"""Progress of background experiments, with estimated completion times.

    python scripts/status.py            # once
    watch -n 30 python scripts/status.py

Each training log line has the form  "it= k/total ... t=<elapsed>s eta=<s>s".
The queue runner (scripts/queue_runner.py) writes runs/queue/state.json; with
the estimated minutes of the jobs still in runs/queue/todo.txt this gives the
time at which the whole queue should be finished.
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
        if ".failed" in name:
            continue
        if done or it + 1 >= tot:
            continue
        rows.append((eta, name, it, tot, t, keep))
    if rows:
        print(f"{'job':34s} {'progress':>10s} {'elapsed':>8s} {'left':>8s} {'done at':>10s}  latest")
        for eta, name, it, tot, t, keep in sorted(rows):
            print(f"{name:34s} {it + 1:>4d}/{tot:<5d} {hm(t):>8s} {hm(eta):>8s} {clock(eta):>10s}  {keep}")
    qd = os.path.join("runs", "queue")
    state_p, todo_p = os.path.join(qd, "state.json"), os.path.join(qd, "todo.txt")
    if os.path.exists(state_p):
        import json
        with open(state_p) as f:
            st = json.load(f)
        alive = time.time() - st["updated"] < 60
        par = st.get("parallel", 5)
        print(f"\n== queue runner: {'RUNNING' if alive else 'NOT RUNNING (start: python scripts/jobs.py start)'}"
              f", {len(st['running'])} jobs active (limited by memory; cap {par})")
        left_active = []
        for j in st["running"]:
            p = parse(os.path.join("runs", "logs", f"{j['tag']}.log"))
            el = time.time() - j["start"]
            eta = p[4] if p and p[4] is not None else max(j["est"] * 60 - el, 0)
            left_active.append(eta)
        queued = []
        if os.path.exists(todo_p):
            with open(todo_p) as f:
                for line in f:
                    parts = [x.strip() for x in line.split("|", 2)]
                    if len(parts) == 3 and not line.startswith("#"):
                        try:
                            queued.append((parts[0], float(parts[1])))
                        except ValueError:
                            queued.append((parts[0], 30.0))
        # memory, not the cap, limits concurrency: use the jobs running now
        eff = max(1, min(par, len(st["running"])))
        work = sum(left_active) + sum(m * 60 for _, m in queued)
        left = max(work / eff, max(left_active, default=0))
        print(f"   {len(queued)} jobs waiting (~{sum(m for _, m in queued) / 60:.1f} job-hours); "
              f"everything done in ~{hm(left)} (around {clock(left)})")
        if left < 3 * 3600:
            print("   WARNING: less than 3 hours of work queued")
        for tag, m in queued[:8]:
            print(f"   next: {tag} (~{m:.0f} min)")
        if len(queued) > 8:
            print(f"   ... and {len(queued) - 8} more")
        done_p = os.path.join(qd, "done.txt")
        if os.path.exists(done_p):
            with open(done_p) as f:
                done = f.read().splitlines()
            fails = [d for d in done if "exit=0" not in d]
            print(f"   finished: {len(done)} jobs ({len(fails)} failed)")
            for d in fails[-3:]:
                print("   FAILED:", d[:140])
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
