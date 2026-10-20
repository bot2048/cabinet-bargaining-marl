"""Manage the experiment queue read by scripts/queue_runner.py.

    python scripts/jobs.py list
    python scripts/jobs.py add <tag> <est_minutes> [--front] [--script=<path>] -- <arguments>
    python scripts/jobs.py remove <tag-substring>
    python scripts/jobs.py start          # start the runner if it is not running
    python scripts/jobs.py parallel 5

Commands added with ``add`` use .venv-gpu when it exists (PPO updates on the
GPU) and otherwise .venv.  A job whose result file already exists is skipped.
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from queue_runner import QDIR, TODO, locked, parse_line  # noqa: E402


def python_and_device():
    gpu = os.path.join(ROOT, ".venv-gpu", "bin", "python")
    if os.path.exists(gpu):
        ok = subprocess.run([gpu, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                            capture_output=True).returncode == 0
        if ok:
            return gpu, "cuda"
    cpu = os.path.join(ROOT, ".venv", "bin", "python")
    return (cpu if os.path.exists(cpu) else "python3"), "cpu"


def read():
    if not os.path.exists(TODO):
        return []
    with open(TODO) as f:
        return f.read().splitlines()


@locked
def write(lines):
    os.makedirs(QDIR, exist_ok=True)
    with open(TODO + ".tmp", "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    os.replace(TODO + ".tmp", TODO)


def add(tag, est, args, threads=2, front=False, script="scripts/run_experiment.py"):
    py, dev = python_and_device()
    cmd = f"{py} {script} {' '.join(args)} --threads {threads} --device {dev}"
    lines = read()
    if any((parse_line(l) or {}).get("tag") == tag for l in lines):
        return False
    line = f"{tag} | {est:g} | {cmd}"
    write([line] + lines if front else lines + [line])
    return True


def total_minutes():
    return sum(j["est"] for j in map(parse_line, read()) if j)


def runner_alive():
    out = subprocess.run(["pgrep", "-f", "scripts/queue_runner.py"], capture_output=True, text=True)
    return bool(out.stdout.strip())


def start():
    if runner_alive():
        print("runner already running")
        return
    py, _ = python_and_device()
    os.makedirs(QDIR, exist_ok=True)
    log = open(os.path.join(QDIR, "runner.log"), "a")
    subprocess.Popen(["nohup", py, "scripts/queue_runner.py"], cwd=ROOT, stdout=log,
                     stderr=subprocess.STDOUT, start_new_session=True)
    print("runner started")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        for l in read():
            j = parse_line(l)
            if j:
                print(f"{j['tag']:40s} ~{j['est']:.0f} min")
        print(f"total ~{total_minutes():.0f} job-minutes queued")
        return
    cmd = sys.argv[1]
    if cmd == "add":
        i = sys.argv.index("--")
        tag, est = sys.argv[2], float(sys.argv[3])
        opts = sys.argv[4:i]
        script = next((o.split("=", 1)[1] for o in opts if o.startswith("--script=")),
                      "scripts/run_experiment.py")
        ok = add(tag, est, sys.argv[i + 1:], front="--front" in opts, script=script)
        print("added" if ok else "already queued")
    elif cmd == "remove":
        keep = [l for l in read() if sys.argv[2] not in ((parse_line(l) or {}).get("tag") or "")]
        removed = len(read()) - len(keep)
        write(keep)
        print(f"removed {removed}")
    elif cmd == "start":
        start()
    elif cmd == "parallel":
        os.makedirs(QDIR, exist_ok=True)
        with open(os.path.join(QDIR, "parallel"), "w") as f:
            f.write(sys.argv[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
