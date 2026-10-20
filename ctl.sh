#!/usr/bin/env bash
# One-word controls for the experiment queue.
#
#   ./ctl.sh            progress of every job, queued hours, expected finish time
#   ./ctl.sh eta        just the expected finish time
#   ./ctl.sh check      is anything wrong? (runner dead, jobs lost, memory) and what to do
#   ./ctl.sh restart    start the runner again (after a crash/reboot); interrupted jobs
#                       are re-queued first and resume from their last checkpoint
#   ./ctl.sh reload     restart only the runner process (e.g. after a code update);
#                       running jobs keep going and are adopted by the new runner
#   ./ctl.sh log <tag>  follow one job's log (tag as shown by ./ctl.sh)
#   ./ctl.sh stop       finish the running jobs, then stop (remove runs/queue/STOP to undo)
#   ./ctl.sh watch      live view, refreshed every 30 s (Ctrl-C to leave)
cd "$(dirname "$0")"
PY=$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )
Q=runs/queue

runner_pid() { pgrep -f "python[0-9.]* scripts/queue_runner\.py$" | head -1; }

case "${1:-status}" in
  status) $PY scripts/status.py ;;
  watch)  watch -n 30 "$PY scripts/status.py" ;;
  eta)    $PY scripts/status.py | grep -E "everything done|WARNING" || echo "no queue information yet" ;;
  log)    tail -f "runs/logs/$2.log" ;;
  stop)   mkdir -p $Q && touch $Q/STOP && echo "runner will stop after the running jobs finish" ;;
  check)
    problems=0
    pid=$(runner_pid)
    if [ -z "$pid" ]; then
      echo "PROBLEM: the queue runner is not running."; problems=1
    else
      age=$($PY -c "import json,time;print(int(time.time()-json.load(open('$Q/state.json'))['updated']))" 2>/dev/null || echo 999)
      if [ "$age" -gt 120 ]; then echo "PROBLEM: runner (pid $pid) has not reported for ${age}s."; problems=1
      else echo "ok: runner alive (pid $pid), last report ${age}s ago"; fi
    fi
    lost=$($PY - <<'EOF'
import json, os
try:
    st = json.load(open("runs/queue/state.json"))
except OSError:
    st = {"running": []}
def alive(p):
    try:
        os.kill(p, 0); return True
    except OSError:
        return False
n = 0
for j in st["running"]:
    log = f"runs/logs/{j['tag']}.log"
    done = os.path.exists(log) and "DONE" in open(log, errors="ignore").read()
    if not alive(j["pid"]) and not done:
        print("  lost:", j["tag"]); n += 1
print(n)
EOF
)
    nlost=$(echo "$lost" | tail -1)
    if [ "$nlost" != "0" ]; then echo "PROBLEM: $nlost job(s) died without finishing:"; echo "$lost" | head -n -1; problems=1
    else echo "ok: no running job has died"; fi
    free_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$free_mb" -lt 1024 ]; then echo "WARNING: only ${free_mb} MB RAM free (the runner keeps >= 1024 MB by starting fewer jobs)"
    else echo "ok: ${free_mb} MB RAM free"; fi
    fails=$(grep -vc "exit=0" $Q/done.txt 2>/dev/null || true)
    [ "${fails:-0}" != "0" ] && echo "note: ${fails} finished job(s) exited with an error (see $Q/done.txt)"
    if [ $problems = 1 ]; then echo; echo ">>> restart needed: run   ./ctl.sh restart"
    else echo; echo ">>> all good, nothing to do"; fi ;;
  restart)
    if [ -n "$(runner_pid)" ]; then echo "runner already running (pid $(runner_pid)); use ./ctl.sh reload to restart it"; exit 0; fi
    rm -f $Q/STOP
    $PY scripts/jobs.py start && sleep 12 && tail -n 15 $Q/runner.log ;;
  reload)
    pid=$(runner_pid)
    if [ -n "$pid" ]; then
      kill "$pid"
      for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    fi
    rm -f $Q/STOP
    $PY scripts/jobs.py start && sleep 12 && tail -n 12 $Q/runner.log ;;
  *) sed -n '2,15p' "$0" ;;
esac
