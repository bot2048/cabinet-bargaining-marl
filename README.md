# Bargaining over Cabinets: MARL for coalition portfolio allocation

**Team:** Arpan Jain (B23CS1089), Aradhya Mahajan (B23CS1088). AI course, Assignment 2 (RL mini-project).

After an election without a majority, parties bargain over who enters government and how the ministries
are split. Real cabinets split portfolios almost exactly in proportion to seats (**Gamson's law**), while
the leading bargaining theory (**Baron-Ferejohn**) predicts that the party that forms the government (the
*formateur*) takes a premium. We model parties as reinforcement-learning agents in a multi-agent environment
that encodes the institutional rules of government formation, check the learners against the analytic
Baron-Ferejohn equilibrium, and then switch institutional mechanisms on one at a time to see which of them
move learned allocations towards proportionality.

Everything here is reinforcement learning: agents learn only from environment rewards (no supervised
targets, no hard-coded fairness); the scripted agents in `cabinet/baselines.py` are comparison baselines.

## Main findings

1. **Validated learner.** In the Baron-Ferejohn game the learned formateur share matches the closed
   form (mean error 0.017 over 10 seeds).
2. **Under proposal (closed-rule) bargaining the formateur premium never disappears; it changes
   currency.** Base game: formateur coefficient 0.25, rising to 0.30 with 4x training. Repetition
   (even patient, with memory), exits, the recognition rule and every learner (IPPO, MAPPO,
   fictitious self-play, PSRO, GRU) leave it in place. Private valuations and policy motives drive
   the *counted* premium to zero, but the formateur keeps it in the ministries it values most or in
   the policy-setting ministries.
3. **Real cabinets have no premium in any currency** (195 cabinets, Seki-Williams + ParlGov):
   share = 0.031 + 0.896 seats + 0.011 formateur; the PM party holds key ministries in proportion;
   Gamson's law predicts real splits best (MAE 0.052, agents 0.087-0.17, Baron-Ferejohn 0.144).
4. **The protocol is what matters.** Under demand bargaining (Morelli 1999), agents trained only on
   random parliaments form cabinets in the real parliaments with slope 0.81-0.85 and no formateur
   premium, matching the real regression including its small-party bias; matched proposal agents
   keep a premium of 0.21-0.31.

## RL formulation (summary)

| | |
|---|---|
| Agents | one per party, all controlled by **one shared, permutation-equivariant policy** (parameter sharing) |
| State / observation | a *set* of party tokens (seat share, recognition probability, roles, offer, history of the last Λ cabinets, ...), a set of ministry tokens (salience, own valuation, policy dimension led), (ministry, party) pair features and a context vector (phase, quota, δ, β_T, λ, ...). No party labels, so the policy works for any number of parties |
| Actions | formateur: masked autoregressive proposal, i.e. include/exclude each party (masks keep the coalition winning) then a pointer over coalition members for each ministry (or a 1%-grid offer per partner for a divisible surplus); voters: accept/reject; governing members: stay/exit |
| Reward | per period, office value of held ministries minus λ × policy distance; exit bonus on walking out; zero otherwise. Discount δ per period |
| Algorithm | PPO (clipped) with GAE on each agent's own decision sequence (semi-MDP), shared-parameter self-play (IPPO); best-response PPO for exploitability |

Full details: `report/report.pdf`; how every requirement, research question and hypothesis of the
specification is covered (and every deviation): [`docs/SPEC_COVERAGE.md`](docs/SPEC_COVERAGE.md).

## Repository map

| Path | What it is |
|---|---|
| [`cabinet/env.py`](cabinet/env.py) | the cabinet bargaining game (formation rounds, votes, governance, exits, elections, private types, policy) |
| [`cabinet/config.py`](cabinet/config.py) | parameters, levels E0-E6, base configuration and mechanism switches |
| [`cabinet/policy.py`](cabinet/policy.py) | shared equivariant attention policy with masked autoregressive proposal heads |
| [`cabinet/ppo.py`](cabinet/ppo.py) | PPO self-play on agent-level decision sequences, best-response training, evaluation |
| [`cabinet/bf_solver.py`](cabinet/bf_solver.py) | Baron-Ferejohn closed form and certified solver for weighted parliaments |
| [`cabinet/metrics.py`](cabinet/metrics.py) | formateur premium, Gamson distance, Hamilton floor, Gamson regression |
| [`cabinet/baselines.py`](cabinet/baselines.py) | scripted baselines: BF agent, Gamson agent, myopic BF heuristic, random |
| [`cabinet/diagnostics.py`](cabinet/diagnostics.py) | probing learned voters' acceptance thresholds |
| [`scripts/run_experiment.py`](scripts/run_experiment.py) | train + evaluate one experiment cell |
| [`scripts/run_all.sh`](scripts/run_all.sh) | every experiment of the report, in parallel |
| [`scripts/make_figures.py`](scripts/make_figures.py) | figures and summary tables from `results/` |
| [`tests/test_core.py`](tests/test_core.py) | tests tying the code to the theory (closed form, masks, equivariance, welfare identity, ...) |
| [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) | run everything on Google Colab |
| `models/` | trained policies (`<experiment>/seed<k>.pt`) |
| `results/` | raw results (JSON) of every run |
| [`scripts/seki_eval.py`](scripts/seki_eval.py) | real cabinets: empirical regression, split prediction vs Gamson/BF, agents' own cabinets in real parliaments |
| [`scripts/key_ministries.py`](scripts/key_ministries.py) | do real prime-minister parties hold the key ministries? |
| [`scripts/parlgov_eval.py`](scripts/parlgov_eval.py) | coalition choice in real ParlGov parliaments |
| `data/` | ParlGov parliaments and Seki-Williams cabinets (see the NOTICE files; raw downloads via `scripts/*download*.sh`) |

## Installation

Python 3.10+ (tested with 3.14).

```bash
git clone https://github.com/bot2048/cabinet-bargaining-marl.git
cd cabinet-bargaining-marl
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU build is enough
pip install -r requirements.txt
python -m pytest -q          # 26 tests, ~35 s
```

## Running

```bash
# one cell, e.g. the Baron-Ferejohn benchmark (about 10 min on 4 CPU threads)
python scripts/run_experiment.py --exp e0 --seed 0 --threads 4

# one mechanism configuration, e.g. base + repeated terms
python scripts/run_experiment.py --exp mech --config BASE+M1_repeat --seed 0

# everything in the report: queues every missing experiment and starts the queue runner
bash scripts/run_all.sh

# figures and tables -> figures/*.png|pdf, results/summary.json
python scripts/make_figures.py
```

### Long campaigns: the job queue

`scripts/run_all.sh` does not run jobs itself; it appends them to a persistent queue
(`runs/queue/todo.txt`, one line per job: `tag | estimated minutes | command`) that
`scripts/queue_runner.py` works through, a few jobs at a time, in the background. The runner keeps
going when the terminal closes, re-reads the queue whenever a slot frees (so jobs can be added,
removed or reordered while it runs) and waits instead of exiting when the queue is empty.

```bash
./ctl.sh             # progress of every job, queued hours, expected finish time
./ctl.sh watch       # live view (refreshes every 30 s)
./ctl.sh eta         # just the expected finish time
./ctl.sh check       # anything wrong? tells you whether a restart is needed
./ctl.sh restart     # after a crash or reboot: interrupted jobs are re-queued and resume
./ctl.sh log <tag>   # follow one job's log
./ctl.sh stop        # finish the running jobs, then stop
python scripts/jobs.py list | remove <tag> | parallel <k>   # edit the queue
```

**Robustness.** Training saves a checkpoint every 25 iterations (`runs/ckpt/`) and resumes from it
when restarted, so a power cut costs at most a few minutes per job. The runner keeps at least 1 GB of
RAM free: it starts a job only if enough memory remains afterwards, and if free memory drops below
600 MB it stops the newest job and re-queues it (`runs/queue/reserve_mb` changes the reserve).
Restarting the runner while jobs are running is safe: it adopts them instead of starting them again.

With a CUDA build of PyTorch in `.venv-gpu`, queued jobs run the PPO update on the GPU
(`--device cuda`, about 2x faster per iteration in the main environment); rollouts always run on the CPU.

Configurations: `E0`...`E6`, `BASE`, `BASE+<M>` (add-one-in), `Estar` (all mechanisms on), `Estar-<M>`
(leave-one-out) and `Estar_Lam0` (memoryless), with `<M>` one of `M1_repeat`, `M4_private`, `M5_hetero`,
`M8_exits`, `M9_policy`, `M10_seatrho`.

### Loading a trained agent

```python
import torch
from cabinet.policy import PolicyNet
from cabinet.config import full_estar
from cabinet.ppo import evaluate
from cabinet.metrics import summarise

net = PolicyNet(); net.load_state_dict(torch.load("models/mech_Estar/seed0.pt"))
cfg = full_estar()
cab, ep = evaluate(cfg, {"learn": net}, lambda env, rng: ["learn"] * env.n, episodes=1000)
print(summarise(cab, ep, cfg.m, cfg.divisible))
```

## Google Colab

A CPU runtime is enough (the bottleneck is the Python game simulation, not the network).

1. Open `notebooks/colab_run.ipynb` in Colab (*File > Open notebook > GitHub*, or upload the file).
2. The repository is private: create a fine-grained GitHub token with *Contents: Read-only* on this
   repository and store it in Colab's **Secrets** panel as `GH_TOKEN` (enable notebook access).
3. Run the cells top to bottom: clone, install and test (~1 min), quick demo (~3 min), optional extra
   seeds, then figures and a zip of the results to download.

## Reproducibility

Every run is seeded (environments, PyTorch, NumPy); the environment is deterministic given its seed
(tested). Hyper-parameters are in `cabinet/ppo.py` (`DEFAULT_HP`) and were tuned on the benchmark
level E0 only.
