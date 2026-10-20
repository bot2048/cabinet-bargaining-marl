"""Shared-parameter PPO self-play and best-response training.

* ``Runner`` steps a batch of environments.  Each seat is driven by a
  *controller*: the learning network, a frozen network or a scripted agent.
  Decisions of the learning controller are stored; its rewards are the
  interval rewards between its own decisions (spec Lemma 3.5).
* ``ppo_update`` is the clipped objective of spec Definition 3.9 with GAE on
  agent-level decision sequences (spec Definition 3.6).
* ``train_selfplay`` is algorithm LA1 (IPPO with shared parameters) or,
  with ``algo="fsp"``, LA3 (fictitious self-play against past checkpoints).
* ``exploitability`` estimates Gain (spec Definition 3.10) by training a
  best response for a random focal seat against a frozen profile.
"""
import copy
import time
from collections import defaultdict

import numpy as np
import torch

from .baselines import Scripted
from .env import CabinetEnv, PROPOSE
from .metrics import summarise
from .policy import PolicyNet, pad_obs, to_torch


class Runner:
    def __init__(self, cfg, n_envs, seed, controllers, assign):
        self.cfg = cfg
        self.envs = [CabinetEnv(cfg, seed=seed * 1000 + e) for e in range(n_envs)]
        self.rng = np.random.default_rng(seed)
        self.controllers = controllers          # name -> PolicyNet | Scripted
        self.assign = assign                    # (env, rng) -> list of names per seat
        self.N = cfg.n_max
        self.M = 1 if cfg.divisible else cfg.m
        self.seq_counter = 0

    def _reset(self, e):
        env = self.envs[e].reset()
        env.seats = self.assign(env, self.rng)
        env.seq_base = self.seq_counter
        self.seq_counter += env.n

    def collect(self, min_decisions, learner="learn"):
        S = defaultdict(list)                   # storage of learner decisions
        last = {}
        for e in range(len(self.envs)):
            self._reset(e)
        active = set(range(len(self.envs)))
        n_dec, episodes = 0, []
        while active:
            groups = defaultdict(list)          # controller -> [(e, decision, obs)]
            for e in active:
                env = self.envs[e]
                for d in env.pending():
                    r, dsc = env.take_interval(d.agent)
                    key = (e, d.agent)
                    if key in last:
                        S["rew"][last[key]], S["disc"][last[key]] = r, dsc
                    groups[env.seats[d.agent]].append((e, d, env.observe(d.agent, d.kind)))
            actions = defaultdict(dict)
            for name, items in groups.items():
                ctrl = self.controllers[name]
                if isinstance(ctrl, Scripted):
                    for e, d, _ in items:
                        actions[e][d.agent] = ctrl.act(self.envs[e], d.agent, d.kind)
                    continue
                batch = pad_obs([o for _, _, o in items], self.N, self.M)
                with torch.no_grad():
                    out, logp, _, value = ctrl(to_torch(batch))
                out = {k: v.numpy() for k, v in out.items()}
                for b, (e, d, o) in enumerate(items):
                    actions[e][d.agent] = _env_action(o, out, b)
                    if name == learner:
                        last[(e, d.agent)] = len(S["kind"])
                        S["obs"].append(o)
                        for k in ("bit", "inc", "mu", "shares"):
                            S[k].append(out[k][b])
                        S["logp"].append(float(logp[b]))
                        S["value"].append(float(value[b]))
                        S["kind"].append(d.kind)
                        S["seq"].append(self.envs[e].seq_base + d.agent)
                        S["rew"].append(0.0)
                        S["disc"].append(1.0)
                        S["done"].append(False)
                        n_dec += 1
            for e in list(active):
                env = self.envs[e]
                if e not in actions:
                    continue
                env.act(actions[e])
                if env.done:
                    for j in range(env.n):
                        r, _ = env.take_interval(j)
                        if (e, j) in last:
                            idx = last.pop((e, j))
                            S["rew"][idx], S["done"][idx] = r, True
                    episodes.append(dict(env.episode_log[-1], seats=list(env.seats)))
                    if n_dec >= min_decisions:
                        active.discard(e)
                    else:
                        self._reset(e)
        return S, episodes

    def logs(self):
        cab, ep = [], []
        for env in self.envs:
            cab += env.cabinet_log
            ep += env.episode_log
            env.cabinet_log, env.episode_log = [], []
        return cab, ep


def _env_action(o, out, b):
    if o["kind"] != PROPOSE:
        return int(out["bit"][b])
    n, f = o["n"], o["self_idx"]
    C = np.zeros(n, bool)
    C[f] = True
    for t, j in enumerate(o["order"]):
        if out["inc"][b, t]:
            C[j] = True
    if o["divisible"]:
        return C, out["shares"][b, :n]
    return C, out["mu"][b, :o["m"]]


def gae(S, lam):
    rew, disc = np.array(S["rew"]), np.array(S["disc"])
    val, done = np.array(S["value"]), np.array(S["done"])
    adv = np.zeros(len(rew))
    seqs = defaultdict(list)
    for i, s in enumerate(S["seq"]):
        seqs[s].append(i)
    for idxs in seqs.values():
        a_next, v_next = 0.0, 0.0
        for i in reversed(idxs):
            if done[i]:
                a_next, v_next = 0.0, 0.0
            td = rew[i] + disc[i] * v_next - val[i]
            a_next = td + disc[i] * lam * a_next
            adv[i] = a_next
            v_next = val[i]
    return adv, adv + val


def ppo_update(net, opt, S, N, M, hp):
    adv, ret = gae(S, hp["lam"])
    batch = pad_obs(S["obs"], N, M)
    batch["bit"] = np.array(S["bit"])
    batch["inc"] = np.stack(S["inc"])
    batch["mu"] = np.stack(S["mu"])
    batch["shares"] = np.stack(S["shares"]).astype(np.float32)
    T = to_torch(batch)
    old_logp = torch.tensor(S["logp"], dtype=torch.float32)
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(ret, dtype=torch.float32)
    n = len(adv)
    stats = defaultdict(list)
    for _ in range(hp["epochs"]):
        perm = torch.randperm(n)
        for mb in perm.split(max(n // hp["minibatches"], 1)):
            ob = {k: v[mb] for k, v in T.items()}
            act = {k: ob[k] for k in ("bit", "inc", "mu", "shares")}
            _, logp, ent, value = net(ob, act=act)
            # normalise advantages per decision type: voters' stakes are much
            # smaller than formateurs', and a pooled normalisation drowns them
            a = adv_t[mb].clone()
            for k in range(3):
                sel = ob["kind"] == k
                if sel.sum() > 1:
                    a[sel] = (a[sel] - a[sel].mean()) / (a[sel].std(unbiased=False) + 1e-8)
            ratio = torch.exp((logp - old_logp[mb]).clamp(-20, 20))
            pg = -torch.min(ratio * a, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * a).mean()
            vl = ((value - ret_t[mb]) ** 2).mean()
            # voters and governing members get a larger entropy weight so that
            # rejecting / exiting keeps being tried until its value is learned
            ent_w = torch.where(ob["kind"] == PROPOSE, torch.tensor(1.0),
                                torch.tensor(float(hp["ent_vote_mult"])))
            loss = pg + hp["vf"] * vl - hp["ent"] * (ent_w * ent).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats["pg"].append(pg.item())
            stats["vl"].append(vl.item())
            stats["ent"].append(ent.mean().item())
            stats["kl"].append((old_logp[mb] - logp).mean().item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


DEFAULT_HP = dict(lr=3e-4, lam=0.95, clip=0.2, epochs=4, minibatches=4, vf=0.5,
                  ent_start=2e-2, ent_end=2e-3, ent_vote_mult=1.0, n_envs=64, rollout=4096, d=64,
                  algo="ippo", snap_every=10, pool_size=10, p_learner=0.5)


def all_learn(env, rng):
    return ["learn"] * env.n


def train_selfplay(cfg, iters, seed=0, hp=None, net=None, log_every=5, verbose=True,
                   controllers=None, assign=all_learn):
    hp = {**DEFAULT_HP, **(hp or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = net or PolicyNet(d=hp["d"])
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    ctrls = {"learn": net, **(controllers or {})}
    pool = []
    if hp["algo"] == "fsp":
        # LA3, fictitious self-play: each seat is the learner with probability
        # p_learner (at least one seat), otherwise a past checkpoint drawn
        # uniformly from the pool.  Only learner seats provide training data.
        def assign(env, rng, _pool=pool):
            seats = ["learn" if (not _pool or rng.random() < hp["p_learner"]) else
                     _pool[rng.integers(len(_pool))] for _ in range(env.n)]
            if "learn" not in seats:
                seats[rng.integers(env.n)] = "learn"
            return seats
    runner = Runner(cfg, hp["n_envs"], seed, ctrls, assign)
    history = []
    t0, steps = time.time(), 0
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        hp["ent"] = hp["ent_start"] + frac * (hp["ent_end"] - hp["ent_start"])
        for g in opt.param_groups:
            g["lr"] = hp["lr"] * (1 - 0.9 * frac)
        S, eps = runner.collect(hp["rollout"])
        steps += len(S["kind"])
        st = ppo_update(net, opt, S, runner.N, runner.M, hp)
        if hp["algo"] == "fsp" and it % hp["snap_every"] == 0:
            name = f"pool{it}"
            ctrls[name] = copy.deepcopy(net).eval()
            pool.append(name)
            if len(pool) > hp["pool_size"]:
                del ctrls[pool.pop(0)]
        cab, ep = runner.logs()
        if it % log_every == 0 or it == iters - 1:
            summ = summarise(cab, ep, cfg.m, cfg.divisible)
            rec = dict(iter=it, steps=steps, time=time.time() - t0, **st, **summ)
            history.append(rec)
            if verbose:
                print(f"[{cfg.name} s{seed}] it={it:4d} steps={steps:8d} "
                      f"t={rec['time']:6.0f}s phi={summ.get('phi', np.nan):+.3f} "
                      f"phi~={summ.get('phi_adj', np.nan):+.3f} "
                      f"dg~={summ.get('dg_adj', np.nan):.3f} gapBF={summ.get('gap_bf', np.nan):+.3f} "
                      f"fail={summ.get('fail_rate', np.nan):.2f} ent={st['ent']:.2f}", flush=True)
    return net, history


def seat_payoffs(episodes, name):
    """Average per-period payoff of seats driven by controller ``name``."""
    vals = [e["ret"][j] / e["omega"] for e in episodes for j, s in enumerate(e["seats"])
            if s == name and e["omega"] > 0]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(max(len(vals), 1)))


def exploitability(cfg, profile, iters, seed=0, hp=None, warm_start=None, eval_eps=3000):
    """Estimate Gain(profile): a best response for one random focal seat is
    trained against the frozen profile; Gain = U(BR; profile) - U(profile; profile)."""
    hp = {**DEFAULT_HP, **(hp or {})}
    frozen = profile

    def focal(env, rng):
        seats = ["frozen"] * env.n
        seats[rng.integers(env.n)] = "learn"
        return seats

    br = copy.deepcopy(warm_start) if warm_start is not None else PolicyNet(d=hp["d"])
    br, hist = train_selfplay(cfg, iters, seed=seed, hp=hp, net=br,
                              controllers={"frozen": frozen}, assign=focal, verbose=False)
    _, ep_br = evaluate(cfg, {"learn": br, "frozen": frozen}, focal, eval_eps, seed + 7)
    _, ep_base = evaluate(cfg, {"frozen": frozen}, lambda env, rng: ["frozen"] * env.n,
                              eval_eps, seed + 7)
    u_br, se_br = seat_payoffs(ep_br, "learn")
    u_base, se_base = seat_payoffs(ep_base, "frozen")
    return dict(gain=u_br - u_base, u_br=u_br, u_base=u_base,
                se=float(np.hypot(se_br, se_base)), br_history=hist)


def evaluate(cfg, controllers, assign, episodes=2000, seed=123):
    """Batched evaluation (all controllers sample their actions)."""
    runner = Runner(cfg, 64, seed, controllers, assign)
    eps = []
    while len(eps) < episodes:
        _, e = runner.collect(0, learner="__none__")
        eps += e
    return runner.logs()[0], eps     # (cabinet log, episode log)
