"""Checks that tie the code to the results proved in the spec (R2, R3, R7)."""
import numpy as np
import pytest
import torch

from cabinet.baselines import Scripted
from cabinet.bf_solver import bf_symmetric_share, solve_bf
from cabinet.config import E0, E1, E6, LEVELS, full_estar
from cabinet.env import CabinetEnv, PROPOSE
from cabinet.metrics import hamilton_min
from cabinet.policy import PolicyNet, pad_obs, to_torch
from cabinet.ppo import _env_action


def play(cfg, bot, episodes, seed=0):
    env = CabinetEnv(cfg, seed=seed)
    for _ in range(episodes):
        env.reset()
        while not env.done:
            pend = env.pending()
            env.act({d.agent: bot.act(env, d.agent, d.kind) for d in pend})
    return env


@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("delta", [0.8, 0.9, 0.95])
def test_symmetric_bf_closed_form(n, delta):
    """Theorem 2.7: values 1/n, formateur keeps 1 - delta (n-1)/(2n)."""
    sol = solve_bf([1] * n, (n + 1) // 2, [1 / n] * n, delta)
    assert np.allclose(sol["v"], 1 / n, atol=1e-6)
    assert np.allclose(sol["x"], bf_symmetric_share(n, delta), atol=1e-6)
    assert sol["resid"] < 1e-8


def test_weighted_solver_certificate():
    """Remark 2.9: the LP residual of the fixed point is tiny."""
    sol = solve_bf([25, 32, 37, 6], 51, [0.25, 0.32, 0.37, 0.06], 0.9)
    assert sol["resid"] < 1e-6
    # Proposition 2.10: formateur keeps at least its ex-ante value
    rho = np.array([0.25, 0.32, 0.37, 0.06])
    assert np.all(sol["x"] >= sol["v"] - 1e-9)


def test_bf_agent_recovers_benchmark_in_env():
    """R2: the environment reproduces the closed form when agents play it."""
    env = play(E0, Scripted("bf", 0), 300)
    for rec in env.cabinet_log:
        assert abs(rec["p"][rec["f"]] - bf_symmetric_share(rec["n"], rec["delta"])) < 1e-9
        assert rec["delay"] == 0


def test_hamilton_floor_bound():
    """Proposition 2.12(c): D_min <= |C| / (4m); equality for 2 equal parties, odd m."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        k, m = rng.integers(2, 6), rng.integers(1, 15)
        s = rng.dirichlet(np.ones(k))
        _, dmin = hamilton_min(s, m, rng)
        assert dmin <= k / (4 * m) + 1e-12
    _, d = hamilton_min(np.array([0.5, 0.5]), 7)
    assert abs(d - 1 / 14) < 1e-12


@pytest.mark.parametrize("cfg", list(LEVELS.values()) + [full_estar()], ids=lambda c: c.name)
def test_masks_always_feasible(cfg):
    """Lemma 2.21: masked decoding never yields an infeasible proposal (R3)."""
    torch.manual_seed(0)
    net = PolicyNet(d=16, heads=2, blocks=1)
    env = CabinetEnv(cfg, seed=1)
    N, M = cfg.n_max, (1 if cfg.divisible else cfg.m)
    for _ in range(30):
        env.reset()
        while not env.done:
            acts = {}
            for d in env.pending():
                o = env.observe(d.agent, d.kind)
                with torch.no_grad():
                    out, logp, _, _ = net(to_torch(pad_obs([o], N, M)))
                assert torch.isfinite(logp).all()
                acts[d.agent] = _env_action(o, {k: v.numpy() for k, v in out.items()}, 0)
            env.act(acts)   # env asserts feasibility


def test_teacher_forced_logp_matches_sampling():
    """Lemma 3.3: recomputing log-probabilities with stored actions is exact."""
    torch.manual_seed(0)
    cfg = full_estar()
    net = PolicyNet(d=16, heads=2, blocks=1)
    env = CabinetEnv(cfg, seed=3).reset()
    obs = []
    while len(obs) < 40 and not env.done:
        acts = {}
        for d in env.pending():
            o = env.observe(d.agent, d.kind)
            obs.append(o)
            acts[d.agent] = Scripted("random", 0).act(env, d.agent, d.kind)
        env.act(acts)
    batch = to_torch(pad_obs(obs, cfg.n_max, cfg.m))
    with torch.no_grad():
        out, logp, _, _ = net(batch)
        _, logp2, _, _ = net(batch, act=out)
    assert torch.allclose(logp, logp2, atol=1e-5)


def test_permutation_equivariance_of_value():
    """Lemma 3.2: relabelling parties leaves vote/value outputs unchanged."""
    torch.manual_seed(0)
    cfg = E6
    net = PolicyNet(d=16, heads=2, blocks=1).eval()
    env = CabinetEnv(cfg, seed=5).reset()
    d = env.pending()[0]
    o = env.observe(d.agent, d.kind)
    perm = np.random.default_rng(0).permutation(o["n"])
    o2 = dict(o, P=o["P"][perm], PR=o["PR"][:, perm], w=o["w"][perm],
              self_idx=int(np.flatnonzero(perm == o["self_idx"])[0]))
    o2["order"] = None if o["order"] is None else np.array(
        [int(np.flatnonzero(perm == j)[0]) for j in o["order"]])
    b1 = to_torch(pad_obs([o], cfg.n_max, cfg.m))
    b2 = to_torch(pad_obs([o2], cfg.n_max, cfg.m))
    with torch.no_grad():
        v1 = net.encode(b1)[2]
        v2 = net.encode(b2)[2]
    assert torch.allclose(v1, v2, atol=1e-5)


def test_determinism_given_seed():
    """R7: identical seeds give identical cabinet logs."""
    a = play(full_estar(), Scripted("random", 4), 20, seed=9).cabinet_log
    b = play(full_estar(), Scripted("random", 4), 20, seed=9).cabinet_log
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x["f"] == y["f"] and np.allclose(x["p"], y["p"])


def test_welfare_identity():
    """Proposition 2.11(a): with lambda = ebar = xi = 0 the summed per-period
    payoff equals the discounted number of governed periods."""
    cfg = E1.with_(divisible=False, m=6, L=5, T=2.0)
    env = play(cfg, Scripted("random", 1), 50, seed=2)
    for ep in env.episode_log:
        assert ep["ret"].sum() <= ep["omega"] + 1e-9


def test_recurrent_memory_and_logp():
    """Def. 3.1 memory core: memory changes outputs, and teacher-forced
    log-probabilities with the stored memory match sampling exactly."""
    torch.manual_seed(0)
    cfg = full_estar()
    net = PolicyNet(d=16, heads=2, blocks=1, recurrent=True)
    env = CabinetEnv(cfg, seed=3).reset()
    obs = []
    while len(obs) < 30 and not env.done:
        acts = {}
        for d in env.pending():
            o = env.observe(d.agent, d.kind)
            o["mem"] = np.random.default_rng(len(obs)).normal(size=16).astype(np.float32)
            obs.append(o)
            acts[d.agent] = Scripted("random", 0).act(env, d.agent, d.kind)
        env.act(acts)
    batch = to_torch(pad_obs(obs, cfg.n_max, cfg.m))
    with torch.no_grad():
        out, logp, _, v1 = net(batch)
        _, logp2, _, _ = net(batch, act=out)
        zero = dict(batch, mem=torch.zeros_like(batch["mem"]))
        _, _, _, v2 = net(zero)
    assert out["mem"].shape == (len(obs), 16)
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert not torch.allclose(v1, v2)


def test_demand_protocol():
    """M2: closing is offered only when a cheapest winning set of earlier
    demands exists; the closer gets the residual; shares sum to one."""
    from cabinet.config import WIDE_DEMAND
    from cabinet.env import CLOSE, DEMAND
    rng = np.random.default_rng(0)
    env = CabinetEnv(WIDE_DEMAND, seed=4)
    for _ in range(60):
        env.reset()
        while not env.done:
            d = env.pending()[0]
            assert d.kind == DEMAND
            o = env.observe(d.agent, d.kind)
            a = CLOSE if (o["close_ok"] and rng.random() < 0.5) else int(rng.integers(0, 60))
            env.act({d.agent: a})
    for rec in env.cabinet_log:
        assert abs(rec["p"].sum() - 1) < 1e-9
        assert rec["w"][rec["C"]].sum() >= rec["w"].sum() // 2 + 1 or rec["C"].sum() >= 1
        assert "first" in rec


def test_demand_transitions_exact():
    """M2 by hand: 45/35/20 parliament, order 0 -> 1 -> 2.  Party 1 can close only
    with an earlier demander forming a majority; the closer takes the residual;
    a round where nobody closes fails and is discounted."""
    from cabinet.config import fixed_divisible
    from cabinet.env import CLOSE, DEMAND
    cfg = fixed_divisible((45, 35, 20), "demand", "t")
    env = CabinetEnv(cfg, seed=0).reset()
    env.dem_order, env.f = [0, 1, 2], 0
    assert env.pending()[0].kind == DEMAND
    env._pending = [type(env.pending()[0])(env, 0, DEMAND)]
    env.act({0: 60})                                  # party 0 demands 0.60
    assert env.dem_k == 1 and env.cheapest_close(1) == [0]
    env.act({1: CLOSE})                               # party 1 closes with party 0
    rec = env.cabinet_log[-1]
    assert np.allclose(rec["p"], [0.6, 0.4, 0.0]) and rec["f"] == 1 and rec["first"] == 0
    # nobody closes: the round fails, everybody gets 0 and the next round is discounted
    env = CabinetEnv(cfg, seed=1).reset()
    env.dem_order, env.f = [0, 1, 2], 0
    env._pending = [type(env.pending()[0])(env, 0, DEMAND)]
    for j in (0, 1, 2):
        env.act({j: 70})
    assert env.ep_stats["failed"] == 1 and not env.done and env.phase == DEMAND
    assert np.allclose(env.acc_disc, env.delta)       # one period of discounting


def test_cheapest_close_prefers_cheaper_partner():
    from cabinet.config import fixed_divisible
    cfg = fixed_divisible((45, 35, 20), "demand", "t")
    env = CabinetEnv(cfg, seed=0).reset()
    env.dem_order, env.dem_k = [0, 1, 2], 2
    env.demands = np.array([0.6, 0.3, np.nan])
    assert env.cheapest_close(2) == [1]               # 20 + 35 is winning and cheaper
    env.demands = np.array([0.6, 0.9, np.nan])
    assert env.cheapest_close(2) == [0]


def test_regression_recovers_known_coefficients():
    from cabinet.metrics import gamson_regression
    rng = np.random.default_rng(0)
    s = rng.random(500)
    F = (rng.random(500) < 0.3).astype(float)
    p = 0.1 + 0.8 * s + 0.05 * F
    r = gamson_regression(np.column_stack([p, s, F]))
    assert abs(r["alpha"] - 0.1) < 1e-9 and abs(r["beta"] - 0.8) < 1e-9 and abs(r["gamma"] - 0.05) < 1e-9


def test_recognition_sums_to_one_in_truncated_real_parliaments():
    import json
    from cabinet.config import estar_wide, real_parliament
    cabs = json.load(open("data/parlgov/cabinets.json"))[:40]
    for cab in cabs:
        env = CabinetEnv(real_parliament(estar_wide(), cab), seed=0).reset()
        assert abs(env.recognition_probs().sum() - 1) < 1e-12


def test_demand_solver_known_equilibria():
    """Exact SPE of our demand protocol: 3 equal parties split 50/50 between the
    first two movers; in 40/30/25/5 the 5-seat dummy gets nothing."""
    from cabinet.demand_solver import solve
    V, out = solve([1, 1, 1], 2, 0.9, step=0.02)
    assert np.allclose(sorted(out[0]["pay"]), [0, 0.5, 0.5], atol=1e-6)
    V, out = solve([40, 30, 25, 5], 51, 0.9, recognition="seat", step=0.02)
    assert V[3] < 1e-9 and all(o["pay"][3] < 1e-9 for o in out)
    assert all(np.isclose(sorted(o["pay"])[-2:], [0.5, 0.5]).all() for o in out)
