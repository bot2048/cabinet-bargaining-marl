"""Shared permutation-equivariant policy (spec Definition 3.1, "-rec" variant).

Every party is controlled by a copy of this network.  It reads a *set* of
party tokens and a set of ministry tokens (no party labels, no positional
encodings), contextualises them with attention, and has masked heads:

* coalition head  - include/exclude the next party, masked so that the
  coalition can always still be winning (spec Lemma 2.21);
* ministry head   - pointer over coalition members, one ministry at a time;
* Dirichlet head  - shares of a divisible surplus (levels E0-E1);
* vote and exit heads (Bernoulli) and a value head (decentralised critic).

Memory across terms comes from the history window Lambda in the tokens
(spec R9); the recurrent core of Definition 3.1 is left out to keep PPO
simple and fast on a CPU.  Setting Lambda = 0 gives the memoryless ablation.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import FP, FM, FPAIR, FC, FX, PROPOSE, VOTE, GOVERN


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth - 1):
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Block(nn.Module):
    """Self-attention over parties, cross-attention parties -> ministries, MLP."""

    def __init__(self, d, heads):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = mlp(d, 2 * d, d)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, u, v, pmask, mmask):
        u = self.n1(u + self.sa(u, u, u, key_padding_mask=~pmask)[0])
        u = self.n2(u + self.ca(u, v, v, key_padding_mask=~mmask)[0])
        return self.n3(u + self.ff(u))


class PolicyNet(nn.Module):
    def __init__(self, d=64, heads=4, blocks=2):
        super().__init__()
        self.d = d
        self.fp, self.fm = mlp(FP, d, d), mlp(FM, d, d)
        self.fc, self.fx = mlp(FC, d, d), mlp(FX, d, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(blocks)])
        self.trunk = mlp(4 * d, 2 * d, d)
        self.value = mlp(d, d, 1)
        self.vote = mlp(d, d, 1)
        self.exit = mlp(d, d, 1)
        self.include = mlp(2 * d + 3, d, 1)
        self.minister = mlp(3 * d + FPAIR + 4, d, 1)
        self.dir_logit = mlp(2 * d + 2, d, 1)
        self.dir_conc = mlp(d, d, 1)

    # ------------------------------------------------------------ encoder
    def encode(self, ob):
        P, M = ob["P"], ob["M"]
        pmask, mmask = ob["pmask"], ob["mmask"]
        u, v = self.fp(P), self.fm(M)
        for blk in self.blocks:
            u = blk(u, v, pmask, mmask)
        pm = pmask.unsqueeze(-1).float()
        pooled = (u * pm).sum(1) / pm.sum(1).clamp(min=1)
        own = u[torch.arange(u.shape[0]), ob["self_idx"]]
        h = self.trunk(torch.cat([own, pooled, self.fc(ob["ctx"]), self.fx(ob["X"])], -1))
        return u, v, h

    # ------------------------------------------------------------ heads
    def forward(self, ob, act=None, greedy=False):
        """Return (action dict, log-prob, entropy, value) for a batch.
        With ``act`` given, the log-probability of those actions is computed
        (teacher forcing) instead of sampling."""
        u, v, h = self.encode(ob)
        B = h.shape[0]
        value = self.value(h).squeeze(-1)
        kind = ob["kind"]
        logp = torch.zeros(B)
        ent = torch.zeros(B)
        out = {"bit": torch.zeros(B, dtype=torch.long),
               "inc": torch.zeros(B, ob["P"].shape[1] - 1, dtype=torch.long),
               "mu": torch.zeros(B, ob["M"].shape[1], dtype=torch.long),
               "shares": torch.zeros(B, ob["P"].shape[1])}
        for k, head in ((VOTE, self.vote), (GOVERN, self.exit)):
            idx = (kind == k).nonzero(as_tuple=True)[0]
            if len(idx):
                dist = torch.distributions.Bernoulli(logits=head(h[idx]).squeeze(-1))
                a = act["bit"][idx].float() if act is not None else (
                    (dist.probs > 0.5).float() if greedy else dist.sample())
                out["bit"][idx] = a.long()
                logp[idx] = dist.log_prob(a)
                ent[idx] = dist.entropy()
        idx = (kind == PROPOSE).nonzero(as_tuple=True)[0]
        if len(idx):
            sub = {key: val[idx] for key, val in ob.items()}
            a_sub = {key: val[idx] for key, val in act.items()} if act is not None else None
            res = self._propose(sub, u[idx], v[idx], h[idx], a_sub, greedy)
            for key in ("inc", "mu", "shares"):
                out[key][idx] = res[key]
            logp[idx] = res["logp"]
            ent[idx] = res["ent"]
        return out, logp, ent, value

    def _propose(self, ob, u, v, h, act, greedy):
        B, N, _ = u.shape
        ar = torch.arange(B)
        w, q, order = ob["w"], ob["q"], ob["order"]
        f = ob["self_idx"]
        C = torch.zeros(B, N, dtype=torch.bool)
        C[ar, f] = True
        w_inc = w[ar, f].clone()
        safe = order.clamp(min=0)
        w_ord = torch.where(order >= 0, w.gather(1, safe), torch.zeros_like(w[:, :N - 1]))
        suffix = torch.flip(torch.cumsum(torch.flip(w_ord, [1]), 1), [1])  # sum_{u'>=u}
        logp, ent = torch.zeros(B), torch.zeros(B)
        inc = torch.zeros(B, N - 1, dtype=torch.long)
        # ---- coalition decisions in decreasing order of seats
        for t in range(N - 1):
            j = order[:, t]
            valid = j >= 0
            if not valid.any():
                break
            rest_after = suffix[:, t] - w_ord[:, t]
            can_excl = (w_inc + rest_after >= q - 1e-9) & valid
            kappa = torch.stack([w_inc, (q - w_inc).clamp(min=0), rest_after], -1)
            uj = u[ar, j.clamp(min=0)]
            logit = self.include(torch.cat([uj, h, kappa], -1)).squeeze(-1)
            dist = torch.distributions.Bernoulli(logits=logit)
            if act is not None:
                a = act["inc"][:, t].float()
            elif greedy:
                a = (logit > 0).float()
            else:
                a = dist.sample()
            a = torch.where(can_excl, a, torch.ones_like(a))   # forced include
            a = torch.where(valid, a, torch.zeros_like(a))
            free = can_excl.float()
            logp = logp + free * dist.log_prob(a)
            ent = ent + free * dist.entropy()
            inc[:, t] = a.long()
            add = (a > 0.5) & valid
            C[ar[add], j[add]] = True
            w_inc = w_inc + add.float() * w_ord[:, t]
        sC = torch.where(C, w / (w * C).sum(1, keepdim=True), torch.zeros_like(w))
        is_self = F.one_hot(f, N).float()
        res = {"inc": inc, "mu": torch.zeros(B, v.shape[1], dtype=torch.long),
               "shares": torch.zeros(B, N)}
        div = ob["divisible"]
        # ---- divisible surplus: Dirichlet over coalition members
        if div.any():
            d_idx = div.nonzero(as_tuple=True)[0]
            feat = torch.cat([u[d_idx], h[d_idx].unsqueeze(1).expand(-1, N, -1),
                              sC[d_idx].unsqueeze(-1), is_self[d_idx].unsqueeze(-1)], -1)
            logits = self.dir_logit(feat).squeeze(-1)
            Cd = C[d_idx]
            logits = logits.masked_fill(~Cd, -1e9)
            conc = torch.exp(self.dir_conc(h[d_idx]).squeeze(-1).clamp(-2, 7.5)) * 10
            # alpha >= 1 keeps the density bounded at the edges of the simplex
            alpha = torch.softmax(logits, -1) * conc.unsqueeze(-1) + 1.0
            if act is not None:
                x = act["shares"][d_idx]
            elif greedy:
                x = torch.where(Cd, alpha, torch.zeros_like(alpha))
                x = x / x.sum(-1, keepdim=True)
            else:
                g = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
                g = torch.where(Cd, g.clamp(min=1e-6), torch.zeros_like(g))
                x = g / g.sum(-1, keepdim=True)
            lp, en = _masked_dirichlet(alpha, x, Cd)
            logp[d_idx] = logp[d_idx] + lp
            ent[d_idx] = ent[d_idx] + en
            res["shares"][d_idx] = x
        # ---- indivisible ministries: pointer head, one ministry at a time
        nd = ~div
        if nd.any():
            i_idx = nd.nonzero(as_tuple=True)[0]
            ui, vi, hi = u[i_idx], v[i_idx], h[i_idx]
            Ci, sCi, si = C[i_idx], sC[i_idx], is_self[i_idx]
            PR, mmask = ob["PR"][i_idx], ob["mmask"][i_idx]
            sigma = ob["M"][i_idx, :, 0] / ob["M"][i_idx, :, 0].sum(1, keepdim=True).clamp(min=1e-9)
            own_nu = ob["M"][i_idx, :, 1] / ob["M"][i_idx, :, 1].sum(1, keepdim=True).clamp(min=1e-9)
            Bi, Mm = len(i_idx), vi.shape[1]
            ari = torch.arange(Bi)
            got = torch.zeros(Bi, N)
            got_own = torch.zeros(Bi, N)
            mu = torch.zeros(Bi, Mm, dtype=torch.long)
            for k in range(Mm):
                live = mmask[:, k]
                if not live.any():
                    break
                feat = torch.cat([ui, vi[:, k].unsqueeze(1).expand(-1, N, -1),
                                  hi.unsqueeze(1).expand(-1, N, -1), PR[:, k],
                                  got.unsqueeze(-1), got_own.unsqueeze(-1),
                                  sCi.unsqueeze(-1), si.unsqueeze(-1)], -1)
                logits = self.minister(feat).squeeze(-1).masked_fill(~Ci, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                if act is not None:
                    a = act["mu"][i_idx, k]
                elif greedy:
                    a = logits.argmax(-1)
                else:
                    a = dist.sample()
                lv = live.float()
                logp[i_idx] = logp[i_idx] + lv * dist.log_prob(a)
                ent[i_idx] = ent[i_idx] + lv * dist.entropy()
                mu[:, k] = a
                got[ari, a] += lv * sigma[:, k]
                got_own[ari, a] += lv * own_nu[:, k]
            res["mu"][i_idx] = mu
        res["logp"], res["ent"] = logp, ent
        return res


def _masked_dirichlet(alpha, x, mask):
    """Log-density and entropy of a Dirichlet restricted to the coalition."""
    a = torch.where(mask, alpha, torch.ones_like(alpha))
    xs = torch.where(mask, x.clamp(min=1e-12), torch.ones_like(x))
    K = mask.sum(-1).float()
    a0 = (a * mask).sum(-1)
    lgB = (torch.lgamma(a) * mask).sum(-1) - torch.lgamma(a0)
    lp = ((a - 1) * torch.log(xs) * mask).sum(-1) - lgB
    ent = lgB + (a0 - K) * torch.digamma(a0) - ((a - 1) * torch.digamma(a) * mask).sum(-1)
    # a one-member coalition has a degenerate (deterministic) share
    single = K <= 1
    lp = torch.where(single, torch.zeros_like(lp), lp)
    ent = torch.where(single, torch.zeros_like(ent), ent)
    return lp, ent


# ---------------------------------------------------------------- batching
def pad_obs(obs, N, M):
    """Stack a list of environment observations into padded numpy arrays."""
    B = len(obs)
    out = dict(P=np.zeros((B, N, FP), np.float32), M=np.zeros((B, M, FM), np.float32),
               PR=np.zeros((B, M, N, FPAIR), np.float32), ctx=np.zeros((B, FC), np.float32),
               X=np.zeros((B, FX), np.float32), pmask=np.zeros((B, N), bool),
               mmask=np.zeros((B, M), bool), kind=np.zeros(B, np.int64),
               self_idx=np.zeros(B, np.int64), w=np.zeros((B, N), np.float32),
               q=np.zeros(B, np.float32), order=np.full((B, N - 1), -1, np.int64),
               divisible=np.zeros(B, bool))
    for b, o in enumerate(obs):
        n, m = o["n"], o["m"]
        out["P"][b, :n] = o["P"]
        out["M"][b, :m] = o["M"]
        out["PR"][b, :m, :n] = o["PR"]
        out["ctx"][b], out["X"][b] = o["ctx"], o["X"]
        out["pmask"][b, :n], out["mmask"][b, :m] = True, True
        out["kind"][b], out["self_idx"][b] = o["kind"], o["self_idx"]
        out["w"][b, :n], out["q"][b] = o["w"], o["q"]
        if o["order"] is not None:
            out["order"][b, :n - 1] = o["order"]
        out["divisible"][b] = o["divisible"]
    return out


def to_torch(batch):
    return {k: torch.as_tensor(v) for k, v in batch.items()}
