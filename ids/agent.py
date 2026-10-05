"""Lightweight GRU Q-network + Double-DQN local learner with FedProx proximal term.

MDP view (per flow window): state = last-L flows (B,L,K); actions = {allow, rate-limit, block}
(or {allow, block}); reward = asymmetric cost matrix (missed attack is worst, blocking benign next).
The replay buffer stores only integer indices + rewards (states are re-gathered) -> tiny memory.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# reward[y][a]; rows: benign / attack.  Columns: allow, rate-limit, block
REWARD3 = np.array([[+1.0, -0.5, -2.0],
                    [-2.0, +0.5, +1.0]], dtype=np.float32)
REWARD2 = REWARD3[:, [0, 2]]


class GRUQNet(nn.Module):
    """1-layer GRU(64) -> LayerNorm -> MLP head.  ~20k params (vs ~420k for 3-layer 256/128/64 GRU)."""

    def __init__(self, in_dim, n_actions, hidden=64, head=32, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)      # LayerNorm (not BatchNorm): no running stats to average in FL
        self.head = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(head, n_actions))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(self.norm(h[-1]))


def get_vec(net):
    return torch.nn.utils.parameters_to_vector(net.parameters()).detach().clone()


def set_vec(net, vec):
    torch.nn.utils.vector_to_parameters(vec.clone(), net.parameters())


class Replay:
    def __init__(self, cap):
        self.cap, self.n, self.p = cap, 0, 0
        self.e = np.zeros(cap, np.int64); self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32); self.e2 = np.zeros(cap, np.int64)

    def push(self, e, a, r, e2):
        for i in range(len(e)):
            self.e[self.p], self.a[self.p], self.r[self.p], self.e2[self.p] = e[i], a[i], r[i], e2[i]
            self.p = (self.p + 1) % self.cap
        self.n = min(self.cap, self.n + len(e))

    def sample(self, b, rng):
        i = rng.integers(0, self.n, b)
        return self.e[i], self.a[i], self.r[i], self.e2[i]


class DDQNLearner:
    def __init__(self, in_dim, n_actions, args):
        self.a = args
        self.na = n_actions
        self.R = REWARD3 if n_actions == 3 else REWARD2
        self.online = GRUQNet(in_dim, n_actions, args.hidden)
        self.target = copy.deepcopy(self.online).eval()
        self.n_params = sum(p.numel() for p in self.online.parameters())

    # ---- one federated round of local training (also used for the centralized baseline)
    def local_train(self, bundle, idx, steps, eps, mu, rng, w_global=None):
        a = self.a
        self.online.train()
        opt = torch.optim.Adam(self.online.parameters(), lr=a.lr)
        y = bundle.y
        yn = y.numpy(); pos = idx[yn[idx] == 1]; neg = idx[yn[idx] == 0]
        replay = Replay(a.replay)
        gl = None if w_global is None or mu == 0 else [p.detach().clone() for p in self.online.parameters()]
        n_all = len(bundle.y)
        losses = []
        for t in range(steps):
            # ---- act (epsilon-greedy) on a class-balanced batch of new flow windows
            if a.balanced and len(pos) and len(neg):
                e = np.concatenate([rng.choice(pos, a.batch // 2), rng.choice(neg, a.batch - a.batch // 2)])
            else:
                e = rng.choice(idx, a.batch)
            et = torch.from_numpy(e)
            with torch.no_grad():
                greedy = self.online(bundle.gather(et)).argmax(1).numpy()
            act = np.where(rng.random(len(e)) < eps, rng.integers(0, self.na, len(e)), greedy)
            r = self.R[y[et].numpy(), act]
            replay.push(e, act, r, np.minimum(e + 1, n_all - 1))
            # ---- learn (Double DQN target)
            be, ba, br, be2 = replay.sample(a.batch, rng)
            s = bundle.gather(torch.from_numpy(be))
            q = self.online(s).gather(1, torch.from_numpy(ba).unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if a.gamma > 0:
                    s2 = bundle.gather(torch.from_numpy(be2))
                    a_star = self.online(s2).argmax(1, keepdim=True)          # online net selects ...
                    q2 = self.target(s2).gather(1, a_star).squeeze(1)         # ... target net evaluates
                    tgt = torch.from_numpy(br) + a.gamma * q2
                else:
                    tgt = torch.from_numpy(br)
            loss = Fn.smooth_l1_loss(q, tgt)
            if a.adv_eps > 0:                                                 # FGSM consistency (adversarial training)
                s_req = s.clone().requires_grad_(True)
                qq = self.online(s_req)
                g = torch.autograd.grad((qq[:, 1:].max(1).values - qq[:, 0]).sum(), s_req)[0]
                sign = torch.where(y[torch.from_numpy(be)].view(-1, 1, 1) == 1, -1.0, 1.0)   # attacker hides / benign mimics
                s_adv = (s + a.adv_eps * sign * g.sign()).detach()
                q_adv = self.online(s_adv).gather(1, torch.from_numpy(ba).unsqueeze(1)).squeeze(1)
                loss = loss + a.adv_lambda * Fn.smooth_l1_loss(q_adv, tgt)
            opt.zero_grad(); loss.backward()
            if gl is not None:                                                # FedProx: grad += mu (w - w_g)
                for p, g0 in zip(self.online.parameters(), gl):
                    p.grad.add_(mu * (p.detach() - g0))
            nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
            opt.step()
            if (t + 1) % a.target_sync == 0:
                self.target.load_state_dict(self.online.state_dict())
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def predict(self, bundle, idx, bs=4096):
        self.online.eval()
        pred, score = [], []
        for i in range(0, len(idx), bs):
            q = self.online(bundle.gather(torch.from_numpy(idx[i:i + bs])))
            pred.append((q.argmax(1) != 0).numpy())
            score.append((q[:, 1:].max(1).values - q[:, 0]).numpy())
        return np.concatenate(pred).astype(int), np.concatenate(score)
