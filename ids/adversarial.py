"""White-box PGD evasion test in *standardised feature space* (L_inf ball, all window steps).
Upper bound on the threat: real attackers cannot freely change every field (flags, protocol) - see README."""
import numpy as np
import torch


def pgd_eval(learner, bundle, idx, eps_list=(0.0, 0.1, 0.25, 0.5, 1.0), steps=10, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = idx[bundle.y[idx].numpy() == 1]
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    net = learner.online.eval()
    x0 = bundle.gather(torch.from_numpy(idx))
    rows = []
    for eps in eps_list:
        x = x0.clone()
        for _ in range(steps if eps > 0 else 0):
            x.requires_grad_(True)
            q = net(x)
            g = torch.autograd.grad((q[:, 1:].max(1).values - q[:, 0]).sum(), x)[0]
            x = (x.detach() - (2.5 * eps / steps) * g.sign())            # push toward "allow"
            x = x0 + (x - x0).clamp(-eps, eps)
        with torch.no_grad():
            rec = (net(x).argmax(1) != 0).float().mean().item()
        rows.append({"eps_std": eps, "attack_recall": rec})
    return rows
