"""Post-hoc XAI for the trained DDQN policy (runs offline, never in the inference path).
Attribution target = Q-margin = max(Q[block/rate-limit]) - Q[allow]  (how strongly the agent wants to act)."""
import numpy as np
import torch


def _margin(net, x):
    q = net(x)
    return q[:, 1:].max(1).values - q[:, 0]


def grad_x_input(learner, bundle, idx, n=2000, seed=0):
    """Fast global attribution: mean |grad * input| over window steps and samples."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    net = learner.online.eval()
    x = bundle.gather(torch.from_numpy(idx)).clone().requires_grad_(True)
    _margin(net, x).sum().backward()
    att = (x.grad * x).detach()
    return att.abs().mean((0, 1)).numpy(), att.mean((0, 1)).numpy()


def shap_gradient(learner, bundle, idx, n_bg=100, n_ex=200, seed=0):
    """Optional SHAP GradientExplainer (slower). Returns mean |SHAP| per feature or None on failure."""
    try:
        import shap
        rng = np.random.default_rng(seed)
        net = learner.online.eval()

        class W(torch.nn.Module):
            def forward(self, x):
                return _margin(net, x).unsqueeze(1)
        bg = bundle.gather(torch.from_numpy(rng.choice(idx, n_bg, replace=False)))
        ex = bundle.gather(torch.from_numpy(rng.choice(idx, n_ex, replace=False)))
        sv = shap.GradientExplainer(W(), bg).shap_values(ex)
        sv = np.abs(np.asarray(sv[0] if isinstance(sv, list) else sv))
        return sv.reshape(sv.shape[0], sv.shape[1], sv.shape[2], -1).mean((0, 1, 3))
    except Exception as e:                              # pragma: no cover
        print("SHAP GradientExplainer skipped:", repr(e)[:120])
        return None


def lime_local(learner, bundle, end_idx, names, n_bg=500, k=10, seed=0):
    """LIME on the newest flow of one window (history held fixed). Returns [(feature_rule, weight), ...]."""
    from lime.lime_tabular import LimeTabularExplainer
    rng = np.random.default_rng(seed)
    net = learner.online.eval()
    x0 = bundle.gather(torch.tensor([end_idx]))[0]
    bg_ids = rng.choice(len(bundle.X), n_bg, replace=False)
    bg = bundle.X[torch.from_numpy(bg_ids)].numpy()

    def predict(Z):
        xs = x0.unsqueeze(0).repeat(len(Z), 1, 1)
        xs[:, -1, :] = torch.from_numpy(Z.astype(np.float32))
        with torch.no_grad():
            p = torch.sigmoid(_margin(net, xs)).numpy()
        return np.stack([1 - p, p], 1)
    exp = LimeTabularExplainer(bg, feature_names=names, class_names=["benign", "attack"], mode="classification",
                               random_state=seed)
    return exp.explain_instance(x0[-1].numpy(), predict, num_features=k, num_samples=1000).as_list()
