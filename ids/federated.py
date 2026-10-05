"""Federated orchestration: FedProx local training -> (clip, weight) delta -> encrypt -> server adds
ciphertexts -> collaborative decryption -> global update.  Also runs the centralized baseline (1 client)."""
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .agent import DDQNLearner, get_vec, set_vec
from .crypto import Registry, make_backend


def metrics(y, pred, score):
    tn = ((y == 0) & (pred == 0)).sum(); fp = ((y == 0) & (pred == 1)).sum()
    out = {"acc": accuracy_score(y, pred), "prec": precision_score(y, pred, zero_division=0),
           "rec": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "fpr": fp / max(1, tn + fp)}
    try:
        out["auc"] = roc_auc_score(y, score)
    except ValueError:
        out["auc"] = float("nan")
    return out


def evaluate(learner, bundle):
    res, ys, ps, ss = {}, [], [], []
    for dom, idx in bundle.tests.items():
        pred, score = learner.predict(bundle, idx)
        y = bundle.y[idx].numpy()
        res[dom] = metrics(y, pred, score)
        ys.append(y); ps.append(pred); ss.append(score)
    res["ALL"] = metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(ss))
    return res


def run(bundle, args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    n_actions = 3 if args.actions == 3 else 2
    K = bundle.X.shape[1]
    glob = DDQNLearner(K, n_actions, args)           # global model holder / evaluator
    work = DDQNLearner(K, n_actions, args)           # reusable local worker (sequential simulation)
    n_cl = len(bundle.clients)
    mu = args.mu if args.agg == "fedprox" else 0.0
    be = make_backend(args.crypto)
    reg = Registry(n_cl)
    t_setup = be.setup(n_cl)
    n_k = np.array([len(c) for c in bundle.clients], float)
    alpha = n_k / n_k.sum()
    print(f"model params={glob.n_params:,} | clients={n_cl} | agg={args.agg}(mu={mu}) | crypto={be.name} "
          f"| key setup {t_setup:.2f}s")
    hist = []
    steps = args.local_steps * (n_cl if args.mode == "centralized" else 1)   # equal total gradient budget
    for r in range(1, args.rounds + 1):
        eps = max(args.eps_min, args.eps0 * args.eps_decay ** (r - 1))
        wg = get_vec(glob.online)
        T = dict(train=0.0, enc=0.0, agg=0.0, dec=0.0, verify=0.0)
        up_bytes, cts, loss, plain = 0, [], [], np.zeros(glob.n_params, np.float32)
        for k in range(n_cl):
            set_vec(work.online, wg); work.target.load_state_dict(work.online.state_dict())
            t0 = time.time()
            loss.append(work.local_train(bundle, bundle.clients[k], steps, eps, mu, rng, w_global=wg))
            T["train"] += time.time() - t0
            delta = (get_vec(work.online) - wg)
            if k < args.byzantine:                              # simulated poisoning client (sign-flip + scale)
                delta = -args.byz_scale * delta
            nrm = delta.norm().item()
            if nrm > args.clip:                                # bounds any single client's influence (poisoning)
                delta = delta * (args.clip / nrm)
            payload = (delta * float(alpha[k])).numpy()
            plain += payload
            t0 = time.time(); ct, nb = be.encrypt(k, payload); T["enc"] += time.time() - t0
            up_bytes += nb
            pkt = reg.sign(k, r, be.blob(ct))
            t0 = time.time()
            if not reg.verify(pkt, be.blob(ct)):
                print(f"  round {r}: client {k} rejected (auth/integrity)"); continue
            T["verify"] += time.time() - t0
            cts.append(ct)
        t0 = time.time(); agg = be.aggregate(cts); T["agg"] += time.time() - t0
        t0 = time.time(); d = be.decrypt(agg, glob.n_params); T["dec"] += time.time() - t0
        set_vec(glob.online, wg + torch.from_numpy(d))
        glob.target.load_state_dict(glob.online.state_dict())
        row = {"round": r, "eps": eps, "loss": float(np.mean(loss)), "up_MB": up_bytes / 1e6, "crypto_max_err": float(np.abs(d - plain).max()),
               **{f"t_{k}": v for k, v in T.items()}}
        if r % args.eval_every == 0 or r == args.rounds:
            m = evaluate(glob, bundle)
            for dom, v in m.items():
                for kk, vv in v.items():
                    row[f"{dom}_{kk}"] = vv
            print(f"round {r:3d} loss {row['loss']:.4f} | ALL acc {m['ALL']['acc']:.4f} f1 {m['ALL']['f1']:.4f} "
                  f"rec {m['ALL']['rec']:.4f} fpr {m['ALL']['fpr']:.4f} | train {T['train']:.1f}s "
                  f"enc {T['enc']:.2f}s agg {T['agg']:.2f}s dec {T['dec']:.2f}s | up {up_bytes/1e6:.2f}MB")
        hist.append(row)
    return glob, hist


def run_local(bundle, args):
    """'Do we need FL?' baseline: every client trains ONLY on its own data (same budget), no communication.
    Reported metrics = mean over clients of each client's model evaluated on ALL domains' test sets."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    na, K = (3 if args.actions == 3 else 2), bundle.X.shape[1]
    learners, all_m = [], []
    for k, idx in enumerate(bundle.clients):
        L = DDQNLearner(K, na, args)
        for r in range(1, args.rounds + 1):
            eps = max(args.eps_min, args.eps0 * args.eps_decay ** (r - 1))
            L.local_train(bundle, idx, args.local_steps, eps, 0.0, rng)
            L.target.load_state_dict(L.online.state_dict())
        m = evaluate(L, bundle); all_m.append(m)
        print(f"client {k} ({bundle.client_domain[k]}): ALL f1 {m['ALL']['f1']:.4f} rec {m['ALL']['rec']:.4f} "
              f"| per-domain f1 " + " ".join(f"{d}={v['f1']:.3f}" for d, v in m.items() if d != "ALL"))
        learners.append(L)
    final = {d: {k: float(np.nanmean([m[d][k] for m in all_m])) for k in all_m[0][d]} for d in all_m[0]}
    return learners[0], [{"round": args.rounds, "up_MB": 0.0}], final
