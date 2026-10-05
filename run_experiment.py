"""One experiment = one (mode, aggregation, crypto) configuration. See run_all.sh for the full study."""
import argparse, json, os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch

from ids.data import build_bundle
from ids.federated import run, run_local, evaluate
from ids import explain as X


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True,
                   help='name=glob[,glob]  e.g. insdn="data/insdn/*.csv" cicids2017="data/cicids/*.csv"')
    p.add_argument("--sample_frac", default="", help="per-domain block-sampling, e.g. insdn=1,cicddos2019=0.02")
    p.add_argument("--tag", default="run"); p.add_argument("--out", default="results"); p.add_argument("--seed", type=int, default=0)
    # granularity / features
    p.add_argument("--window", type=int, default=8, help="flows per state (1 = flow-level granularity)")
    p.add_argument("--group_by", choices=["none", "dst_ip"], default="none", help="build windows per victim host")
    p.add_argument("--n_features", type=int, default=32); p.add_argument("--keep_ports", action="store_true")
    p.add_argument("--test_frac", type=float, default=0.3); p.add_argument("--test_cap", type=int, default=60000)
    # federation
    p.add_argument("--mode", choices=["federated", "centralized", "local"], default="federated")
    p.add_argument("--partition", choices=["domain", "iid", "dirichlet"], default="domain")
    p.add_argument("--clients", type=int, default=6, help="for iid/dirichlet"); p.add_argument("--clients_per_domain", type=int, default=2)
    p.add_argument("--dirichlet_alpha", type=float, default=0.3)
    p.add_argument("--agg", choices=["fedavg", "fedprox"], default="fedprox"); p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--crypto", choices=["none", "tenseal", "openfhe"], default="none")
    p.add_argument("--clip", type=float, default=5.0, help="L2 clip of each client's delta")
    p.add_argument("--rounds", type=int, default=15); p.add_argument("--local_steps", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1)
    # agent
    p.add_argument("--actions", type=int, choices=[2, 3], default=3); p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=256); p.add_argument("--replay", type=int, default=50000)
    p.add_argument("--target_sync", type=int, default=50); p.add_argument("--balanced", type=int, default=1)
    p.add_argument("--eps0", type=float, default=1.0); p.add_argument("--eps_min", type=float, default=0.05)
    p.add_argument("--eps_decay", type=float, default=0.8)
    p.add_argument("--adv_eps", type=float, default=0.0, help="FGSM adversarial-training strength (std units)")
    p.add_argument("--adv_lambda", type=float, default=0.5)
    p.add_argument("--adv_eval", action="store_true", help="PGD evasion test after training")
    p.add_argument("--byzantine", type=int, default=0, help="# poisoning clients (sign-flip x byz_scale)")
    p.add_argument("--byz_scale", type=float, default=3.0)
    p.add_argument("--explain", action="store_true"); p.add_argument("--shap_grad", action="store_true")
    a = p.parse_args()
    a.sample_frac = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.sample_frac.split(",") if kv}
    return a


if __name__ == "__main__":
    a = parse()
    out = os.path.join(a.out, a.tag); os.makedirs(out, exist_ok=True)
    t0 = time.time(); bundle = build_bundle(a); t_data = time.time() - t0
    t0 = time.time()
    if a.mode == "local":
        model, hist, final = run_local(bundle, a)
    else:
        model, hist = run(bundle, a); final = None
    t_fl = time.time() - t0
    pd.DataFrame(hist).to_csv(f"{out}/history.csv", index=False)
    final = final or evaluate(model, bundle)
    summ = {"args": vars(a), "final": final, "params": model.n_params, "time_data_s": t_data, "time_train_s": t_fl,
            "total_up_MB": float(sum(h["up_MB"] for h in hist)), "info": bundle.info}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1, default=float)
    json.dump(bundle.spec, open(f"{out}/feature_spec.json", "w"))
    torch.save(model.online.state_dict(), f"{out}/ddqn_gru.pt")
    print("\nFINAL (test, natural class distribution):")
    print(pd.DataFrame(final).T.round(4).to_string())
    if a.adv_eval:
        from ids.adversarial import pgd_eval
        ev = pgd_eval(model, bundle, np.concatenate(list(bundle.tests.values())))
        pd.DataFrame(ev).to_csv(f"{out}/adv_pgd.csv", index=False); print("\nPGD evasion (attack recall vs eps):\n", pd.DataFrame(ev).round(4).to_string(index=False))
    if a.explain:
        names = bundle.spec["names"]; allidx = np.concatenate(list(bundle.tests.values()))
        ab, sg = X.grad_x_input(model, bundle, allidx)
        df = pd.DataFrame({"feature": names, "grad_x_input": ab, "signed": sg})
        if a.shap_grad:
            s = X.shap_gradient(model, bundle, allidx)
            if s is not None: df["shap_gradient"] = s
        df.sort_values("grad_x_input", ascending=False).to_csv(f"{out}/explain_global.csv", index=False)
        atk = allidx[bundle.y[allidx].numpy() == 1]
        try:
            pd.DataFrame(X.lime_local(model, bundle, int(atk[0]), names), columns=["rule", "weight"]).to_csv(
                f"{out}/explain_lime_example.csv", index=False)
        except Exception as e:
            print("LIME skipped:", repr(e)[:100])
        print("\nTop features driving the policy:\n", df.sort_values("grad_x_input", ascending=False).head(10).to_string(index=False))
