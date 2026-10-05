"""Feature engineering + FEDERATED XAI feature selection.

Nothing here needs raw rows to leave a client: clients only send (n, sum, X^T X) for the
scaler / correlation matrix and a normalised importance vector (RF-Gini + SHAP).
"""
import numpy as np


# ---------------------------------------------------------------- engineering (local, deterministic)
def engineer(df, cols):
    """Canonical dataframe -> (float32 matrix, names). Adds ratio/rate features, signed-log1p all."""
    d = {c: df[c].to_numpy(np.float64) for c in cols}
    add = {}

    def ratio(name, a, b, eps=1.0):
        if a in d and b in d:
            add[name] = d[a] / (np.abs(d[b]) + eps)

    ratio("r_fwd_bwd_pkts", "tot_fwd_pkts", "tot_bwd_pkts")
    ratio("r_fwd_bwd_bytes", "totlen_fwd_pkts", "totlen_bwd_pkts")
    ratio("bytes_per_pkt_fwd", "totlen_fwd_pkts", "tot_fwd_pkts")
    ratio("bytes_per_pkt_bwd", "totlen_bwd_pkts", "tot_bwd_pkts")
    ratio("iat_cv", "flow_iat_std", "flow_iat_mean", 1e-6)
    ratio("syn_ack_ratio", "syn_flag_cnt", "ack_flag_cnt")
    ratio("active_idle_ratio", "active_mean", "idle_mean")
    fl = [f"{f}_flag_cnt" for f in ("fin", "syn", "rst", "psh", "ack", "urg")]
    if all(k in d for k in fl + ["tot_fwd_pkts", "tot_bwd_pkts"]):
        add["flag_density"] = sum(d[k] for k in fl) / (d["tot_fwd_pkts"] + d["tot_bwd_pkts"] + 1)
    if all(k in d for k in ["fwd_header_len", "bwd_header_len", "totlen_fwd_pkts", "totlen_bwd_pkts"]):
        add["hdr_overhead"] = (d["fwd_header_len"] + d["bwd_header_len"]) / (
            d["totlen_fwd_pkts"] + d["totlen_bwd_pkts"] + 1)
    if all(k in d for k in ["flow_duration", "tot_fwd_pkts", "tot_bwd_pkts"]):
        add["dur_per_pkt"] = d["flow_duration"] / (d["tot_fwd_pkts"] + d["tot_bwd_pkts"] + 1)
    d.update(add)
    names = list(d)
    M = np.stack([d[n] for n in names], 1)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = np.sign(M) * np.log1p(np.abs(M))          # heavy-tail compression, same rule for every client
    return M.astype(np.float32), names


# ---------------------------------------------------------------- federated statistics
def local_stats(M):
    M64 = M.astype(np.float64)
    return {"n": len(M64), "s": M64.sum(0), "ss": M64.T @ M64}


def merge_stats(stats):
    n = sum(s["n"] for s in stats)
    s = sum(x["s"] for x in stats)
    ss = sum(x["ss"] for x in stats)
    mean = s / n
    cov = ss / n - np.outer(mean, mean)
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(std, std)
    return mean.astype(np.float32), std.astype(np.float32), np.nan_to_num(corr)


def local_importance(M, y, seed=0, n=3000, shap_n=300):
    """RF-Gini + mean|SHAP| (TreeExplainer) on a small stratified local sample -> normalised vector."""
    import shap
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        idx.append(rng.choice(ci, min(len(ci), n // 2), replace=False))
    idx = np.concatenate(idx)
    Xs, ys = M[idx], y[idx]
    rf = RandomForestClassifier(n_estimators=60, max_depth=10, n_jobs=-1, random_state=seed,
                                class_weight="balanced_subsample").fit(Xs, ys)
    imp_rf = rf.feature_importances_
    sv = shap.TreeExplainer(rf).shap_values(Xs[rng.choice(len(Xs), min(shap_n, len(Xs)), replace=False)])
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
    imp_shap = np.abs(sv).mean(0)
    f = lambda v: v / (v.sum() + 1e-12)
    return 0.5 * f(imp_rf) + 0.5 * f(imp_shap)


def select_features(names, mean, std, corr, imps, weights, k=32, corr_thr=0.95):
    imp = sum(w * i for w, i in zip(weights, imps)) / sum(weights)
    ok = std > 1e-5
    chosen = []
    for f in np.argsort(-imp):
        if not ok[f]:
            continue
        if all(abs(corr[f, g]) < corr_thr for g in chosen):
            chosen.append(int(f))
        if len(chosen) == k:
            break
    return {"names": [names[i] for i in chosen], "idx": chosen, "mean": mean[chosen].tolist(),
            "std": std[chosen].tolist(), "importance": [float(imp[i]) for i in chosen]}


def apply_spec(M, spec):
    X = (M[:, spec["idx"]] - np.asarray(spec["mean"], np.float32)) / np.asarray(spec["std"], np.float32)
    return np.clip(X, -5, 5).astype(np.float32)
