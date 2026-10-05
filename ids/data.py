"""Loading, granularity (flow vs time/host windows), leakage-safe splitting, client partitioning."""
import glob
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from . import features as F
from .schema import FEATURES, canonicalize

BENIGN = {"benign", "normal"}
BLOCK = 512  # windows per split block (train/test are split by whole blocks -> no overlap leakage)


@dataclass
class Bundle:
    X: torch.Tensor                    # (N, K) scaled, selected features (all domains concatenated)
    y: torch.Tensor                    # (N,) 0 benign / 1 attack
    clients: list                      # list[np.ndarray] of window END indices (train)
    client_domain: list
    tests: dict                        # domain -> np.ndarray of window END indices (test)
    spec: dict                         # selected features + scaler (deploy artefact)
    L: int
    info: dict = field(default_factory=dict)

    def gather(self, end):             # end: LongTensor (B,) -> (B, L, K)
        if self.L == 1:
            return self.X[end].unsqueeze(1)
        off = torch.arange(-self.L + 1, 1)
        return self.X[end.unsqueeze(1) + off]


# ------------------------------------------------------------------ loading
def load_domain(name, patterns, frac, group_by, rng, block=256, keep_ports=False):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"[{name}] no files match {patterns}")
    parts, base = [], 0
    for fi, f in enumerate(files):
        for ch in pd.read_csv(f, chunksize=200_000, low_memory=False, encoding_errors="ignore"):
            ch = canonicalize(ch)
            if "label" not in ch:
                continue
            n = len(ch)
            if frac < 1.0:                      # keep random *contiguous blocks* to preserve temporal locality
                nb = math.ceil(n / block)
                keep = np.repeat(rng.random(nb) < frac, block)[:n]
                seg = base + np.arange(n) // block
                base += nb
                ch, seg = ch[keep], seg[keep]
            else:
                seg = np.full(n, fi)
            ch = ch.copy()
            ch["seg"] = seg
            parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in FEATURES if c in df.columns]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates(subset=feats + ["label"]).reset_index(drop=True)
    y = (~df["label"].astype(str).str.strip().str.lower().isin(BENIGN)).astype(np.int64).to_numpy()
    if "timestamp" in df:
        t = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
        t = t.astype("int64").to_numpy() / 1e9 if t.notna().mean() > 0.9 else np.arange(len(df), dtype=float)
    else:
        t = np.arange(len(df), dtype=float)
    if group_by == "dst_ip" and "dst_ip" in df:
        grp = pd.factorize(df["dst_ip"])[0]
    else:
        if group_by == "dst_ip":
            print(f"[{name}] no Dst IP column -> falling back to group_by=none")
        grp = np.zeros(len(df), dtype=np.int64)
    order = np.lexsort((t, grp, df["seg"].to_numpy()))
    df, y, grp = df.iloc[order].reset_index(drop=True), y[order], grp[order]
    key = df["seg"].to_numpy().astype(np.int64) * (int(grp.max()) + 2) + grp
    print(f"[{name}] {len(df):,} flows from {len(files)} file(s) | attack share {y.mean():.2%} | "
          f"{len(feats)} canonical features")
    return {"df": df, "y": y, "key": key, "feats": feats}


def valid_ends(key, L):
    n = len(key)
    if L == 1:
        return np.arange(n)
    return np.flatnonzero(key[L - 1:] == key[: n - L + 1]) + (L - 1)


def split_blocks(ends, L, test_frac, rng):
    """Split the valid windows into contiguous blocks; whole blocks go to train or test."""
    nb = math.ceil(len(ends) / BLOCK)
    blk = np.arange(len(ends)) // BLOCK
    pos = np.arange(len(ends)) % BLOCK
    is_test_blk = rng.random(nb) < test_frac
    if is_test_blk.all() or not is_test_blk.any():
        is_test_blk[rng.integers(nb)] = ~is_test_blk[0]
    keep = pos >= (L - 1) if L > 1 else np.ones(len(ends), bool)   # drop block-border windows that overlap
    tr = [ends[(blk == b) & keep] for b in range(nb) if not is_test_blk[b]]
    te = ends[keep & is_test_blk[blk]]
    return [b for b in tr if len(b)], te


# ------------------------------------------------------------------ partitioning
def partition(train_blocks, dom_of_block, y_all, mode, n_clients, k_per_domain, alpha, rng):
    """train_blocks: list[np.ndarray] (global end idx); returns (list of arrays, list of domain names)."""
    doms = sorted(set(dom_of_block))
    if mode == "domain":
        out, names = [], []
        for d in doms:
            bl = [b for b, dd in zip(train_blocks, dom_of_block) if dd == d]
            rng.shuffle(bl)
            for chunk in np.array_split(np.arange(len(bl)), k_per_domain):
                if len(chunk):
                    out.append(np.concatenate([bl[i] for i in chunk]))
                    names.append(d)
        return out, names
    order = rng.permutation(len(train_blocks))
    if mode == "iid":
        groups = np.array_split(order, n_clients)
    elif mode == "dirichlet":                       # label-skew non-IID across pooled domains
        lab = np.array([int(y_all[b].mean() > 0.5) for b in train_blocks])
        assign = [[] for _ in range(n_clients)]
        for c in (0, 1):
            ids = order[lab[order] == c]
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                assign[k].extend(part.tolist())
        groups = [np.array(a, dtype=int) for a in assign if len(a)]
    else:
        raise ValueError(mode)
    out = [np.concatenate([train_blocks[i] for i in g]) for g in groups]
    return out, ["mixed"] * len(out)


# ------------------------------------------------------------------ full pipeline
def build_bundle(args):
    rng = np.random.default_rng(args.seed)
    doms = {}
    for spec in args.data:                                  # name=glob[,glob...]
        name, pats = spec.split("=", 1)
        frac = args.sample_frac.get(name, 1.0)
        doms[name] = load_domain(name, pats.split(","), frac, args.group_by, rng, keep_ports=args.keep_ports)
    common = [c for c in FEATURES if all(c in d["feats"] for d in doms.values())]
    print(f"common canonical features across domains: {len(common)}")
    if len(common) < 20:
        print("WARNING: few common features -> check column names in your CSVs")

    Ms, ys, tests_local, blocks, dom_of_block, off = [], [], {}, [], [], 0
    for name, d in doms.items():
        M, names = F.engineer(d["df"], common)
        ends = valid_ends(d["key"], args.window) + off
        tb, te = split_blocks(ends - off, args.window, args.test_frac, rng)
        blocks += [b + off for b in tb]
        dom_of_block += [name] * len(tb)
        if len(te) > args.test_cap:
            te = np.sort(rng.choice(te, args.test_cap, replace=False))
        tests_local[name] = te + off
        Ms.append(M); ys.append(d["y"]); off += len(M)
    M_all, y_all = np.concatenate(Ms), np.concatenate(ys)
    del Ms

    clients, cdom = partition(blocks, dom_of_block, y_all, args.partition, args.clients, args.clients_per_domain,
                              args.dirichlet_alpha, rng)
    if args.mode == "centralized":
        clients, cdom = [np.concatenate(clients)], ["pooled"]
    print(f"{len(clients)} clients, train windows/client: {[len(c) for c in clients]}")

    # ---- federated feature selection (only aggregate statistics + importance vectors are shared)
    stats, imps, wts = [], [], []
    for i, c in enumerate(clients):
        rows = M_all[c]
        stats.append(F.local_stats(rows))
        if len(np.unique(y_all[c])) == 2:
            imps.append(F.local_importance(rows, y_all[c], seed=args.seed + i)); wts.append(len(c))
    mean, std, corr = F.merge_stats(stats)
    spec = F.select_features(names, mean, std, corr, imps, wts, k=args.n_features)
    print(f"selected {len(spec['names'])}/{len(names)} features: {spec['names'][:8]} ...")
    X = torch.from_numpy(F.apply_spec(M_all, spec))
    info = {"n_flows": {k: len(v["y"]) for k, v in doms.items()}, "n_engineered": len(names),
            "attack_share": {k: float(v["y"].mean()) for k, v in doms.items()},
            "client_sizes": [len(c) for c in clients], "client_domains": cdom}
    return Bundle(X, torch.from_numpy(y_all), clients, cdom, tests_local, spec, args.window, info)
