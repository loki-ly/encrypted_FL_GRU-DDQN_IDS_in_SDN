"""Generate SMALL fake CICFlowMeter-style CSVs (native column names of each dataset) to smoke-test the
pipeline. Numbers obtained on this data are meaningless -- use the real InSDN / CICIDS2017 / CICDDoS2019."""
import argparse, os
import numpy as np, pandas as pd
from ids.schema import A, FEATURES

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="synthetic"); ap.add_argument("--n", type=int, default=8000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
sig = rng.random(len(FEATURES)) < 0.35                       # attack-informative features
cfg = {"insdn": (-1, "", ["Normal", "DDoS", "Probe"]), "cicids2017": (0, "", ["BENIGN", "DDoS", "PortScan"]),
       "cicddos2019": (0, " ", ["BENIGN", "DrDoS_UDP", "Syn"])}
for k, (ai, pad, labs) in cfg.items():
    n = a.n; shift = rng.uniform(0.7, 1.5, len(FEATURES))
    atk = np.zeros(n, int)
    for i in range(1, n):                                    # bursty attacks
        atk[i] = atk[i - 1] if rng.random() < 0.985 else int(rng.random() < 0.5)
    base = rng.lognormal(3, 1.2, (n, len(FEATURES))) * shift
    base[atk == 1] *= np.where(sig, rng.uniform(2.5, 8, len(FEATURES)), 1.0)
    df = pd.DataFrame(base, columns=[pad + (A[f][ai] if ai else A[f][0]) for f in FEATURES])
    df["Flow ID"] = "x"; df["Src IP"] = "10.0.0.1"
    df["Dst IP"] = np.where(atk == 1, "10.0.0.99", rng.choice([f"10.0.0.{i}" for i in range(2, 8)], n))
    df["Timestamp"] = pd.date_range("2025-01-01", periods=n, freq="s").astype(str)
    df["Label"] = np.where(atk == 1, np.where(rng.random(n) < .7, labs[1], labs[2]), labs[0])
    df.to_csv(f"{a.out}/{k}.csv", index=False); print(k, df.shape, atk.mean())
