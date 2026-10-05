import glob, json, os, sys
import pandas as pd
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "results", "*", "summary.json"))):
    s = json.load(open(f)); a, fin = s["args"], s["final"]
    r = {"tag": a["tag"], "mode": a["mode"], "agg": a["agg"] if a["mode"] == "federated" else "-", "crypto": a["crypto"],
         "window": a["window"], "gamma": a["gamma"], "adv_eps": a.get("adv_eps", 0), "byz": a.get("byzantine", 0),
         "params": s["params"], "up_MB_total": round(s["total_up_MB"], 1), "train_s": round(s["time_train_s"], 1)}
    for k in ("acc", "prec", "rec", "f1", "fpr", "auc"):
        r[k] = round(fin["ALL"][k], 4)
    for d, v in fin.items():
        if d != "ALL": r[f"f1_{d}"] = round(v["f1"], 4)
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv("results/summary_table.csv", index=False); print(df.to_string(index=False))
