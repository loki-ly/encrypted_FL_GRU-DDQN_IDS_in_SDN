# FedProx-DDQN-GRU: encrypted federated intrusion *prevention* for SDN (P3 architecture)

Datasets: **InSDN**, **CICIDS2017**, **CICDDoS2019** (all CICFlowMeter-based -> one shared feature schema).

## 1. What changed from P2 -> P3

| P2 (NSL-KDD)                         | P3 (this code)                                                                 | Comes from |
|--------------------------------------|--------------------------------------------------------------------------------|------------|
| DQN, 2 actions (normal/attack)       | **Double DQN**, 3 actions: allow / rate-limit / block, asymmetric reward        | Paper 3 (Fed-DDQN, rate-limit prevention) |
| MLP 30-256-128-2, aggregate online+target | **1-layer GRU(64)** Q-net, ~21k params; only the online net is exchanged | Paper 1/2 (GRU) + efficiency |
| FedAvg, plaintext weights            | **FedProx** (mu=0.01) + **CKKS-encrypted deltas**, additions-only server       | Paper 3 (FedProx/non-IID) + Papers 1/2 (HE) |
| Random partition of one dataset      | **Clients = SDN domains** (InSDN / CICIDS / CICDDoS domains) => real cross-domain heterogeneity | Paper 3, panel point 1 |
| Per-record, no context               | **Windowed granularity** (last L flows, optionally per victim Dst-IP)           | panel point 1 |
| RF+SHAP+LIME on a pooled 1% sample   | **Feature engineering + federated XAI selection** (RF-Gini + SHAP, only importance vectors and aggregate stats leave clients) | panel point 2 |
| -                                    | HMAC + SHA-256 + nonce/timestamp on every update packet; L2 clipping            | Paper 1 |
| -                                    | OpenFHE **multiparty** keygen (no trusted dealer), collaborative decryption     | Paper 3 (DKG) |
| -                                    | PGD evasion test, FGSM adversarial training, poisoning-client simulation        | thesis title (adversarial) |
| -                                    | Gradient x input, SHAP GradientExplainer, LIME on the *trained policy*          | XAI |

## 2. Pipeline (rearranged architecture, panel point 3)

1. **Data layer (local to each domain)** - canonical schema -> clean/dedupe -> sort by (segment, Dst-IP, time) -> engineered features
   (ratios, flag density, header overhead, IAT coefficient of variation ...) + signed log1p.
2. **Leakage-safe split** - windows are grouped in blocks of 512; whole blocks go to train or test (no overlapping windows across the split).
3. **Federated feature selection** - each client sends `(n, sum, X^T X)` + a normalised RF/SHAP importance vector; the server builds the global scaler,
   correlation matrix, and picks the top-K uncorrelated features (`feature_spec.json` is the deployable artefact).
4. **Local DDQN-GRU learning** - class-balanced sampling, epsilon-greedy, index-only replay buffer, Double-DQN target, FedProx gradient term.
5. **Secure update** - `delta = w_local - w_global`, L2-clipped, pre-multiplied by `n_k/N`, encrypted (CKKS), signed (HMAC).
6. **Server** - verifies hash/tag/freshness/nonce, **adds ciphertexts only** (depth 0 -> smallest CKKS parameters), never sees plaintext.
7. **Collaborative decryption** -> new global model; target net re-synced locally.
8. **Offline XAI** on the converged policy; online inference = plain GRU forward pass (no crypto/XAI in the data path).

## 3. Efficiency choices (why it is cheaper than the papers / P2)

* Model ~21k params (paper 1/2: ~420k for GRU 256/128/64; P2: 41k x 2 networks exchanged) -> 4x-20x fewer numbers to encrypt.
* Server only does `EvalAdd` (weights folded into the client payload): no relinearisation/rescale, no multiplicative depth.
* Index-only replay, lazy window gathering (no L x F copy of the dataset), vectorised batch environment (no per-record Python loop).
* Local RF/SHAP on <=3000 rows per client; LIME only for a handful of post-hoc explanations.
* Measured per run and written to `history.csv`: `up_MB`, `t_train`, `t_enc`, `t_agg`, `t_dec`, `crypto_max_err`.

## 4. Install / run

```bash
pip install -r requirements.txt            # Python 3.10-3.12
# put the original CSVs under data/insdn, data/cicids2017, data/cicddos2019  (or edit paths in run_all.sh)

# smoke test without datasets (fake CICFlowMeter CSVs; numbers are meaningless):
python make_synthetic.py --n 6000
python run_experiment.py --data insdn=synthetic/insdn.csv cicids2017=synthetic/cicids2017.csv \
       cicddos2019=synthetic/cicddos2019.csv --rounds 3 --local_steps 30 --crypto tenseal --explain --adv_eval

# real single dataset:
python run_experiment.py --data insdn="data/insdn/*.csv" --tag insdn_fedprox --crypto tenseal --explain
# full study (17 configurations incl. ablations) + summary table:
./run_all.sh
```
Outputs per run in `results/<tag>/`: `summary.json`, `history.csv` (per round), `feature_spec.json`, `ddqn_gru.pt`,
`explain_global.csv`, `explain_lime_example.csv`, `adv_pgd.csv`. `results/summary_table.csv` merges everything.

Key flags: `--mode federated|centralized|local`, `--agg fedavg|fedprox --mu`, `--crypto none|tenseal|openfhe`,
`--window L` (1 = flow-level), `--group_by dst_ip`, `--partition domain|iid|dirichlet`, `--actions 2|3`, `--gamma`,
`--adv_eps`, `--byzantine k`, `--clip`. CICDDoS2019 is huge: use `--sample_frac cicddos2019=0.01` (random contiguous blocks are kept so windows stay temporal).

## 5. Answers to the panel's three comments

1. **Need / granularity of FL** - `1b_local_only` (no collaboration) vs `2/3` (FL) vs `1_centralized` (upper bound) quantifies *why* FL is needed
   (each domain sees only its own attacks). Granularity ablation: `8_flow_level` (L=1) vs `L=8` vs `L=16` vs `10_window8_dstip` (per-victim).
2. **Feature engineering** - engineered ratio/rate features + federated SHAP/RF selection with correlation pruning; `explain_global.csv` shows what the policy uses.
3. **Architecture rearrangement** - data->features->federated-selection->DDQN-GRU->encrypted FedProx->offline XAI/adversarial checks (section 2).

## 6. Honest notes / limits (please read before the defense)

* I had **no access to the real datasets** in my sandbox and no P2 code files were uploaded (I rebuilt from the thesis text). The pipeline was verified end-to-end
  only on synthetic CSVs. **Do not quote any number from the synthetic run**; your accuracy comes from running on the real data.
* "CICIDS2019" in your message was read as **CICIDS2017 + CICDDoS2019**. Column-name aliasing covers the standard CICFlowMeter CSVs; the loader prints how many
  features are common. Check that line on your CSVs (files with a different export format may need extra aliases in `ids/schema.py`).
* Classification is a one-step decision problem: the chosen action does not change the next flow. With `gamma>0` the DDQN target only adds an
  action-independent term, so `gamma=0` is a legitimate ablation (`11_gamma0`). The RL justification is the cost-sensitive **3-action prevention**
  and reward-driven adaptation, not long-horizon planning. Metrics count `rate-limit` and `block` as "attack". The +0.5 / -0.5 rate-limit rewards are a design choice - tune/justify them.
* OpenFHE backend = native **n-of-n multiparty** CKKS (no dealer). Paper 3's **t-of-n Shamir layer is custom C++** and is not reproduced. TenSEAL backend uses a trusted key setup like Paper 1.
* With encrypted aggregation the server cannot inspect updates: poisoning is limited by client-side clipping + authentication, **not detected**. `--byzantine` shows the effect.
* Feature-selection statistics (aggregate sums, importance vectors) are sent in plaintext; they can be passed through the same secure aggregation if required.
* PGD/FGSM operate in standardised feature space without protocol constraints (upper bound on attacker power). Domain-constrained perturbations are future work.
* No GAN oversampling: imbalance handled by class-balanced sampling + asymmetric reward; test sets keep the natural class ratio. Attack-type-specific / zero-day evaluation (hold out one attack family) is not included.
* Client training is simulated sequentially in one process; communication sizes are real serialized ciphertext sizes, latency is not network latency.
