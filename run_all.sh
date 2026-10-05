#!/usr/bin/env bash
# Full P3 study.  Edit the three paths below (folders containing the original CSV files).
set -e
INSDN="data/insdn/*.csv"                 # Normal_data.csv, OVS.csv, metasploitable-2.csv
CICIDS="data/cicids2017/*.csv"           # the 8 daily CSVs
CICDDOS="data/cicddos2019/*.csv"         # 01-12 / 03-11 CSVs (huge -> sampled)
FRAC="insdn=1.0,cicids2017=0.2,cicddos2019=0.01"     # block-sampling fraction per domain (memory control)
COMMON="--data insdn=$INSDN cicids2017=$CICIDS cicddos2019=$CICDDOS --sample_frac $FRAC --rounds 15 --local_steps 100 --explain"

python3 run_experiment.py $COMMON --tag 1_centralized        --mode centralized
python3 run_experiment.py $COMMON --tag 1b_local_only        --mode local   # 'do we need FL?' baseline
python3 run_experiment.py $COMMON --tag 2_fedavg_plain       --agg fedavg
python3 run_experiment.py $COMMON --tag 3_fedprox_plain      --agg fedprox
python3 run_experiment.py $COMMON --tag 4_fedprox_tenseal    --agg fedprox --crypto tenseal
python3 run_experiment.py $COMMON --tag 5_fedprox_openfhe    --agg fedprox --crypto openfhe
# non-IID stress test (label skew over pooled domains)
python3 run_experiment.py $COMMON --tag 6_dirichlet_fedavg   --partition dirichlet --clients 6 --agg fedavg
python3 run_experiment.py $COMMON --tag 7_dirichlet_fedprox  --partition dirichlet --clients 6 --agg fedprox
# granularity ablation (flow-level vs windows vs per-victim windows)
python3 run_experiment.py $COMMON --tag 8_flow_level         --window 1
python3 run_experiment.py $COMMON --tag 9_window16           --window 16
python3 run_experiment.py $COMMON --tag 10_window8_dstip     --group_by dst_ip
# RL ablation
python3 run_experiment.py $COMMON --tag 11_gamma0            --gamma 0.0
python3 run_experiment.py $COMMON --tag 12_binary_actions    --actions 2
# adversarial robustness + poisoning
python3 run_experiment.py $COMMON --tag 13_advtrain_pgd      --adv_eps 0.1 --adv_eval
python3 run_experiment.py $COMMON --tag 14_noadv_pgd         --adv_eval
python3 run_experiment.py $COMMON --tag 15_poison_clip       --byzantine 1 --clip 5
python3 run_experiment.py $COMMON --tag 16_poison_noclip     --byzantine 1 --clip 1e9
python3 summarize.py results
