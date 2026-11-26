#!/usr/bin/env bash
# Chapter 6 experiment suite.  Every number printed in the chapter comes from
# the CSV/JSON files this script writes; nothing is entered by hand.
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-8}
SEEDS=${SEEDS:-30}
SWEEP_SEEDS=${SWEEP_SEEDS:-20}

echo "== main comparison: 4 variants x $SEEDS seeds x all policies"
python3 -m online_grouping.experiment --seeds "$SEEDS" --jobs "$JOBS" --out results_repro/main

# Sensitivity axes.  Variant D (high density, clustered) is the setting where
# consolidation matters; A is used where the effect is about information only.
declare -A SWEEPS=(
  [noise_A]='--variants A_low_uniform --set policy.sigma_meas=0.020'
  [noise_C]='--variants C_high_uniform --set policy.sigma_meas=0.020'
  [clean_A]='--variants A_low_uniform --set policy.sigma_meas=0.001 policy.p_missing=0.0 policy.quantisation=0.0'
  [missing_A]='--variants A_low_uniform --set policy.p_missing=0.20'
  [eps01_A]='--variants A_low_uniform --set scenario.epsilon=0.01'
  [eps10_A]='--variants A_low_uniform --set scenario.epsilon=0.10'
  [eps25_A]='--variants A_low_uniform --set scenario.epsilon=0.25'
  [shock_C]='--variants C_high_uniform --set policy.shock_rate=0.005'
  [shock_D]='--variants D_high_cluster --set policy.shock_rate=0.005'
  [delay3_D]='--variants D_high_cluster --set scenario.delta_delay=3'
  [delay10_D]='--variants D_high_cluster --set scenario.delta_delay=10'
  [delay20_D]='--variants D_high_cluster --set scenario.delta_delay=20'
  [adv6_D]='--variants D_high_cluster --set scenario.delta_adv=6'
  [win20_D]='--variants D_high_cluster --set scenario.L_win=20'
  [cap5_D]='--variants D_high_cluster --set scenario.K_clus=5'
  [nolock_S5]='--variants D_high_cluster --set policy.lock_days=0 policy.eta_stab=0.0'
  [cap3_D]='--variants D_high_cluster --set scenario.K_clus=3'
  [jitter8_D]='--variants D_high_cluster --set scenario.cluster_jitter=8.0'
  [jitter15_D]='--variants D_high_cluster --set scenario.cluster_jitter=15.0'
  [eta20_D]='--variants D_high_cluster --set scenario.eta_plus=20.0'
)

for name in "${!SWEEPS[@]}"; do
  echo "== sweep $name"
  # shellcheck disable=SC2086
  python3 -m online_grouping.experiment ${SWEEPS[$name]} --seeds "$SWEEP_SEEDS" \
      --jobs "$JOBS" --out "results_repro/sweep_$name"
done
echo "== trace for the timeline figure"
python3 -m online_grouping.experiment --dump-trace --variant D_high_cluster \
    --seed 3 --policy S5_proposed --out results_repro/trace_run
cp results_repro/trace_run/trace.csv results_repro/trace.csv

echo "== all sweeps written under results_repro/"
