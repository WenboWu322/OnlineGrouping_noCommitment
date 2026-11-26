#!/usr/bin/env bash
# Copy generated Chapter 6 assets into the thesis repository.
# Figures and tables in the thesis are always the output of the code in this
# repository; they are never edited by hand.
set -euo pipefail
cd "$(dirname "$0")"
THESIS=${THESIS:-../thesis-overleaf}

test -d "$THESIS/Thesis/figures" || { echo "thesis figures dir missing"; exit 1; }
mkdir -p "$THESIS/Thesis/tables"

if compgen -G "results/figures/fig_ch6_*.pdf" > /dev/null; then
  cp -v results/figures/fig_ch6_*.pdf "$THESIS/Thesis/figures/"
fi
# Chapter 2 figures: statistics of the real data extracts, never edited by hand
if compgen -G "results/figures/fig_data_*.pdf" > /dev/null; then
  cp -v results/figures/fig_data_*.pdf "$THESIS/Thesis/figures/"
fi
if [ -f results/industrial/industrial_data_summary.json ]; then
  cp -v results/industrial/industrial_data_summary.json "$THESIS/Thesis/tables/"
fi
if compgen -G "results/tables/*.tex" > /dev/null; then
  cp -v results/tables/*.tex "$THESIS/Thesis/tables/"
fi
cp -v results/main/summary.csv "$THESIS/Thesis/tables/ch6_summary_main.csv"
# rows.csv (per-seed rows, ~0.8 MB) stays in this repository; the thesis
# carries the summary so that every quoted median remains traceable
echo "assets synced to $THESIS"
