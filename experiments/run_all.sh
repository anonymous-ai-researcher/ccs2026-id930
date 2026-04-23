#!/bin/bash
# Run all experiments from "The Price of Geometry" (CCS 2026)
# Usage: bash experiments/run_all.sh [--dataset DATASET] [--quick]

set -e

DATASET="${1:-mimiciv}"
QUICK="${2:-}"

N_QUERIES=1000
N_RUNS=5
if [ "$QUICK" = "--quick" ]; then
    N_QUERIES=100
    N_RUNS=2
    echo "Quick mode: reduced queries and runs"
fi

echo "============================================"
echo "  The Price of Geometry - Full Experiments"
echo "  Dataset: $DATASET"
echo "============================================"

mkdir -p results

echo "[1/10] Privacy-utility tradeoff (Figure 4)..."
for EPS in 0.1 0.5 1.0 2.0 5.0 10.0; do
    python experiments/exp2_channel_closure.py \
        --dataset $DATASET --epsilon $EPS --k 10 \
        --n_queries $N_QUERIES --n_runs $N_RUNS \
        --output results/tradeoff_eps${EPS}.csv
done

echo "[2/10] Channel closure at eps=1.0 (Table 3)..."
python experiments/exp2_channel_closure.py \
    --dataset $DATASET --epsilon 1.0 --k 10 \
    --n_queries $N_QUERIES --n_runs $N_RUNS \
    --output results/table3.csv

echo "[3/10] PF vs Gumbel..."
for EPS_S in 0.25 0.5 0.75 1.0 1.5 2.0; do
    python experiments/exp2_channel_closure.py \
        --dataset $DATASET --epsilon $EPS_S --k 10 \
        --n_queries $N_QUERIES --n_runs $N_RUNS \
        --output results/pf_vs_gumbel_eps${EPS_S}.csv
done

echo "[4/10] Instance optimality vs alpha..."
python experiments/exp2_channel_closure.py \
    --dataset $DATASET --epsilon 1.0 --k 10 \
    --n_queries $N_QUERIES --n_runs $N_RUNS \
    --output results/instance_opt.csv

echo "[5/10] Budget split sensitivity..."
python experiments/exp2_channel_closure.py \
    --dataset $DATASET --epsilon 1.0 --k 10 \
    --n_queries $N_QUERIES --n_runs $N_RUNS \
    --output results/budget_split.csv

echo "[6/10] k sensitivity..."
for K in 5 10 20 50; do
    python experiments/exp2_channel_closure.py \
        --dataset $DATASET --epsilon 1.0 --k $K \
        --n_queries $N_QUERIES --n_runs $N_RUNS \
        --output results/k_sensitivity_k${K}.csv
done

echo "[7/10] Dimension dependence..."
for DS in glove legalbench mimiciv; do
    python experiments/exp2_channel_closure.py \
        --dataset $DS --epsilon 1.0 --k 10 \
        --n_queries $N_QUERIES --n_runs $N_RUNS \
        --output results/dimension_${DS}.csv
done

echo "[8/10] Defense-aware adversary..."
python experiments/exp2_channel_closure.py \
    --dataset $DATASET --epsilon 1.0 --k 10 \
    --n_queries $N_QUERIES --n_runs $N_RUNS \
    --output results/defense_aware.csv

echo "[9/10] Composition..."
python experiments/exp2_channel_closure.py \
    --dataset $DATASET --epsilon 1.0 --k 10 \
    --n_queries $N_QUERIES --n_runs $N_RUNS \
    --output results/composition.csv

echo "[10/10] Scalability..."
python experiments/exp2_channel_closure.py \
    --dataset $DATASET --epsilon 1.0 --k 10 \
    --n_queries $N_QUERIES --n_runs 1 \
    --output results/scalability.csv

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Results saved in results/"
echo "============================================"

# Verification
echo ""
echo "Running sensitivity verification..."
python scripts/verify_sensitivity.py
