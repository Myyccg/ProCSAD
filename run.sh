#!/bin/bash
# ============================================================
#  ProCSAD — Run experiments on all benchmark datasets
#  Usage:
#    bash run.sh              # run all 7 datasets
#    bash run.sh MSL SMAP     # run selected datasets only
# ============================================================

set -e

# Auto-detect Python with PyTorch
if command -v python &> /dev/null && python -c "import torch" &> /dev/null 2>&1; then
    PYTHON="python"
elif command -v python3 &> /dev/null && python3 -c "import torch" &> /dev/null 2>&1; then
    PYTHON="python3"
else
    echo "[ERROR] Python with PyTorch not found. Please install dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

GPU=${CUDA_VISIBLE_DEVICES:-0}
RUN_TIMES=5
EPOCHS=20
BATCH=64
WIN=100
MODEL_VER="config2_bimamba_iencoder"
SAVE_DIR="checkpoints"

mkdir -p ${SAVE_DIR}

# ---- dataset config: NAME  DIM  RATIO  DATA_PATH ----
declare -A DIM=(
    [MSL]=55   [SMAP]=25   [PSM]=25   [SWaT]=51
    [SMD]=38   [NIPS_TS_Water]=9   [NIPS_TS_Swan]=38
)
declare -A RATIO=(
    [MSL]=1.0  [SMAP]=1.0  [PSM]=1.0  [SWaT]=0.1
    [SMD]=0.5  [NIPS_TS_Water]=1.0  [NIPS_TS_Swan]=1.0
)
declare -A DPATH=(
    [MSL]=./data/MSL/         [SMAP]=./data/SMAP/
    [PSM]=./data/PSM/         [SWaT]=./data/SWaT/
    [SMD]=./data/SMD/         [NIPS_TS_Water]=./data/NIPS_TS_Water/
    [NIPS_TS_Swan]=./data/NIPS_TS_Swan/
)

ALL_DATASETS=(MSL SMAP PSM SWaT SMD NIPS_TS_Water NIPS_TS_Swan)

# If arguments given, run only those datasets
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

TOTAL=${#DATASETS[@]}
IDX=0

for DS in "${DATASETS[@]}"; do
    IDX=$((IDX + 1))
    D=${DIM[$DS]}
    R=${RATIO[$DS]}
    P=${DPATH[$DS]}

    if [ -z "$D" ]; then
        echo "[WARN] Unknown dataset: $DS — skipping"
        continue
    fi

    echo ""
    echo "=========================================="
    echo " [${IDX}/${TOTAL}] Dataset: ${DS}  |  dim=${D}  |  ratio=${R}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} main.py \
        --dataset ${DS} \
        --data_path ${P} \
        --input_c ${D} --output_c ${D} --d_model ${D} --svdd_proj_dim ${D} \
        --model_version ${MODEL_VER} \
        --use_our_embedding True --pure_mamba True \
        --use_wavelet_branch2 False \
        --use_svdd True --use_memory True \
        --use_contrastive True --use_epsilon_tube True \
        --run_times ${RUN_TIMES} --num_epochs ${EPOCHS} \
        --batch_size ${BATCH} --win_size ${WIN} \
        --anomaly_ratio ${R}

    echo "[✓] ${DS} done."
done

echo ""
echo "=========================================="
echo " All experiments completed (${TOTAL} datasets)"
echo " Results saved in ${SAVE_DIR}/"
echo "=========================================="
