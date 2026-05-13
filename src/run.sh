#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Mode: data exploration (prints report to stdout & train.log) ----
# Uncomment the block below to run data explorer on the platform dataset.
# The platform sets TRAIN_DATA_PATH / TRAIN_LOG_PATH via env.
# python3 -u "${SCRIPT_DIR}/data_explorer.py" \
#     "$@"

# ---- Experiment C: Focal Loss ----
# 在 baseline 上只换 loss，不改模型结构。
# 正负比 1:9，Focal Loss 自动降权简单负样本。
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --no_scheduler \
    --loss_type focal \
    --focal_alpha 0.75 \
    --focal_gamma 2.0 \
    "$@"

# ---- Legacy config: BCE + Pairwise Hinge ----
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --use_target_attention \
#     --no_scheduler \
#     --loss_type bce+pair \
#     --bce_weight 0.5 \
#     --pair_weight 0.5 \
#     --rank_margin 1.0 \
#     "$@"

# ---- Baseline config: BCE only ----
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --use_target_attention \
#     --no_scheduler \
#     "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
