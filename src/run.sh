#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Mode: data exploration (prints report to stdout & train.log) ----
# Uncomment the block below to run data explorer on the platform dataset.
# The platform sets TRAIN_DATA_PATH / TRAIN_LOG_PATH via env.
# python3 -u "${SCRIPT_DIR}/data_explorer.py" \
#     "$@"

# ---- Mode: training (default) ----
# RankMixer NS tokenizer + BCE + InfoNCE(uniformity) + LongerEncoder
# Scheduler: 1-epoch warmup (~4k steps, ~10%) + cosine over 10 epochs (~40k steps)
# LongerEncoder: top-K cross-attention compresses sequences to 256 tokens,
# allowing 4x longer input sequences vs baseline without O(n²) explosion.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --seq_encoder_type longer \
    --seq_top_k 256 \
    --seq_max_lens seq_a:1024,seq_b:1024,seq_c:1024,seq_d:2048 \
    --warmup_epochs 1 \
    --cosine_t_max_epochs 10 \
    --min_lr_ratio 0.1 \
    --dropout_rate 0.05 \
    --dense_weight_decay 0.01 \
    --use_rope \
    --loss_type bce+info \
    --bce_weight 1.0 \
    --info_weight 0.05 \
    --info_tau 0.15 \
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
