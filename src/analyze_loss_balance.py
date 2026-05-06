#!/usr/bin/env python3
"""Diagnose BCE vs Ranking loss balance for combined_pair.

Runs a few batches through the model and reports:
1. Raw loss magnitudes (bce_loss, rank_loss)
2. Gradient norms when each loss term is back-propagated individually
3. Recommended bce_weight to balance either loss values or gradient norms.

Usage:
    python analyze_loss_balance.py
"""

import os
import sys
import json
import glob
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add src/ to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


# ────────────────────────────── Config ─────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "datasets"
SCHEMA_PATH = PROJECT_ROOT / "schema.json"
BATCH_SIZE = 512
NUM_WORKERS = 4
NUM_BATCHES = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model args (must match run.sh active config)
MODEL_KWARGS = dict(
    d_model=64,
    emb_dim=64,
    num_queries=2,
    num_hyformer_blocks=2,
    num_heads=4,
    seq_encoder_type="transformer",
    hidden_mult=4,
    dropout_rate=0.01,
    seq_top_k=50,
    seq_causal=False,
    action_num=1,
    num_time_buckets=NUM_TIME_BUCKETS,
    rank_mixer_mode="full",
    use_rope=False,
    rope_base=10000.0,
    emb_skip_threshold=1_000_000,
    seq_id_threshold=10000,
    ns_tokenizer_type="rankmixer",
    user_ns_tokens=5,
    item_ns_tokens=2,
    use_target_attention=True,
)
RANK_MARGIN = 1.0


# ────────────────────────────── Helpers ────────────────────────────────────


def build_feature_specs(schema: FeatureSchema, per_position_vocab_sizes: list):
    specs = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def collect_loss_stats(model, data_loader, num_batches: int):
    """Collect per-batch bce_loss, rank_loss, and individual gradient norms."""
    bce_vals = []
    rank_vals = []
    bce_grad_norms = []
    rank_grad_norms = []

    model.train()
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= num_batches:
            break

        # Move to device
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(DEVICE, non_blocking=True)
            else:
                device_batch[k] = v

        label = device_batch["label"].float()

        # Build model input
        seq_domains = device_batch["_seq_domains"]
        seq_data = {d: device_batch[d] for d in seq_domains}
        seq_lens = {d: device_batch[f"{d}_len"] for d in seq_domains}
        seq_time_buckets = {}
        for d in seq_domains:
            B, L = device_batch[d].shape[0], device_batch[d].shape[2]
            seq_time_buckets[d] = device_batch.get(
                f"{d}_time_bucket",
                torch.zeros(B, L, dtype=torch.long, device=DEVICE),
            )

        model_input = ModelInput(
            user_int_feats=device_batch["user_int_feats"],
            item_int_feats=device_batch["item_int_feats"],
            user_dense_feats=device_batch["user_dense_feats"],
            item_dense_feats=device_batch["item_dense_feats"],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
        )

        # Forward
        logits = model(model_input).squeeze(-1)

        # ---- BCE loss ----
        bce_loss = F.binary_cross_entropy_with_logits(logits, label)

        # Backward BCE alone and record grad norm
        model.zero_grad()
        bce_loss.backward(retain_graph=True)
        bce_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                bce_norm += p.grad.data.norm(2).item() ** 2
        bce_norm = bce_norm ** 0.5

        # ---- Rank loss ----
        pos_mask = label == 1
        neg_mask = label == 0
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_logits = logits[pos_mask]
            neg_logits = logits[neg_mask]
            diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
            rank_loss = F.relu(RANK_MARGIN - diff).mean()
        else:
            rank_loss = torch.tensor(0.0, device=logits.device)

        # Backward rank alone and record grad norm
        if rank_loss.item() > 0:
            model.zero_grad()
            rank_loss.backward()
            rank_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    rank_norm += p.grad.data.norm(2).item() ** 2
            rank_norm = rank_norm ** 0.5
        else:
            rank_norm = 0.0

        bce_vals.append(bce_loss.item())
        rank_vals.append(rank_loss.item())
        bce_grad_norms.append(bce_norm)
        rank_grad_norms.append(rank_norm)

    return {
        "bce_vals": np.array(bce_vals),
        "rank_vals": np.array(rank_vals),
        "bce_grad_norms": np.array(bce_grad_norms),
        "rank_grad_norms": np.array(rank_grad_norms),
    }


# ────────────────────────────── Main ───────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema.json not found at {SCHEMA_PATH}")

    # Load a single data file directly to avoid IterableDataset complexity
    pq_files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    if not pq_files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR}")

    logging.info("Loading data ...")
    train_loader, _, dataset = get_pcvr_data(
        data_dir=str(DATA_DIR),
        schema_path=str(SCHEMA_PATH),
        batch_size=BATCH_SIZE,
        valid_ratio=0.1,
        train_ratio=1.0,
        num_workers=NUM_WORKERS,
        buffer_batches=10,
        seed=42,
    )

    # Build model
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=[[i] for i in range(len(dataset.user_int_schema.entries))],
        item_ns_groups=[[i] for i in range(len(dataset.item_int_schema.entries))],
        **MODEL_KWARGS,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model loaded: {total_params:,} params, device={DEVICE}")

    # Collect stats
    logging.info(f"Running {NUM_BATCHES} batches for diagnosis ...")
    stats = collect_loss_stats(model, train_loader, NUM_BATCHES)

    bce_vals = stats["bce_vals"]
    rank_vals = stats["rank_vals"]
    bce_gn = stats["bce_grad_norms"]
    rank_gn = stats["rank_grad_norms"]

    # Filter out zero-rank batches (no pos/neg pairs)
    valid_mask = rank_vals > 0
    n_valid = int(valid_mask.sum())

    print("\n" + "=" * 70)
    print("Loss Balance Diagnosis Report")
    print("=" * 70)

    print(f"\nBatches with pos+neg pairs: {n_valid}/{NUM_BATCHES}")
    print(f"BCE loss  -> mean={bce_vals.mean():.6f}, std={bce_vals.std():.6f}")
    print(f"Rank loss -> mean={rank_vals.mean():.6f}, std={rank_vals.std():.6f}")
    print(f"Ratio (rank/bce) -> mean={np.divide(rank_vals, bce_vals, where=bce_vals!=0).mean():.4f}")

    print(f"\nBCE grad_norm  -> mean={bce_gn.mean():.4f}, std={bce_gn.std():.4f}")
    print(f"Rank grad_norm -> mean={rank_gn.mean():.4f}, std={rank_gn.std():.4f}")

    # Recommendations
    print("\n" + "-" * 70)
    print("Recommended bce_weight")
    print("-" * 70)

    # 1. Balance loss values: w * bce = (1-w) * rank => w = rank / (bce + rank)
    avg_bce = bce_vals.mean()
    avg_rank = rank_vals.mean()
    if avg_bce + avg_rank > 0:
        w_by_value = avg_rank / (avg_bce + avg_rank)
        print(f"  [By loss value]   w = rank/(bce+rank) = {w_by_value:.4f}")
    else:
        w_by_value = None

    # 2. Balance gradient norms: w * gn_bce = (1-w) * gn_rank => w = gn_rank / (gn_bce + gn_rank)
    avg_gn_bce = bce_gn.mean()
    avg_gn_rank = rank_gn.mean()
    if avg_gn_bce + avg_gn_rank > 0:
        w_by_grad = avg_gn_rank / (avg_gn_bce + avg_gn_rank)
        print(f"  [By grad norm]    w = gn_rank/(gn_bce+gn_rank) = {w_by_grad:.4f}")
    else:
        w_by_grad = None

    # 3. Fallback: if rank loss is very small, weight BCE more heavily
    if avg_rank < 0.01 * avg_bce:
        print(f"\n  WARNING: rank_loss << bce_loss ({avg_rank:.6f} vs {avg_bce:.6f})")
        print(f"  Consider increasing rank_margin or using a smaller bce_weight.")

    # Practical recommendation (blend both signals)
    if w_by_value is not None and w_by_grad is not None:
        w_blend = 0.5 * w_by_value + 0.5 * w_by_grad
        print(f"\n  [Blended]         w = {w_blend:.4f}  <-- RECOMMENDED")
    elif w_by_value is not None:
        print(f"\n  [Fallback]        w = {w_by_value:.4f}  <-- RECOMMENDED")
    else:
        print(f"\n  [Fallback]        w = 0.7  <-- DEFAULT")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
