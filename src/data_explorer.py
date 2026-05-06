#!/usr/bin/env python3
"""PCVR Data Explorer — Standalone data profiling tool.

Reads Parquet training data and produces a comprehensive statistical report
without touching any training code. Can run locally on demo data or (when
available) on full-scale training data.

Usage:
    python scripts/data_explorer.py --data_path datasets/demo_1000.parquet --output report.md
    python scripts/data_explorer.py --data_path /data_ams/academic_training_data --output report.md
"""

import os
import sys
import argparse
import glob
import json
import logging
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from utils import create_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVR Data Explorer")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to Parquet file or directory containing *.parquet. "
                             "If omitted, uses TRAIN_DATA_PATH env var or falls back to demo data.")
    parser.add_argument("--schema_path", type=str, default=None,
                        help="Optional schema.json path")
    parser.add_argument("--output", type=str, default="-",
                        help="Output destination: '-' for stdout (default), or a file path")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory. If omitted, uses TRAIN_LOG_PATH env var.")
    parser.add_argument("--max_sample_rows", type=int, default=10000,
                        help="Maximum rows to sample for profiling. "
                             "Default 10000 avoids OOM on full-scale data.")
    parser.add_argument("--max_sample_files", type=int, default=50,
                        help="Maximum Parquet files to read. "
                             "Default 50 avoids opening too many files.")

    args = parser.parse_args()

    # Align with train.py: environment variables take precedence.
    # Priority: CLI > env var > local fallback
    args.data_path = args.data_path or os.environ.get('TRAIN_DATA_PATH')
    if not args.data_path:
        # Local fallback: relative to current working directory (no ..)
        args.data_path = "datasets/demo_1000.parquet"

    args.log_dir = args.log_dir or os.environ.get('TRAIN_LOG_PATH') or "."

    return args


def discover_parquet_files(data_path: str) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path]
    files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {data_path}")
    return files


def collect_basic_info(files: List[str]) -> Dict[str, Any]:
    total_rows = 0
    total_row_groups = 0
    schema = None
    for f in files:
        pf = pq.ParquetFile(f)
        total_rows += pf.metadata.num_rows
        total_row_groups += pf.metadata.num_row_groups
        if schema is None:
            schema = pf.schema_arrow
    return {
        "num_files": len(files),
        "total_rows": total_rows,
        "total_row_groups": total_row_groups,
        "num_columns": len(schema.names) if schema else 0,
        "column_names": schema.names if schema else [],
    }


def sample_table(files: List[str], max_rows: int, max_files: int = 50) -> "pa.Table":
    """Read up to max_rows across at most max_files files.

    Randomly shuffles files before sampling to avoid temporal bias
    (e.g. only reading the earliest files).
    """
    import pyarrow as pa
    import random

    # Shuffle files for unbiased sampling
    shuffled = files.copy()
    random.shuffle(shuffled)
    candidate_files = shuffled[:max_files] if max_files > 0 else shuffled

    tables = []
    rows_so_far = 0
    for f in candidate_files:
        if max_rows > 0 and rows_so_far >= max_rows:
            break
        pf = pq.ParquetFile(f)
        for rg_idx in range(pf.metadata.num_row_groups):
            if max_rows > 0 and rows_so_far >= max_rows:
                break
            batch = pf.read_row_group(rg_idx)
            tables.append(batch)
            rows_so_far += len(batch)
    if not tables:
        raise ValueError("No data read")
    combined = pa.concat_tables(tables)
    if max_rows > 0 and len(combined) > max_rows:
        combined = combined.slice(0, max_rows)
    return combined


def analyze_label(table) -> Dict[str, Any]:
    if "label_type" not in table.schema.names:
        return {"available": False}
    label_type = table.column("label_type").to_pylist()
    counter = Counter(label_type)
    total = len(label_type)
    # Map: assuming label_type==2 is positive (conversion)
    pos = counter.get(2, 0)
    neg = total - pos
    # Also report all distinct label types
    dist = {int(k): v for k, v in sorted(counter.items())}
    return {
        "available": True,
        "total": total,
        "positive": pos,
        "negative": neg,
        "positive_ratio": pos / total if total else 0,
        "label_type_distribution": dist,
    }


def analyze_timestamp(table) -> Dict[str, Any]:
    if "timestamp" not in table.schema.names:
        return {"available": False}
    ts = table.column("timestamp").to_numpy()
    return {
        "available": True,
        "min": int(ts.min()),
        "max": int(ts.max()),
        "span_seconds": int(ts.max() - ts.min()),
        "span_hours": (ts.max() - ts.min()) / 3600.0,
        "span_days": (ts.max() - ts.min()) / 86400.0,
        "median": int(np.median(ts)),
        "p01": int(np.percentile(ts, 1)),
        "p99": int(np.percentile(ts, 99)),
    }


def analyze_user_item(table) -> Dict[str, Any]:
    result = {}
    for col in ["user_id", "item_id"]:
        if col not in table.schema.names:
            result[col] = {"available": False}
            continue
        arr = table.column(col).to_numpy()
        unique = len(np.unique(arr))
        total = len(arr)
        # Frequency distribution (how many times each ID appears)
        counts = Counter(arr.tolist())
        freq_values = list(counts.values())
        result[col] = {
            "available": True,
            "total_records": total,
            "unique_ids": unique,
            "repeat_ratio": 1.0 - unique / total if total else 0.0,
            "max_freq": max(freq_values) if freq_values else 0,
            "median_freq": float(np.median(freq_values)) if freq_values else 0.0,
            "p90_freq": float(np.percentile(freq_values, 90)) if freq_values else 0.0,
            "p99_freq": float(np.percentile(freq_values, 99)) if freq_values else 0.0,
        }
    return result


def analyze_sequences(table) -> Dict[str, Any]:
    """Analyze all columns matching *_seq_* pattern."""
    seq_cols = [n for n in table.schema.names if "_seq_" in n]
    # Group by domain prefix (e.g., domain_a_seq_38 -> domain_a)
    domains: Dict[str, List[str]] = {}
    for col in seq_cols:
        # Extract prefix like "domain_a"
        parts = col.split("_seq_")
        if len(parts) >= 2:
            prefix = parts[0]
            domains.setdefault(prefix, []).append(col)

    result = {}
    for domain, cols in sorted(domains.items()):
        all_lens = []
        empty_count = 0
        total_entries = 0
        for col in cols:
            col_data = table.column(col)
            for i in range(len(col_data)):
                val = col_data[i].as_py()
                total_entries += 1
                if val:
                    all_lens.append(len(val))
                else:
                    all_lens.append(0)
                    empty_count += 1

        arr = np.array(all_lens)
        result[domain] = {
            "num_features": len(cols),
            "total_entries": total_entries,
            "empty_entries": empty_count,
            "empty_ratio": empty_count / total_entries if total_entries else 0.0,
            "mean_len": float(arr.mean()),
            "median_len": float(np.median(arr)),
            "p80_len": float(np.percentile(arr, 80)),
            "p90_len": float(np.percentile(arr, 90)),
            "p95_len": float(np.percentile(arr, 95)),
            "p99_len": float(np.percentile(arr, 99)),
            "max_len": int(arr.max()),
        }
    return result


def analyze_dense_features(table) -> Dict[str, Any]:
    """Simple stats for dense (float list) features."""
    dense_cols = [n for n in table.schema.names if n.startswith("user_dense_feats_") or n.startswith("item_dense_feats_")]
    result = {}
    for col in dense_cols:
        col_data = table.column(col)
        lens = []
        for i in range(len(col_data)):
            val = col_data[i].as_py()
            lens.append(len(val) if val else 0)
        arr = np.array(lens)
        result[col] = {
            "mean_dim": float(arr.mean()),
            "max_dim": int(arr.max()),
            "min_dim": int(arr.min()),
            "empty_ratio": float((arr == 0).mean()),
        }
    return result


def analyze_int_features(table) -> Dict[str, Any]:
    """Cardinality estimate for int features."""
    import pyarrow as pa
    int_cols = [n for n in table.schema.names if "_int_feats_" in n]
    result = {}
    for col in int_cols:
        col_data = table.column(col)
        # Check if it's a list-type column (ChunkedArray of ListArray)
        is_list = pa.types.is_list(col_data.type) or pa.types.is_large_list(col_data.type)
        if is_list:
            # Manually flatten all sublists
            flat = []
            for i in range(len(col_data)):
                val = col_data[i].as_py()
                if val is not None:
                    if isinstance(val, (list, tuple)):
                        flat.extend([v for v in val if v is not None])
                    else:
                        flat.append(val)
            values = np.array(flat, dtype=np.int64)
        else:
            values = col_data.to_numpy()
            if values.dtype.kind == 'f':
                values = values[~np.isnan(values)].astype(np.int64)
            elif values.dtype == object:
                values = np.array([v for v in values if v is not None], dtype=np.int64)
        if len(values) == 0:
            result[col] = {"unique_values": 0, "min": "N/A", "max": "N/A"}
            continue
        unique = len(np.unique(values))
        result[col] = {
            "unique_values": unique,
            "min": int(values.min()),
            "max": int(values.max()),
        }
    return result


def generate_report(args, basic, label, timestamp, user_item, sequences, dense, int_feats) -> str:
    lines = []
    lines.append("# PCVR Data Explorer Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Data Path**: `{args.data_path}`")
    lines.append(f"**Sampled Rows**: {args.max_sample_rows if args.max_sample_rows > 0 else 'ALL ({} rows)'.format(basic['total_rows'])}")
    lines.append("\n---\n")

    # Basic Info
    lines.append("## 1. Basic Information")
    lines.append(f"- **Parquet Files**: {basic['num_files']}")
    lines.append(f"- **Total Rows**: {basic['total_rows']:,}")
    lines.append(f"- **Total Row Groups**: {basic['total_row_groups']}")
    lines.append(f"- **Columns**: {basic['num_columns']}")
    lines.append("\n---\n")

    # Label Analysis
    lines.append("## 2. Label Distribution (CVR Task)")
    if label["available"]:
        lines.append(f"- **Total Samples**: {label['total']:,}")
        lines.append(f"- **Positive (label_type=2)**: {label['positive']:,} ({label['positive_ratio']*100:.2f}%)")
        lines.append(f"- **Negative**: {label['negative']:,} ({(1-label['positive_ratio'])*100:.2f}%)")
        lines.append(f"- **All label_type values**: {label['label_type_distribution']}")
        if label['positive_ratio'] < 0.05:
            lines.append(
                "\n> ⚠️ **Imbalanced**: Positive ratio is very low. "
                "Consider Focal Loss or weighted sampling."
            )
        elif label['positive_ratio'] > 0.1:
            lines.append(
                "\n> ℹ️ **Note**: Positive ratio is relatively high (>10%). "
                "This may be demo data or up-sampled. Real CVR is usually <1%."
            )
    else:
        lines.append("- `label_type` column not found.")
    lines.append("\n---\n")

    # Timestamp Analysis
    lines.append("## 3. Timestamp Analysis")
    if timestamp["available"]:
        lines.append(f"- **Min**: {timestamp['min']} ({datetime.fromtimestamp(timestamp['min']).strftime('%Y-%m-%d %H:%M:%S')})")
        lines.append(f"- **Max**: {timestamp['max']} ({datetime.fromtimestamp(timestamp['max']).strftime('%Y-%m-%d %H:%M:%S')})")
        lines.append(f"- **Span**: {timestamp['span_seconds']:,} sec = {timestamp['span_hours']:.1f} hrs = {timestamp['span_days']:.1f} days")
        if timestamp['span_days'] < 1.0:
            lines.append(
                "\n> ⚠️ **Warning**: Time span is very short (<1 day). "
                "If this is demo data, temporal features may not be representative."
            )
        else:
            lines.append(
                "\n> ✅ Time span looks reasonable for temporal feature engineering."
            )
    else:
        lines.append("- `timestamp` column not found.")
    lines.append("\n---\n")

    # User / Item Analysis
    lines.append("## 4. User & Item Analysis")
    for entity in ["user_id", "item_id"]:
        info = user_item.get(entity, {})
        if not info.get("available"):
            lines.append(f"- `{entity}` not found.")
            continue
        name = entity.replace("_id", "").title()
        lines.append(f"### {name}")
        lines.append(f"- **Total Records**: {info['total_records']:,}")
        lines.append(f"- **Unique IDs**: {info['unique_ids']:,}")
        lines.append(f"- **Repeat Ratio**: {info['repeat_ratio']*100:.1f}%")
        lines.append(f"- **Max Frequency**: {info['max_freq']}")
        lines.append(f"- **Median Frequency**: {info['median_freq']:.1f}")
        lines.append(f"- **P90 Frequency**: {info['p90_freq']:.1f}")
        lines.append(f"- **P99 Frequency**: {info['p99_freq']:.1f}")
        if info['repeat_ratio'] < 0.1:
            lines.append(
                f"\n> ⚠️ **Low repetition**: Most {name}s appear only once. "
                f"ID-based Embedding may not learn meaningful representations. "
                f"Consider target statistics or side-info features."
            )
        else:
            lines.append(
                f"\n> ✅ {name} repetition is healthy for Embedding learning."
            )
    lines.append("\n---\n")

    # Sequence Analysis
    lines.append("## 5. Sequence Feature Analysis")
    if sequences:
        lines.append("| Domain | Features | Mean | Median | P80 | P90 | P95 | P99 | Max | Empty% |")
        lines.append("|--------|----------|------|--------|-----|-----|-----|-----|-----|--------|")
        for domain, s in sorted(sequences.items()):
            lines.append(
                f"| {domain} | {s['num_features']} | "
                f"{s['mean_len']:.0f} | {s['median_len']:.0f} | "
                f"{s['p80_len']:.0f} | {s['p90_len']:.0f} | "
                f"{s['p95_len']:.0f} | {s['p99_len']:.0f} | "
                f"{s['max_len']} | {s['empty_ratio']*100:.1f}% |"
            )
        lines.append("\n### Truncation Impact (Current: seq_a:256, seq_b:256, seq_c:512, seq_d:512)")
        for domain, s in sorted(sequences.items()):
            trunc_map = {"domain_a": 256, "domain_b": 256, "domain_c": 512, "domain_d": 512}
            trunc = trunc_map.get(domain, 256)
            # Use P95 (or mean if P95 not available) to judge truncation adequacy
            over_trunc = (s['p95_len'] > trunc)
            lines.append(
                f"- **{domain}**: trunc={trunc}, mean={s['mean_len']:.0f}, "
                f"P95={s['p95_len']:.0f} → "
                f"{'⚠️ UNDER-TRUNCATED' if over_trunc else '✅ OK'}"
            )
        lines.append(
            "\n> 💡 **Suggestion**: If GPU memory allows, increase truncation lengths "
            "to at least P95 values. For domain_d, consider 2048+ or LongerEncoder."
        )
    else:
        lines.append("- No sequence features found.")
    lines.append("\n---\n")

    # Dense Features
    lines.append("## 6. Dense Feature Analysis")
    if dense:
        lines.append("| Feature | Mean Dim | Max Dim | Empty% |")
        lines.append("|---------|----------|---------|--------|")
        for col, s in sorted(dense.items()):
            lines.append(f"| {col} | {s['mean_dim']:.1f} | {s['max_dim']} | {s['empty_ratio']*100:.1f}% |")
    else:
        lines.append("- No dense features found.")
    lines.append("\n---\n")

    # Int Features
    lines.append("## 7. Integer Feature Cardinality")
    if int_feats:
        lines.append("| Feature | Unique Values | Min | Max | Cardinality Level |")
        lines.append("|---------|---------------|-----|-----|-------------------|")
        for col, s in sorted(int_feats.items()):
            uv = s['unique_values']
            level = "Low" if uv < 100 else ("Medium" if uv < 10000 else "High")
            lines.append(f"| {col} | {uv:,} | {s['min']} | {s['max']} | {level} |")
    else:
        lines.append("- No integer features found.")
    lines.append("\n---\n")

    # Recommendations
    lines.append("## 8. Strategic Recommendations")
    lines.append("\nBased on the above profiling, here are actionable suggestions:\n")

    recs = []
    if label["available"] and label["positive_ratio"] < 0.01:
        recs.append("1. **Use Focal Loss**: Positive ratio <1% indicates extreme imbalance. `--loss_type=focal --focal_alpha=0.75`.")
    elif label["available"] and label["positive_ratio"] > 0.1:
        recs.append("1. **Demo Data Alert**: Positive ratio >10% suggests demo/up-sampled data. Do NOT tune hyperparameters based on this ratio alone.")

    if timestamp["available"] and timestamp["span_days"] < 1.0:
        recs.append("2. **Temporal Features Limited**: Time span <1 day. Real data likely spans weeks. Design time-window stats for multi-day granularity.")
    elif timestamp["available"]:
        recs.append("2. **Strict Time Split**: Ensure validation timestamp > training max timestamp to prevent Future Leakage.")

    # Sequence truncation recommendation
    long_seq_domains = [d for d, s in sequences.items() if s['p95_len'] > 512]
    if long_seq_domains:
        recs.append(
            f"3. **Sequence Truncation**: {', '.join(long_seq_domains)} have P95 lengths > 512. "
            f"Current truncation loses information. Options: (a) increase seq_max_lens if memory allows, "
            f"(b) use `--seq_encoder_type=longer --seq_top_k=200` for compression, (c) implement recency-weighted sampling."
        )
    else:
        recs.append("3. **Sequence Truncation**: Looks reasonable, but monitor actual GPU memory usage.")

    # User/item repetition
    low_repeat = [k for k, v in user_item.items() if v.get("available") and v.get("repeat_ratio", 1.0) < 0.1]
    if low_repeat:
        recs.append(
            f"4. **ID Embedding Risk**: {', '.join(low_repeat)} have <10% repeat ratio. "
            f"Pure ID Embedding will be noisy. Prioritize: (a) target statistics features, (b) side-info aggregation, (c) ID hashing/sharing."
        )

    recs.append("5. **Learning Rate Schedule**: No scheduler detected in baseline. Add Linear Warm-up + Cosine Annealing for stable Transformer training.")
    recs.append("6. **AdamW Weight Decay**: Current AdamW lacks `weight_decay`. Add `1e-2` to prevent Dense layer overfitting.")

    for r in recs:
        lines.append(r)

    lines.append("\n---\n")
    lines.append("*End of Report*")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # Initialize logger (same pattern as train.py)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train.log")
    create_logger(log_path)

    logging.info(f"[Data Explorer] Scanning: {args.data_path}")

    files = discover_parquet_files(args.data_path)
    logging.info(f"[Data Explorer] Found {len(files)} Parquet file(s)")

    basic = collect_basic_info(files)
    logging.info(f"[Data Explorer] Total rows: {basic['total_rows']:,}")

    # Read sampled data for detailed analysis
    table = sample_table(files, args.max_sample_rows, args.max_sample_files)
    logging.info(f"[Data Explorer] Analyzing {len(table):,} rows (from up to {args.max_sample_files} files)...")

    label = analyze_label(table)
    timestamp = analyze_timestamp(table)
    user_item = analyze_user_item(table)
    sequences = analyze_sequences(table)
    dense = analyze_dense_features(table)
    int_feats = analyze_int_features(table)

    report = generate_report(args, basic, label, timestamp, user_item, sequences, dense, int_feats)

    if args.output == "-":
        for line in report.split("\n"):
            logging.info(line)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        logging.info(f"[Data Explorer] Report written to: {args.output}")


if __name__ == "__main__":
    main()
