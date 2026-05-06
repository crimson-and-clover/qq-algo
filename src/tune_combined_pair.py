#!/usr/bin/env python3
"""Tune bce_weight for bce+pair loss on local demo data.

Runs short training sessions (few epochs, tight patience) for each candidate
bce_weight and reports the best validation AUC.  Designed to run standalone
without interfering with the main training pipeline.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import re
from pathlib import Path
from typing import List, Tuple

# ────────────────────────────── Configuration ──────────────────────────────

BCE_WEIGHTS: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]
RANK_MARGIN: float = 1.0
NUM_EPOCHS: int = 3
PATIENCE: int = 1
BATCH_SIZE: int = 512
NUM_WORKERS: int = 4
SEED: int = 42

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_SRC = PROJECT_ROOT / "datasets"
SCHEMA_SRC = PROJECT_ROOT / "schema.json"

# ────────────────────────────── Helpers ────────────────────────────────────


def parse_best_auc(log_path: Path) -> float:
    """Scan train.log for the highest Validation AUC."""
    best = 0.0
    if not log_path.exists():
        return best
    text = log_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        m = re.search(r"Validation \| AUC:\s*([0-9.]+)", line)
        if m:
            auc = float(m.group(1))
            if auc > best:
                best = auc
    return best


def run_one(bce_weight: float, out_dir: Path) -> Tuple[float, str, str]:
    """Launch a single training run and return (best_auc, stdout, stderr)."""
    # Prepare a self-contained temporary data directory.
    tmp_data = out_dir / "data"
    tmp_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCHEMA_SRC, tmp_data / "schema.json")
    for pq in DATA_SRC.glob("*.parquet"):
        link = tmp_data / pq.name
        if not link.exists():
            link.symlink_to(pq)

    ckpt_dir = out_dir / "ckpt"
    log_dir = out_dir / "logs"
    tf_dir = out_dir / "tf_events"

    cmd = [
        sys.executable, "-u", str(SCRIPT_DIR / "train.py"),
        "--data_dir", str(tmp_data),
        "--schema_path", str(tmp_data / "schema.json"),
        "--ckpt_dir", str(ckpt_dir),
        "--log_dir", str(log_dir),
        "--tf_events_dir", str(tf_dir),
        "--ns_tokenizer_type", "rankmixer",
        "--user_ns_tokens", "5",
        "--item_ns_tokens", "2",
        "--num_queries", "2",
        "--ns_groups_json", "",
        "--emb_skip_threshold", "1000000",
        "--seq_id_threshold", "10000",
        "--use_target_attention",
        "--loss_type", "bce+pair",
        "--bce_weight", str(bce_weight),
        "--pair_weight", str(1.0 - bce_weight),
        "--rank_margin", str(RANK_MARGIN),
        "--num_epochs", str(NUM_EPOCHS),
        "--patience", str(PATIENCE),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--no_scheduler",
        "--seed", str(SEED),
    ]

    env = os.environ.copy()
    # Ensure platform env vars do not override our local paths.
    for k in ("TRAIN_DATA_PATH", "TRAIN_CKPT_PATH", "TRAIN_LOG_PATH", "TRAIN_TF_EVENTS_PATH"):
        env.pop(k, None)

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    best_auc = parse_best_auc(log_dir / "train.log")
    return best_auc, proc.stdout, proc.stderr


# ────────────────────────────── Main ───────────────────────────────────────


def main() -> None:
    results: List[Tuple[float, float]] = []
    base_tmp = Path(tempfile.gettempdir()) / "pcvr_tune_bce_pair"

    for w in BCE_WEIGHTS:
        print(f"\n{'='*60}")
        print(f"Running bce_weight={w}")
        print(f"{'='*60}")
        out_dir = base_tmp / f"bw{w}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        best_auc, stdout, stderr = run_one(w, out_dir)
        results.append((w, best_auc))
        print(f"[Result] bce_weight={w} -> best AUC={best_auc:.6f}")

        # Dump stderr on failure so we can debug.
        if best_auc == 0.0 and ("Error" in stderr or "Exception" in stderr):
            print("[STDERR snippet]", file=sys.stderr)
            print(stderr[-2000:], file=sys.stderr)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for w, auc in results:
        marker = " <-- BEST" if (w, auc) == max(results, key=lambda x: x[1]) else ""
        print(f"  bce_weight={w:.1f}  best_valid_AUC={auc:.6f}{marker}")

    best_w, best_auc = max(results, key=lambda x: x[1])
    print(f"\nRecommended bce_weight: {best_w} (best AUC={best_auc:.6f})")


if __name__ == "__main__":
    main()
