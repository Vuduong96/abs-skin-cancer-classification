# -*- coding: utf-8 -*-
"""
ISIC 2020 — ABS (Adaptive Balanced Sampling), PATIENT-LEVEL SPLIT,
BINARY TASK (malignant vs. benign).

Train/test are split by patient_id, not by image row, so no patient's
images can straddle both sets. ISIC 2020 has ~16 images/patient on
average (2,056 patients, 33,126 images) — an image-level split would
leak patient-specific appearance into "unseen" test images. See
common/splits.py's group_split() for the mechanics.

ISIC 2020's own `diagnosis` column is ~82% "unknown", so it can't be
used as a multiclass target the way PAD-UFES-20's `diagnostic` or
HAM10000's `dx` are. The dataset's actual designed task is binary
melanoma detection, using `benign_malignant` (equivalently `target`) as
the label instead:
    benign    : 32,542 images (98.24%)
    malignant :    584 images ( 1.76%)
This is far more imbalanced than PAD-UFES-20 or HAM10000 (~55:1 vs.
their ~5:1 rarest-class ratios), making it a stress test for ABS's
class-balancing minority bonus.

Because malignant images are so scarce, the AL loop stops early once
every malignant training image has been acquired -- there is nothing
left to teach the model about the minority class at that point, and
continuing would just be random majority-class labeling.

The only acquisition strategy in this file is "full_abs" -- the complete
method (Phase 1 class-balanced candidate filter with minority bonus,
Phase 2 uncertainty + diversity selection with adaptive weight decay).
Baseline/ablation acquisition strategies used during development (plain
margin sampling, phase-1-only, phase-2-only, fixed-weight variants,
BADGE variants) are not part of this release.

Run:
    python train_full_abs_binary.py
    python train_full_abs_binary.py --total_rounds 15
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Tuple
import warnings
import logging
warnings.filterwarnings('ignore')
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.seed import set_seed
from common.model import EfficientNetBasic
from common.training import train_one_epoch
from common.transforms import build_aggressive_transform, build_val_transform
from common.metrics import evaluate_metrics
from common.abs_strategy import query_ablation
from common.splits import group_split
from dataset import ISIC2020Dataset

# ─────────────────────────────────────────────
# PATHS & BASE CONFIG
# ─────────────────────────────────────────────
METADATA_PATH = "data/isic2020/metadata.csv"
IMAGES_DIR    = "data/isic2020/images"
BASE_OUTPUT   = "outputs/isic2020_binary"

SEED         = 42
IMG_SIZE     = 300
BATCH_SIZE   = 16
NUM_WORKERS  = 4
TOTAL_ROUNDS = 35
INIT_SIZE    = 100
QUERY_SIZE   = 50
POOL_FACTOR  = 10
EPOCHS       = 20

# ─────────────────────────────────────────────
# ABLATION CONFIGS — plain-English names, no letters
# ─────────────────────────────────────────────
ABLATION_CONFIGS = {
    "full_abs": {
        "description":        "Full Adaptive Balanced Sampling (Phase 1 + Phase 2)",
        "use_balanced":       True,
        "use_minority_bonus": True,
        "uncertainty_type":   "margin",
        "use_diversity":      True,
        "adaptive_weight":    True,
        "uncertainty_weight": 0.7,
    },
}


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(
        output_dir,
        f'ablation_{datetime.now():%Y%m%d_%H%M%S}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# ─────────────────────────────────────────────
# RUN ONE ABLATION CONFIG
# ─────────────────────────────────────────────
def run_ablation(config_name: str, cfg: dict,
                 train_df, test_df,
                 class_names, num_classes,
                 device, logger, total_rounds: int) -> Tuple[list, float, float]:

    output_dir = Path(BASE_OUTPUT) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Config: {config_name}")
    logger.info(f"Description: {cfg['description']}")
    logger.info(
        f"Settings: {json.dumps({k: v for k, v in cfg.items() if k != 'description'}, indent=2)}")
    logger.info(f"{'='*60}")

    set_seed(SEED)

    train_tf = build_aggressive_transform(IMG_SIZE)
    val_tf   = build_val_transform(IMG_SIZE)

    train_ds = ISIC2020Dataset(train_df, IMAGES_DIR, transform=train_tf)
    test_ds  = ISIC2020Dataset(test_df,  IMAGES_DIR, transform=val_tf)
    query_ds = ISIC2020Dataset(train_df, IMAGES_DIR, transform=val_tf)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    all_indices = list(range(len(train_df)))
    rng         = np.random.RandomState(SEED)
    rng.shuffle(all_indices)
    labeled_idx   = all_indices[:INIT_SIZE]
    unlabeled_idx = all_indices[INIT_SIZE:]

    # Early-stop condition: terminate once every malignant (cancerous)
    # training image has been acquired -- no need to keep running once
    # there's nothing left to teach the model about the minority class.
    malignant_idx = class_names.index("malignant")
    total_malignant = int((train_df["label"] == malignant_idx).sum())
    logger.info(f"Total malignant in train pool: {total_malignant} "
                f"(early-stop once all are labeled)")

    model     = EfficientNetBasic(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4, weight_decay=1e-4)

    history      = []
    best_f1      = -np.inf
    best_mal_f1  = -np.inf

    for round_idx in range(1, total_rounds + 1):
        if not unlabeled_idx:
            break

        logger.info(f"\n  [{config_name}] Round {round_idx}/{total_rounds} | "
                    f"Labeled: {len(labeled_idx)}")

        train_loader = DataLoader(
            Subset(train_ds, labeled_idx),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )

        for epoch in range(1, EPOCHS + 1):
            loss, acc = train_one_epoch(
                model, train_loader, optimizer, device, criterion)

        metrics = evaluate_metrics(
            model, test_loader, device,
            num_classes, class_names,
            binary_positive_class="malignant")

        entry = {
            "config":       config_name,
            "round":        round_idx,
            "labeled_size": len(labeled_idx),
            **metrics
        }
        history.append(entry)

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            torch.save(model.state_dict(),
                       output_dir / "best_model.pth")

        mal_f1 = metrics.get("f1_malignant", 0.0)
        if mal_f1 > best_mal_f1:
            best_mal_f1 = mal_f1

        logger.info(
            f"  [{config_name}] R{round_idx:02d} | "
            f"N={len(labeled_idx):5d} | "
            f"F1={metrics['f1_macro']:.4f} | "
            f"malignant={mal_f1:.4f} | "
            f"benign={metrics.get('f1_benign', 0):.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f}")

        if round_idx < total_rounds:
            selected = query_ablation(
                model=model,
                unlabeled_indices=unlabeled_idx,
                dataset=query_ds,
                device=device,
                query_size=QUERY_SIZE,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                num_classes=num_classes,
                pool_factor=POOL_FACTOR,
                cfg=cfg
            )

            sel_labels = [train_df.iloc[i]["label"] for i in selected]
            logger.info(f"  Selection: {dict(Counter(sel_labels))}")

            labeled_idx   = list(labeled_idx) + selected
            unlabeled_idx = list(
                set(unlabeled_idx) - set(selected))

            labeled_malignant = int((train_df.iloc[labeled_idx]["label"] == malignant_idx).sum())
            if labeled_malignant >= total_malignant:
                logger.info(f"  All {total_malignant} malignant images acquired "
                            f"(labeled_malignant={labeled_malignant}) -- stopping early "
                            f"at round {round_idx}/{total_rounds}.")
                break

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(output_dir / "history.csv", index=False)

    logger.info(f"\n  [{config_name}] DONE | "
                f"Best Macro F1: {best_f1:.4f} | "
                f"Best Malignant F1: {best_mal_f1:.4f}")

    return history, best_f1, best_mal_f1

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="all",
        help="Config name e.g. full_abs or 'all'")
    parser.add_argument(
        "--total_rounds", type=int, default=TOTAL_ROUNDS,
        help="Number of active learning rounds per config")
    args = parser.parse_args()

    Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(BASE_OUTPUT)

    logger.info("="*60)
    logger.info("ISIC 2020 — ABS Query Strategy (patient-level split)")
    logger.info("="*60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Rounds: {args.total_rounds} | "
                f"Query size: {QUERY_SIZE} | "
                f"Init: {INIT_SIZE}")

    logger.info("Loading ISIC 2020 metadata...")
    meta = pd.read_csv(METADATA_PATH)

    if "benign_malignant" not in meta.columns:
        raise ValueError("Expected 'benign_malignant' column in ISIC 2020 metadata.")
    if "image_name" not in meta.columns:
        raise ValueError("Expected 'image_name' column in ISIC 2020 metadata.")
    if "patient_id" not in meta.columns:
        raise ValueError("Expected 'patient_id' column in ISIC 2020 metadata.")

    meta = meta[["image_name", "benign_malignant", "patient_id"]].dropna(subset=["benign_malignant"]).copy()
    meta["benign_malignant"] = meta["benign_malignant"].str.strip().str.lower()
    le             = LabelEncoder()
    meta["label"]  = le.fit_transform(meta["benign_malignant"])
    meta["class_name"] = meta["benign_malignant"].astype(str)
    class_names    = list(le.classes_)   # ['benign', 'malignant']
    num_classes    = len(class_names)

    logger.info(f"Classes ({num_classes}): {class_names}")
    logger.info(f"Total images: {len(meta)} | Total patients: {meta['patient_id'].nunique()}")
    logger.info(f"Class distribution: {meta['benign_malignant'].value_counts().to_dict()}")

    train_df, temp_df = group_split(
        meta, test_size=0.20,
        label_col="label", group_col="patient_id", seed=SEED)
    _, test_df = group_split(
        temp_df, test_size=0.50,
        label_col="label", group_col="patient_id", seed=SEED)

    overlap = set(train_df["patient_id"]) & set(test_df["patient_id"])
    assert not overlap, f"Patient leakage between train/test: {overlap}"

    logger.info(f"Train: {len(train_df)} images / {train_df['patient_id'].nunique()} patients | "
                f"Test: {len(test_df)} images / {test_df['patient_id'].nunique()} patients | "
                f"Patient overlap: {len(overlap)}")

    if args.config == "all":
        configs_to_run = ABLATION_CONFIGS
    else:
        configs_to_run = {
            k: v for k, v in ABLATION_CONFIGS.items()
            if k == args.config
        }
        if not configs_to_run:
            logger.error(
                f"Config '{args.config}' not found. "
                f"Available: {list(ABLATION_CONFIGS.keys())}")
            sys.exit(1)

    logger.info(f"Running {len(configs_to_run)} configs: "
                f"{list(configs_to_run.keys())}")

    all_results = []
    summary     = []

    for config_name, cfg in configs_to_run.items():
        history, best_f1, best_mal = run_ablation(
            config_name=config_name,
            cfg=cfg,
            train_df=train_df,
            test_df=test_df,
            class_names=class_names,
            num_classes=num_classes,
            device=device,
            logger=logger,
            total_rounds=args.total_rounds
        )
        all_results.extend(history)
        best_roc_auc = max((h.get("roc_auc", 0.0) for h in history), default=0.0)
        summary.append({
            "config":             config_name,
            "description":        cfg["description"],
            "uncertainty_weight": cfg["uncertainty_weight"],
            "adaptive_weight":    cfg["adaptive_weight"],
            "best_f1_macro":      best_f1,
            "best_f1_malignant":  best_mal,
            "best_roc_auc":       best_roc_auc,
        })

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(
        Path(BASE_OUTPUT) / "all_results.csv", index=False)

    df_summary = pd.DataFrame(summary).sort_values(
        "best_f1_macro", ascending=False)
    df_summary.to_csv(
        Path(BASE_OUTPUT) / "ablation_summary.csv", index=False)

    logger.info(f"\n{'='*70}")
    logger.info("ABLATION SUMMARY — ISIC 2020")
    logger.info(f"{'='*70}")
    logger.info(f"{'Config':<24} {'Description':<48} "
                f"{'Macro F1':>9} {'Malig F1':>9}")
    logger.info("-"*70)
    for _, row in df_summary.iterrows():
        logger.info(
            f"  {row['config']:<22} "
            f"{row['description']:<48} "
            f"{row['best_f1_macro']:>9.4f} "
            f"{row['best_f1_malignant']:>9.4f}")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {BASE_OUTPUT}")


if __name__ == "__main__":
    main()
