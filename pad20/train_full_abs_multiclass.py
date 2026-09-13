# -*- coding: utf-8 -*-
"""
PAD-UFES-20 — ABS (Adaptive Balanced Sampling), patient-level split, 6-class classification task.

Train and test partitions are created at the patient level using
`patient_id`, ensuring that all images from a given patient are assigned
to a single split. Since PAD-UFES-20 contains multiple images per
patient, this protocol helps reduce overlap between training and
evaluation data and provides a more conservative assessment of
generalization. The splitting procedure is implemented in
`common/splits.py` through `group_split()`.

This script evaluates the full ABS framework, consisting of:
    - Phase 1: class-balanced candidate allocation with rarity-aware
      adjustment.
    - Phase 2: uncertainty-diversity sample selection.

The repository release focuses on the complete ABS method used in the
paper. 

    python train_full_abs_multiclass.py
    python train_full_abs_multiclass.py --total_rounds 15
"""

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List
import warnings
import logging
warnings.filterwarnings('ignore')
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, average_precision_score, roc_auc_score,
    accuracy_score, f1_score,
    balanced_accuracy_score, pairwise_distances
)

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from common.seed import set_seed
from common.model import EfficientNetBasic
from common.training import train_one_epoch
from common.transforms import build_aggressive_transform, build_val_transform
from common.metrics import evaluate_metrics
from common.abs_strategy import query_ablation
from common.splits import group_split
from dataset import PAD20Dataset



# ─────────────────────────────────────────────
# PATHS & BASE CONFIG
# ─────────────────────────────────────────────
METADATA_PATH = "data/pad20/metadata.csv"
IMAGES_DIR    = "data/pad20/images"
BASE_OUTPUT   = "outputs/pad20_multiclass"

SEED         = 42
IMG_SIZE     = 300
BATCH_SIZE   = 16
NUM_WORKERS  = 4
TOTAL_ROUNDS = 35    # reduced — this run is to demonstrate the idea, not full convergence
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

# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────




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

    train_ds = PAD20Dataset(train_df, IMAGES_DIR, transform=train_tf)
    test_ds  = PAD20Dataset(test_df,  IMAGES_DIR, transform=val_tf)
    query_ds = PAD20Dataset(train_df, IMAGES_DIR, transform=val_tf)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    all_indices = list(range(len(train_df)))
    rng         = np.random.RandomState(SEED)
    rng.shuffle(all_indices)
    labeled_idx   = all_indices[:INIT_SIZE]
    unlabeled_idx = all_indices[INIT_SIZE:]

    model     = EfficientNetBasic(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4, weight_decay=1e-4)

    history      = []
    best_f1      = -np.inf
    best_mel_f1  = -np.inf

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
            num_classes, class_names)

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

        mel_f1 = metrics.get("f1_MEL", metrics.get("f1_mel", 0.0))
        if mel_f1 > best_mel_f1:
            best_mel_f1 = mel_f1

        logger.info(
            f"  [{config_name}] R{round_idx:02d} | "
            f"N={len(labeled_idx):5d} | "
            f"F1={metrics['f1_macro']:.4f} | "
            f"mel={mel_f1:.4f} | "
            f"nev={metrics.get('f1_NEV', 0):.4f} | "
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

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(output_dir / "history.csv", index=False)

    logger.info(f"\n  [{config_name}] DONE | "
                f"Best Macro F1: {best_f1:.4f} | "
                f"Best Mel F1: {best_mel_f1:.4f}")

    return history, best_f1, best_mel_f1

# ─────────────────────────────────────────────
# PATIENT-LEVEL SPLIT
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="all",
        help="Config name e.g. full_abs, margin_baseline or 'all'")
    parser.add_argument(
        "--total_rounds", type=int, default=TOTAL_ROUNDS,
        help="Number of active learning rounds per config")
    args = parser.parse_args()

    Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(BASE_OUTPUT)

    logger.info("="*60)
    logger.info("PAD-UFES-20 — 2-Phase Query Strategy Ablation Study (patient-level split)")
    logger.info("="*60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Rounds: {args.total_rounds} | "
                f"Query size: {QUERY_SIZE} | "
                f"Init: {INIT_SIZE}")

    logger.info("Loading PAD-UFES-20 metadata...")
    meta = pd.read_csv(METADATA_PATH)

    # PAD-UFES-20 uses 'diagnostic' column (string class name, e.g. NEV, BCC, MEL)
    # and 'img_id' for the image filename (already includes .png extension).
    if "diagnostic" not in meta.columns:
        raise ValueError("Expected 'diagnostic' column in PAD-UFES-20 metadata.")
    if "img_id" not in meta.columns:
        raise ValueError("Expected 'img_id' column in PAD-UFES-20 metadata.")
    if "patient_id" not in meta.columns:
        raise ValueError("Expected 'patient_id' column in PAD-UFES-20 metadata.")

    meta = meta[["img_id", "diagnostic", "patient_id"]].dropna(subset=["diagnostic"]).copy()
    meta["diagnostic"] = meta["diagnostic"].str.strip().str.upper()
    le             = LabelEncoder()
    meta["label"]  = le.fit_transform(meta["diagnostic"])
    meta["class_name"] = meta["diagnostic"].astype(str)
    class_names    = list(le.classes_)   # e.g. [ACK, BCC, MEL, NEV, SCC, SEK]
    num_classes    = len(class_names)

    logger.info(f"Classes ({num_classes}): {class_names}")
    logger.info(f"Total images: {len(meta)} | Total patients: {meta['patient_id'].nunique()}")

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
        history, best_f1, best_mel = run_ablation(
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
            "best_f1_mel":        best_mel,
            "best_roc_auc":        best_roc_auc,
        })

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(
        Path(BASE_OUTPUT) / "all_results.csv", index=False)

    df_summary = pd.DataFrame(summary).sort_values(
        "best_f1_macro", ascending=False)
    df_summary.to_csv(
        Path(BASE_OUTPUT) / "ablation_summary.csv", index=False)

    logger.info(f"\n{'='*70}")
    logger.info("ABLATION SUMMARY — PAD-UFES-20")
    logger.info(f"{'='*70}")
    logger.info(f"{'Config':<24} {'Description':<48} "
                f"{'Macro F1':>9} {'Mel F1':>9}")
    logger.info("-"*70)
    for _, row in df_summary.iterrows():
        logger.info(
            f"  {row['config']:<22} "
            f"{row['description']:<48} "
            f"{row['best_f1_macro']:>9.4f} "
            f"{row['best_f1_mel']:>9.4f}")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {BASE_OUTPUT}")


if __name__ == "__main__":
    main()