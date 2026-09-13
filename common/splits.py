"""Group-aware, label-stratified splitting -- verbatim mechanism used
for both PAD-UFES-20 (grouped by patient_id) and HAM10000 (grouped by
lesion_id): no patient's/lesion's images may straddle two splits, or
appearance information leaks from train into the \"unseen\" test set.
"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def group_split(df: pd.DataFrame, test_size: float,
                        label_col: str, group_col: str,
                        seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware, label-stratified split.

    Splits by group_col so no group's rows straddle both output sets. StratifiedGroupKFold approximates the class
    balance of a plain stratified split while respecting that
    constraint. n_splits is chosen so the held-out fold is ~test_size
    of the data — mirrors train_test_split(df, test_size=...)'s
    (bigger, smaller) return order.
    """
    n_splits = max(2, round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    keep_idx, held_out_idx = next(
        sgkf.split(df, df[label_col], groups=df[group_col]))
    df_keep     = df.iloc[keep_idx].reset_index(drop=True)
    df_held_out = df.iloc[held_out_idx].reset_index(drop=True)
    return df_keep, df_held_out

# ─────────────────────────────────────────────
# K-FOLD VALIDATION (single config, patient-grouped folds)
# ─────────────────────────────────────────────
