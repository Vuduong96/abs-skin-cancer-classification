# Adaptive Balanced Sampling (ABS)

**Adaptive Balanced Sampling (ABS) is a two-phase active learning framework designed for class-imbalanced medical image classification.**

### Phase 1 — Balanced Candidate Allocation
The unlabeled pool is partitioned according to the model's predicted classes. A rarity-scaled quota is then assigned to each class, ensuring that under-represented categories receive sufficient representation within the candidate pool.

### Phase 2 — Informative Sample Selection
Samples are selected from the balanced candidate pool by jointly considering predictive uncertainty and feature-space diversity, reducing redundancy while retaining informative examples.

By combining class-balanced allocation with uncertainty- and diversity-based selection, 
  ABS improves minority-class performance while maintaining strong performance on majority classes. 
  The framework is evaluated on three dermatological benchmarks (HAM10000, PAD-UFES-20, and ISIC-2020) 
using patient- and lesion-level group-aware splits to better assess generalization to previously unseen patients and lesions.
"""
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import pairwise_distances


def query_ablation(model, unlabeled_indices, dataset, device,
                   query_size, batch_size, num_workers,
                   num_classes, pool_factor, cfg):
    if not unlabeled_indices:
        return []
    query_size = min(query_size, len(unlabeled_indices))
    if query_size >= len(unlabeled_indices):
        return list(unlabeled_indices)

    subset = Subset(dataset, unlabeled_indices)
    loader = DataLoader(subset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    margins, preds, feats_list = [], [], []
    model.eval()

    with torch.no_grad():
        for xb, _, _ in loader:
            xb     = xb.to(device)
            logits, feats = model(xb, return_features=True)
            prob   = F.softmax(logits, dim=1).cpu().numpy()

            srt = -np.sort(-prob, axis=1)
            margins.extend((srt[:, 0] - srt[:, 1]).tolist())

            preds.extend(prob.argmax(axis=1).tolist())
            feats_list.append(feats.cpu().numpy())

    margins   = np.array(margins)
    preds     = np.array(preds)
    features  = np.concatenate(feats_list, axis=0)   # raw penultimate features (N, d)

    norms       = np.linalg.norm(features, axis=1, keepdims=True)
    features_l2 = features / (norms + 1e-10)

    unc_score = 1 - margins   # lower margin = higher uncertainty

    # ── PHASE 1: Candidate pool + per-class quota ─────────────
    target_k      = min(query_size * pool_factor, len(unlabeled_indices))
    class_counts  = Counter(preds)

    if cfg["use_balanced"]:
        k_base = int(target_k / num_classes) + 2
        k_per_class = {}
        for c in range(num_classes):
            idx_c = np.where(preds == c)[0]
            if len(idx_c) == 0:
                k_per_class[c] = 0
                continue
            if cfg["use_minority_bonus"]:
                c_count = class_counts.get(c, 1)
                total_s = len(preds)
                bonus   = int((total_s / (c_count * num_classes)) * k_base * 0.3)
                k_per_class[c] = min(k_base + bonus, len(idx_c))
            else:
                k_per_class[c] = min(k_base, len(idx_c))

        candidates = []
        for c in range(num_classes):
            idx_c = np.where(preds == c)[0]
            if len(idx_c) == 0:
                continue
            k = k_per_class[c]
            top = np.argsort(-unc_score[idx_c])[:k]
            candidates.extend(idx_c[top].tolist())
        candidates = list(set(candidates))

        if len(candidates) < query_size:
            global_sort = np.argsort(-unc_score)
            for idx in global_sort:
                if idx not in candidates:
                    candidates.append(int(idx))
                if len(candidates) >= query_size * pool_factor:
                    break
    else:
        k_per_class = None
        candidates = np.argsort(-unc_score)[:target_k].tolist()

    if not candidates:
        candidates = list(range(len(unlabeled_indices)))

    print(f"  [Phase 1] Pool: {len(candidates)} | "
          f"Dist: {dict(Counter(preds[candidates].tolist()))}")

    if len(candidates) <= query_size:
        return [unlabeled_indices[i] for i in candidates]

    cand_feats_l2 = features_l2[candidates]
    cand_unc      = unc_score[candidates]
    unc_norm      = cand_unc / (cand_unc.max() + 1e-10)

    # ── PHASE 2a: no diversity → balance-preserving top-K ─────
    if not cfg["use_diversity"]:
        if cfg["use_balanced"] and k_per_class is not None:
            # Allocate query_size proportionally to each class's
            # Phase-1 quota, so balancing survives into the final
            # selection instead of being erased by a global re-rank.
            total_quota = sum(k_per_class.values())
            if total_quota == 0:
                top = np.argsort(-cand_unc)[:query_size]
                selected = [candidates[i] for i in top]
            else:
                query_per_class = {
                    c: max(0, round(k_per_class[c] / total_quota * query_size))
                    for c in range(num_classes)
                }
                diff = query_size - sum(query_per_class.values())
                classes_desc = sorted(range(num_classes), key=lambda c: -k_per_class[c])
                i = 0
                while diff != 0 and classes_desc:
                    c = classes_desc[i % len(classes_desc)]
                    idx_c = np.where(preds == c)[0]
                    if diff > 0 and query_per_class[c] < len(idx_c):
                        query_per_class[c] += 1; diff -= 1
                    elif diff < 0 and query_per_class[c] > 0:
                        query_per_class[c] -= 1; diff += 1
                    i += 1
                    if i > 10 * num_classes:
                        break

                selected = []
                for c in range(num_classes):
                    n_select = query_per_class.get(c, 0)
                    if n_select == 0:
                        continue
                    idx_c = np.where(preds == c)[0]
                    if len(idx_c) == 0:
                        continue
                    n_select = min(n_select, len(idx_c))
                    top_c = np.argsort(-unc_score[idx_c])[:n_select]
                    selected.extend(idx_c[top_c].tolist())

                if len(selected) < query_size:
                    remaining = list(set(range(len(preds))) - set(selected))
                    need = query_size - len(selected)
                    if remaining:
                        top_remaining = np.argsort(-unc_score[remaining])[:need]
                        selected.extend([remaining[i] for i in top_remaining])
                selected = selected[:query_size]
        else:
            # No balancing was applied in Phase 1 anyway (e.g. a
            # hypothetical unbalanced+no-diversity config) — plain
            # top-K by uncertainty within the candidate pool is correct.
            top = np.argsort(-cand_unc)[:query_size]
            selected = [candidates[i] for i in top]

        print(f"  [Phase 2] Balance-preserving top-K (no diversity): {len(selected)} | "
              f"Dist: {dict(Counter(preds[selected].tolist()))}")
        return [unlabeled_indices[i] for i in selected]

    # ── PHASE 2b: greedy hybrid uncertainty + diversity, in L2-normalized
    # feature space ──
    uw = cfg["uncertainty_weight"]

    dists    = pairwise_distances(cand_feats_l2, metric='euclidean')
    selected = [int(np.argmax(unc_norm))]
    min_d    = dists[selected[0]].copy()

    for it in range(min(query_size - 1, len(candidates) - 1)):
        min_d[selected] = -np.inf
        div             = min_d / (np.max(min_d) + 1e-10)

        if cfg["adaptive_weight"]:
            w = uw * (1 - it / query_size * 0.3)
        else:
            w = uw

        score = w * unc_norm + (1 - w) * div
        nxt   = int(np.argmax(score))
        selected.append(nxt)
        min_d = np.minimum(min_d, dists[nxt])

    final = [unlabeled_indices[candidates[i]]
             for i in selected][:query_size]
    sel_preds = preds[[candidates[i] for i in selected[:len(final)]]]
    print(f"  [Phase 2] greedy hybrid final: {len(final)} | "
          f"Dist: {dict(Counter(sel_preds.tolist()))}")

    return final
