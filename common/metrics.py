"""Per-round evaluation metrics: accuracy, balanced accuracy, per-class
and macro/weighted F1, ROC-AUC, and classification_report-based
precision/recall. ROC-AUC uses sklearn's multiclass one-vs-rest form for
the native multiclass tasks, and the standard binary form (via the
positive class's own probability column) for the binary tasks -- pass
`binary_positive_class` to select the latter.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, f1_score, roc_auc_score,
)


@torch.no_grad()
def evaluate_metrics(model, loader, device, num_classes, class_names,
                     binary_positive_class=None):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    for xb, yb, _ in loader:
        xb     = xb.to(device)
        logits = model(xb)
        probs  = F.softmax(logits, dim=1)
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_targets.extend(yb.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_probs   = np.concatenate(all_probs, axis=0)
    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)

    f1_per = f1_score(all_targets, all_preds,
                      average=None, zero_division=0)

    metrics = {
        "acc":          accuracy_score(all_targets, all_preds),
        "balanced_acc": balanced_accuracy_score(all_targets, all_preds),
        "f1_macro":     f1_score(all_targets, all_preds,
                                 average='macro', zero_division=0),
        "f1_weighted":  f1_score(all_targets, all_preds,
                                 average='weighted', zero_division=0),
    }

    for i, name in enumerate(class_names):
        metrics[f"f1_{name}"] = (f1_per[i]
                                  if i < len(f1_per) else 0.0)

    try:
        if binary_positive_class is not None:
            # Binary task: sklearn's roc_auc_score wants the positive
            # class's own probability column, not the full (N, 2) matrix.
            pos_idx = (class_names.index(binary_positive_class)
                       if binary_positive_class in class_names else 1)
            y_true_pos = (all_targets == pos_idx).astype(int)
            metrics["roc_auc"] = roc_auc_score(y_true_pos, all_probs[:, pos_idx])
        else:
            metrics["roc_auc"] = roc_auc_score(
                all_targets, all_probs, multi_class='ovr', average='macro')
    except Exception:
        metrics["roc_auc"] = 0.0

    # Precision/recall, appended last so they land at the end of
    # all_results.csv without reordering the existing columns.
    report = classification_report(
        all_targets, all_preds, labels=list(range(num_classes)),
        target_names=class_names, output_dict=True, zero_division=0)

    metrics["precision_macro"]    = report["macro avg"]["precision"]
    metrics["recall_macro"]       = report["macro avg"]["recall"]
    metrics["precision_weighted"] = report["weighted avg"]["precision"]
    metrics["recall_weighted"]    = report["weighted avg"]["recall"]

    for name in class_names:
        metrics[f"precision_{name}"] = report[name]["precision"]
        metrics[f"recall_{name}"]    = report[name]["recall"]

    return metrics
