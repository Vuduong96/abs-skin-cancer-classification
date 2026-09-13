"""Verbatim from every reported ABS run's train_one_epoch()."""
import torch


def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb, _ in loader:
        xb, yb = (xb.to(device, non_blocking=True),
                  yb.to(device, non_blocking=True))
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += yb.size(0)
            loss_sum += loss.item() * yb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)
