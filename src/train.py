"""Training loop — notebook Step 5, plus the bundle save from the final cell."""

import torch
import torch.nn as nn

from src import config, dataset
from src.model import build_model, save_bundle


def train(epochs=config.EPOCHS, rebuild_data=False, batch_size=config.BATCH_SIZE,
          lr=config.LR, resume=False):
    dev = config.device()
    print(f"Device: {dev}")

    X, y, class_names = dataset.load_arrays(rebuild=rebuild_data)
    dataset.sanity_check(X, y)
    num_classes = len(class_names)

    train_loader, val_loader = dataset.make_loaders(X, y, batch_size=batch_size)

    model = build_model(num_classes).to(dev)
    if resume and config.BEST_CKPT.exists():
        model.load_state_dict(torch.load(config.BEST_CKPT, map_location=dev))
        print(f"Resumed from {config.BEST_CKPT}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        correct, total, train_loss = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        scheduler.step()

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total += yb.size(0)
        val_acc = v_correct / v_total

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(correct / total)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), config.BEST_CKPT)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: loss={train_loss/len(train_loader):.4f} "
                  f"train_acc={correct/total:.3f} val_acc={val_acc:.3f} (best={best_val:.3f})")

    print("Training done. Best val accuracy:", best_val)

    # Reload the best checkpoint before bundling, so the saved model is the best one.
    model.load_state_dict(torch.load(config.BEST_CKPT, map_location=dev))
    path = save_bundle(model, list(class_names), config.MODEL_BUNDLE, num_classes=num_classes)
    print(f"Saved bundle: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    return model, class_names, val_loader, best_val, history
