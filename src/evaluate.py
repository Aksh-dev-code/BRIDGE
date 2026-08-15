"""Validation metrics and plots — notebook Step 6.

Charts are both saved to config.RESULTS_DIR and popped up on screen, the way
the notebook renders them inline. Pass show=False (or --no-show) to only save,
which is what you want on a headless box.
"""

import sys

import numpy as np
import torch

from src import config, dataset
from src.model import load_bundle

# Backends that can't open a window — plotting falls back to save-only.
_HEADLESS_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "cairo"}


def _cache_path():
    return config.RESULTS_DIR / "eval_cache.npz"


def _save_cache(preds, labels, confs, class_names, topk):
    """Stash raw predictions so `eval` can be replayed without the dataset.

    Training usually happens somewhere with the data (Kaggle, a GPU box) while
    you want the numbers and charts locally — this is what makes that work.
    """
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        preds=np.asarray(preds), labels=np.asarray(labels), confs=np.asarray(confs),
        class_names=np.asarray(class_names, dtype=object).astype("U"),
        topk_k=np.array(sorted(topk)), topk_v=np.array([topk[k] for k in sorted(topk)]),
    )
    return path


def _load_cache():
    d = np.load(_cache_path(), allow_pickle=False)
    topk = {int(k): float(v) for k, v in zip(d["topk_k"], d["topk_v"])}
    return d["preds"], d["labels"], d["confs"], d["class_names"], topk


def _no_data_message(reason):
    return (
        f"{reason}\n\n"
        "`eval` measures accuracy against labelled clips, so it needs the INCLUDE\n"
        f"dataset at {config.DATA_ROOT}\n"
        "(or BRIDGE_DATA_ROOT pointing at your copy).\n\n"
        "Options:\n"
        "  1. Download it — see 'Getting the dataset' in the README.\n"
        "  2. Run `train` or `eval` where the data lives (Kaggle, a GPU box).\n"
        f"     Results get cached to {_cache_path().name} in results/, and copying\n"
        "     that file back here lets `eval` replay the numbers and charts offline.\n"
        "  3. Just want to use the model? `python main.py run` needs no dataset."
    )


def can_show_plots():
    """True if matplotlib can actually open a window here."""
    import matplotlib

    if not sys.stdout.isatty():
        return False
    return matplotlib.get_backend().lower() not in _HEADLESS_BACKENDS


def predict_loader(model, loader, dev):
    """Run the model over a loader -> (preds, labels, confidences)."""
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb)
            probs = torch.softmax(out, dim=1)
            conf, pred = probs.max(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_confs.extend(conf.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_confs)


def topk_accuracy(model, loader, k, dev):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb)
            topk = out.topk(k, dim=1).indices
            correct += sum(yb[i] in topk[i] for i in range(len(yb)))
            total += yb.size(0)
    return correct / total


def evaluate(model=None, val_loader=None, class_names=None, plots=True, show=None,
             history=None, allow_cache=True):
    """Score the model, print the notebook's tables, then draw the charts.

    show=None auto-detects whether a window can be opened. allow_cache lets a
    machine without the dataset replay the last cached evaluation instead of
    failing outright.
    """
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    dev = config.device()
    if show is None:
        show = can_show_plots()

    all_preds = topk = None
    replayed = False

    if model is None or val_loader is None:
        try:
            model, class_names, _ = load_bundle(map_location=dev)
            X, y, class_names_arr = dataset.load_arrays()
            class_names = np.array(class_names if len(class_names) else class_names_arr)
            _, val_loader = dataset.make_loaders(X, y)
        except FileNotFoundError as e:
            # No dataset here. Fall back to the last cached evaluation if we have one.
            if not (allow_cache and _cache_path().exists()):
                raise FileNotFoundError(_no_data_message(str(e).split("\n")[0])) from None
            all_preds, all_labels, all_confs, class_names, topk = _load_cache()
            replayed = True
            print(f"No dataset here — replaying the last evaluation from {_cache_path()}")

    class_names = np.array(class_names)
    num_classes = len(class_names)

    if all_preds is None:
        all_preds, all_labels, all_confs = predict_loader(model, val_loader, dev)
        topk = {k: topk_accuracy(model, val_loader, k, dev) for k in (1, 3, 5)}
        _save_cache(all_preds, all_labels, all_confs, class_names, topk)

    val_acc = (all_preds == all_labels).mean()
    print(f"Validation accuracy: {val_acc:.4f}")

    report = classification_report(all_labels, all_preds, target_names=class_names,
                                   zero_division=0, output_dict=True)
    df = pd.DataFrame(report).T.iloc[:-3].sort_values("f1-score")

    print("\nWorst 15 classes:")
    print(df.head(15)[["precision", "recall", "f1-score", "support"]])
    print("\nBest 10 classes:")
    print(df.tail(10)[["precision", "recall", "f1-score", "support"]])

    for k, acc in sorted(topk.items()):
        print(f"Top-{k} accuracy: {acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    cm_noself = cm.copy()
    np.fill_diagonal(cm_noself, 0)

    pairs = [(cm_noself[t, p], class_names[t], class_names[p])
             for t in range(num_classes) for p in range(num_classes) if cm_noself[t, p] > 0]
    pairs.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 30 Confused Class Pairs")
    print("=" * 60)
    for count, true_class, predicted_class in pairs[:30]:
        print(f"True: {str(true_class):20s} → Predicted: {str(predicted_class):20s} | Errors: {count}")

    summary = {
        "val_acc": float(val_acc),
        "topk": {k: float(v) for k, v in topk.items()},
        "num_classes": int(num_classes),
        "val_samples": int(len(all_labels)),
        "mean_conf_correct": float(all_confs[all_preds == all_labels].mean()) if (all_preds == all_labels).any() else 0.0,
        "mean_conf_wrong": float(all_confs[all_preds != all_labels].mean()) if (all_preds != all_labels).any() else 0.0,
        "perfect_classes": int((df["f1-score"] == 1.0).sum()),
        "failed_classes": int((df["f1-score"] == 0.0).sum()),
    }
    summary["replayed"] = replayed
    _print_summary(summary)

    if plots:
        _make_plots(df, cm, cm_noself, pairs, class_names, all_preds, all_labels,
                    all_confs, show, history)

    return {"val_acc": float(val_acc), "report": df, "confusion_matrix": cm, "summary": summary}


def _print_summary(s):
    """Short 'how did it do' block at the very end of a run."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  Validation accuracy : {s['val_acc']:.2%}")
    print(f"  Top-3 / Top-5       : {s['topk'][3]:.2%} / {s['topk'][5]:.2%}")
    print(f"  Classes             : {s['num_classes']}  ({s['val_samples']} val samples)")
    print(f"  Perfect classes     : {s['perfect_classes']} (F1 = 1.00)")
    print(f"  Failed classes      : {s['failed_classes']} (F1 = 0.00)")
    print(f"  Mean confidence     : {s['mean_conf_correct']:.2f} when right, "
          f"{s['mean_conf_wrong']:.2f} when wrong")
    if s.get("replayed"):
        print("  (replayed from cache — no dataset on this machine)")
    print("=" * 60)


def _make_plots(df, cm, cm_noself, pairs, class_names, all_preds, all_labels,
                all_confs, show, history=None):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    # Training curves (only available right after a training run)
    if history and history.get("epoch"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(history["epoch"], history["train_loss"], color="tab:red")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(alpha=0.3)

        axes[1].plot(history["epoch"], history["train_acc"], label="train")
        axes[1].plot(history["epoch"], history["val_acc"], label="val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"Accuracy (best val = {max(history['val_acc']):.3f})")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=150)
        saved.append("training_curves.png")

    # Top confusion pairs
    top_n = min(20, len(pairs))
    if top_n:
        top_pairs = pairs[:top_n]
        labels = [f"{t} → {p}" for _, t, p in top_pairs]
        values = [count for count, _, _ in top_pairs]

        fig = plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), values)
        plt.yticks(range(top_n), labels, fontsize=9)
        plt.xlabel("Number of Misclassifications")
        plt.ylabel("True → Predicted")
        plt.title("Top 20 Sign Confusion Pairs")
        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "confusion_pairs.png", dpi=150)
        saved.append("confusion_pairs.png")

    # Worst-performing classes by F1
    fig = plt.figure(figsize=(10, 8))
    worst30 = df.head(30)
    plt.barh(worst30.index, worst30["f1-score"], color="crimson")
    plt.xlabel("F1 Score")
    plt.title("30 Worst-Performing Classes")
    fig.tight_layout()
    fig.savefig(out / "worst_classes_f1.png", dpi=150)
    saved.append("worst_classes_f1.png")

    # Confidence: correct vs wrong
    fig = plt.figure(figsize=(6, 4))
    plt.hist(all_confs[all_preds == all_labels], bins=30, alpha=0.6, label="Correct")
    plt.hist(all_confs[all_preds != all_labels], bins=30, alpha=0.6, label="Wrong")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Prediction Confidence: Correct vs Wrong")
    fig.tight_layout()
    fig.savefig(out / "confidence_dist.png", dpi=150)
    saved.append("confidence_dist.png")

    # Confusion matrix over the most-confused classes
    n_show = min(40, len(class_names))
    top_confused_idx = np.argsort(cm_noself.sum(axis=1))[-n_show:]
    fig = plt.figure(figsize=(14, 12))
    sns.heatmap(cm[np.ix_(top_confused_idx, top_confused_idx)],
                xticklabels=class_names[top_confused_idx],
                yticklabels=class_names[top_confused_idx],
                cmap="Blues", annot=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {n_show} Most-Confused Classes")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix_top40_normalized.png", dpi=150)
    saved.append("confusion_matrix_top40_normalized.png")

    print(f"\nSaved {len(saved)} charts to {out}:")
    for name in saved:
        print(f"  - {name}")

    if show:
        print("\nOpening charts... close the windows to finish.")
        plt.show()
    else:
        plt.close("all")
