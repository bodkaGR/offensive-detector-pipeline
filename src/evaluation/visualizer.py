from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

plt.rcParams.update({
    "figure.dpi": 120, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})


class TrainingVisualizer:
    """Visualisation of learning outcomes and assessment"""

    def __init__(self, plots_dir: str):
        self._plots_dir = plots_dir

    def _save(self, figure: plt.Figure, name: str) -> str:
        path = os.path.join(self._plots_dir, name)
        figure.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(figure)
        return path

    def plot_training_curves(self, history: dict, title: str = "Learning") -> str:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for ax, (train_k, val_k, ylabel) in zip(axes, [
            ("train_loss", "val_loss", "Loss"),
            ("train_f1", "val_f1", "F1 (weighted)"),
            ("train_auc", "val_auc", "ROC-AUC"),
        ]):
            epochs = range(1, len(history[train_k]) + 1)
            ax.plot(epochs, history[train_k], label="Train", color="#3498db", lw=2)
            ax.plot(epochs, history[val_k], label="Val", color="#e74c3c", lw=2, ls="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend()

        fig.tight_layout()
        path = self._save(fig, "training_curves.png")

        if "lr" in history and history["lr"]:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.plot(history["lr"], color="#9b59b6", lw=2)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("LR")
            ax2.set_title("Learning Rate Schedule")
            fig2.tight_layout()
            self._save(fig2, "lr_schedule.png")

        return path

    def plot_confusion_matrix(self, cm: np.ndarray, title: str = "Confusion matrix") -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ["clean", "offensive"]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[0], linewidths=0.5)
        axes[0].set_title(f"{title} (абсолютні)")
        axes[0].set_ylabel("Справжні")
        axes[0].set_xlabel("Передбачені")

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="RdYlGn",
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
        axes[1].set_title(f"{title} (нормалізована)")
        axes[1].set_ylabel("Справжні")
        axes[1].set_xlabel("Передбачені")

        fig.tight_layout()
        return self._save(fig, "confusion_matrix.png")

    def plot_roc_pr(self, labels: np.ndarray, probs: np.ndarray, name: str = "Model") -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc_val = auc(fpr, tpr)

        ax1.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC={roc_auc_val:.4f}")
        ax1.plot([0, 1], [0, 1], color="#95a5a6", lw=1, ls="--")
        ax1.fill_between(fpr, tpr, alpha=0.1, color="#3498db")
        ax1.set(xlabel="FPR", ylabel="TPR", title=f"ROC - {name}")
        ax1.legend(loc="lower right")

        prec, rec, _ = precision_recall_curve(labels, probs)
        pr_val = auc(rec, prec)
        ax2.plot(rec, prec, color="#e74c3c", lw=2, label=f"PR-AUC={pr_val:.4f}")
        ax2.axhline(labels.mean(), color="#95a5a6", lw=1, ls="--",
                    label=f"Baseline={labels.mean():.3f}")
        ax2.fill_between(rec, prec, alpha=0.1, color="#e74c3c")
        ax2.set(xlabel="Recall", ylabel="Precision", title=f"PR - {name}")
        ax2.legend(loc="upper right")

        fig.tight_layout()
        return self._save(fig, f"roc_pr_{name.replace(' ', '_')}.png")

    def plot_probability_distribution(self, labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5,) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))

        for lbl, color, name in [(0, "#2ecc71", "Clean"), (1, "#e74c3c", "Offensive")]:
            mask = labels == lbl
            ax.hist(probs[mask], bins=60, alpha=0.6, color=color,
                    label=f"{name} (n={mask.sum():,})", density=True)
        ax.axvline(threshold, color="#2c3e50", lw=2, ls="--",
                   label=f"Threshold={threshold:.2f}")
        ax.set(xlabel="P(offensive)", ylabel="Density",
               title="Probability distribution by class")
        ax.legend()

        fig.tight_layout()
        return self._save(fig, "probability_distribution.png")