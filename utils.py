# utils.py
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Metrics
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# ============================================================
# REPRO
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def pretty_config(cfg) -> str:
    d = asdict(cfg)
    lines = [f"{k}: {v}" for k, v in sorted(d.items(), key=lambda x: x[0])]
    return "\n".join(lines)

def to_jsonable(x):
    """Recursively convert objects to JSON-serializable forms."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


# ============================================================
# METRICS
# ============================================================
def emotion_name(label_idx: int) -> str:
    return {0: "Angry", 1: "Happy", 2: "Sad", 3: "Neutral"}.get(label_idx, f"Class {label_idx}")


def compute_metrics(labels, predictions, probabilities=None, inv_label_map=None):
    labels_np = labels.detach().cpu().numpy()
    preds_np  = predictions.detach().cpu().numpy()

    # Force stable label order: [0,1,2,3] (or whatever your map uses)
    if inv_label_map is not None:
        label_ids = sorted(list(inv_label_map.keys()))
    else:
        label_ids = sorted(list(set(labels_np.tolist())))

    cm = confusion_matrix(labels_np, preds_np, labels=label_ids)
    recalls = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    uar = float(np.mean(recalls))

    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels_np, preds_np, labels=label_ids, average=None, zero_division=0
    )

    metrics = {
        "uar": uar,
        "uf1": float(np.mean(f1_per_class)),
        "weighted_f1": float(f1_score(labels_np, preds_np, labels=label_ids, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "macro_f1": float(f1_score(labels_np, preds_np, labels=label_ids, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds_np)),
        "confusion_matrix": cm,
        "class_recalls": recalls.tolist(),
    }

    # Per-class dict
    per_class = {}
    if inv_label_map is not None:
        for i, lab_id in enumerate(label_ids):
            name = inv_label_map.get(lab_id, str(lab_id))
            per_class[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1_per_class[i]),
            }
    metrics["per_class"] = per_class

    # ECE (optional)
    if probabilities is not None:
        probs_np = probabilities.detach().cpu().numpy()
        max_probs = np.max(probs_np, axis=1)

        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                acc_in_bin = accuracy_score(labels_np[in_bin], preds_np[in_bin])
                conf_in_bin = float(np.mean(max_probs[in_bin]))
                ece += abs(float(acc_in_bin) - conf_in_bin) * (in_bin.sum() / len(max_probs))
        metrics["ece"] = float(ece)

    return metrics



# ============================================================
# RETRIEVAL
# ============================================================
@torch.no_grad()
def retrieval_metrics_from_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, ks=(1, 5, 10)) -> Dict:
    """
    Retrieval on embeddings:
      - R@k, R@k_std
      - MRR
    Positives: same label, excludes self.
    """
    Z = F.normalize(embeddings, dim=-1)
    Y = labels

    sim = Z @ Z.t()
    sim.fill_diagonal_(-1e9)

    ranks = sim.argsort(dim=1, descending=True)

    out = {}
    for k in ks:
        topk = ranks[:, :k]
        hit = (Y[topk] == Y[:, None]).any(dim=1).float()
        out[f"r@{k}"] = float(hit.mean().item())
        out[f"r@{k}_std"] = float(hit.std().item())

    # MRR
    mrr_sum = 0.0
    n = int(len(Y))
    for i in range(n):
        same = (Y == Y[i])
        same[i] = False
        if not bool(same.any()):
            continue
        pos = torch.where(same)[0]
        # rank positions (1-indexed)
        pos_ranks = (ranks[i][:, None] == pos).nonzero(as_tuple=True)[1] + 1
        if len(pos_ranks) > 0:
            mrr_sum += float((1.0 / pos_ranks.min().float()).item())
    out["mrr"] = float(mrr_sum / max(1, n))
    return out


# ============================================================
# VISUALIZATION
# ============================================================
class VisualizationGenerator:
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.figsize"] = (10, 8)
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["legend.fontsize"] = 10

    def _get_emotion_name(self, label_idx: int) -> str:
        return emotion_name(label_idx)

    def plot_umap(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        title: str = "UMAP Projection of Emotion Embeddings",
        filename: str = "umap.png",
    ):
        X = embeddings.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()

        emotion_colors = {
            0: "#8B0000",  # Angry - Dark Red
            1: "#FFD700",  # Happy - Gold
            2: "#000080",  # Sad - Navy
            3: "#006400",  # Neutral - Dark Green
        }
        colors = [emotion_colors.get(int(i), "#95A5A6") for i in y]

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        X_umap = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            c=colors,
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        legend_elements = []
        for lab, col in emotion_colors.items():
            if lab in np.unique(y):
                legend_elements.append(
                    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=10, label=self._get_emotion_name(lab))
                )
        ax.legend(handles=legend_elements, loc="best", frameon=True)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_tsne(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        title: str = "t-SNE Visualization of Emotion Embeddings",
        filename: str = "tsne.png",
    ):
        X = embeddings.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()

        pca = PCA(n_components=min(50, X.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X)

        tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric="cosine")
        X_tsne = tsne.fit_transform(X_pca)

        emotion_colors = {0: "#8B0000", 1: "#FFD700", 2: "#000080", 3: "#006400"}
        colors = [emotion_colors.get(int(i), "#95A5A6") for i in y]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=50, alpha=0.7, edgecolors="black", linewidth=0.5)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png",
    ):
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            title=title,
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                )

        fig.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_modality_robustness(self, robustness_results: Dict, filename: str = "modality_robustness.png"):
        scenarios, uars, uf1s = [], [], []
        for _, res in robustness_results.items():
            scenarios.append(res["display_name"])
            uars.append(res["uar"])
            uf1s.append(res["uf1"])

        x = np.arange(len(scenarios))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 8))
        b1 = ax.bar(x - width / 2, uars, width, label="UAR", alpha=0.8)
        b2 = ax.bar(x + width / 2, uf1s, width, label="UF1", alpha=0.8)

        ax.set_xlabel("Modality Condition")
        ax.set_ylabel("Score")
        ax.set_title("Model Robustness Under Missing Modalities", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1])

        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_per_class_performance(self, per_class_metrics: Dict, filename: str = "per_class_performance.png"):
        classes = list(per_class_metrics.keys())
        precisions = [m["precision"] for m in per_class_metrics.values()]
        recalls = [m["recall"] for m in per_class_metrics.values()]
        f1s = [m["f1"] for m in per_class_metrics.values()]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width, precisions, width, label="Precision", alpha=0.8)
        ax.bar(x, recalls, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1s, width, label="F1-Score", alpha=0.8)

        ax.set_xlabel("Emotion Class")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Performance Metrics", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_confidence_calibration(self, labels: torch.Tensor, probs: torch.Tensor, filename: str = "calibration.png"):
        labels_np = labels.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        preds_np = probs_np.argmax(axis=1)
        conf = probs_np.max(axis=1)

        n_bins = 10
        bounds = np.linspace(0, 1, n_bins + 1)
        centers = (bounds[:-1] + bounds[1:]) / 2

        accs, counts = [], []
        for i in range(n_bins):
            in_bin = (conf > bounds[i]) & (conf <= bounds[i + 1])
            counts.append(int(in_bin.sum()))
            if in_bin.sum() > 0:
                accs.append(float(accuracy_score(labels_np[in_bin], preds_np[in_bin])))
            else:
                accs.append(0.0)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
        ax.plot(centers, accs, "o-", linewidth=2, markersize=8, label="Model")

        ax2 = ax.twinx()
        ax2.bar(centers, counts, width=0.09, alpha=0.3, label="Sample Count")
        ax2.set_ylabel("Number of Samples")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Calibration Plot (Reliability Diagram)", fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
