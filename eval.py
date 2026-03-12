from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json
import warnings
import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import umap


from config import Config
from model.model import CAMEOModel
from torch.utils.data import DataLoader
from functools import partial
from data.dataloader import IEMOCAPUtteranceDataset, collate_fn

from utils import (
    to_jsonable,
    compute_metrics,
    retrieval_metrics_from_embeddings,
    emotion_name,
)

warnings.filterwarnings("ignore")

CLASS_COLORS = [
    "#4E79A7",  
    "#F28E2B",
    "#59A14F",  
    "#E15759", 
]
CLASS_CMAP = ListedColormap(CLASS_COLORS)

class EvaluationPlotter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _labels_to_names(self, y: np.ndarray):
        return [emotion_name(int(i)) for i in y]

    def plot_umap(self, embeddings, labels, title="UMAP Projection"):


        X = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)
        y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        Z = reducer.fit_transform(X)

        plt.figure(figsize=(8, 6))
        for cls_idx in sorted(np.unique(y)):
            idx = y == cls_idx
            plt.scatter(
                Z[idx, 0],
                Z[idx, 1],
                s=22,
                alpha=0.85,
                c=CLASS_COLORS[int(cls_idx)],
                label=emotion_name(int(cls_idx)),
                edgecolors="white",
                linewidths=0.35,
            )

        plt.title(title, fontsize=13)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend(frameon=True)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(self.out_dir / "umap_validation.png", dpi=220, bbox_inches="tight")
        plt.close()

    def plot_tsne(self, embeddings, labels, title="t-SNE Projection"):
        X = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)
        y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(X) // 10)),
            learning_rate="auto",
            init="pca",
            random_state=42,
        )
        Z = tsne.fit_transform(X)

        plt.figure(figsize=(8, 6))
        for cls_idx in sorted(np.unique(y)):
            idx = y == cls_idx
            plt.scatter(
                Z[idx, 0],
                Z[idx, 1],
                s=22,
                alpha=0.85,
                c=CLASS_COLORS[int(cls_idx)],
                label=emotion_name(int(cls_idx)),
                edgecolors="white",
                linewidths=0.35,
            )

        plt.title(title, fontsize=13)
        plt.xlabel("t-SNE-1")
        plt.ylabel("t-SNE-2")
        plt.legend(frameon=True)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(self.out_dir / "tsne_validation.png", dpi=220, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, class_names, title="Confusion Matrix"):
        plt.figure(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", values_format="d", colorbar=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.out_dir / "confusion_matrix_validation.png", dpi=220, bbox_inches="tight")
        plt.close()

    def plot_modality_robustness(self, robustness: Dict):
        names = [v["display_name"] for v in robustness.values()]
        uars = [v["uar"] for v in robustness.values()]

        plt.figure(figsize=(10, 5.5))
        bars = plt.bar(range(len(names)), uars)
        for i, b in enumerate(bars):
            b.set_alpha(0.9)

        # softer but distinct bar coloring
        bar_colors = [
            "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
            "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"
        ]
        for bar, c in zip(bars, bar_colors):
            bar.set_color(c)

        plt.xticks(range(len(names)), names, rotation=25, ha="right")
        plt.ylabel("UAR")
        plt.title("Robustness Under Missing-Modality Conditions")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(self.out_dir / "modality_robustness.png", dpi=220, bbox_inches="tight")
        plt.close()

    def plot_per_class_performance(self, per_class: Dict):
        if not per_class:
            return

        class_names = list(per_class.keys())
        f1_scores = [per_class[k]["f1"] for k in class_names]

        plt.figure(figsize=(7, 5))
        bars = plt.bar(class_names, f1_scores, color=CLASS_COLORS[:len(class_names)], alpha=0.9)
        plt.ylabel("F1 Score")
        plt.title("Per-Class F1 Scores")
        plt.ylim(0, 1.0)
        plt.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, f1_scores):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(self.out_dir / "per_class_f1.png", dpi=220, bbox_inches="tight")
        plt.close()

    def plot_confidence_calibration(self, labels, probabilities, n_bins: int = 10):
        y_true = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        probs = probabilities.detach().cpu().numpy() if torch.is_tensor(probabilities) else np.asarray(probabilities)

        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correctness = (predictions == y_true).astype(np.float32)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(confidences, bins) - 1
        bin_acc, bin_conf, counts = [], [], []

        for b in range(n_bins):
            idx = bin_ids == b
            if np.sum(idx) == 0:
                bin_acc.append(0.0)
                bin_conf.append((bins[b] + bins[b + 1]) / 2)
                counts.append(0)
            else:
                bin_acc.append(float(np.mean(correctness[idx])))
                bin_conf.append(float(np.mean(confidences[idx])))
                counts.append(int(np.sum(idx)))

        plt.figure(figsize=(7, 6))
        x = np.linspace(0, 1, n_bins)
        width = 1.0 / n_bins * 0.9
        plt.bar(x, bin_acc, width=width, alpha=0.85, color="#4E79A7", label="Accuracy")
        plt.plot([0, 1], [0, 1], "--", color="#E15759", linewidth=1.5, label="Perfect calibration")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Confidence Calibration")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(self.out_dir / "confidence_calibration.png", dpi=220, bbox_inches="tight")
        plt.close()


# =========================================================
# Evaluator
# =========================================================
class EmotionEvaluator:
    def __init__(self, model: CAMEOModel, cfg: Config, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.inv_label_map = {v: k for k, v in cfg.label_map.items()}

    def _move_batch(self, batch: Dict) -> Dict:
        for k in [
            "audio", "video_embed", "mocap", "mocap_len",
            "has_audio", "has_text", "has_video", "has_mocap",
            "text_input_ids", "text_attention_mask",
            "input_ids", "attention_mask",
        ]:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device, non_blocking=True)
        return batch

    def _apply_modality_mask(self, batch: Dict, mask_type: str) -> Dict:
        if not mask_type or mask_type == "all_modalities":
            return batch

        batch = copy.deepcopy(batch)

        def zlike(x):
            return torch.zeros_like(x)

        def olike(x):
            return torch.ones_like(x)

        if mask_type == "drop_audio":
            batch["has_audio"] = zlike(batch["has_audio"])
        elif mask_type == "drop_text":
            batch["has_text"] = zlike(batch["has_text"])
        elif mask_type == "drop_video":
            batch["has_video"] = zlike(batch["has_video"])
        elif mask_type == "drop_mocap":
            batch["has_mocap"] = zlike(batch["has_mocap"])
        elif mask_type == "audio_only":
            batch["has_audio"] = olike(batch["has_audio"])
            batch["has_text"] = zlike(batch["has_text"])
            batch["has_video"] = zlike(batch["has_video"])
            batch["has_mocap"] = zlike(batch["has_mocap"])
        elif mask_type == "text_only":
            batch["has_audio"] = zlike(batch["has_audio"])
            batch["has_text"] = olike(batch["has_text"])
            batch["has_video"] = zlike(batch["has_video"])
            batch["has_mocap"] = zlike(batch["has_mocap"])
        elif mask_type == "video_only":
            batch["has_audio"] = zlike(batch["has_audio"])
            batch["has_text"] = zlike(batch["has_text"])
            batch["has_video"] = olike(batch["has_video"])
            batch["has_mocap"] = zlike(batch["has_mocap"])
        elif mask_type == "mocap_only":
            batch["has_audio"] = zlike(batch["has_audio"])
            batch["has_text"] = zlike(batch["has_text"])
            batch["has_video"] = zlike(batch["has_video"])
            batch["has_mocap"] = olike(batch["has_mocap"])

        return batch

    def _labels_to_tensor(self, labels):
        if torch.is_tensor(labels):
            return labels.to(self.device)

        y = []
        for l in labels:
            if isinstance(l, str):
                y.append(self.cfg.label_map.get(str(l), -1))
            else:
                y.append(int(l))
        return torch.tensor(y, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def extract_embeddings_and_predictions(self, dataloader, modality_mask: Optional[str] = None) -> Dict:
        self.model.eval()

        embs, labs, preds, probs = [], [], [], []
        utt_ids, mod_flags = [], []

        for batch in tqdm(dataloader, desc=f"Extracting ({modality_mask})"):
            batch = self._move_batch(batch)
            if modality_mask:
                batch = self._apply_modality_mask(batch, modality_mask)

            # CAMEO flow
            h = self.model.encode(batch, self.device)
            reps = self.model.project_distributions(h)

            hasA = batch["has_audio"].float().to(self.device)
            hasV = batch["has_video"].float().to(self.device)
            hasT = batch["has_text"].float().to(self.device)
            hasM = batch["has_mocap"].float().to(self.device)

            # IMPORTANT: order must match model modal order
            full_mask = torch.stack([hasA, hasV, hasT, hasM], dim=1)

            z = self.model.fuse(reps, full_mask)
            z_refined = self.model.refiner(z)
            logits = self.model.classifier(z_refined)

            p = F.softmax(logits, dim=-1)
            pr = logits.argmax(dim=-1)

            y = self._labels_to_tensor(batch["label"])
            valid = y >= 0
            if valid.sum() == 0:
                continue

            embs.append(z_refined[valid].detach().cpu())
            labs.append(y[valid].detach().cpu())
            preds.append(pr[valid].detach().cpu())
            probs.append(p[valid].detach().cpu())

            if "utt_id" in batch:
                utt_ids.extend([batch["utt_id"][i] for i in range(len(batch["utt_id"])) if bool(valid[i])])

            flags = torch.stack([hasA, hasV, hasT, hasM], dim=1)[valid].detach().cpu()
            mod_flags.append(flags)

        out = {
            "embeddings": torch.cat(embs, dim=0) if len(embs) else torch.empty(0),
            "labels": torch.cat(labs, dim=0) if len(labs) else torch.empty(0, dtype=torch.long),
            "predictions": torch.cat(preds, dim=0) if len(preds) else torch.empty(0, dtype=torch.long),
            "probabilities": torch.cat(probs, dim=0) if len(probs) else torch.empty(0, self.cfg.num_classes),
            "utt_ids": utt_ids,
            "modality_flags": torch.cat(mod_flags, dim=0) if len(mod_flags) else None,
        }
        return out

    def evaluate_modality_robustness(self, dataloader) -> Dict:
        scenarios = [
            ("all_modalities", "All Modalities"),
            ("drop_audio", "Missing Audio"),
            ("drop_text", "Missing Text"),
            ("drop_video", "Missing Video"),
            ("drop_mocap", "Missing MoCap"),
            ("audio_only", "Audio Only"),
            ("text_only", "Text Only"),
            ("video_only", "Video Only"),
            ("mocap_only", "MoCap Only"),
        ]

        results = {}
        for mask_type, display_name in scenarios:
            data = self.extract_embeddings_and_predictions(dataloader, modality_mask=mask_type)
            m = compute_metrics(
                data["labels"],
                data["predictions"],
                data["probabilities"],
                inv_label_map=self.inv_label_map,
            )
            results[mask_type] = {
                "display_name": display_name,
                "uar": m["uar"],
                "uf1": m["uf1"],
                "accuracy": m["accuracy"],
                "metrics": m,
            }
        return results

def evaluate_model(model_path: Path, cfg: Config, output_dir: Path, val_loader, test_loader=None) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = CAMEOModel(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    evaluator = EmotionEvaluator(model, cfg, device)
    plotter = EvaluationPlotter(output_dir / "figures")

    results: Dict = {
        "model_path": str(model_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "config": to_jsonable(cfg.__dict__),
    }

    val_data = evaluator.extract_embeddings_and_predictions(val_loader)
    val_metrics = compute_metrics(
        val_data["labels"],
        val_data["predictions"],
        val_data["probabilities"],
        inv_label_map=evaluator.inv_label_map,
    )
    results["validation"] = val_metrics


    retrieval = retrieval_metrics_from_embeddings(val_data["embeddings"], val_data["labels"])
    results["retrieval"] = retrieval
    for k, v in retrieval.items():
        if "std" not in k:
            print(f"{k}: {v:.4f}")


    robustness = evaluator.evaluate_modality_robustness(val_loader)
    results["robustness"] = {
        k: {
            "uar": v["uar"],
            "uf1": v["uf1"],
            "accuracy": v["accuracy"],
        }
        for k, v in robustness.items()
    }


    if len(val_data["embeddings"]) > 0:
        plotter.plot_umap(val_data["embeddings"], val_data["labels"], title="UMAP Projection (Validation)")
        plotter.plot_tsne(val_data["embeddings"], val_data["labels"], title="t-SNE Projection (Validation)")

    plotter.plot_confusion_matrix(
        np.array(val_metrics["confusion_matrix"]),
        [emotion_name(i) for i in range(len(cfg.label_map))],
        title="Confusion Matrix (Validation)",
    )
    plotter.plot_modality_robustness(robustness)
    plotter.plot_per_class_performance(val_metrics.get("per_class", {}))
    plotter.plot_confidence_calibration(val_data["labels"], val_data["probabilities"])


    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(to_jsonable(results), f, indent=2)


    with open(output_dir / "evaluation_summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("CAMEO EMOTION RECOGNITION EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', '?')}\n\n")

        f.write("VALIDATION METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"UAR (Primary):      {val_metrics['uar']:.4f}\n")
        f.write(f"UF1:                {val_metrics['uf1']:.4f}\n")
        f.write(f"Weighted F1:        {val_metrics['weighted_f1']:.4f}\n")
        f.write(f"Accuracy:           {val_metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1:           {val_metrics['macro_f1']:.4f}\n")
        f.write(f"Balanced Accuracy:  {val_metrics['balanced_accuracy']:.4f}\n")
        if "ece" in val_metrics:
            f.write(f"ECE:                {val_metrics['ece']:.4f}\n")

        f.write("\nRETRIEVAL METRICS:\n")
        f.write("-" * 30 + "\n")
        for k, v in retrieval.items():
            if "std" not in k:
                f.write(f"{k}: {v:.4f}\n")

        f.write("\nMODALITY ROBUSTNESS (UAR):\n")
        f.write("-" * 30 + "\n")
        for _, res in robustness.items():
            f.write(f"{res['display_name']}: {res['uar']:.4f}\n")

        f.write("\nPER-CLASS PERFORMANCE (F1):\n")
        f.write("-" * 30 + "\n")
        for cls_name, m in val_metrics.get("per_class", {}).items():
            f.write(f"{cls_name}: {m['f1']:.4f}\n")

    return results

if __name__ == "__main__":
    cfg = Config()

    val_ds = IEMOCAPUtteranceDataset(cfg.val_manifest, cfg.data_root, cfg, split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=partial(collate_fn, cfg=cfg),
    )

    evaluate_model(
        model_path=Path("checkpoints/best.pt"),
        cfg=cfg,
        output_dir=Path("evaluation_results"),
        val_loader=val_loader,
    )