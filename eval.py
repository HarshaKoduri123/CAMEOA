# eval.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from model.model import ShaSpecEAUModel
from torch.utils.data import DataLoader
from functools import partial
from data.dataloader import IEMOCAPUtteranceDataset, collate_fn

from utils import (
    to_jsonable,
    compute_metrics,
    retrieval_metrics_from_embeddings,
    VisualizationGenerator,
    emotion_name,
)

warnings.filterwarnings("ignore")


class EmotionEvaluator:

    def __init__(self, model: ShaSpecEAUModel, cfg: Config, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.inv_label_map = {v: k for k, v in cfg.label_map.items()}

    def _move_batch(self, batch: Dict) -> Dict:
        for k in ["audio", "video_embed", "mocap", "mocap_len", "has_audio", "has_text", "has_video", "has_mocap"]:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device, non_blocking=True)
        return batch

    def _apply_modality_mask(self, batch: Dict, mask_type: str) -> Dict:
        if not mask_type or mask_type == "all_modalities":
            return batch

        def zlike(x): return torch.zeros_like(x)
        def olike(x): return torch.ones_like(x)

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

    @torch.no_grad()
    def extract_embeddings_and_predictions(self, dataloader, modality_mask: Optional[str] = None) -> Dict:
        self.model.eval()
        embs, labs, preds, probs = [], [], [], []
        utt_ids, mod_flags = [], []

        for batch in tqdm(dataloader, desc=f"Extracting ({modality_mask})"):
            batch = self._move_batch(batch)
            if modality_mask:
                batch = self._apply_modality_mask(batch, modality_mask)

            h = self.model.encode(batch, self.device)
            reps = self.model.shared_private(h)

            hasA = batch["has_audio"].float()
            hasT = batch["has_text"].float()
            hasV = batch["has_video"].float()
            hasM = batch["has_mocap"].float()
            full_mask = torch.stack([hasA, hasT, hasV, hasM], dim=1)

            z = self.model.fuse(reps, full_mask)
            logits = self.model.classifier(z)
            p = F.softmax(logits, dim=-1)
            pr = logits.argmax(dim=-1)

            y = torch.tensor([self.cfg.label_map.get(str(l), -1) for l in batch["label"]], device=self.device)
            valid = y >= 0
            if valid.sum() == 0:
                continue

            embs.append(z[valid].detach().cpu())
            labs.append(y[valid].detach().cpu())
            preds.append(pr[valid].detach().cpu())
            probs.append(p[valid].detach().cpu())

            if "utt_id" in batch:
                utt_ids.extend([batch["utt_id"][i] for i in range(len(batch["utt_id"])) if bool(valid[i])])

            flags = torch.stack([hasA, hasT, hasV, hasM], dim=1)[valid].detach().cpu()
            mod_flags.append(flags)

        out = {
            "embeddings": torch.cat(embs, dim=0),
            "labels": torch.cat(labs, dim=0),
            "predictions": torch.cat(preds, dim=0),
            "probabilities": torch.cat(probs, dim=0),
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
            ("drop_mocap", "Missing Mocap"),
            ("audio_only", "Audio Only"),
            ("text_only", "Text Only"),
            ("video_only", "Video Only"),
            ("mocap_only", "Mocap Only"),
        ]

        results = {}
        for mask_type, display_name in scenarios:
            data = self.extract_embeddings_and_predictions(dataloader, modality_mask=mask_type)
            m = compute_metrics(data["labels"], data["predictions"], data["probabilities"], inv_label_map=self.inv_label_map)
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

    model = ShaSpecEAUModel(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    evaluator = EmotionEvaluator(model, cfg, device)
    viz = VisualizationGenerator(output_dir / "figures")

    results: Dict = {
        "model_path": str(model_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "config": to_jsonable(cfg.__dict__),
    }

    # ---- Validation ----
    val_data = evaluator.extract_embeddings_and_predictions(val_loader)
    val_metrics = compute_metrics(
        val_data["labels"],
        val_data["predictions"],
        val_data["probabilities"],
        inv_label_map=evaluator.inv_label_map,
    )
    results["validation"] = val_metrics

    # ---- Retrieval ----
    retrieval = retrieval_metrics_from_embeddings(val_data["embeddings"], val_data["labels"])
    results["retrieval"] = retrieval
    for k, v in retrieval.items():
        if "std" not in k:
            print(f"{k}: {v:.4f}")

    # ---- Robustness ----
    robustness = evaluator.evaluate_modality_robustness(val_loader)
    results["robustness"] = {k: {"uar": v["uar"], "uf1": v["uf1"], "accuracy": v["accuracy"]} for k, v in robustness.items()}

    # ---- Plots ----
    viz.plot_umap(val_data["embeddings"], val_data["labels"], title="UMAP Projection (Validation)")
    viz.plot_tsne(val_data["embeddings"], val_data["labels"], title="t-SNE Visualization (Validation)")
    viz.plot_confusion_matrix(
        np.array(val_metrics["confusion_matrix"]),
        [emotion_name(i) for i in range(len(cfg.label_map))],
        title="Confusion Matrix (Validation)",
    )
    viz.plot_modality_robustness(robustness)
    viz.plot_per_class_performance(val_metrics.get("per_class", {}))
    viz.plot_confidence_calibration(val_data["labels"], val_data["probabilities"])

    # ---- Save JSON ----
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(to_jsonable(results), f, indent=2)

    # ---- Save text summary ----
    with open(output_dir / "evaluation_summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EMOTION RECOGNITION EVALUATION SUMMARY\n")
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
