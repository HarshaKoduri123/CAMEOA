from __future__ import annotations

from pathlib import Path
from functools import partial
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import torch.multiprocessing as mp

from config import Config
from utils import set_seed, pretty_config
from data.dataloader import IEMOCAPUtteranceDataset, collate_fn
from model.model import CAMEOModel
from eval import evaluate_model


def move_batch_to_device(batch, device):
    out = dict(batch)
    for key in ["audio", "mocap", "mocap_len", "video_embed"]:
        if out.get(key) is not None and torch.is_tensor(out[key]):
            out[key] = out[key].to(device, non_blocking=True)

    for key in ["has_audio", "has_text", "has_video", "has_mocap"]:
        if out.get(key) is not None and torch.is_tensor(out[key]):
            out[key] = out[key].to(device, non_blocking=True)

    if "text_input_ids" in out and torch.is_tensor(out["text_input_ids"]):
        out["text_input_ids"] = out["text_input_ids"].to(device, non_blocking=True)
    if "text_attention_mask" in out and torch.is_tensor(out["text_attention_mask"]):
        out["text_attention_mask"] = out["text_attention_mask"].to(device, non_blocking=True)

    if "input_ids" in out and torch.is_tensor(out["input_ids"]):
        out["input_ids"] = out["input_ids"].to(device, non_blocking=True)
    if "attention_mask" in out and torch.is_tensor(out["attention_mask"]):
        out["attention_mask"] = out["attention_mask"].to(device, non_blocking=True)

    return out


def safe_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(path)


def save_checkpoint(cfg, model, optimizer, epoch, out_path, **extra):
    safe_torch_save(
        {
            "cfg": cfg.__dict__,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            **extra,
        },
        out_path,
    )


def metric_is_higher_better(name: str) -> bool:
    return name.lower() != "loss"


def get_ckpt_score(ckpt_metric, val_loss, eval_results):
    m = ckpt_metric.lower()
    if m == "loss":
        return float(val_loss)
    if eval_results is None:
        return None
    v = eval_results["validation"]
    mapping = {
        "uar": "uar",
        "uf1": "uf1",
        "accuracy": "accuracy",
        "weighted_f1": "weighted_f1",
        "macro_f1": "macro_f1",
    }
    if m in mapping:
        return float(v[mapping[m]])
    return None


def run_epoch(model, loader, optimizer, device, cfg, train: bool, epoch_desc: str = "", apply_mask: bool = False):
    model.train(train)

    total_loss = 0.0
    total_align = 0.0
    total_kl = 0.0
    total_cls = 0.0
    n = 0

    pbar = tqdm(loader, desc=epoch_desc, leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        losses = model.compute_losses(batch, device=device, train=train, apply_mask=apply_mask)
        loss = losses["total"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

        bs = len(batch["label"])
        total_loss += float(loss.item()) * bs
        total_align += float(losses["align"].item()) * bs
        total_kl += float(losses["kl"].item()) * bs
        total_cls += float(losses["cls"].item()) * bs
        n += bs

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{total_loss / max(1, n):.4f}",
        })

    return {
        "loss": total_loss / max(1, n),
        "align": total_align / max(1, n),
        "kl": total_kl / max(1, n),
        "cls": total_cls / max(1, n),
    }


def main():
    cfg = Config()
    print("CONFIG:\n" + pretty_config(cfg))
    set_seed(int(cfg.seed))

    use_cuda = torch.cuda.is_available() and str(cfg.device).lower() == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    ckpt_metric = str(cfg.ckpt_metric)

    print("Loading datasets...")
    train_ds = IEMOCAPUtteranceDataset(cfg.train_manifest, cfg.data_root, cfg, split="train")
    val_ds = IEMOCAPUtteranceDataset(cfg.val_manifest, cfg.data_root, cfg, split="val")
    collate = partial(collate_fn, cfg=cfg)

    loader_kwargs = dict(
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=(int(cfg.num_workers) > 0),
        prefetch_factor=2 if int(cfg.num_workers) > 0 else None,
        collate_fn=collate,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model = CAMEOModel(cfg).to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    scheduler_type = str(getattr(cfg, "scheduler", "plateau")).lower()
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.epochs),
            eta_min=float(getattr(cfg, "min_lr", 1e-6)),
        )
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max" if metric_is_higher_better(ckpt_metric) else "min",
            factor=float(getattr(cfg, "plateau_factor", 0.5)),
            patience=int(getattr(cfg, "plateau_patience", 3)),
            min_lr=float(getattr(cfg, "min_lr", 1e-6)),
        )
    else:
        scheduler = None

    out_dir = Path(cfg.ckpt_dir)
    eval_root = Path(cfg.eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    eval_history_path = eval_root / "evaluation_history.json"
    try:
        eval_history = json.loads(eval_history_path.read_text()) if eval_history_path.exists() else []
        if not isinstance(eval_history, list):
            eval_history = []
    except Exception:
        eval_history = []

    higher_better = metric_is_higher_better(ckpt_metric)
    best_score = float("-inf") if higher_better else float("inf")
    bad_epochs = 0

    print("Starting training...")
    with tqdm(total=int(cfg.epochs), desc="Training", position=0) as epoch_pbar:
        for epoch in range(1, int(cfg.epochs) + 1):
            train_stats = run_epoch(
                model, train_loader, optimizer, device, cfg,
                train=True,
                apply_mask=True,
                epoch_desc=f"Epoch {epoch}/{cfg.epochs} [train]",
            )

            val_stats = run_epoch(
                model, val_loader, optimizer, device, cfg,
                train=False,
                apply_mask=False,
                epoch_desc=f"Epoch {epoch}/{cfg.epochs} [val]",
            )

            epoch_ckpt = out_dir / f"epoch_{epoch:03d}.pt"
            if bool(cfg.save_every_epoch):
                save_checkpoint(
                    cfg, model, optimizer, epoch, epoch_ckpt,
                    val_loss=float(val_stats["loss"]),
                )

            eval_results = None
            if epoch % int(cfg.eval_frequency) == 0:
                epoch_eval_dir = eval_root / f"epoch_{epoch:03d}"
                epoch_eval_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n{'='*60}")
                print(f"COMPREHENSIVE EVALUATION @ epoch {epoch}")
                print(f"  ckpt : {epoch_ckpt}")
                print(f"  out  : {epoch_eval_dir}")
                print("=" * 60)

                eval_results = evaluate_model(
                    model_path=epoch_ckpt,
                    cfg=cfg,
                    output_dir=epoch_eval_dir,
                    val_loader=val_loader,
                    test_loader=None,
                )

            current_score = get_ckpt_score(ckpt_metric, val_stats["loss"], eval_results)
            did_improve = False

            if current_score is not None:
                did_improve = (current_score > best_score) if higher_better else (current_score < best_score)

            if did_improve:
                best_score = float(current_score)
                best_ckpt = out_dir / "best.pt"
                save_checkpoint(
                    cfg, model, optimizer, epoch, best_ckpt,
                    best_score=float(best_score),
                    val_loss=float(val_stats["loss"]),
                    ckpt_metric=ckpt_metric,
                    eval_results=eval_results,
                )
                tqdm.write(f"saved best.pt ({ckpt_metric}: {best_score:.4f})")
                bad_epochs = 0
            else:
                if current_score is not None:
                    bad_epochs += 1

            if eval_results is not None:
                summary = {
                    "epoch": int(epoch),
                    "ckpt": str(epoch_ckpt),
                    "val_loss": float(val_stats["loss"]),
                    "best_score": float(best_score),
                    "ckpt_metric": ckpt_metric,
                    "metrics": {
                        "uar": float(eval_results["validation"]["uar"]),
                        "uf1": float(eval_results["validation"]["uf1"]),
                        "accuracy": float(eval_results["validation"]["accuracy"]),
                        "weighted_f1": float(eval_results["validation"]["weighted_f1"]),
                        "macro_f1": float(eval_results["validation"]["macro_f1"]),
                        "ece": float(eval_results["validation"].get("ece", 0.0)),
                    },
                    "retrieval": {
                        k: float(v)
                        for k, v in eval_results.get("retrieval", {}).items()
                        if "std" not in k
                    },
                    "train_losses": {
                        "loss": round(train_stats["loss"], 6),
                        "align": round(train_stats["align"], 6),
                        "kl": round(train_stats["kl"], 6),
                        "cls": round(train_stats["cls"], 6),
                    },
                    "val_losses": {
                        "loss": round(val_stats["loss"], 6),
                        "align": round(val_stats["align"], 6),
                        "kl": round(val_stats["kl"], 6),
                        "cls": round(val_stats["cls"], 6),
                    },
                }
                eval_history.append(summary)
                eval_history_path.write_text(json.dumps(eval_history, indent=2))
                print("Eval complete.\n")

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(current_score if current_score is not None else float(val_stats["loss"]))
                else:
                    scheduler.step()

            if int(cfg.early_stop_patience) > 0 and bad_epochs >= int(cfg.early_stop_patience):
                print(f"Early stopping: no improvement for {bad_epochs} epochs.")
                break

            lr_now = optimizer.param_groups[0]["lr"]
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                "lr": f"{lr_now:.2e}",
                "tr": f"{train_stats['loss']:.4f}",
                "va": f"{val_stats['loss']:.4f}",
                "best": f"{best_score:.4f}",
                "m": ckpt_metric,
            })

            tqdm.write(
                f"Epoch {epoch:02d} | lr {lr_now:.2e} | "
                f"train {train_stats['loss']:.4f} "
                f"(align {train_stats['align']:.4f} kl {train_stats['kl']:.4f} cls {train_stats['cls']:.4f}) | "
                f"val {val_stats['loss']:.4f} "
                f"(align {val_stats['align']:.4f} kl {val_stats['kl']:.4f} cls {val_stats['cls']:.4f})"
            )


if __name__ == "__main__":
    mp.freeze_support()
    main()