# train.py
from __future__ import annotations

from pathlib import Path
from functools import partial
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch.multiprocessing as mp

from config import Config
from utils import set_seed, pretty_config
from data.dataloader import IEMOCAPUtteranceDataset, collate_fn
from model.model import ShaSpecEAUModel
from eval import evaluate_model 


def move_batch_to_device(batch, device):
    out = dict(batch)

    if out.get("audio") is not None:
        out["audio"] = out["audio"].to(device, non_blocking=True)

    if out.get("mocap") is not None:
        out["mocap"] = out["mocap"].to(device, non_blocking=True)
    if out.get("mocap_len") is not None:
        out["mocap_len"] = out["mocap_len"].to(device, non_blocking=True)

    # Option A video: (B,512)
    if out.get("video_embed") is not None:
        out["video_embed"] = out["video_embed"].to(device, non_blocking=True)

    # flags
    for k in ["has_audio", "has_text", "has_video", "has_mocap"]:
        if out.get(k) is not None and torch.is_tensor(out[k]):
            out[k] = out[k].to(device, non_blocking=True)

    return out


def safe_torch_save(obj, path: Path):

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(path)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    cfg,
    train: bool,
    epoch_desc: str = "",
    apply_mask=None,  # None => defaults to model behavior; bool => force
):
    model.train(train)
    total_loss = 0.0
    n = 0
    sums = {"con": 0.0, "cons": 0.0, "orth": 0.0, "task": 0.0, "sigma": 0.0}

    pbar = tqdm(loader, desc=epoch_desc, leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        losses = model.compute_losses(
            batch,
            device=device,
            train=train,
            apply_mask=apply_mask,
        )
        loss = losses["total"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

        bs = len(batch["label"])
        total_loss += float(loss.item()) * bs
        for k in sums.keys():
            if k in losses:
                sums[k] += float(losses[k]) * bs
        n += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{(total_loss/max(1,n)):.4f}"})

    avg = total_loss / max(1, n)
    avg_parts = {k: v / max(1, n) for k, v in sums.items()}
    return avg, avg_parts


def save_checkpoint(
    cfg: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    out_path: Path,
    **extra,
):
    payload = {
        "cfg": cfg.__dict__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        **extra,
    }
    safe_torch_save(payload, out_path)


def main():
    cfg = Config()
    print("CONFIG:\n" + pretty_config(cfg))
    set_seed(int(cfg.seed))

    use_cuda = torch.cuda.is_available() and getattr(cfg, "device", "cuda") == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    ckpt_metric = getattr(cfg, "ckpt_metric", "val_full")  # "val_full" or "val_masked"

    print("Loading datasets...")
    train_ds = IEMOCAPUtteranceDataset(cfg.train_manifest, cfg.data_root, cfg, split="train")
    val_ds = IEMOCAPUtteranceDataset(cfg.val_manifest, cfg.data_root, cfg, split="val")

    collate = partial(collate_fn, cfg=cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=(int(cfg.num_workers) > 0),
        prefetch_factor=2 if int(cfg.num_workers) > 0 else None,
        drop_last=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=(int(cfg.num_workers) > 0),
        prefetch_factor=2 if int(cfg.num_workers) > 0 else None,
        drop_last=False,
        collate_fn=collate,
    )

    model = ShaSpecEAUModel(cfg).to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    out_dir = Path(getattr(cfg, "ckpt_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # evaluation output root
    eval_root = Path(getattr(cfg, "eval_dir", "evaluation_results"))
    eval_root.mkdir(parents=True, exist_ok=True)

    # evaluation history (lightweight summary only)
    eval_history_path = eval_root / "evaluation_history.json"
    if eval_history_path.exists():
        try:
            eval_history = json.loads(eval_history_path.read_text())
            if not isinstance(eval_history, list):
                eval_history = []
        except Exception:
            eval_history = []
    else:
        eval_history = []

    best_score = float("inf")

    eval_frequency = int(getattr(cfg, "eval_frequency", 5))
    save_every_epoch = bool(getattr(cfg, "save_every_epoch", True))

    print("Starting training...")
    with tqdm(total=int(cfg.epochs), desc="Training", position=0) as epoch_pbar:
        for epoch in range(1, int(cfg.epochs) + 1):

            tr_loss, tr_parts = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                train=True,
                apply_mask=True,
                epoch_desc=f"Epoch {epoch}/{cfg.epochs} [train]",
            )


            val_full, val_full_parts = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                train=False,
                apply_mask=False,
                epoch_desc=f"Epoch {epoch}/{cfg.epochs} [val_full]",
            )

            val_masked, val_masked_parts = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                train=False,
                apply_mask=True,
                epoch_desc=f"Epoch {epoch}/{cfg.epochs} [val_masked]",
            )

            current_score = val_full if ckpt_metric == "val_full" else val_masked

            if save_every_epoch:
                epoch_ckpt = out_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    out_path=epoch_ckpt,
                    val_full=float(val_full),
                    val_masked=float(val_masked),
                    ckpt_metric=str(ckpt_metric),
                )

            if current_score < best_score:
                best_score = float(current_score)
                best_ckpt = out_dir / "best.pt"
                save_checkpoint(
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    out_path=best_ckpt,
                    best_score=float(best_score),
                    val_full=float(val_full),
                    val_masked=float(val_masked),
                    ckpt_metric=str(ckpt_metric),
                )
                tqdm.write(f"  saved: {best_ckpt} (best {ckpt_metric}: {best_score:.4f})")

            if (epoch % eval_frequency == 0):

                model_path = out_dir / "best.pt"
                epoch_eval_dir = eval_root / f"epoch_{epoch:03d}"
                epoch_eval_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n{'='*60}")
                print(f"COMPREHENSIVE EVALUATION (eval.py) @ epoch {epoch}")
                print(f"Checkpoint: {model_path}")
                print(f"Output: {epoch_eval_dir}")
                print("=" * 60)

                results = evaluate_model(
                    model_path=model_path,
                    cfg=cfg,
                    output_dir=epoch_eval_dir,
                    val_loader=val_loader,
                    test_loader=None,
                )
                summary = {
                    "epoch": int(epoch),
                    "ckpt": str(model_path),
                    "val_loss_full": float(val_full),
                    "val_loss_masked": float(val_masked),
                    "best_score": float(best_score),
                    "ckpt_metric": str(ckpt_metric),
                    "metrics": {
                        "uar": float(results["validation"]["uar"]),
                        "uf1": float(results["validation"]["uf1"]),
                        "accuracy": float(results["validation"]["accuracy"]),
                        "weighted_f1": float(results["validation"]["weighted_f1"]),
                        "macro_f1": float(results["validation"]["macro_f1"]),
                        "ece": float(results["validation"].get("ece", 0.0)),
                    },
                    "retrieval": {k: float(v) for k, v in results.get("retrieval", {}).items() if "std" not in k},
                }
                eval_history.append(summary)
                eval_history_path.write_text(json.dumps(eval_history, indent=2))

                print("eval.py evaluation complete.\n")

            epoch_pbar.update(1)
            epoch_pbar.set_postfix(
                {
                    "tr": f"{tr_loss:.4f}",
                    "vf": f"{val_full:.4f}",
                    "vm": f"{val_masked:.4f}",
                    "best": f"{best_score:.4f}",
                    "ckpt": str(ckpt_metric),
                }
            )

            tqdm.write(
                f"Epoch {epoch:02d} | "
                f"train {tr_loss:.4f} (con {tr_parts['con']:.3f} cons {tr_parts['cons']:.3f} orth {tr_parts['orth']:.3f} task {tr_parts['task']:.3f} sigma {tr_parts['sigma']:.4f}) | "
                f"val_full {val_full:.4f} (con {val_full_parts['con']:.3f} cons {val_full_parts['cons']:.3f} orth {val_full_parts['orth']:.3f} task {val_full_parts['task']:.3f} sigma {val_full_parts['sigma']:.4f}) | "
                f"val_masked {val_masked:.4f} (con {val_masked_parts['con']:.3f} cons {val_masked_parts['cons']:.3f} orth {val_masked_parts['orth']:.3f} task {val_masked_parts['task']:.3f} sigma {val_masked_parts['sigma']:.4f})"
            )


if __name__ == "__main__":
    mp.freeze_support()
    main()
