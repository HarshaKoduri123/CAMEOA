# train.py  (UPDATED v2)
# Key fixes vs v1:
#   1. eval_history logs the CORRECT best_score (not the pre-update value)
#   2. run_epoch sums track all new loss components (con_cross, con_hub, dao, dco, prior)
#   3. tqdm log line updated to show new components
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
from model.model import ShaSpecEAUModel
from eval import evaluate_model


def move_batch_to_device(batch, device):
    out = dict(batch)
    for key in ["audio", "mocap", "mocap_len", "video_embed"]:
        if out.get(key) is not None:
            out[key] = out[key].to(device, non_blocking=True)
    for key in ["has_audio", "has_text", "has_video", "has_mocap"]:
        if out.get(key) is not None and torch.is_tensor(out[key]):
            out[key] = out[key].to(device, non_blocking=True)
    return out


def safe_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(path)



_LOSS_KEYS = ["con", "con_cross", "con_hub", "cons", "orth", "task",
              "dao", "dco", "prior", "sigma", "speaker"]


def run_epoch(model, loader, optimizer, device, cfg, train: bool,
              epoch_desc: str = "", apply_mask=None):
    model.train(train)
    total_loss = 0.0
    n = 0
    sums = {k: 0.0 for k in _LOSS_KEYS}

    pbar = tqdm(loader, desc=epoch_desc, leave=False)
    for batch in pbar:
        batch  = move_batch_to_device(batch, device)
        losses = model.compute_losses(batch, device=device, train=train, apply_mask=apply_mask)
        loss   = losses["total"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

        bs          = len(batch["label"])
        total_loss += float(loss.item()) * bs
        for k in sums:
            if k in losses:
                sums[k] += float(losses[k]) * bs
        n += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "avg":  f"{total_loss / max(1, n):.4f}"})

    avg       = total_loss / max(1, n)
    avg_parts = {k: v / max(1, n) for k, v in sums.items()}
    return avg, avg_parts


def save_checkpoint(cfg, model, optimizer, epoch, out_path, **extra):
    safe_torch_save(
        {
            "cfg":       cfg.__dict__,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch":     int(epoch),
            **extra,
        },
        out_path,
    )


def metric_is_higher_better(name: str) -> bool:
    name = name.lower()
    if name in {"val_full", "val_masked", "loss", "ece"}:
        return False
    return True  

def get_ckpt_score(ckpt_metric, val_full, val_masked, eval_results):
    m = ckpt_metric.lower()
    if m == "val_full":
        return float(val_full)
    if m == "val_masked":
        return float(val_masked)
    if eval_results is None:
        return None
    v = eval_results["validation"]
    mapping = {
        "uar": "uar", "uf1": "uf1", "accuracy": "accuracy",
        "weighted_f1": "weighted_f1", "macro_f1": "macro_f1",
    }
    if m in mapping:
        return float(v[mapping[m]])
    if m == "ece":
        return float(v.get("ece", 0.0))
    return None


def main():
    cfg = Config()
    print("CONFIG:\n" + pretty_config(cfg))
    set_seed(int(cfg.seed))

    use_cuda = torch.cuda.is_available() and getattr(cfg, "device", "cuda") == "cuda"
    device   = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    ckpt_metric = str(cfg.ckpt_metric)   

    # ── Datasets ────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = IEMOCAPUtteranceDataset(cfg.train_manifest, cfg.data_root, cfg, split="train")
    val_ds   = IEMOCAPUtteranceDataset(cfg.val_manifest,   cfg.data_root, cfg, split="val")
    collate  = partial(collate_fn, cfg=cfg)

    _loader_kwargs = dict(
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=(int(cfg.num_workers) > 0),
        prefetch_factor=2 if int(cfg.num_workers) > 0 else None,
        collate_fn=collate,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **_loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **_loader_kwargs)

    # ── Model + optimiser ───────────────────────────────────────────────────
    model     = ShaSpecEAUModel(cfg).to(device)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    # ── Scheduler ───────────────────────────────────────────────────────────
    scheduler_type = str(getattr(cfg, "scheduler", "cosine")).lower()
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=int(cfg.epochs),
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

    # ── Output dirs ─────────────────────────────────────────────────────────
    out_dir   = Path(cfg.ckpt_dir);   out_dir.mkdir(parents=True, exist_ok=True)
    eval_root = Path(cfg.eval_dir);   eval_root.mkdir(parents=True, exist_ok=True)

    eval_history_path = eval_root / "evaluation_history.json"
    try:
        eval_history = json.loads(eval_history_path.read_text()) if eval_history_path.exists() else []
        if not isinstance(eval_history, list):
            eval_history = []
    except Exception:
        eval_history = []

    # ── Best-score tracker ──────────────────────────────────────────────────
    higher_better    = metric_is_higher_better(ckpt_metric)
    best_score       = float("-inf") if higher_better else float("inf")
    eval_frequency   = int(getattr(cfg, "eval_frequency", 1))
    save_every_epoch = bool(getattr(cfg, "save_every_epoch", True))
    early_stop_patience = int(getattr(cfg, "early_stop_patience", 0))
    bad_epochs       = 0

    # ── Training loop ───────────────────────────────────────────────────────
    print("Starting training...")
    with tqdm(total=int(cfg.epochs), desc="Training", position=0) as epoch_pbar:
        for epoch in range(1, int(cfg.epochs) + 1):

            unfreeze_epoch = int(getattr(cfg, "unfreeze_epoch", 5))
            if epoch == unfreeze_epoch:
                print(f"\nEpoch {epoch}: Unfreezing top encoder layers...")
                audio_n = int(getattr(cfg, "audio_unfreeze_top_n", 2))
                text_n  = int(getattr(cfg, "text_unfreeze_top_n",  2))
                if audio_n > 0:
                    n = len(model.enc_audio.backbone.encoder.layers)
                    for i in range(n - audio_n, n):
                        for p in model.enc_audio.backbone.encoder.layers[i].parameters():
                            p.requires_grad = True
                    for p in model.enc_audio.backbone.encoder.layer_norm.parameters():
                        p.requires_grad = True
                if text_n > 0:
                    layers = model.enc_text.backbone.encoder.layer
                    n = len(layers)
                    for i in range(n - text_n, n):
                        for p in layers[i].parameters():
                            p.requires_grad = True
                    for p in model.enc_text.backbone.pooler.parameters():
                        p.requires_grad = True

                base_lr = float(cfg.lr)
                mul     = float(getattr(cfg, "unfreeze_lr_multiplier", 0.1))
                backbone_params = [
                    p for n_, p in model.named_parameters()
                    if p.requires_grad and ("enc_audio.backbone" in n_ or "enc_text.backbone" in n_)
                ]
                other_params = [
                    p for n_, p in model.named_parameters()
                    if p.requires_grad and ("enc_audio.backbone" not in n_ and "enc_text.backbone" not in n_)
                ]
                optimizer = AdamW([
                    {"params": other_params,   "lr": base_lr},
                    {"params": backbone_params, "lr": base_lr * mul},
                ], weight_decay=float(cfg.weight_decay))
                print(f"  Rebuilt optimizer: {len(other_params)} main params @ {base_lr:.1e}, "
                      f"{len(backbone_params)} backbone params @ {base_lr*mul:.1e}")


            adv_alpha = min(1.0, epoch / 20.0)
            model.set_adv_alpha(adv_alpha)

            tr_loss,  tr_parts  = run_epoch(
                model, train_loader, optimizer, device, cfg,
                train=True, apply_mask=True,
                epoch_desc=f"Epoch {epoch}/{cfg.epochs} [train]",
            )
            vf_loss,  vf_parts  = run_epoch(
                model, val_loader, optimizer, device, cfg,
                train=False, apply_mask=False,
                epoch_desc=f"Epoch {epoch}/{cfg.epochs} [val_full]",
            )
            vm_loss,  vm_parts  = run_epoch(
                model, val_loader, optimizer, device, cfg,
                train=False, apply_mask=True,
                epoch_desc=f"Epoch {epoch}/{cfg.epochs} [val_masked]",
            )

            epoch_ckpt = out_dir / f"epoch_{epoch:03d}.pt"
            if save_every_epoch:
                save_checkpoint(
                    cfg, model, optimizer, epoch, epoch_ckpt,
                    val_full=float(vf_loss), val_masked=float(vm_loss),
                    ckpt_metric=ckpt_metric,
                )

            eval_results = None
            if epoch % eval_frequency == 0:
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


            current_score = get_ckpt_score(ckpt_metric, vf_loss, vm_loss, eval_results)
            did_improve   = False

            if current_score is not None:
                did_improve = (
                    current_score > best_score if higher_better
                    else current_score < best_score
                )

            if did_improve:
                best_score = float(current_score)  
                best_ckpt  = out_dir / "best.pt"
                save_checkpoint(
                    cfg, model, optimizer, epoch, best_ckpt,
                    best_score=float(best_score),
                    val_full=float(vf_loss), val_masked=float(vm_loss),
                    ckpt_metric=ckpt_metric,
                    eval_results=eval_results,
                )
                tqdm.write(f"saved best.pt  ({ckpt_metric}: {best_score:.4f})")
                bad_epochs = 0
            else:
                if current_score is not None:
                    bad_epochs += 1

            if eval_results is not None:
                summary = {
                    "epoch":           int(epoch),
                    "ckpt":            str(epoch_ckpt),
                    "val_loss_full":   float(vf_loss),
                    "val_loss_masked": float(vm_loss),
                    "best_score":      float(best_score),
                    "ckpt_metric":     ckpt_metric,
                    "metrics": {
                        "uar":         float(eval_results["validation"]["uar"]),
                        "uf1":         float(eval_results["validation"]["uf1"]),
                        "accuracy":    float(eval_results["validation"]["accuracy"]),
                        "weighted_f1": float(eval_results["validation"]["weighted_f1"]),
                        "macro_f1":    float(eval_results["validation"]["macro_f1"]),
                        "ece":         float(eval_results["validation"].get("ece", 0.0)),
                    },
                    "retrieval": {
                        k: float(v)
                        for k, v in eval_results.get("retrieval", {}).items()
                        if "std" not in k
                    },

                    "train_parts":  {k: round(v, 5) for k, v in tr_parts.items()},
                    "val_parts":    {k: round(v, 5) for k, v in vf_parts.items()},
                }
                eval_history.append(summary)
                eval_history_path.write_text(json.dumps(eval_history, indent=2))
                print("Eval complete.\n")

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    plate_score = current_score if current_score is not None else float(vf_loss)
                    scheduler.step(plate_score)
                else:
                    scheduler.step()

            if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
                print(f"Early stopping: no improvement for {bad_epochs} epochs "
                      f"(metric={ckpt_metric}).")
                break

            lr_now = optimizer.param_groups[0]["lr"]
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                "lr":   f"{lr_now:.2e}",
                "tr":   f"{tr_loss:.4f}",
                "vf":   f"{vf_loss:.4f}",
                "vm":   f"{vm_loss:.4f}",
                "best": f"{best_score:.4f}",
                "m":    ckpt_metric,
            })

            tqdm.write(
                f"Epoch {epoch:02d} | lr {lr_now:.2e} | "
                f"train {tr_loss:.4f} "
                f"(cross {tr_parts['con_cross']:.3f} hub {tr_parts['con_hub']:.3f} "
                f"cons {tr_parts['cons']:.3f} orth {tr_parts['orth']:.3f} "
                f"task {tr_parts['task']:.3f} dao {tr_parts['dao']:.3f} "
                f"dco {tr_parts['dco']:.3f} σ {tr_parts['sigma']:.4f}) | "
                f"val_full {vf_loss:.4f} | val_masked {vm_loss:.4f}"
            )


if __name__ == "__main__":
    mp.freeze_support()
    main()