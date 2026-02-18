# precompute_video_embeds.py
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import decord
from decord import VideoReader

from config import Config


# ---------------------------------------------------------
# Backbone extractor: r3d_18 -> 512-d features (no adapter)
# ---------------------------------------------------------
def build_r3d18_backbone() -> nn.Module:
    from torchvision.models.video import r3d_18, R3D_18_Weights

    weights = R3D_18_Weights.DEFAULT
    m = r3d_18(weights=weights)
    m.fc = nn.Identity()  # output (B,512)
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


@torch.no_grad()
def sample_utterance_frames(vr: VideoReader, start: float, end: float, num_frames: int) -> torch.Tensor:
    """
    Returns uint8 frames (T,H,W,3) as torch tensor using decord torch bridge.
    """
    if end <= start:
        end = start + 1e-3

    nframes = len(vr)
    fps = float(vr.get_avg_fps())
    if fps <= 1e-3:
        fps = 30.0

    s_idx = int(start * fps)
    e_idx = int(end * fps)

    s_idx = max(0, min(s_idx, nframes - 1))
    e_idx = max(0, min(e_idx, nframes - 1))
    if e_idx < s_idx:
        e_idx = s_idx

    if num_frames <= 1:
        idxs = [s_idx]
    else:
        idxs = np.linspace(s_idx, e_idx, num_frames).astype(np.int64).tolist()

    frames = vr.get_batch(idxs)  # torch uint8 (T,H,W,3)
    return frames.contiguous()


@torch.no_grad()
def preprocess_for_r3d(frames_u8: torch.Tensor, target_hw) -> torch.Tensor:
    """
    frames_u8: (T,H,W,3) uint8
    returns: (1,T,3,H,W) float32 in [0,1], resized
    """
    T, H, W, C = frames_u8.shape
    x = frames_u8.float().div_(255.0)          # (T,H,W,3)
    x = x.permute(0, 3, 1, 2).contiguous()     # (T,3,H,W)

    th, tw = int(target_hw[0]), int(target_hw[1])
    if (H != th) or (W != tw):
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

    x = x.unsqueeze(0)                         # (1,T,3,H,W)
    return x


# ---------------------------------------------------------
# LRU cache for VideoReaders
# ---------------------------------------------------------
class VideoReaderCache:
    def __init__(self, max_items: int = 6):
        self.max_items = int(max_items)
        self.cache = OrderedDict()  # key -> VideoReader

    def get(self, video_path: Path) -> VideoReader:
        key = str(video_path)

        # hit
        if key in self.cache:
            vr = self.cache.pop(key)
            self.cache[key] = vr
            return vr

        # miss
        vr = VideoReader(key, ctx=decord.cpu(0))
        self.cache[key] = vr

        # evict
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

        return vr


def find_all_manifests(data_root: Path):
    """
    Auto-detect manifests: <data_root>/<split>/manifest.csv
    """
    manifests = []
    for p in sorted(data_root.glob("*/manifest.csv")):
        split_name = p.parent.name
        manifests.append((split_name, p))
    return manifests


def main():
    cfg = Config()
    data_root = Path(cfg.data_root)

    # Decord torch bridge: vr.get_batch returns torch tensors
    decord.bridge.set_bridge("torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    backbone = build_r3d18_backbone().to(device)

    out_dir = data_root / "assets" / "video_embeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect all splits that exist
    manifests = find_all_manifests(data_root)
    if not manifests:
        raise RuntimeError(f"No manifests found under: {data_root} (expected */manifest.csv)")

    print("Found manifests:")
    for name, p in manifests:
        print(f"  - {name}: {p}")

    # Cache readers per dialog avi
    vr_cache = VideoReaderCache(max_items=int(getattr(cfg, "video_cache_size", 6)))

    total_saved = 0
    num_frames = int(cfg.video_num_frames)
    target_hw = cfg.video_resize

    for split_name, manifest_path in manifests:
        df = pd.read_csv(manifest_path)
        print(f"\n{split_name}: rows={len(df)}")

        for r in tqdm(df.itertuples(index=False), total=len(df), desc=f"precompute [{split_name}]"):
            utt_id = str(r.utt_id)

            video_rel = getattr(r, "video_path", "")
            if not isinstance(video_rel, str) or video_rel.strip() == "":
                continue

            vid_path = data_root / video_rel
            if not vid_path.exists():
                continue

            out_path = out_dir / f"{utt_id}.pt"
            if out_path.exists():
                continue

            start = float(getattr(r, "start"))
            end = float(getattr(r, "end"))

            vr = vr_cache.get(vid_path)

            frames_u8 = sample_utterance_frames(vr, start, end, num_frames=num_frames)
            x = preprocess_for_r3d(frames_u8, target_hw=target_hw).to(device, non_blocking=True)

            # r3d expects (B,3,T,H,W)
            x = x.permute(0, 2, 1, 3, 4).contiguous()

            feat512 = backbone(x).squeeze(0).detach().float().cpu()  # (512,)
            torch.save(feat512, out_path)
            total_saved += 1

    print(f"\nSaved {total_saved} utterance video embeddings to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
