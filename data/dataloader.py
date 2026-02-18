# dataloader.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import soundfile as sf
from scipy.signal import resample_poly


@dataclass
class Sample:
    utt_id: str
    dialog_id: str
    session: int
    speaker: str
    start: float
    end: float
    label: str

    audio_path: str
    transcript_path: str
    video_path: str
    mocap_rotated_path: str
    mocap_head_path: str
    mocap_hand_path: str

    has_audio: int
    has_text: int
    has_video: int
    has_mocap: int


class IEMOCAPUtteranceDataset(Dataset):
    """
    Strategy:
      - audio loaded per utterance (padded to cfg.max_audio_seconds)
      - text loaded as string
      - video uses precomputed utterance embedding: {data_root}/assets/video_embeds/{utt_id}.pt
      - mocap loads 3 streams; collate pads/aligns
    """

    def __init__(self, manifest_path: Path, data_root: Path, cfg, split: str):
        self.df = pd.read_csv(manifest_path)
        self.data_root = Path(data_root)
        self.cfg = cfg
        self.split = split

        self.df["label"] = self.df["label"].astype(str).str.strip().str.lower()

        self.df.loc[self.df["label"] == "exc", "label"] = "hap"
        if bool(getattr(cfg, "use_label_subset", False)):
            allowed = {"ang", "hap", "sad", "neu"}
            before = len(self.df)
            self.df = self.df[self.df["label"].isin(allowed)].reset_index(drop=True)
            after = len(self.df)


        self.video_embed_dir = self.data_root / "assets" / "video_embeds"

        self.mocap_file_dim = int(getattr(cfg, "mocap_file_dim", 192))
        self.mocap_max_len = int(getattr(cfg, "mocap_max_len", 400))
        self.mocap_feat_dim = int(getattr(cfg, "mocap_feat_dim", 576))

        self.audio_sr = int(getattr(cfg, "audio_sr", 16000))
        self.max_audio_seconds = float(getattr(cfg, "max_audio_seconds", 8.0))
        self.max_audio_len = int(self.audio_sr * self.max_audio_seconds)
        print(f"[{split}] total after filter:", len(self.df))
        print(self.df["label"].value_counts())


    def __len__(self):
        return len(self.df)

    def _abs(self, rel: str) -> Optional[Path]:
        if not isinstance(rel, str) or rel.strip() == "":
            return None
        return self.data_root / rel

    # -----------------------------
    # AUDIO
    # -----------------------------
    def _load_audio(self, path: Path) -> torch.Tensor:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        if sr != self.audio_sr:
            wav = resample_poly(wav, self.audio_sr, sr).astype(np.float32)

        wav = torch.from_numpy(wav)
        if wav.numel() > self.max_audio_len:
            wav = wav[: self.max_audio_len]
        else:
            wav = F.pad(wav, (0, self.max_audio_len - wav.numel()))
        return wav.contiguous()

    # -----------------------------
    # TEXT
    # -----------------------------
    def _load_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore").strip()

    # -----------------------------
    # MOCAP STREAM FILE
    # -----------------------------
    def _load_mocap_file(self, path: Path) -> torch.Tensor:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 4:
            return torch.zeros((1, self.mocap_file_dim), dtype=torch.float32)

        data_rows: List[List[float]] = []
        for ln in lines[2:]:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            vals = []
            ok = True
            for x in parts:
                try:
                    if str(x).lower() == "nan":
                        vals.append(0.0)
                    else:
                        vals.append(float(x))
                except Exception:
                    ok = False
                    break
            if ok:
                data_rows.append(vals)

        if not data_rows:
            return torch.zeros((1, self.mocap_file_dim), dtype=torch.float32)

        arr = torch.tensor(data_rows, dtype=torch.float32)

        # drop Frame# and Time
        if arr.size(1) >= 3:
            arr = arr[:, 2:]
        elif arr.size(1) >= 2:
            arr = arr[:, 1:]
        else:
            return torch.zeros((arr.size(0), self.mocap_file_dim), dtype=torch.float32)

        # downsample time to mocap_max_len
        T = arr.size(0)
        if T > self.mocap_max_len:
            idx = torch.linspace(0, T - 1, self.mocap_max_len).long()
            arr = arr[idx]

        # pad/crop feature dim
        if arr.size(1) < self.mocap_file_dim:
            arr = F.pad(arr, (0, self.mocap_file_dim - arr.size(1), 0, 0))
        elif arr.size(1) > self.mocap_file_dim:
            arr = arr[:, : self.mocap_file_dim]

        return arr.contiguous()

    # -----------------------------
    # VIDEO EMBED (Option A)
    # -----------------------------
    def _load_video_embed(self, utt_id: str) -> Optional[torch.Tensor]:
        p = self.video_embed_dir / f"{utt_id}.pt"
        if not p.exists():
            return None
        emb = torch.load(p, map_location="cpu")
        if not torch.is_tensor(emb):
            raise RuntimeError(f"Video embedding is not a tensor: {p}")
        return emb.float().view(-1).contiguous()

    # -----------------------------
    # GET ITEM
    # -----------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]

        label = str(r["label"]).strip().lower()  # already merged exc->hap

        sample = Sample(
            utt_id=str(r["utt_id"]),
            dialog_id=str(r["dialog_id"]),
            session=int(r["session"]),
            speaker=str(r["speaker"]),
            start=float(r["start"]),
            end=float(r["end"]),
            label=label,

            audio_path=str(r["audio_path"]),
            transcript_path=str(r["transcript_path"]),
            video_path=str(r["video_path"]),
            mocap_rotated_path=str(r["mocap_rotated_path"]),
            mocap_head_path=str(r["mocap_head_path"]),
            mocap_hand_path=str(r["mocap_hand_path"]),

            has_audio=int(r.get("has_audio", 1)),
            has_text=int(r.get("has_text", 1)),
            has_video=int(r.get("has_video", 0)),
            has_mocap=int(r.get("has_mocap", 1)),
        )

        item: Dict[str, Any] = {
            "utt_id": sample.utt_id,
            "dialog_id": sample.dialog_id,
            "session": sample.session,
            "speaker": sample.speaker,
            "start": sample.start,
            "end": sample.end,
            "label": sample.label,
        }

        # audio
        ap = self._abs(sample.audio_path)
        if ap and sample.has_audio and ap.exists():
            item["audio"] = self._load_audio(ap)
            item["has_audio"] = 1
        else:
            item["audio"] = None
            item["has_audio"] = 0

        # text
        tp = self._abs(sample.transcript_path)
        if tp and sample.has_text and tp.exists():
            txt = self._load_text(tp)
            item["text"] = txt
            item["has_text"] = 1 if (isinstance(txt, str) and txt.strip() != "") else 0
        else:
            item["text"] = None
            item["has_text"] = 0

        # video embed
        vemb = self._load_video_embed(sample.utt_id) if sample.has_video else None
        item["video_embed"] = vemb
        item["has_video"] = 1 if vemb is not None else 0

        # mocap streams
        mr = self._abs(sample.mocap_rotated_path)
        mh = self._abs(sample.mocap_head_path)
        mk = self._abs(sample.mocap_hand_path)

        moc_list = []
        if sample.has_mocap:
            for pth in [mr, mh, mk]:
                if pth and pth.exists():
                    moc_list.append(self._load_mocap_file(pth))
        item["mocap_list"] = moc_list
        item["has_mocap"] = 1 if len(moc_list) > 0 else 0

        return item


# ============================================================
# COLLATE
# ============================================================
def collate_fn(batch: List[Dict[str, Any]], cfg=None) -> Dict[str, Any]:
    assert cfg is not None, "cfg must be provided"
    out: Dict[str, Any] = {}

    B = len(batch)
    out["utt_id"] = [b["utt_id"] for b in batch]
    out["label"] = [b["label"] for b in batch]

    # flags
    out["has_audio"] = torch.tensor([int(b.get("has_audio", 0)) for b in batch], dtype=torch.long)
    out["has_text"]  = torch.tensor([int(b.get("has_text", 0)) for b in batch], dtype=torch.long)
    out["has_video"] = torch.tensor([int(b.get("has_video", 0)) for b in batch], dtype=torch.long)
    out["has_mocap"] = torch.tensor([int(b.get("has_mocap", 0)) for b in batch], dtype=torch.long)

    # audio
    max_audio_len = int(getattr(cfg, "audio_sr", 16000) * float(getattr(cfg, "max_audio_seconds", 8.0)))
    aud = []
    for b in batch:
        if b["audio"] is None:
            aud.append(torch.zeros((max_audio_len,), dtype=torch.float32))
        else:
            a = b["audio"].float()
            if a.numel() != max_audio_len:
                if a.numel() > max_audio_len:
                    a = a[:max_audio_len]
                else:
                    a = F.pad(a, (0, max_audio_len - a.numel()))
            aud.append(a)
    out["audio"] = torch.stack(aud, dim=0)

    # text
    out["text"] = [(b["text"] if isinstance(b["text"], str) else "") for b in batch]

    # video embed
    Dv = int(getattr(cfg, "video_embed_dim", 512))
    vemb = []
    for b in batch:
        ve = b.get("video_embed", None)
        if ve is None:
            vemb.append(torch.zeros((Dv,), dtype=torch.float32))
        else:
            ve = ve.float().view(-1)
            if ve.numel() != Dv:
                if ve.numel() > Dv:
                    ve = ve[:Dv]
                else:
                    ve = F.pad(ve, (0, Dv - ve.numel()))
            vemb.append(ve)
    out["video_embed"] = torch.stack(vemb, dim=0)

    # mocap
    fixedF = int(getattr(cfg, "mocap_feat_dim", 576))
    maxT_cap = int(getattr(cfg, "mocap_max_len", 400))

    moc_tensors = []
    moc_lens = []
    for b in batch:
        arrs = b.get("mocap_list", None)
        if not arrs:
            moc_tensors.append(None)
            moc_lens.append(0)
            continue

        target_T = min(maxT_cap, max(a.size(0) for a in arrs))
        target_T = max(1, target_T)

        resampled = []
        for a in arrs:
            T = a.size(0)
            if T != target_T:
                idx = torch.linspace(0, T - 1, target_T).long()
                a = a[idx]
            resampled.append(a)

        moc = torch.cat(resampled, dim=-1)
        if moc.size(1) < fixedF:
            moc = F.pad(moc, (0, fixedF - moc.size(1), 0, 0))
        elif moc.size(1) > fixedF:
            moc = moc[:, :fixedF]

        moc_tensors.append(moc)
        moc_lens.append(moc.size(0))

    maxT = min(maxT_cap, max(1, max(moc_lens)))
    padded = []
    for m in moc_tensors:
        if m is None:
            padded.append(torch.zeros((maxT, fixedF), dtype=torch.float32))
        else:
            if m.size(0) > maxT:
                m = m[:maxT]
            elif m.size(0) < maxT:
                m = F.pad(m, (0, 0, 0, maxT - m.size(0)))
            padded.append(m.float())

    out["mocap"] = torch.stack(padded, dim=0)
    out["mocap_len"] = torch.tensor(moc_lens, dtype=torch.long)

    return out
