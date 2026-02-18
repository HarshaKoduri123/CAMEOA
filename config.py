# config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional


@dataclass
class Config:
    # -------------------------
    # Data
    # -------------------------
    data_root: Path = Path(r"C:\Users\PRASANTH\CAMEOA\dataset\iemocap_data")
    train_manifest: Path = data_root / "train" / "manifest.csv"
    val_manifest: Path = data_root / "val" / "manifest.csv"

    # -------------------------
    # Training
    # -------------------------
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 16
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 2
    ckpt_metric: str = "val_full"   # "val_full" or "val_masked"
    eval_frequency: int = 1

    # -------------------------
    # Loss weights
    # -------------------------
    temperature: float = 0.07
    lambda_cons: float = 1.0
    lambda_orth: float = 0.1
    lambda_task: float = 0.2

    # uncertainty stability regularizer
    lambda_sigma: float = 1e-4      # start tiny: 1e-4 or 5e-4
    sigma_reg_mode: str = "log2"    # "log2" (safe) or "mean"

    # -------------------------
    # Modality masking (training-time robustness)
    # -------------------------
    p_keep_all: float = 0.35
    p_drop_video: float = 0.25
    p_drop_text: float = 0.10
    p_drop_audio: float = 0.10
    p_drop_mocap: float = 0.10
    p_drop_two: float = 0.10

    # -------------------------
    # Encoders
    # -------------------------
    freeze_audio: bool = True
    freeze_text: bool = True
    freeze_video: bool = True

    audio_model_name: str = "facebook/wav2vec2-base-960h"
    audio_sr: int = 16000
    max_audio_seconds: float = 8.0

    text_model_name: str = "roberta-base"
    max_text_len: int = 64

    # -------------------------
    # Video
    # -------------------------
    use_video_embeds: bool = True
    video_embed_dir: Path = data_root / "video_embeds"  # where .pt files will be stored

    video_backbone: str = "r3d_18"
    video_embed_dim: int = 512
    video_embed_dtype: str = "fp16"

    video_num_frames: int = 8
    video_fps: int = 25
    video_resize: Tuple[int, int] = (112, 112)

    video_cache_size: int = 6
    video_decode_threads: int = 2

    # -------------------------
    # Mocap
    # -------------------------
    mocap_file_dim: int = 192
    mocap_feat_dim: int = 576
    mocap_max_len: int = 400
    mocap_nhead: int = 4
    mocap_layers: int = 2

    # -------------------------
    # Shared latent dims
    # -------------------------
    d_model: int = 256
    d_shared: int = 256
    d_private: int = 256

    # -------------------------
    # Labels / classes
    # -------------------------
    # Your model currently uses 4 classes; make that explicit here.
    num_classes: int = 4

    # If True, map only {ang,hap,sad,neu} and ignore others in supervised loss.
    # (Contrastive still uses all utterances; CE uses only these.)
    use_label_subset: bool = True

    label_map: Optional[Dict[str, int]] = None

    def __post_init__(self):
        self.label_map = {
            "ang": 0,
            "hap": 1,
            "sad": 2,
            "neu": 3,
        }
        num_classes = 4
        use_label_subset = True
