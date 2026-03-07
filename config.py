
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class Config:
    # ─────────────────────────────────────────────
    #  Paths
    # ─────────────────────────────────────────────
    data_root:      Path = Path(r"C:\Users\PRASANTH\CAMEOA\dataset\iemocap_data")
    train_manifest: Path = data_root / "train" / "manifest.csv"
    val_manifest:   Path = data_root / "val"   / "manifest.csv"
    ckpt_dir:       Path = Path("checkpoints")
    eval_dir:       Path = Path("evaluation_results")

    # ─────────────────────────────────────────────
    #  Training
    # ─────────────────────────────────────────────
    seed:         int   = 42
    device:       str   = "cuda"
    batch_size:   int   = 12         
    epochs:       int   = 50
    lr:           float = 1e-4     
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0
    num_workers:  int   = 2

    # ── Evaluation / checkpointing ───────────────
    eval_frequency:    int  = 1
    ckpt_metric:       str  = "uar"
    save_every_epoch:  bool = True

    # ── Encoder unfreeze schedule ────────────────
    unfreeze_epoch:         int   = 5   
    audio_unfreeze_top_n:   int   = 2
    text_unfreeze_top_n:    int   = 2
    unfreeze_lr_multiplier: float = 0.1  
    # ── Scheduler ───────────────────────────────
    scheduler:        str   = "plateau"
    min_lr:           float = 1e-7
    plateau_factor:   float = 0.5
    plateau_patience: int   = 4

    # ── Early stopping ───────────────────────────
    early_stop_patience: int = 10

    # ─────────────────────────────────────────────
    #  Core losses
    # ─────────────────────────────────────────────
    temperature:      float = 0.10
    lambda_con_cross: float = 1.0
    lambda_con_hub:   float = 0.5
    lambda_cons:      float = 0.5
    lambda_orth:      float = 0.05
    lambda_task:      float = 2.0
    label_smoothing:  float = 0.05

    use_class_weights: bool                         = True
    class_weights:     Optional[Tuple[float, ...]] = (1.2, 1.0, 1.4, 1.0)

    # ─────────────────────────────────────────────
    #  ShaSpec — DISABLED
    # ─────────────────────────────────────────────
    lambda_dao: float = 0.0  
    lambda_dco: float = 0.0  

    # ─────────────────────────────────────────────
    #  Distributional representations
    # ─────────────────────────────────────────────
    log_sigma_min:    float = -4.0
    log_sigma_max:    float =  2.0
    lambda_prior:     float = 0.0    
    lambda_sigma_reg: float = 1e-5
    sigma_reg_mode:   str   = "log2"

    # ─────────────────────────────────────────────
    #  Speaker adversarial
    # ─────────────────────────────────────────────
    lambda_speaker_adv: float = 0.1
    n_speakers:         int   = 10    # IEMOCAP has 10 speakers

    # ─────────────────────────────────────────────
    #  Modality-dropout masking
    # ─────────────────────────────────────────────
    p_keep_all:   float = 0.35
    p_drop_video: float = 0.20
    p_drop_text:  float = 0.15
    p_drop_audio: float = 0.15
    p_drop_mocap: float = 0.10
    p_drop_two:   float = 0.05

    # ─────────────────────────────────────────────
    #  Audio encoder 
    # ─────────────────────────────────────────────
    freeze_audio:      bool  = True
    audio_model_name:  str   = "facebook/wav2vec2-base-960h"
    audio_sr:          int   = 16000
    max_audio_seconds: float = 8.0
    audio_dropout:     float = 0.1

    # ─────────────────────────────────────────────
    #  Text encoder 
    # ─────────────────────────────────────────────
    freeze_text:     bool  = True
    text_model_name: str   = "roberta-base"
    max_text_len:    int   = 96
    text_dropout:    float = 0.1

    # ─────────────────────────────────────────────
    #  Video encoder 
    # ─────────────────────────────────────────────
    freeze_video:         bool  = True
    use_video_embeds:     bool  = True
    video_embed_dir:      Path  = data_root / "video_embeds"
    video_backbone:       str   = "r3d_18"
    video_embed_dim:      int   = 512
    video_temporal_embed: bool  = False
    video_dropout:        float = 0.1

    # ─────────────────────────────────────────────
    #  MoCap encoder 
    # ─────────────────────────────────────────────
    mocap_feat_dim: int   = 576
    mocap_max_len:  int   = 400
    mocap_nhead:    int   = 8
    mocap_layers:   int   = 6      
    mocap_n_groups: int   = 4
    mocap_dropout:  float = 0.1

    # ─────────────────────────────────────────────
    #  Representation dims
    # ─────────────────────────────────────────────
    d_model:            int   = 256
    d_shared:           int   = 256
    d_private:          int   = 128  
    d_proj:             int   = 128
    enc_dropout:        float = 0.1
    classifier_dropout: float = 0.3   

    # ─────────────────────────────────────────────
    #  Labels
    # ─────────────────────────────────────────────
    num_classes:      int  = 4
    use_label_subset: bool = True
    label_map: Optional[Dict[str, int]] = None

    def __post_init__(self):
        self.label_map = {"ang": 0, "hap": 1, "sad": 2, "neu": 3}

        valid_metrics = {
            "val_full", "val_masked", "uar", "uf1",
            "accuracy", "weighted_f1", "macro_f1", "ece",
        }
        assert self.ckpt_metric in valid_metrics, (
            f"Unknown ckpt_metric='{self.ckpt_metric}'. "
            f"Choose from: {sorted(valid_metrics)}"
        )
        assert self.d_model % self.mocap_n_groups == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"mocap_n_groups ({self.mocap_n_groups})"
        )