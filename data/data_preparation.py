import os
import shutil
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
IEMOCAP_ROOT = Path(r"C:\Users\PRASANTH\CAMEOA\dataset\IEMOCAP_full_release\IEMOCAP_full_release")
OUT_ROOT = Path(r"C:\Users\PRASANTH\CAMEOA\dataset\iemocap_data")

SESSIONS = [1,2,3,4,5]

SEED = 42
TRAIN_RATIO = 0.8

LINK_MODE = "hardlink"
WRITE_TRANSCRIPT_FILES = True
USE_SENTENCE_LEVEL_MOCAP = True
DIALOG_SAFE_SPLIT = True

# -----------------------------
# HELPERS
# -----------------------------
def ensure_parent(p):
    p.parent.mkdir(parents=True, exist_ok=True)

def link_or_copy(src, dst):
    if dst.exists():
        return
    ensure_parent(dst)
    os.link(src, dst)

def infer_dialog_and_speaker(utt_id):
    parts = utt_id.split("_")
    dialog_id = "_".join(parts[:-1])
    speaker = parts[-1][0]
    return dialog_id, speaker

# -----------------------------
# REGEX
# -----------------------------
EVAL_MAIN_RE = re.compile(
    r"\[(?P<start>\d+\.?\d*)\s*-\s*(?P<end>\d+\.?\d*)\]\s+"
    r"(?P<utt_id>\S+)\s+(?P<emo>\w+)"
)

TRANS_RE = re.compile(
    r"^(?P<utt_id>\S+)\s+\[(.*?)\]\:\s*(?P<text>.*)$"
)

# -----------------------------
# BUILD DATAFRAME (ALL SESSIONS)
# -----------------------------
def build_dataset():

    rows = []

    for sess in SESSIONS:

        print(f"Loading Session {sess}")

        root = IEMOCAP_ROOT / f"Session{sess}"

        eval_dir = root / "dialog" / "EmoEvaluation"
        trans_dir = root / "dialog" / "transcriptions"

        # transcripts map
        tmap = {}

        for tf in trans_dir.glob("Ses*.txt"):
            for line in tf.read_text(errors="ignore").splitlines():
                m = TRANS_RE.match(line.strip())
                if m:
                    tmap[m.group("utt_id")] = m.group("text")

        # main annotation files
        for ef in eval_dir.glob("Ses*.txt"):

            for line in ef.read_text(errors="ignore").splitlines():

                m = EVAL_MAIN_RE.match(line.strip())

                if not m:
                    continue

                utt_id = m.group("utt_id")

                dialog_id, speaker = infer_dialog_and_speaker(utt_id)

                rows.append(dict(
                    utt_id = utt_id,
                    dialog_id = dialog_id,
                    session = sess,
                    speaker = speaker,
                    start = float(m.group("start")),
                    end = float(m.group("end")),
                    duration = float(m.group("end")) - float(m.group("start")),
                    label = m.group("emo").lower(),
                    transcript = tmap.get(utt_id,"")
                ))

    df = pd.DataFrame(rows)

    # -----------------------------
    # PATHS
    # -----------------------------
    def audio_path(r):
        p = IEMOCAP_ROOT / f"Session{r.session}" / "sentences" / "wav" / r.dialog_id / f"{r.utt_id}.wav"
        return str(p) if p.exists() else None

    def video_path(r):
        p = IEMOCAP_ROOT / f"Session{r.session}" / "dialog" / "avi" / "DivX" / f"{r.dialog_id}.avi"
        return str(p) if p.exists() else None

    def mocap_rot(r):
        p = IEMOCAP_ROOT / f"Session{r.session}" / "sentences" / "MOCAP_rotated" / r.dialog_id / f"{r.utt_id}.txt"
        return str(p) if p.exists() else None

    # <<< ADDED
    def mocap_head(r):
        p = IEMOCAP_ROOT / f"Session{r.session}" / "sentences" / "MOCAP_head" / r.dialog_id / f"{r.utt_id}.txt"
        return str(p) if p.exists() else None

    # <<< ADDED
    def mocap_hand(r):
        p = IEMOCAP_ROOT / f"Session{r.session}" / "sentences" / "MOCAP_hand" / r.dialog_id / f"{r.utt_id}.txt"
        return str(p) if p.exists() else None

    df["audio_path"] = df.apply(audio_path, axis=1)
    df["video_path"] = df.apply(video_path, axis=1)
    df["mocap_rotated_path"] = df.apply(mocap_rot, axis=1)

    # <<< ADDED
    df["mocap_head_path"] = df.apply(mocap_head, axis=1)
    df["mocap_hand_path"] = df.apply(mocap_hand, axis=1)

    df["has_audio"] = df.audio_path.notna()
    df["has_video"] = df.video_path.notna()
    df["has_text"] = df.transcript != ""

    # <<< CHANGED (recommended): require ALL 3 mocap files
    df["has_mocap"] = (
        df["mocap_rotated_path"].notna()
        & df["mocap_head_path"].notna()
        & df["mocap_hand_path"].notna()
    )

    # If you want to require ONLY rotated, use this instead:
    # df["has_mocap"] = df["mocap_rotated_path"].notna()

    # require full multimodal
    df = df[df.has_audio & df.has_video & df.has_text & df.has_mocap]

    return df.reset_index(drop=True)

# -----------------------------
# SPLIT (dialog-safe)
# -----------------------------
def split_train_val(df):

    dialogs = df.dialog_id.unique().tolist()
    random.Random(SEED).shuffle(dialogs)

    n_train = int(TRAIN_RATIO * len(dialogs))

    train_d = set(dialogs[:n_train])
    val_d = set(dialogs[n_train:])

    train = df[df.dialog_id.isin(train_d)]
    val = df[df.dialog_id.isin(val_d)]

    return train.reset_index(drop=True), val.reset_index(drop=True)

# -----------------------------
# MATERIALIZE
# -----------------------------
def materialize(name, df):

    print(f"Writing {name}")

    split_root = OUT_ROOT / name
    assets = OUT_ROOT / "assets"

    aud = assets / "audios"
    vid = assets / "videos"
    moc = assets / "mocap"
    txt = assets / "transcripts"

    for d in [aud,vid,moc,txt]:
        d.mkdir(parents=True,exist_ok=True)

    video_seen=set()

    audio_rel=[]
    video_rel=[]
    moc_rot_rel=[]
    moc_head_rel=[]  # <<< ADDED
    moc_hand_rel=[]  # <<< ADDED
    text_rel=[]

    for _,r in df.iterrows():

        # audio
        ap = Path(r.audio_path)
        adst = aud / f"{r.utt_id}.wav"
        link_or_copy(ap, adst)
        audio_rel.append(str(adst.relative_to(OUT_ROOT)))

        # video (dialog level)
        vp = Path(r.video_path)
        vdst = vid / f"{r.dialog_id}.avi"

        if r.dialog_id not in video_seen:
            link_or_copy(vp,vdst)
            video_seen.add(r.dialog_id)

        video_rel.append(str(vdst.relative_to(OUT_ROOT)))

        # mocap rotated
        mp = Path(r.mocap_rotated_path)
        mdst = moc / f"{r.utt_id}_rotated.txt"
        link_or_copy(mp,mdst)
        moc_rot_rel.append(str(mdst.relative_to(OUT_ROOT)))

        # <<< ADDED: mocap head
        hp = Path(r.mocap_head_path)
        hdst = moc / f"{r.utt_id}_head.txt"
        link_or_copy(hp, hdst)
        moc_head_rel.append(str(hdst.relative_to(OUT_ROOT)))

        # <<< ADDED: mocap hand
        kp = Path(r.mocap_hand_path)
        kdst = moc / f"{r.utt_id}_hand.txt"
        link_or_copy(kp, kdst)
        moc_hand_rel.append(str(kdst.relative_to(OUT_ROOT)))

        # transcript
        tp = txt / f"{r.utt_id}.txt"
        tp.write_text(r.transcript, encoding="utf-8")
        text_rel.append(str(tp.relative_to(OUT_ROOT)))

    manifest = df.copy()

    manifest["audio_path"]=audio_rel
    manifest["video_path"]=video_rel
    manifest["mocap_rotated_path"]=moc_rot_rel
    manifest["mocap_head_path"]=moc_head_rel     # <<< ADDED
    manifest["mocap_hand_path"]=moc_hand_rel     # <<< ADDED
    manifest["transcript_path"]=text_rel

    split_root.mkdir(parents=True,exist_ok=True)

    manifest.to_csv(split_root/"manifest.csv",index=False, encoding="utf-8")

# -----------------------------
# MAIN
# -----------------------------
def main():

    OUT_ROOT.mkdir(parents=True,exist_ok=True)

    df = build_dataset()

    print("Total samples:",len(df))

    train,val = split_train_val(df)

    materialize("train",train)
    materialize("val",val)

    print("DONE")

if __name__=="__main__":
    main()
