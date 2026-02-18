import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

IEMOCAP_ROOT = Path(r"C:\Users\PRASANTH\CAMEOA\dataset\IEMOCAP_full_release\IEMOCAP_full_release")
SESSIONS = [1, 2, 3, 4, 5]

USE_SENTENCE_LEVEL_MOCAP = True

CAT_TO_SHORT = {
    "Neutral state": "neu",
    "Neutral": "neu",
    "Anger": "ang",
    "Happiness": "hap",
    "Sadness": "sad",
}

# ============================================================
# REGEX
# ============================================================

EVAL_MAIN_RE = re.compile(
    r"^\s*\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s+"
    r"(?P<utt_id>\S+)\s+(?P<emo>[A-Za-z]+)"
)

TRANS_RE = re.compile(
    r"^(?P<utt_id>\S+)\s+\[(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\]\s*:\s*(?P<text>.*)$"
)

CAT_RE = re.compile(
    r"^(?P<utt_id>\S+)\s*:(?P<label>.*?);\s*\(.*\)\s*$"
)

# ============================================================
# HELPERS
# ============================================================

def infer_dialog_and_speaker(utt_id: str) -> Tuple[str, str]:
    parts = utt_id.split("_")
    dialog_id = "_".join(parts[:-1])
    speaker = parts[-1][0]
    return dialog_id, speaker


# ============================================================
# LOAD ONE SESSION
# ============================================================

def build_session_df(root: Path, sess: int):

    eval_dir = root / f"Session{sess}" / "dialog" / "EmoEvaluation"
    trans_dir = root / f"Session{sess}" / "dialog" / "transcriptions"
    cat_dir = eval_dir / "Categorical"

    rows = []

    # ---- main evaluation labels ----
    for eval_file in sorted(eval_dir.glob("Ses*.txt")):
        for line in eval_file.read_text(errors="ignore").splitlines():

            m = EVAL_MAIN_RE.match(line.strip())
            if not m:
                continue

            utt_id = m.group("utt_id")
            dialog_id, speaker = infer_dialog_and_speaker(utt_id)

            rows.append({
                "utt_id": utt_id,
                "dialog_id": dialog_id,
                "session": sess,
                "speaker": speaker,
                "emo_short": m.group("emo").lower()
            })

    df = pd.DataFrame(rows).drop_duplicates("utt_id")

    # ---- categorical labels ----
    cmap = {}

    for cf in sorted(cat_dir.glob("*_cat.txt")):
        for line in cf.read_text(errors="ignore").splitlines():
            mm = CAT_RE.match(line.strip())
            if mm:
                cmap[mm.group("utt_id")] = mm.group("label").strip()

    df["cat_label_raw"] = df["utt_id"].map(cmap)

    def final_label(r):
        raw = r["cat_label_raw"]
        if isinstance(raw, str) and raw in CAT_TO_SHORT:
            return CAT_TO_SHORT[raw]
        return r["emo_short"]

    df["label"] = df.apply(final_label, axis=1)

    # ---- modality paths ----

    def p_audio(r):
        p = root / f"Session{sess}" / "sentences" / "wav" / r.dialog_id / f"{r.utt_id}.wav"
        return str(p) if p.exists() else None

    def p_video(r):
        p = root / f"Session{sess}" / "dialog" / "avi" / "DivX" / f"{r.dialog_id}.avi"
        return str(p) if p.exists() else None

    def p_mocap_rot(r):
        p = root / f"Session{sess}" / "sentences" / "MOCAP_rotated" / r.dialog_id / f"{r.utt_id}.txt"
        return str(p) if p.exists() else None

    def p_mocap_head(r):
        p = root / f"Session{sess}" / "sentences" / "MOCAP_head" / r.dialog_id / f"{r.utt_id}.txt"
        return str(p) if p.exists() else None

    def p_mocap_hand(r):
        p = root / f"Session{sess}" / "sentences" / "MOCAP_hand" / r.dialog_id / f"{r.utt_id}.txt"
        return str(p) if p.exists() else None

    df["audio_path"] = df.apply(p_audio, axis=1)
    df["video_path"] = df.apply(p_video, axis=1)
    df["mocap_rotated_path"] = df.apply(p_mocap_rot, axis=1)
    df["mocap_head_path"] = df.apply(p_mocap_head, axis=1)
    df["mocap_hand_path"] = df.apply(p_mocap_hand, axis=1)

    return df


# ============================================================
# LOAD ALL SESSIONS
# ============================================================

def load_all():

    dfs = []

    for s in SESSIONS:
        print(f"Loading Session{s}...")
        dfs.append(build_session_df(IEMOCAP_ROOT, s))

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# PRINT MOCAP MISSING STATS
# ============================================================

def print_missing_stats(df):

    print("\n==============================")
    print("TOTAL LABEL DISTRIBUTION")
    print("==============================")

    print(df["label"].value_counts())

    mocap_cols = [
        "mocap_rotated_path",
        "mocap_head_path",
        "mocap_hand_path",
    ]

    print("\n==============================")
    print("MOCAP MISSING COUNTS BY LABEL")
    print("==============================")

    for col in mocap_cols:

        print(f"\n--- {col} ---")

        missing_mask = df[col].isna()

        stats = (
            df.assign(missing=missing_mask)
              .groupby("label")["missing"]
              .agg(total="count", missing_count="sum")
        )

        stats["missing_percent"] = (
            stats["missing_count"] / stats["total"] * 100
        )

        print(stats)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_all()

    print_missing_stats(df)
