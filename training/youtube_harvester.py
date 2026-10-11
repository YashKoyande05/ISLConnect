"""
YouTube Dataset Harvester for Indian Sign Language (ISL).

Automates:
1. Downloading ISL tutorial videos and subtitles from YouTube playlist.
2. Parsing sign demonstration intervals and labels from subtitles:
   - Letters A-Z (qcdivQfA41Y)
   - Numbers 1-15 (vnH2BmcSRMA)
   - Basic Words 1 (VtbYvVDItvg)
   - Basic Words 2 (lffGJ29IhZQ)
   - Question Words (DOFPRw6Epl0)
3. Extracting MediaPipe hand landmarks (126-D features: 2 hands) from video demonstration intervals.
4. Exporting to dataset/processed/landmarks.csv and training the model.
"""

from pathlib import Path
import subprocess
import json
import re
import cv2
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from core.feature_extractor import LandmarkFeatureExtractor

YOUTUBE_DIR = Path("dataset/youtube")
VIDEOS_DIR = YOUTUBE_DIR / "videos"
MANIFESTS_DIR = YOUTUBE_DIR / "manifests"
PROCESSED_DIR = YOUTUBE_DIR / "processed"
MAIN_LANDMARKS_CSV = Path("dataset/processed/landmarks.csv")


def ensure_dirs():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def to_sec(ts: str) -> float:
    parts = ts.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def parse_vtt_cues(vtt_path: Path):
    if not vtt_path or not vtt_path.exists():
        return []
    text = vtt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.findall(
        r'(\d\d:\d\d:\d\d\.\d+)\s*-->\s*(\d\d:\d\d:\d\d\.\d+)\s*\n(.*?)(?=\n\d\d:\d\d:\d\d|\Z)',
        text,
        re.DOTALL
    )
    results = []
    for s, e, cue in blocks:
        clean = " ".join(cue.strip().split()).replace("&nbsp;", " ")
        results.append((to_sec(s), to_sec(e), clean))
    return results


def parse_alphabet_intervals(vtt_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts timestamps for all 26 letters (A-Z) from Lesson 1 (qcdivQfA41Y).
    """
    cues = parse_vtt_cues(vtt_path)
    import string
    letters = {}

    for start_sec, end_sec, cue in cues:
        for char in string.ascii_uppercase:
            # Match single letter alone or followed by comma/space/period
            if re.match(rf'^{char}\b', cue) or re.match(rf'^{char}[\,\.\s]', cue):
                if char not in letters or start_sec < letters[char]['start_sec']:
                    letters[char] = {
                        "label": char,
                        "start_sec": start_sec,
                        "end_sec": max(end_sec, start_sec + 5.0),
                        "cue": cue
                    }

    intervals = [letters[k] for k in sorted(letters.keys())]
    print(f"Extracted {len(intervals)}/26 alphabet letters.")
    return intervals


def parse_numbers_intervals(vtt_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts timestamps for numbers 1 to 15 from Lesson 2 (vnH2BmcSRMA).
    """
    cues = parse_vtt_cues(vtt_path)
    numbers = {}

    for start_sec, end_sec, cue in cues:
        m = re.match(r'^(\d{1,2})\b', cue)
        if m:
            num = m.group(1)
            if int(num) <= 20 and num not in numbers:
                numbers[num] = {
                    "label": f"Num_{num}",
                    "start_sec": start_sec,
                    "end_sec": max(end_sec, start_sec + 3.0),
                    "cue": cue
                }

    intervals = [numbers[k] for k in sorted(numbers.keys(), key=lambda x: int(x))]
    print(f"Extracted {len(intervals)} numbers.")
    return intervals


def parse_general_words_intervals(vtt_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts timestamps for words from Lessons 3, 4, 5.
    """
    cues = parse_vtt_cues(vtt_path)
    pattern = re.compile(
        r'(?:first|second|third|fourth|\d+(?:st|nd|rd|th)?|next)\s*(?:word|phrase|sign)?\s*(?:is|\:)\s*([A-Za-z\s\-\'\.]+)',
        re.IGNORECASE
    )

    signs = []
    seen = set()
    for start_sec, end_sec, cue in cues:
        m = pattern.search(cue)
        if m:
            raw_sign = m.group(1).strip().strip('.').strip('"')
            if len(raw_sign.split()) <= 3 and len(raw_sign) > 1 and not raw_sign.lower().startswith(('that', 'to', 'for', 'our', 'what', 'the reply')):
                clean_label = "".join(w.capitalize() for w in raw_sign.replace("-", " ").split())
                if clean_label not in seen:
                    seen.add(clean_label)
                    signs.append({
                        "label": clean_label,
                        "start_sec": start_sec,
                        "end_sec": start_sec + 6.0,
                        "cue": cue
                    })

    print(f"Extracted {len(signs)} word intervals from {vtt_path.name}.")
    return signs


def harvest_landmarks_from_video(video_path: Path, intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts 126-D MediaPipe landmarks for frames in the given intervals.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    extractor = LandmarkFeatureExtractor()
    records = []

    print(f"Processing {len(intervals)} items from {video_path.name}...")
    for item in intervals:
        label = item["label"]
        start_frame = int(item["start_sec"] * fps)
        end_frame = int(item["end_sec"] * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame = start_frame
        detected_count = 0

        # Sample every 2 frames
        step = 2
        while curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (curr_frame - start_frame) % step == 0:
                features, results = extractor.extract(frame)
                if features is not None and len(features) == 126 and results and results.multi_hand_landmarks:
                    # Only collect frame if at least one hand is actively raised in signing space (wrist y <= 0.72)
                    has_active = any(h.landmark[0].y <= 0.72 for h in results.multi_hand_landmarks)
                    if has_active:
                        row = {f"f{i}": features[i] for i in range(126)}
                        row["label"] = label
                        records.append(row)
                        detected_count += 1

            curr_frame += 1

        print(f"  - [{label}]: {detected_count} samples extracted.", flush=True)

    cap.release()
    return records


def run_full_harvest_and_train():
    ensure_dirs()

    tasks = [
        {
            "id": "qcdivQfA41Y",
            "type": "alphabet",
            "vtt": MANIFESTS_DIR / "qcdivQfA41Y.en.vtt",
            "video": VIDEOS_DIR / "qcdivQfA41Y.mp4"
        },
        {
            "id": "vnH2BmcSRMA",
            "type": "numbers",
            "vtt": MANIFESTS_DIR / "vnH2BmcSRMA.en.vtt",
            "video": VIDEOS_DIR / "vnH2BmcSRMA.mp4"
        },
        {
            "id": "VtbYvVDItvg",
            "type": "words",
            "vtt": MANIFESTS_DIR / "VtbYvVDItvg.en-IN.vtt",
            "video": VIDEOS_DIR / "VtbYvVDItvg.mp4"
        },
        {
            "id": "lffGJ29IhZQ",
            "type": "words",
            "vtt": MANIFESTS_DIR / "lffGJ29IhZQ.en-IN.vtt",
            "video": VIDEOS_DIR / "lffGJ29IhZQ.mp4"
        },
        {
            "id": "DOFPRw6Epl0",
            "type": "words",
            "vtt": MANIFESTS_DIR / "DOFPRw6Epl0.en-IN.vtt",
            "video": VIDEOS_DIR / "DOFPRw6Epl0.mp4"
        }
    ]

    all_dfs = []

    for task in tasks:
        vid_id = task["id"]
        vtt = task["vtt"]
        vid = task["video"]

        if not vid.exists():
            print(f"Skipping {vid_id}: video not found.")
            continue
        if not vtt.exists():
            print(f"Skipping {vid_id}: subtitles not found.")
            continue

        print(f"\n--- Harvesting {vid_id} ({task['type']}) ---")
        if task["type"] == "alphabet":
            intervals = parse_alphabet_intervals(vtt)
        elif task["type"] == "numbers":
            intervals = parse_numbers_intervals(vtt)
        else:
            intervals = parse_general_words_intervals(vtt)

        if not intervals:
            continue

        records = harvest_landmarks_from_video(vid, intervals)
        if records:
            df = pd.DataFrame(records)
            out_path = PROCESSED_DIR / f"{vid_id}_landmarks.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {len(df)} samples to {out_path}")
            all_dfs.append(df)

    if not all_dfs:
        print("No samples harvested!")
        return

    # Merge ALL harvested datasets
    full_df = pd.concat(all_dfs, ignore_index=True)
    feat_cols = [f"f{i}" for i in range(126)]
    full_df = full_df.drop_duplicates(subset=feat_cols)

    MAIN_LANDMARKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(MAIN_LANDMARKS_CSV, index=False)

    print(f"\n==========================================")
    print(f"Dataset successfully compiled exclusively from playlist!")
    print(f"Total clean samples: {len(full_df)}")
    classes = sorted(full_df["label"].unique().tolist())
    print(f"Total classes ({len(classes)}): {classes}")
    print(f"==========================================")

    # Trigger model retraining
    print("\n--- Retraining Model on Playlist Dataset ---")
    from training.train import main as train_model
    train_model()


if __name__ == "__main__":
    run_full_harvest_and_train()
