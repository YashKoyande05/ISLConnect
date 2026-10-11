"""
Convert an image-folder dataset into a CSV of normalized MediaPipe landmarks.

Expected dataset:

dataset/raw/
    A/
       image1.jpg
       image2.jpg
    B/
       ...
    0/
       ...

The folder name becomes the class label.
"""

from pathlib import Path
import argparse
import cv2
import pandas as pd

from core.feature_extractor import LandmarkFeatureExtractor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images_from_roboflow(dataset_dir, max_per_class=0):
    """
    Extracts images and ground-truth labels from a Roboflow dataset directory.
    Uses _classes.csv inside train/valid/test splits.
    """
    dataset_dir = Path(dataset_dir)
    class_counts = {}

    splits = [dataset_dir / s for s in ["train", "valid", "test"] if (dataset_dir / s).exists()]
    if not splits:
        splits = [dataset_dir]

    for current_dir in splits:
        csv_path = current_dir / "_classes.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                filename = row["filename"]
                fname_lower = filename.lower()
                
                # Heuristics for ISL-words dataset filenames
                if "hello" in fname_lower:
                    label = "Hello"
                elif "iloveyou" in fname_lower or "i-love-you" in fname_lower:
                    label = "ILoveYou"
                elif "please" in fname_lower:
                    label = "Please"
                elif "thanks" in fname_lower:
                    label = "Thanks"
                elif "yes" in fname_lower:
                    label = "Yes"
                elif "no" in fname_lower:
                    label = "No"
                else:
                    # One-hot column match for alphabet/numbers
                    label = None
                    for c in df.columns[1:]:
                        if row[c] == 1:
                            label = c.strip()
                            break

                if label:
                    if max_per_class > 0 and class_counts.get(label, 0) >= max_per_class:
                        continue
                    img_path = current_dir / filename
                    if img_path.exists():
                        class_counts[label] = class_counts.get(label, 0) + 1
                        yield label, img_path
        else:
            for label_dir in sorted(current_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                label = label_dir.name
                for image_path in sorted(label_dir.rglob("*")):
                    if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        if max_per_class > 0 and class_counts.get(label, 0) >= max_per_class:
                            break
                        class_counts[label] = class_counts.get(label, 0) + 1
                        yield label, image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset/processed/landmarks.csv")
    parser.add_argument("--max-per-class", type=int, default=120, help="Max images per letter/number class")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = LandmarkFeatureExtractor()
    rows = []
    total = 0
    detected = 0

    # 1. Harvest all word images from ISL-words-2 (Hello, Please, Thanks, Yes, No, ILoveYou)
    words_dir = Path("dataset/ISL-words-2")
    if words_dir.exists():
        print(f"Extracting words from {words_dir}...")
        for label, img_path in collect_images_from_roboflow(words_dir, max_per_class=0):
            total += 1
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            features, _ = extractor.extract(img)
            if features is None:
                continue
            rows.append(list(features) + [label])
            detected += 1
            if detected % 200 == 0:
                print(f"Processed {detected} samples with detected hands...")

    # 2. Harvest letters and numbers from indian-sign-language-1
    letters_dir = Path("dataset/indian-sign-language-1")
    if letters_dir.exists():
        print(f"\nExtracting letters & numbers from {letters_dir} (capped at {args.max_per_class}/class)...")
        for label, img_path in collect_images_from_roboflow(letters_dir, max_per_class=args.max_per_class):
            total += 1
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            features, _ = extractor.extract(img)
            if features is None:
                continue
            rows.append(list(features) + [label])
            detected += 1
            if detected % 200 == 0:
                print(f"Processed {detected} samples with detected hands...")

    if not rows:
        raise SystemExit("No hand landmarks were extracted. Check dataset directories.")

    n_features = len(rows[0]) - 1
    columns = [f"f{i}" for i in range(n_features)] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)

    print("\nExtraction complete.")
    print("Images scanned:", total)
    print("Hands detected:", detected)
    print(f"Saved to: {output_path}")
    print("\nClass summary:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()

