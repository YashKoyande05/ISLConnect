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


def collect_images(dataset_dir):
    for label_dir in sorted(dataset_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield label, image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/raw")
    parser.add_argument("--output", default="dataset/processed/landmarks.csv")
    args = parser.parse_args()

    dataset_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    extractor = LandmarkFeatureExtractor()
    rows = []
    total = 0
    detected = 0

    for label, image_path in collect_images(dataset_dir):
        total += 1
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[SKIP] Cannot read: {image_path}")
            continue

        features, _ = extractor.extract(image)
        if features is None:
            print(f"[SKIP] No hand detected: {image_path}")
            continue

        rows.append(list(features) + [label])
        detected += 1

        if detected % 250 == 0:
            print(f"Processed {detected} images with detected hands...")

    if not rows:
        raise SystemExit(
            "No hand landmarks were extracted. Check dataset paths/images."
        )

    columns = [f"f{i}" for i in range(63)] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)

    print("\nExtraction complete.")
    print("Images scanned:", total)
    print("Hands detected:", detected)
    print("CSV:", output_path)
    print("\nClass counts:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
