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
    # Support Roboflow format with _classes.csv
    for split_dir in ["train", "valid", "test", ""]:
        current_dir = dataset_dir / split_dir if split_dir else dataset_dir
        if not current_dir.exists():
            continue
            
        csv_path = current_dir / "_classes.csv"
        if csv_path.exists():
            print(f"Found _classes.csv in {current_dir}")
            df = pd.read_csv(csv_path)
            # Find label column
            classes = df.columns[1:]
            for _, row in df.iterrows():
                filename = row["filename"]
                
                # Heuristic for mislabeled Roboflow ISL Words dataset
                fname_lower = filename.lower()
                if 'hello' in fname_lower: label = 'Hello'
                elif 'iloveyou' in fname_lower or 'i-love-you' in fname_lower: label = 'ILoveYou'
                elif 'please' in fname_lower: label = 'Please'
                elif 'thanks' in fname_lower: label = 'Thanks'
                elif 'yes' in fname_lower: label = 'Yes'
                elif 'no' in fname_lower: label = 'No'
                else:
                    # Fallback to CSV for alphabet dataset
                    label = None
                    for c in classes:
                        if row[c] == 1:
                            label = c.strip()
                            break
                if label:
                    img_path = current_dir / filename
                    if img_path.exists():
                        yield label, img_path
        else:
            # Fallback to directory structure
            for label_dir in sorted(current_dir.iterdir()):
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
    parser.add_argument("--max-images", type=int, default=0, help="Max images to process (0 for all)")
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
        if args.max_images > 0 and total >= args.max_images:
            print(f"Reached max images limit ({args.max_images})")
            break
            
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
