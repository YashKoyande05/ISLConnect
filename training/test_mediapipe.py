from pathlib import Path
import cv2

from core.feature_extractor import LandmarkFeatureExtractor

IMAGE_PATH = Path("dataset/raw/")


def main():
    if not IMAGE_PATH.exists():
        print(f"ERROR: {IMAGE_PATH} does not exist.")
        print("Put a clear hand image there and run again.")
        return

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        print("ERROR: OpenCV could not read the image.")
        return

    extractor = LandmarkFeatureExtractor()
    features, results = extractor.extract(image)

    if features is None:
        print("No hand detected.")
    else:
        print("Hand detected!")
        print("Landmarks: 21")
        print("Feature vector length:", len(features))
        print("First 9 values:", features[:9])

        extractor.detector.draw(image, results)
        cv2.imshow("ISLENS - MediaPipe Test", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
