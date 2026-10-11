import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path

# Path to the hand landmark model (downloaded separately)
_DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "model" / "hand_landmarker.task"


class _FakeLandmarks:
    """Mimics the old mp.solutions.hands result structure."""

    def __init__(self, landmarks_list):
        self.multi_hand_landmarks = landmarks_list or None


class _HandLandmarkWrapper:
    """Wraps a single hand's landmarks to expose .landmark attribute."""

    def __init__(self, landmark_list):
        self.landmark = landmark_list


class HandDetector:
    """MediaPipe Tasks-based single-hand detector (MediaPipe >= 1.0).

    Returns results that are compatible with the rest of the codebase
    (results.multi_hand_landmarks is a list of objects with .landmark).
    """

    # MediaPipe landmark connection pairs (same as classic mp.solutions.hands)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),            # index
        (0, 9), (9, 10), (10, 11), (11, 12),        # middle
        (0, 13), (13, 14), (14, 15), (15, 16),      # ring
        (0, 17), (17, 18), (18, 19), (19, 20),      # pinky
        (5, 9), (9, 13), (13, 17),                  # palm
    ]

    def __init__(self, max_num_hands=2, min_detection_confidence=0.55,
                 model_path=None):
        model_path = Path(model_path) if model_path else _DEFAULT_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmark model not found at: {model_path}\n"
                "Download it from:\n"
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.1,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, image_bgr):
        """Process a BGR image and return a result with .multi_hand_landmarks."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._landmarker.detect(mp_image)

        if not detection.hand_landmarks:
            return _FakeLandmarks(None)

        # Wrap each hand's normalized landmarks into a compatible object
        wrapped = [
            _HandLandmarkWrapper(hand_lms)
            for hand_lms in detection.hand_landmarks
        ]
        return _FakeLandmarks(wrapped)

    def draw(self, image_bgr, results):
        """Draw hand landmarks on the image."""
        if not results.multi_hand_landmarks:
            return image_bgr

        h, w = image_bgr.shape[:2]
        for hand in results.multi_hand_landmarks:
            pts = hand.landmark
            # Draw connections
            for (a, b) in self.HAND_CONNECTIONS:
                x1 = int(pts[a].x * w)
                y1 = int(pts[a].y * h)
                x2 = int(pts[b].x * w)
                y2 = int(pts[b].y * h)
                cv2.line(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw landmark dots
            for lm in pts:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image_bgr, (cx, cy), 4, (255, 0, 0), -1)

        return image_bgr
