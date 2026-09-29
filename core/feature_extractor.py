import numpy as np
from .hand_detector import HandDetector


class LandmarkFeatureExtractor:
    """
    Converts a detected hand into a normalized 63-value vector.

    21 landmarks × (x, y, z) = 63 features.
    Coordinates are translated relative to the wrist and scaled
    by the largest absolute coordinate value.
    """

    def __init__(self):
        self.detector = HandDetector()

    def extract_from_results(self, results):
        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark],
            dtype=np.float32
        )

        # Wrist is landmark 0.
        points = points - points[0]

        scale = np.max(np.abs(points))
        if scale > 1e-8:
            points = points / scale

        return points.flatten()

    def extract(self, image_bgr):
        results = self.detector.process(image_bgr)
        features = self.extract_from_results(results)
        return features, results
