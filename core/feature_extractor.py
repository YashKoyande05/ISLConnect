import numpy as np
from .hand_detector import HandDetector


class LandmarkFeatureExtractor:
    """
    Converts detected hand(s) into a normalized 126-value vector.

    Supports both 1-handed and 2-handed Indian Sign Language (ISL) signs:
    - Hand 1 (Left hand): 21 landmarks × 3 (x, y, z) = 63 features
    - Hand 2 (Right hand): 21 landmarks × 3 (x, y, z) = 63 features
    If only one hand is raised, Hand 2 is padded with zeros.
    Total: 126 features.
    """

    def __init__(self):
        self.detector = HandDetector(max_num_hands=2)

    def extract_from_results(self, results):
        if not results or not results.multi_hand_landmarks:
            return None

        raw_hands = results.multi_hand_landmarks
        
        # In webcam signing or tutorial videos, a hand resting near the lap (wrist y > 0.72)
        # while another hand is raised in signing space (wrist y <= 0.72) is an inactive hand.
        # Filter out resting hands when at least one active hand is present.
        active_hands = [h for h in raw_hands if h.landmark[0].y <= 0.72]
        
        if active_hands:
            hands = active_hands
        else:
            # If all detected hands are lower down, treat them as the available hands
            hands = raw_hands
        
        # Sort hands deterministically left-to-right based on wrist x-coordinate
        sorted_hands = sorted(hands, key=lambda h: h.landmark[0].x)

        features = []
        for i in range(2):
            if i < len(sorted_hands):
                hand = sorted_hands[i]
                points = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand.landmark],
                    dtype=np.float32
                )
                # Wrist is landmark 0
                points = points - points[0]
                scale = np.max(np.abs(points))
                if scale > 1e-8:
                    points = points / scale
                features.append(points.flatten())
            else:
                # 63 zeros when second hand is absent
                features.append(np.zeros(63, dtype=np.float32))

        return np.concatenate(features)

    def extract(self, image_bgr):
        results = self.detector.process(image_bgr)
        features = self.extract_from_results(results)
        return features, results
