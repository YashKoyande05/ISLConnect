from pathlib import Path
import json
import joblib
import numpy as np

from .feature_extractor import LandmarkFeatureExtractor


class ISLPredictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "isl_model.joblib"
        self.labels_path = self.model_dir / "labels.json"

        self.extractor = LandmarkFeatureExtractor()
        self.model = None
        self.labels = []

        self._load()

    def _load(self):
        if self.model_path.exists() and self.labels_path.exists():
            self.model = joblib.load(self.model_path)
            self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))

    @property
    def ready(self):
        return self.model is not None and bool(self.labels)

    def reload(self):
        self.model = None
        self.labels = []
        self._load()

    def predict(self, image_bgr):
        features, results = self.extractor.extract(image_bgr)

        if features is None:
            return {
                "prediction": None,
                "confidence": 0.0,
                "hand_detected": False,
                "message": "No hand detected. Try a clear, well-lit image."
            }

        probabilities = self.model.predict_proba(
            np.asarray(features, dtype=np.float32).reshape(1, -1)
        )[0]

        index = int(np.argmax(probabilities))
        prediction = self.labels[index]
        confidence = float(probabilities[index])

        return {
            "prediction": prediction,
            "confidence": confidence,
            "hand_detected": True,
            "message": "Prediction generated."
        }
