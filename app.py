from pathlib import Path
import base64
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from core.predictor import ISLPredictor

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
predictor = ISLPredictor(BASE_DIR / "model")


def decode_image(data_url: str):
    """Decode a browser data URL or a raw base64 image into OpenCV BGR."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        raw = base64.b64decode(data_url)
        array = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "model_ready": predictor.ready,
        "classes": predictor.labels,
        "message": (
            "Model loaded."
            if predictor.ready
            else "Model not trained yet. Run: python -m training.train"
        )
    })


@app.post("/api/predict")
def predict():
    if not predictor.ready:
        return jsonify({
            "success": False,
            "error": "Model not found. Train the Phase 1 model first."
        }), 503

    if "image" in request.files:
        file = request.files["image"]
        image_bytes = file.read()
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    else:
        payload = request.get_json(silent=True) or {}
        data_url = payload.get("image", "")
        image = decode_image(data_url)

    if image is None:
        return jsonify({
            "success": False,
            "error": "Could not read the supplied image."
        }), 400

    result = predictor.predict(image)

    return jsonify({
        "success": True,
        **result
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
