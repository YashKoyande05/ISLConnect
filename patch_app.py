import re
import numpy as np
import cv2
content = open('app.py', 'r', encoding='utf-8').read()
new_func = '''@app.route("/api/recognize-sequence", methods=["POST"])
def recognize_sequence():
    """
    Takes a single frame (like predict-image) but processes it through 
    the temporal sequence buffer to identify continuous words/glosses.
    """
    if not predictor.ready:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
        
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
        return jsonify({"success": False, "error": "Invalid image"}), 400

    # Get static prediction for this frame
    result = predictor.predict(image)
    
    new_gloss = None
    if result["prediction"] and result["confidence"] > 0:
        new_gloss = sequence_buffer.add_prediction(result["prediction"], result["confidence"])
    
    return jsonify({
        "success": True,
        "frame_prediction": result,
        "new_gloss": new_gloss,
        "full_sequence": sequence_buffer.get_sequence()
    })
'''

new_content = re.sub(
    r'@app\.route\("/api/recognize-sequence".*?return jsonify\(\{\n.*?"full_sequence": sequence_buffer\.get_sequence\(\)\n\s+\}\)\n', 
    new_func, 
    content, 
    flags=re.DOTALL
)

open('app.py', 'w', encoding='utf-8').write(new_content)
