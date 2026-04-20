import os
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image, ImageOps
from ai_edge_litert.interpreter import Interpreter

# ==============================
# 1. CONFIGURATION
# ==============================
MODEL_PATH = "waste_classifier.tflite"
IMAGE_PATH = "captured.jpg"   # captured image
IMG_SIZE = (96, 96)           # match your model
CLASSES = ['Organic', 'Paper', 'Plastic']
CONFIDENCE_THRESHOLD = 0.85

app = Flask(__name__)

# ==============================
# 2. LOAD MODEL
# ==============================
print("[infer] loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("[infer] model ready")

# ==============================
# 3. CAPTURE IMAGE FROM CAMERA
# ==============================
def capture_image():
    print("[infer] capturing image...", flush=True)
    os.system(f"rpicam-jpeg -o {IMAGE_PATH} --width 96 --height 96 --immediate")
    print("[infer] image captured", flush=True)

# ==============================
# 4. PREDICTION FUNCTION
# ==============================
def predict_waste(image_path):
    print("\n[infer] entered predict_waste()", flush=True)

    img = Image.open(image_path).convert('RGB')
    img = ImageOps.contain(img, IMG_SIZE, method=Image.BILINEAR)
    canvas = Image.new("RGB", IMG_SIZE, (0, 0, 0))
    canvas.paste(img, ((IMG_SIZE[0] - img.width) // 2, (IMG_SIZE[1] - img.height) // 2))
    img = canvas

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    probs = output_data[0]

    max_conf = np.max(probs)
    class_idx = np.argmax(probs)
    original_guess = CLASSES[class_idx].upper()

    conf_percent = f"{max_conf * 100:.1f}"

    print("Probabilities:", probs, flush=True)
    print("Predicted:", original_guess, flush=True)

    # SAME LOGIC (unchanged)
    if max_conf < CONFIDENCE_THRESHOLD:
        final_label = "OTHER"
        detailed_label = f"{original_guess} (LOW CONFIDENCE)"
    else:
        final_label = original_guess
        detailed_label = original_guess

    print(f"Final Output: {final_label} | Predicted: {detailed_label}", flush=True)

    return final_label, conf_percent, detailed_label

# ==============================
# 5. WEB UI
# ==============================
HTML_PAGE = """
<!doctype html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waste Classifier</title>
</head>
<body style="text-align:center; font-family:sans-serif;">

<h1>Smart Waste Sorter</h1>

<form method="post">
    <button type="submit" style="padding:20px; font-size:20px;">
        Capture & Analyze
    </button>
</form>

{% if result %}
    <h2>Result: {{ result }}</h2>
    <p>Prediction: {{ original }}</p>
    <p>Confidence: {{ confidence }}%</p>
{% endif %}

</body>
</html>
"""

# ==============================
# 6. ROUTE
# ==============================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    original_guess = None

    if request.method == 'POST':
        print("[infer] button pressed from phone", flush=True)

        capture_image()  # take photo

        prediction, confidence, original_guess = predict_waste(IMAGE_PATH)

    return render_template_string(HTML_PAGE,
                                  result=prediction,
                                  confidence=confidence,
                                  original=original_guess)

# ==============================
# 7. RUN
# ==============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
