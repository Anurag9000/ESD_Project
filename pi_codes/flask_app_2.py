import os
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image, ImageOps
from ai_edge_litert.interpreter import Interpreter

# ==============================
# 1. CONFIGURATION
# ==============================
MODEL_PATH = "waste_classifier.tflite"
UPLOAD_FOLDER = "uploads"
TEMP_FILENAME = "test_image.jpg"
IMG_SIZE = (96, 96)
CLASSES = ['Organic', 'Paper', 'Plastic']
CONFIDENCE_THRESHOLD = 0.85 # 85% Threshold

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)

# ==============================
# 2. LOAD TFLITE MODEL
# ==============================
print("[infer] loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("[infer] model ready")

# ==============================
# 3. PREDICTION FUNCTION
# ==============================
def predict_waste(image_path):
    print("\n[infer] entered predict_waste()", flush=True)

    # Open and Resize
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.contain(img, IMG_SIZE, method=Image.BILINEAR)
    canvas = Image.new("RGB", IMG_SIZE, (0, 0, 0))
    canvas.paste(img, ((IMG_SIZE[0] - img.width) // 2, (IMG_SIZE[1] - img.height) // 2))
    img = canvas
    
    # Preprocess
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    print("Input shape:", img_array.shape, flush=True)

    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    # Get Result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probs = output_data[0]

    print("Probabilities:", probs, flush=True)
    print("Max confidence:", np.max(probs), flush=True)

    max_conf = np.max(probs)
    class_idx = np.argmax(probs)
    original_guess = CLASSES[class_idx].upper()

    print("Predicted class:", original_guess, flush=True)
    
    conf_percent = f"{max_conf * 100:.1f}"

    # ==============================
    # Updated logic for the current inference flow.
    # ==============================
    if max_conf < CONFIDENCE_THRESHOLD:
        final_label = "OTHER"
        detailed_label = f"{original_guess} (LOW CONFIDENCE)"
    else:
        final_label = original_guess
        detailed_label = original_guess

    # Extra line every inference (terminal).
    print(f"Final Output: {final_label} | Predicted: {detailed_label}", flush=True)

    return final_label, conf_percent, detailed_label

# ==============================
# 4. WEB INTERFACE
# ==============================
HTML_PAGE = """
<!doctype html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waste Classifier</title>
    <style>
        body { font-family: sans-serif; text-align: center; padding: 20px; background-color: #f4f4f9; }
        .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .btn { background-color: #007bff; color: white; padding: 15px 30px; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; width: 100%; margin-top: 10px;}
        input[type=file] { padding: 10px; border: 2px dashed #ddd; width: 90%; margin-bottom: 20px; }
        
        .result-box { margin-top: 20px; padding: 20px; border-radius: 8px; }
        .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
        
        .main-label { font-size: 36px; font-weight: bold; margin: 10px 0; }
        .details { font-size: 16px; color: #555; }
        .threshold-note { font-size: 12px; color: #999; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Smart Waste Sorter</h1>
        <form action="/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" capture="camera">
            <br>
            <button type="submit" class="btn">Analyze Waste</button>
        </form>

        {% if result %}
            {% if result == 'OTHER' %}
                <div class="result-box warning">
                    <div class="main-label">OTHER</div>
                    <p class="details">
                        Predicted: <b>{{ original }}</b><br>
                        Confidence: <b>{{ confidence }}%</b>
                    </p>
                    <p class="threshold-note">(Needs > {{ threshold * 100 }}% to be sure)</p>
                </div>
            {% else %}
                <div class="result-box success">
                    <div class="main-label">{{ result }}</div>
                    <p class="details">
                        Prediction: <b>{{ original }}</b><br>
                        Confidence: <b>{{ confidence }}%</b>
                    </p>
                </div>
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""

# ==============================
# 5. SERVER ROUTES
# ==============================
@app.route('/', methods=['GET', 'POST'])
def index():
    print("[infer] index function called", flush=True)

    prediction = None
    confidence = None
    original_guess = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        save_path = os.path.join(UPLOAD_FOLDER, TEMP_FILENAME)
        file.save(save_path)
        
        prediction, confidence, original_guess = predict_waste(save_path)
        
        os.remove(save_path)

    return render_template_string(HTML_PAGE, 
                                  result=prediction, 
                                  confidence=confidence, 
                                  original=original_guess,
                                  threshold=CONFIDENCE_THRESHOLD)

# ==============================
# 6. START SERVER
# ==============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
