import os
import re
import subprocess
from datetime import datetime
from flask import Flask, request, render_template_string

# ==============================
# 1. CONFIGURATION
# ==============================
SAVE_FOLDER = "captured_images"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

app = Flask(__name__)

# Try new + old camera commands
CAPTURE_COMMANDS = [
    "rpicam-still",
    "libcamera-still",
]

# ==============================
# 2. WEB INTERFACE
# ==============================
HTML_PAGE = """
<!doctype html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pi Capture</title>
</head>
<body style="background:black;color:white;text-align:center;font-family:sans-serif;">
    <h2>📸 Pi Camera</h2>

    <form method="post">
        <input type="text" name="item_name" placeholder="Enter name" required
               style="padding:10px;font-size:18px;">
        <br><br>
        <button type="submit" style="padding:10px 20px;font-size:18px;">
            Capture
        </button>
    </form>

    {% if message %}
        <p>{{ message }}</p>
    {% endif %}
</body>
</html>
"""

# ==============================
# 3. HELPERS
# ==============================
def safe_filename(text):
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text if text else "unnamed"


def capture_image(path):
    for cmd in CAPTURE_COMMANDS:
        try:
            subprocess.run(
                [cmd, "-o", path, "-t", "1000", "--nopreview"],
                check=True
            )
            return True, cmd
        except Exception:
            continue
    return False, "Camera command failed"


# ==============================
# 4. ROUTE
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    message = None

    if request.method == "POST":
        name = request.form.get("item_name", "")
        safe_name = safe_filename(name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.jpg"
        filepath = os.path.join(SAVE_FOLDER, filename)

        ok, info = capture_image(filepath)

        if ok:
            message = f"Saved: {filename}"
        else:
            message = f"Error: {info}"

    return render_template_string(HTML_PAGE, message=message)


# ==============================
# 5. RUN
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)