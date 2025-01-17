import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from datetime import datetime
from facedb import FaceDB

# Initialize Flask app and FaceDB instance
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a strong secret key
db = FaceDB(
            path="facedata",
            metric="euclidean",
            embedding_dim=128,
            module="face_recognition",
        )

# Function to save image with a timestamp
def save_image_with_timestamp(img_bytes, folder="uploads"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    file_path = os.path.join(folder, filename)
    with open(file_path, "wb") as f:
        f.write(img_bytes)
    return file_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            name = request.form.get("name", "").strip()
            img_file = request.files.get("image")

            if not name:
                flash("Name is required!", "error")
                return redirect(url_for("index"))

            if not img_file or img_file.filename == "":
                flash("No file selected!", "error")
                return redirect(url_for("index"))

            img_bytes = img_file.read()
            if not img_bytes:
                flash("Uploaded file is empty!", "error")
                return redirect(url_for("index"))

            # Save the image with a timestamp
            save_image_with_timestamp(img_bytes)

            # Add face to database
            face_id = db.add(name, img=img_bytes)
            flash(f"Face added successfully with ID: {face_id}", "success")
            return redirect(url_for("index"))
        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("index"))
    return render_template("index.html")

@app.route("/recognize", methods=["GET", "POST"])
def recognize():
    if request.method == "POST":
        img_file = request.files.get("image")

        if not img_file or img_file.filename == "":
            flash("No file selected!", "error")
            return redirect(url_for("recognize"))

        img_bytes = img_file.read()
        if not img_bytes:
            flash("Uploaded file is empty!", "error")
            return redirect(url_for("recognize"))

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = db.recognize(img=img, include=["name"])

        if result and result["id"]:
            flash(f"Recognized as {result['name']} ({result['confidence']:.2f}%)", "success")
        else:
            flash("Unknown face", "error")
        return redirect(url_for("recognize"))
    return render_template("recognize.html")

if __name__ == "__main__":
    app.run(debug=True)
