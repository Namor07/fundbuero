from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from supabase import create_client
import uuid
import io

# ================================
# Supabase Konfiguration
# ================================
SUPABASE_URL = "https://DEIN-PROJEKT.supabase.co"
SUPABASE_KEY = "DEIN_SERVICE_ROLE_KEY"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================================
# Flask & KI Setup
# ================================
app = Flask(__name__)
model = tf.keras.models.load_model("model/keras_model.h5")

with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# ================================
# Hauptseite
# ================================
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            # Eindeutiger Dateiname
            filename = f"{uuid.uuid4()}.jpg"

            # Bild in Bytes umwandeln
            image_bytes = file.read()

            # Upload zu Supabase Storage
            supabase.storage.from_("fundbilder").upload(
                filename,
                image_bytes,
                {"content-type": "image/jpeg"}
            )

            # Öffentliche Bild-URL
            image_url = supabase.storage.from_("fundbilder").get_public_url(filename)

            # KI-Bildverarbeitung
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = image.resize((224, 224))
            image_array = np.asarray(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array)[0]

            results = []
            for label, value in zip(labels, predictions):
                results.append({
                    "label": label,
                    "percent": round(float(value) * 100, 2)
                })

            results.sort(key=lambda x: x["percent"], reverse=True)

    return render_template(
        "index.html",
        results=results,
        image_url=image_url
    )

if __name__ == "__main__":
    app.run(debug=True)
