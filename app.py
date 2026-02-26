"""
Fundbüro-App
--------------
Diese App ermöglicht:
- Upload von Kleidungsbildern
- KI-Kategorisierung mit TensorFlow (Teachable Machine)
- Dominante Farberkennung
- Speicherung in Supabase

Geeignet für Schul- und Lernprojekte
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from supabase import create_client, Client
import io

# ==============================
# 🔹 KONFIGURATION
# ==============================

MODEL_PATH = "model/model.h5"
LABELS_PATH = "labels.txt"

SUPABASE_URL = "DEINE_SUPABASE_URL"
SUPABASE_KEY = "DEIN_SUPABASE_ANON_KEY"

# ==============================
# 🔹 SUPABASE VERBINDUNG
# ==============================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==============================
# 🔹 MODELL LADEN
# ==============================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Klassenlabels laden
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# ==============================
# 🔹 BILDPREPROCESSING
# ==============================

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Standardgröße Teachable Machine
    img_array = np.array(image)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================
# 🔹 KI-VORHERSAGE
# ==============================

def predict_image(image: Image.Image):
    processed = preprocess_image(image)
    predictions = model.predict(processed)[0]

    results = {}
    for i, prob in enumerate(predictions):
        results[class_names[i]] = float(prob * 100)

    best_category = max(results, key=results.get)

    return best_category, results

# ==============================
# 🔹 FARBANALYSE
# ==============================

def detect_dominant_colors(image: Image.Image, k=3):
    """
    Bestimmt dominante Farben mit K-Means Clustering
    """
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.reshape((-1, 3))

    img = np.float32(img)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    colors = []
    for center in centers:
        colors.append(f"RGB{tuple(center)}")

    return colors

# ==============================
# 🔹 SUPABASE SPEICHERN
# ==============================

def save_to_supabase(image_file, category, probabilities, colors):
    timestamp = datetime.now().isoformat()

    # Bild als Bytes speichern
    image_bytes = image_file.getvalue()

    # Bild in Supabase Storage hochladen
    file_name = f"uploads/{timestamp}.png"
    supabase.storage.from_("fundbilder").upload(file_name, image_bytes)

    public_url = supabase.storage.from_("fundbilder").get_public_url(file_name)

    # Daten in Tabelle speichern
    supabase.table("fundstuecke").insert({
        "bild_url": public_url,
        "kategorie": category,
        "wahrscheinlichkeiten": probabilities,
        "farben": colors,
        "zeitstempel": timestamp
    }).execute()

# ==============================
# 🔹 STREAMLIT UI
# ==============================

st.set_page_config(page_title="Digitales Fundbüro", page_icon="👕")

st.title("👕 Digitales Fundbüro für Kleidungsstücke")
st.write("Lade ein Bild eines gefundenen Kleidungsstücks hoch.")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Hochgeladenes Bild")
    st.image(image, use_column_width=True)

    # KI-Analyse
    st.subheader("🤖 KI-Analyse läuft...")
    category, probabilities = predict_image(image)

    st.success(f"Erkannte Kategorie: **{category}**")

    st.subheader("📊 Wahrscheinlichkeiten")
    for label, prob in probabilities.items():
        st.write(f"{label}: {prob:.2f} %")

    # Farbanalyse
    st.subheader("🎨 Dominante Farben")
    colors = detect_dominant_colors(image)
    for color in colors:
        st.write(color)

    # Speichern Button
    if st.button("💾 In Fundbüro speichern"):
        save_to_supabase(uploaded_file, category, probabilities, colors)
        st.success("Erfolgreich in Supabase gespeichert!")

st.markdown("---")
st.caption("Projekt für Schulzwecke – KI mit TensorFlow & Supabase")
