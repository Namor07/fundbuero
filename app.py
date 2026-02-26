import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from supabase import create_client
import uuid
import io

# ------------------------------
# Supabase Konfiguration
# ------------------------------
SUPABASE_URL = "https://DEIN-PROJEKT.supabase.co"
SUPABASE_KEY = "DEIN_SERVICE_ROLE_KEY"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------
# KI-Modell laden
# ------------------------------
model = tf.keras.models.load_model("model/keras_model.h5", compile=False)

# Labels laden
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("👕 Digitales Fundbüro")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])

if uploaded_file:
    # Bild anzeigen
    st.image(uploaded_file, caption="Hochgeladenes Bild", use_column_width=True)

    # --------------------------
    # Supabase Upload
    # --------------------------
    filename = f"{uuid.uuid4()}.jpg"
    image_bytes = uploaded_file.read()
    supabase.storage.from_("fundbilder").upload(filename, image_bytes, {"content-type":"image/jpeg"})
    public_url = supabase.storage.from_("fundbilder").get_public_url(filename)
    st.write(f"📦 Bild gespeichert: [Link]({public_url})")

    # --------------------------
    # KI-Klassifikation
    # --------------------------
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.asarray(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)[0]
    # Ergebnisse sortieren
    results = sorted(
        [{"label": l, "percent": float(p)*100} for l,p in zip(labels, predictions)],
        key=lambda x: x["percent"],
        reverse=True
    )

    # --------------------------
    # Ergebnisse anzeigen
    # --------------------------
    st.subheader("🔍 KI-Ergebnis")
    for i, r in enumerate(results):
        if i==0:
            st.markdown(f"**{r['label']} – {r['percent']:.2f}%**")
        else:
            st.write(f"{r['label']} – {r['percent']:.2f}%")
