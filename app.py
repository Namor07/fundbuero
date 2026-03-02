import streamlit as st
import tensorflow as tf
import numpy as np
import uuid
from supabase import create_client
from PIL import Image

# =========================
# STREAMLIT SEITENLAYOUT
# =========================
st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🧥",
    layout="centered"
)

# =========================
# STYLING (CSS)
# =========================
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}
.card {
    background-color: #f7f7f7;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}
.result {
    font-size: 24px;
    font-weight: bold;
    color: #2E8B57;
    text-align: center;
}
</style>

<div class="title">🧥 Digitales Fundbüro</div>
<div class="subtitle">
Lade ein Bild eines gefundenen Kleidungsstücks hoch<br>
und lasse es automatisch erkennen
</div>
""", unsafe_allow_html=True)

# =========================
# SUPABASE VERBINDUNG
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# KI-MODELL LADEN
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/keras_model.h5")

model = load_model()

# Labels laden
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# =========================
# BILD UPLOAD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Bild eines Kleidungsstücks hochladen",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# VERARBEITUNG NACH UPLOAD
# =========================
if uploaded_file is not None:

    # =========================
    # BILD ANZEIGEN
    # =========================
    st.image(uploaded_file, caption="📷 Hochgeladenes Fundstück", use_column_width=True)

    # =========================
    # BILD FÜR KI VORBEREITEN
    # =========================
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # =========================
    # KI VORHERSAGE
    # =========================
    predictions = model.predict(image_array)[0]
    best_index = np.argmax(predictions)
    best_label = labels[best_index]
    best_confidence = predictions[best_index] * 100

    st.success(f"✅ Erkannte Kategorie: {best_label} ({best_confidence:.1f} %)")

    # =========================
    # DATEI VORBEREITEN
    # =========================
    image_bytes = uploaded_file.getvalue()
    filename = f"{uuid.uuid4()}.jpg"

    # =========================
    # SUPABASE STORAGE UPLOAD
    # =========================
    supabase.storage.from_("fundbilder").upload(
        path=filename,
        file=image_bytes,
        file_options={"content-type": "image/jpeg"}
    )

    image_url = supabase.storage.from_("fundbilder").get_public_url(filename)

    # =========================
    # DATENBANK SPEICHERN
    # =========================
    supabase.table("fundstuecke").insert({
        "image_url": image_url,
        "category": best_label,
        "confidence": float(best_confidence)
    }).execute()

    st.success("📦 Fundstück erfolgreich gespeichert!")

# =========================
# FUNDBÜRO DURCHSUCHEN
# =========================
st.markdown("## 🔍 Fundstücke durchsuchen")

# Kategorien aus Labels
selected_category = st.selectbox(
    "Kategorie auswählen",
    options=["Alle"] + labels
)

# Daten abrufen
if selected_category == "Alle":
    response = supabase.table("fundstuecke").select("*").order("created_at", desc=True).execute()
else:
    response = supabase.table("fundstuecke") \
        .select("*") \
        .eq("category", selected_category) \
        .order("created_at", desc=True) \
        .execute()

data = response.data

# =========================
# ERGEBNISSE ANZEIGEN
# =========================
if not data:
    st.info("Keine Fundstücke gefunden.")
else:
    for item in data:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.image(item["image_url"], use_column_width=True)
        st.markdown(
            f"""
            <b>Kategorie:</b> {item['category']}<br>
            <b>KI-Sicherheit:</b> {item['confidence']:.1f} %
            """,
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)
