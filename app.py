import streamlit as st
import tensorflow as tf
import numpy as np
import uuid
from supabase import create_client
from PIL import Image

CATEGORY_TO_STORAGE = {
    "trinkflasche": "trinkflasche",
    "handschuhe": "handschuhe",
    "mütze": "muetze"   # ❗ Umlaut entfernt
}

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "image_saved" not in st.session_state:
    st.session_state.image_saved = False
    
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

/* Gesamte App schmaler & zentriert */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 900px;
}

/* Header-Leiste oben entfernen */
header {visibility: hidden;}

/* Footer entfernen */
footer {visibility: hidden;}

/* Abstand zwischen Elementen harmonisieren */
.stMarkdown, .stImage, .stButton, .stSelectbox {
    margin-bottom: 1rem;
}

/* Karten-Stil */
.card {
    background-color: #f9fafb;
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
}

/* Ergebnis hervorheben */
.result {
    background-color: #ecfeff;
    border-left: 6px solid #06b6d4;
    padding: 1rem;
    border-radius: 8px;
}

/* Überschriften */
h1, h2, h3 {
    color: #0f172a;
}

</style>
""", unsafe_allow_html=True)

st.title("🧥 Fundbüro – Gefundene Kleidung")
st.markdown(
    "Lade ein Bild eines gefundenen Kleidungsstücks hoch. "
    "Die KI erkennt automatisch die Kategorie und speichert den Fund."
)

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
    "📤 Bild hochladen",
    type=["jpg", "jpeg", "png"],
    key="fundbild_upload"
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    current_file_id = (uploaded_file.name, uploaded_file.size)

    if st.session_state.last_uploaded_file != current_file_id:
        st.session_state.last_uploaded_file = current_file_id
        st.session_state.image_saved = False

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# VERARBEITUNG NACH UPLOAD
# =========================
if uploaded_file is not None:

    # Bild vorbereiten
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # KI-Vorhersage
    predictions = model.predict(image_array)[0]
    best_index = np.argmax(predictions)
    best_label = labels[best_index]
    best_confidence = predictions[best_index] * 100

    # ✅ ERGEBNISANZEIGE MUSS HIER STEHEN
    st.markdown(
        f"""
        <div class="result">
            <b>✅ Erkannte Kategorie:</b><br>
            <span style="font-size:1.2rem;">{best_label}</span><br>
            <span style="color:#0369a1;">Sicherheit: {best_confidence:.1f} %</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ⬇️⬇️⬇️ WICHTIG: Upload passiert HIER und NUR HIER
    if not st.session_state.image_saved:

        image_bytes = uploaded_file.getvalue()
        display_category = best_label.strip().lower()
        storage_category = CATEGORY_TO_STORAGE.get(display_category)

        if storage_category is None:
            st.error(f"❌ Unbekannte Kategorie: '{display_category}'")
            st.stop()

        storage_path = f"{storage_category}/{uuid.uuid4()}.jpg"
        
        supabase.storage.from_("fundbilder").upload(
            path=storage_path,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        image_url = supabase.storage.from_("fundbilder").get_public_url(storage_path)

        supabase.table("fundstuecke").insert({
            "image_url": image_url,
            "storage_path": storage_path,
            "category": display_category,   # für Anzeige
            "confidence": float(best_confidence)
        }).execute()
        
        st.session_state.image_saved = True
        st.success("📦 Fundstück wurde gespeichert!")
    # =========================
    # NUR EINMAL SPEICHERN
    # =========================
    if not st.session_state.image_saved:

        image_bytes = uploaded_file.getvalue()
        filename = f"{uuid.uuid4()}.jpg"

        supabase.storage.from_("fundbilder").upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )

        image_url = supabase.storage.from_("fundbilder").get_public_url(filename)

        supabase.table("fundstuecke").insert({
            "image_url": image_url,
            "category": best_label,
            "confidence": float(best_confidence)
        }).execute()

        st.session_state.image_saved = True
        st.success("📦 Fundstück wurde gespeichert!")

    else:
        st.info("ℹ️ Dieses Bild wurde bereits gespeichert.")
        
# =========================
# FUNDBÜRO DURCHSUCHEN
# =========================

st.markdown("---")
st.header("🔍 Fundstücke durchsuchen")

selected_category = st.selectbox(
    "Kategorie wählen",
    ["Alle"] + labels
)

results = []

if selected_category == "Alle":
    results = supabase.table("fundstuecke").select("*").execute().data
else:
    results = supabase.table("fundstuecke") \
        .select("*") \
        .eq("category", selected_category) \
        .execute().data

if results:
    for item in results:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            st.image(item["image_url"])
        except:
            st.caption("⚠️ Bild nicht verfügbar")
        st.write(f"**Kategorie:** {item['category']}")
        st.write(f"**Sicherheit:** {item['confidence']:.1f} %")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Keine Fundstücke gefunden.")
