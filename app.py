import streamlit as st
from PIL import Image

from ki_kleidung import predict_clothing
from farbanalyse import detect_dominant_colors

st.title("👕 Digitales Fundbüro")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    category, probs = predict_clothing(image)
    colors = detect_dominant_colors(image)

    st.subheader("🤖 KI-Ergebnis")
    st.write("Kategorie:", category)

    st.subheader("📊 Wahrscheinlichkeiten")
    st.json(probs)

    st.subheader("🎨 Farben")
    st.write(colors)
