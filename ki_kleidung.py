import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "labels.txt"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_clothing(image: Image.Image):
    img = preprocess_image(image)
    predictions = model.predict(img)[0]

    results = {
        class_names[i]: float(pred * 100)
        for i, pred in enumerate(predictions)
    }

    best_category = max(results, key=results.get)
    return best_category, results
