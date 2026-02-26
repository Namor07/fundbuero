import numpy as np
from PIL import Image

def detect_dominant_colors(image: Image.Image, k=3):
    image = image.resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)

    # Zufällige Startzentren
    centers = pixels[np.random.choice(len(pixels), k, replace=False)]

    for _ in range(10):
        distances = np.linalg.norm(pixels[:, None] - centers[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        for i in range(k):
            centers[i] = pixels[labels == i].mean(axis=0)

    centers = centers.astype(int)
    return [f"RGB{tuple(color)}" for color in centers]
