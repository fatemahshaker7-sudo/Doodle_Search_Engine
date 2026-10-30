"""Shared preprocessing, embedding and search. Used by every script and the app."""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

MODEL_SIZE = 96   # MobileNetV2 input size (small, so fast on a CPU)

def load_model():
    """MobileNetV2 without its classifier: one 1280-number vector per image."""
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                       input_shape=(MODEL_SIZE, MODEL_SIZE, 3))

def to_model_input(images_28, invert=True):
    """Batch of 28x28 doodles -> array the model expects."""
    x = np.asarray(images_28, dtype="float32").reshape(-1, 28, 28, 1)
    if invert:
        x = 255.0 - x                                   # black strokes on white
    x = tf.image.resize(x, (MODEL_SIZE, MODEL_SIZE)).numpy()
    return preprocess_input(np.repeat(x, 3, axis=-1))   # gray -> 3 channels

def embed(model, images_28, invert=True, batch_size=64):
    """Embed in batches and L2-normalize, so cosine similarity = dot product."""
    batches = [model.predict(to_model_input(images_28[i:i + batch_size], invert), verbose=0)
               for i in range(0, len(images_28), batch_size)]
    return tf.math.l2_normalize(np.concatenate(batches), axis=1).numpy()

def _to_gray(image):
    """PIL image -> 2D grayscale array (transparent areas become white)."""
    image = image.convert("RGBA")
    white = Image.new("RGBA", image.size, "white")
    return np.asarray(Image.alpha_composite(white, image).convert("L"))

def preprocess_query(image):
    """Any drawing -> 28x28 uint8, white strokes on black like the gallery.
    Returns None if the drawing is blank."""
    gray = _to_gray(image)
    low, high = int(gray.min()), int(gray.max())
    if high - low < 40:                                        # no contrast: blank
        return None
    mask = gray < (low + high) / 2                             # dark pen on light paper

    strokes = Image.fromarray(mask.astype("uint8") * 255)
    strokes = strokes.crop(strokes.getbbox())                  # crop to the strokes
    w, h = strokes.size
    size = max(w, h) + 2 * (int(max(w, h) * 0.15) + 1)         # square with a small margin
    square = Image.new("L", (size, size))                      # black square
    square.paste(strokes, ((size - w) // 2, (size - h) // 2))  # strokes in the center

    return np.asarray(square.resize((28, 28), Image.BOX))

def search(query_vector, embeddings, k=25):
    """Indices and cosine scores of the k most similar gallery vectors."""
    scores = embeddings @ query_vector                         # normalized -> dot product
    best = np.argsort(-scores)[:k]
    return best, scores[best]