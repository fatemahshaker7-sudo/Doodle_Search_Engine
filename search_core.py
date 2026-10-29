"""Shared preprocessing, embedding and search. Used by every script and the app."""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

SIZE = 96          # MobileNetV2 input (smallest sensible size -> fast on CPU)
INVERT = True      # black strokes on white look more like ImageNet photos


def load_model():
    # 1280-d vector per image, no classification layer
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                       input_shape=(SIZE, SIZE, 3))


def to_model_input(images_28, invert=INVERT):
    x = np.asarray(images_28, dtype="float32").reshape(-1, 28, 28, 1)
    if invert:
        x = 255.0 - x
    x = tf.image.resize(x, (SIZE, SIZE)).numpy()
    x = np.repeat(x, 3, axis=-1)
    return preprocess_input(x)


def embed(model, images_28, invert=INVERT, batch=64):
    out = []
    for i in range(0, len(images_28), batch):
        v = model.predict(to_model_input(images_28[i:i + batch], invert), verbose=0)
        out.append(v)
    v = np.concatenate(out)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def _otsu(g):
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    var = (m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-12)
    return int(np.argmax(var))


def preprocess_query(image):
    """PIL image or array (gray/RGB/RGBA) -> 28x28 uint8, white strokes on black.
    Returns None if there are no strokes."""
    img = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype("uint8"))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    g = np.asarray(img.convert("L"))
    if int(g.max()) - int(g.min()) < 40:       # blank
        return None
    mask = g < _otsu(g)                         # dark pen on light paper
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype("uint8") * 255
    h, w = crop.shape
    side = max(h, w)
    pad = int(side * 0.15) + 1                  # margin
    canvas = np.zeros((side + 2 * pad,) * 2, dtype="uint8")
    y0, x0 = (canvas.shape[0] - h) // 2, (canvas.shape[1] - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    pil = Image.fromarray(canvas)
    k = max(3, int(canvas.shape[0] / 14) | 1)   # thicken so strokes survive shrinking
    pil = pil.filter(ImageFilter.MaxFilter(k)).resize((28, 28), Image.BOX)
    out = np.asarray(pil).astype("float32")
    out = out / max(out.max(), 1) * 255
    return out.astype("uint8")


def search(query_vector, embeddings, k=25):
    scores = embeddings @ query_vector
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]
