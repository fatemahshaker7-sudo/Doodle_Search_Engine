"""Task 3 check: raw query | preprocessed 28x28 | random gallery doodle."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from search_core import preprocess_query

paths = [ "images/phone.png", "images/cat.jpg"]
if not paths:
    sys.exit("Give at least one image path")
gallery = np.load("index/gallery_images.npy")
rng = np.random.default_rng(1)
fig, rows = plt.subplots(len(paths), 3, figsize=(7, 2.4 * len(paths)), squeeze=False)

for (raw_ax, processed_ax, gallery_ax), path in zip(rows, paths):
    raw = Image.open(path)
    processed = preprocess_query(raw)
    raw_ax.imshow(raw)
    if processed is None:
        processed_ax.text(0.5, 0.5, "None (no strokes)", ha="center")
    else:
        processed_ax.imshow(processed, cmap="gray", vmin=0, vmax=255)
    gallery_ax.imshow(gallery[rng.integers(len(gallery))], cmap="gray")
    for ax, title in zip((raw_ax, processed_ax, gallery_ax), ("raw query", "preprocessed 28x28", "random gallery doodle")):
        ax.set_title(title, fontsize=9)
        ax.axis("off")

plt.tight_layout()
plt.savefig("preprocess_check.png", dpi=120)
print("saved preprocess_check.png")
