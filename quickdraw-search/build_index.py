"""Task 2: embed the gallery and save the index."""
import os
import numpy as np
from search_core import load_model, embed

os.makedirs("index", exist_ok=True)
data = np.load("data/subset.npz")
embeddings = embed(load_model(), data["gallery"])
np.save("index/embeddings.npy", embeddings)
np.save("index/gallery_images.npy", data["gallery"])
np.save("index/gallery_labels.npy", data["gallery_labels"])   # used only by evaluate.py
