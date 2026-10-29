"""Task 2: embed the gallery and save the index."""
import numpy as np
from search_core import load_model, embed

d = np.load("data/subset.npz")
emb = embed(load_model(), d["gallery"])
np.save("index/embeddings.npy", emb)
np.save("index/gallery_images.npy", d["gallery"])
np.save("index/gallery_labels.npy", d["gallery_labels"])  # evaluation only
print(emb.shape, "row norms ~1:", np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4))
