"""Task 1: download Quick, Draw! bitmaps and sample a small subset."""
import os, urllib.request, urllib.parse
import numpy as np
import matplotlib.pyplot as plt

INDEX_CATS = ["cat", "dog", "sun", "smiley face", "cup", "bucket",
              "apple", "house", "car", "fish"]
HELD_OUT = ["cell phone"]
SEED = 0
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

def load(cat):
    path = f"data/raw/{cat}.npy"
    if not os.path.exists(path):
        print("downloading", cat)
        urllib.request.urlretrieve(URL.format(urllib.parse.quote(cat)), path)
    return np.load(path, mmap_mode="r")

rng = np.random.default_rng(SEED)
g_img, g_lab, q_img, q_lab, h_img, h_lab = [], [], [], [], [], []
for i, cat in enumerate(INDEX_CATS):
    arr = load(cat)
    idx = np.sort(rng.choice(len(arr), 500, replace=False))
    s = np.array(arr[idx]).reshape(-1, 28, 28)
    g_img.append(s[:450]); g_lab += [i] * 450
    q_img.append(s[450:]); q_lab += [i] * 50
for j, cat in enumerate(HELD_OUT):
    arr = load(cat)
    idx = np.sort(rng.choice(len(arr), 50, replace=False))
    h_img.append(np.array(arr[idx]).reshape(-1, 28, 28)); h_lab += [j] * 50

np.savez_compressed("data/subset.npz",
    gallery=np.concatenate(g_img), gallery_labels=np.array(g_lab),
    queries=np.concatenate(q_img), query_labels=np.array(q_lab),
    heldout=np.concatenate(h_img), heldout_labels=np.array(h_lab),
    categories=np.array(INDEX_CATS), heldout_categories=np.array(HELD_OUT))

gal = np.concatenate(g_img)
fig, axes = plt.subplots(5, 5, figsize=(5, 5))
for ax, i in zip(axes.ravel(), rng.choice(len(gal), 25, replace=False)):
    ax.imshow(gal[i], cmap="gray"); ax.axis("off")
plt.savefig("grid_check.png"); print("saved data/subset.npz and grid_check.png")
