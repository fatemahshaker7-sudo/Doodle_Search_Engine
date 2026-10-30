"""Task 1: download Quick, Draw! bitmaps and sample a small subset.
Data: Quick, Draw! dataset by Google, Inc., licensed CC BY 4.0."""
import os, sys
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# the chosen categories
INDEX_CATEGORIES = ["cat", "dog", "sun", "smiley face", "cup", "bucket",
                    "apple", "house", "car", "fish"]
HELD_OUT_CATEGORIES = ["cell phone"]   # never indexed: tests the threshold

# 450 gallery + 50 query doodles per index category, 50 per held-out category, the 50 queries are used to calculate (p@5, p@25)
GALLERY_PER_CATEGORY, QUERIES_PER_CATEGORY, HELD_OUT_PER_CATEGORY = 450, 50, 50

# the url to download from
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

# stop if the data file already exists, otherwise make the folders
if os.path.exists("data/subset.npz"):
    sys.exit("data/subset.npz already exists, nothing to do. Delete it to rebuild.")
os.makedirs("data/raw", exist_ok=True)   # creates data/ and data/raw/ if missing
np.random.seed(0)                        # fixed seed: same random picks on every run

def sample_doodles(name, count):
    """Download a category if needed and pick `count` random 28x28 doodles."""
    path = f"data/raw/{name}.npy"
    if not os.path.exists(path):
        print("downloading", name)
        web_name = name.replace(" ", "%20")        # "cell phone" -> "cell%20phone"
        address = URL.format(web_name)             # put the name into the download link
        urllib.request.urlretrieve(address, path)  # download and save to path
    doodles = np.load(path)
    picked = np.sort(np.random.choice(len(doodles), count, replace=False))
    return doodles[picked].reshape(-1, 28, 28)

# index categories: split each into gallery and queries
gallery, gallery_labels, queries, query_labels = [], [], [], []
for label, name in enumerate(INDEX_CATEGORIES):
    doodles = sample_doodles(name, GALLERY_PER_CATEGORY + QUERIES_PER_CATEGORY)
    gallery.append(doodles[:GALLERY_PER_CATEGORY])
    queries.append(doodles[GALLERY_PER_CATEGORY:])
    gallery_labels += [label] * GALLERY_PER_CATEGORY
    query_labels += [label] * QUERIES_PER_CATEGORY

# held-out categories
held_out, held_out_labels = [], []
for label, name in enumerate(HELD_OUT_CATEGORIES):
    held_out.append(sample_doodles(name, HELD_OUT_PER_CATEGORY))
    held_out_labels += [label] * HELD_OUT_PER_CATEGORY

# save everything, with labels
gallery = np.concatenate(gallery)
np.savez_compressed(
    "data/subset.npz",
    gallery=gallery, gallery_labels=np.array(gallery_labels),
    queries=np.concatenate(queries), query_labels=np.array(query_labels),
    heldout=np.concatenate(held_out), heldout_labels=np.array(held_out_labels),
    categories=np.array(INDEX_CATEGORIES), heldout_categories=np.array(HELD_OUT_CATEGORIES))

# sanity check: 5 x 5 grid of random gallery doodles
fig, axes = plt.subplots(5, 5, figsize=(5, 5))
for ax, n in zip(axes.ravel(), np.random.choice(len(gallery), 25, replace=False)):
    ax.imshow(gallery[n], cmap="gray")
    ax.axis("off")
plt.savefig("grid_check.png")
print("saved data/subset.npz and grid_check.png")