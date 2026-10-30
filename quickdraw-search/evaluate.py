"""Task 4: precision@k, raw-pixel baseline, threshold, and good/bad examples.
Labels are used only here."""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from search_core import load_model, embed, search

data = np.load("data/subset.npz")
gallery_labels, query_labels = data["gallery_labels"], data["query_labels"]
categories = list(data["categories"])

def normalize_pixels(images):
    """Raw-pixel baseline: flatten each doodle and L2-normalize it."""
    flat = images.reshape(len(images), -1).astype("float32")
    return tf.math.l2_normalize(flat, axis=1).numpy()

def precision_per_query(gallery_vecs, query_vecs, k):
    """For each query: share of its top k results that have the same category."""
    return np.array([np.mean(gallery_labels[search(v, gallery_vecs, k)[0]] == label) for v, label in zip(query_vecs, query_labels)])

def top1_scores(gallery_vecs, query_vecs):
    """Best similarity score for each query."""
    return np.array([search(v, gallery_vecs, 1)[1][0] for v in query_vecs])

# 1. Precision@k: raw pixels vs the model
model = load_model()
methods = { "Raw pixels": (normalize_pixels(data["gallery"]), normalize_pixels(data["queries"])),
    "Your model": (np.load("index/embeddings.npy"), embed(model, data["queries"])),
    "Your model, inverted colors off": (embed(model, data["gallery"], invert=False), embed(model, data["queries"], invert=False)),}
print(f"{'Method':35s} p@5   p@25")
for name, (g_vecs, q_vecs) in methods.items():
    p5 = precision_per_query(g_vecs, q_vecs, 5)
    p25 = precision_per_query(g_vecs, q_vecs, 25)
    print(f"{name:35s} {p5.mean():.3f} {p25.mean():.3f}")
    if name == "Your model":
        model_p25 = p25                       # reused in step 3

# 2. "Nothing similar" threshold
g_vecs, q_vecs = methods["Your model"]
in_scores = top1_scores(g_vecs, q_vecs)
out_scores = top1_scores(g_vecs, embed(model, data["heldout"]))
candidates = np.linspace(min(in_scores.min(), out_scores.min()), max(in_scores.max(), out_scores.max()), 200)
balanced = [((in_scores >= t).mean() + (out_scores < t).mean()) / 2 for t in candidates]
threshold = float(candidates[np.argmax(balanced)])
print(f"threshold={threshold:.3f}  "f"in-index wrongly rejected: {(in_scores < threshold).sum()}/{len(in_scores)}  "f"held-out wrongly accepted: {(out_scores >= threshold).sum()}/{len(out_scores)}")
json.dump({"threshold": round(threshold, 3), "top_k": 25}, open("index/config.json", "w"))

plt.hist(in_scores, 20, alpha=.6, label="in-index queries")
plt.hist(out_scores, 20, alpha=.6, label="held-out")
plt.axvline(threshold, color="k")
plt.legend()
plt.xlabel("top-1 cosine score")
plt.savefig("threshold_hist.png")

# 3. Three good and three bad queries
def pick_from_distinct_categories(query_order, count=3):
    """First query from each new category, following the given order."""
    chosen = {}
    for q in query_order:
        chosen.setdefault(query_labels[q], q)
        if len(chosen) == count:
            break
    return list(chosen.values())

worst_first = np.argsort(model_p25)
rows = ([("GOOD", q) for q in pick_from_distinct_categories(worst_first[::-1])] +
        [("BAD", q) for q in pick_from_distinct_categories(worst_first)])

gallery_images = np.load("index/gallery_images.npy")
fig, axes = plt.subplots(len(rows), 6, figsize=(9, 9))
for row_axes, (tag, q) in zip(axes, rows):
    best, scores = search(q_vecs[q], g_vecs, 25)
    wrong = np.bincount(gallery_labels[best], minlength=len(categories))
    wrong[query_labels[q]] = 0                     # ignore the correct category
    print(f"{tag:4s} {categories[query_labels[q]]:12s} p@25={model_p25[q]:.2f}  "f"most common wrong category: {categories[wrong.argmax()]}")
    row_axes[0].imshow(data["queries"][q], cmap="gray")
    row_axes[0].set_title(f"{tag}: {categories[query_labels[q]]}\np@25={model_p25[q]:.2f}",fontsize=8)
    for col in range(5):
        row_axes[col + 1].imshow(gallery_images[best[col]], cmap="gray")
        row_axes[col + 1].set_title(f"{scores[col]:.2f}", fontsize=8)
    for ax in row_axes:
        ax.axis("off")
plt.tight_layout()
plt.savefig("good_bad_results.png", dpi=120)