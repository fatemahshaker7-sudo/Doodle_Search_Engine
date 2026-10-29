"""Task 4: precision@k, raw-pixel baseline, threshold calibration. Labels used only here."""
import json
import numpy as np
import matplotlib.pyplot as plt
from search_core import load_model, embed, search

d = np.load("data/subset.npz")
gl, ql = d["gallery_labels"], d["query_labels"]

def norm(x):
    x = x.reshape(len(x), -1).astype("float32")
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

def precision(gallery_emb, query_emb, k):
    p = []
    for q, lab in zip(query_emb, ql):
        idx, _ = search(q, gallery_emb, k)
        p.append(np.mean(gl[idx] == lab))
    return np.mean(p)

def top1(gallery_emb, query_emb):
    return np.array([search(q, gallery_emb, 1)[1][0] for q in query_emb])

model = load_model()
sets = {
    "Raw pixels": (norm(d["gallery"]), norm(d["queries"]), norm(d["heldout"])),
    "Your model": (np.load("index/embeddings.npy"), embed(model, d["queries"]), embed(model, d["heldout"])),
    "Your model, inverted colors off": (embed(model, d["gallery"], invert=False),
                                        embed(model, d["queries"], invert=False),
                                        embed(model, d["heldout"], invert=False)),
}
print(f"{'Method':35s} p@5   p@25")
for name, (g, q, _) in sets.items():
    print(f"{name:35s} {precision(g, q, 5):.3f} {precision(g, q, 25):.3f}")

g, q, h = sets["Your model"]
s_in, s_out = top1(g, q), top1(g, h)
cands = np.linspace(min(s_in.min(), s_out.min()), max(s_in.max(), s_out.max()), 200)
bal = [((s_in >= t).mean() + (s_out < t).mean()) / 2 for t in cands]
t = float(cands[int(np.argmax(bal))])
print(f"threshold={t:.3f}  in-index wrongly rejected: {(s_in < t).sum()}/{len(s_in)}"
      f"  held-out wrongly accepted: {(s_out >= t).sum()}/{len(s_out)}")
json.dump({"threshold": round(t, 3), "top_k": 25}, open("index/config.json", "w"))

plt.hist(s_in, 20, alpha=.6, label="in-index queries")
plt.hist(s_out, 20, alpha=.6, label="held-out")
plt.axvline(t, color="k"); plt.legend(); plt.xlabel("top-1 cosine score")
plt.savefig("threshold_hist.png")
