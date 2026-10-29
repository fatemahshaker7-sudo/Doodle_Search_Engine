"""Task 5: simplest Streamlit doodle search. Never shows labels."""
import json
import numpy as np
import streamlit as st
from PIL import Image
from search_core import load_model, embed, search, preprocess_query

st.set_page_config(page_title="DoodleDex", layout="wide")

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_index():
    return (np.load("index/embeddings.npy"), np.load("index/gallery_images.npy"),
            json.load(open("index/config.json")))

model = get_model()
emb, gallery, cfg = get_index()

def show(raw_img):
    q28 = preprocess_query(raw_img)
    if q28 is None:
        st.info("Draw something first.")
        return
    a, b = st.columns(2)
    a.image(raw_img, caption="Your input", width=200)
    b.image(q28, caption="What the model sees", width=200, clamp=True)
    idx, scores = search(embed(model, q28[None])[0], emb, cfg["top_k"])
    if scores[0] < cfg["threshold"]:
        st.warning("Nothing similar found")
        return
    for r in range(5):
        cols = st.columns(5)
        for c in range(5):
            i = r * 5 + c
            cols[c].image(gallery[idx[i]], caption=f"{scores[i]:.2f}", clamp=True)

st.title("DoodleDex: who drew it like you?")
draw, upload, cam = st.tabs(["Draw", "Upload", "Camera"])

with draw:
    from streamlit_drawable_canvas import st_canvas
    res = st_canvas(stroke_width=8, stroke_color="#000000", background_color="#FFFFFF",
                    height=280, width=280, drawing_mode="freedraw",
                    update_streamlit=True, key="canvas")
    if res.image_data is not None:
        show(Image.fromarray(res.image_data.astype("uint8")))

with upload:
    f = st.file_uploader("PNG or JPG", type=["png", "jpg", "jpeg"])
    if f:
        show(Image.open(f))

with cam:
    p = st.camera_input("Hold a drawing up to the camera")
    if p:
        show(Image.open(p))
