"""DoodleDex Streamlit app. Shows similar doodles, never shows labels."""
import json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from search_core import load_model, embed, search, preprocess_query

# importing nessesary files
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "index"
REQUIRED_FILES = ["embeddings.npy", "gallery_images.npy", "config.json"]

# starting the page
st.set_page_config(page_title="DoodleDex", layout="wide")

# caching
@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_index():
    """Returns (embeddings, gallery images, config), or None if files are missing."""
    if any(not (INDEX_DIR / name).exists() for name in REQUIRED_FILES):
        return None
    return (np.load(INDEX_DIR / "embeddings.npy"),
            np.load(INDEX_DIR / "gallery_images.npy"),
            json.loads((INDEX_DIR / "config.json").read_text()))

index = get_index()
if index is None:
    st.error("Missing index files. Run prepare_data.py, build_index.py and evaluate.py first.")
    st.stop()
embeddings, gallery, config = index
model = get_model()

def show(raw_image):
    """Preprocess one query, search the index, render the results."""
    query_28 = preprocess_query(raw_image)
    if query_28 is None:
        st.info("Draw something first.")
        return

    left, right = st.columns(2)
    left.image(raw_image, caption="Your input", width=200)
    right.image(query_28, caption="What the model sees", width=200, clamp=True)
    k = min(config.get("top_k", 25), 25, len(embeddings))
    indices, scores = search(embed(model, query_28[None])[0], embeddings, k)

    if scores[0] < config["threshold"]:
        st.warning("Nothing similar found")
        return

    st.subheader("Doodles that look like yours")
    for start in range(0, len(indices), 5):
        for column, n in zip(st.columns(5), range(start, min(start + 5, len(indices)))):
            column.image(gallery[indices[n]], caption=f"{scores[n]:.2f}", clamp=True)

st.title("DoodleDex: who drew it like you?")
draw_tab, upload_tab = st.tabs(["Draw", "Upload"])

with draw_tab:
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.warning("Drawing canvas not installed. Use the Upload tab.")
    else:
        canvas = st_canvas(stroke_width=8, stroke_color="#000000", background_color="#FFFFFF",
                           height=280, width=280, drawing_mode="freedraw",
                           update_streamlit=True, return_image_data=True, key="canvas")
        if canvas.image_data is not None:
            show(Image.fromarray(canvas.image_data.astype("uint8")))

with upload_tab:
    uploaded = st.file_uploader("PNG or JPG", type=["png", "jpg", "jpeg"])
    if uploaded:
        show(Image.open(uploaded))
