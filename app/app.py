"""DoodleSearch Streamlit app. Shows similar doodles, never shows labels."""
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from search_core import load_model, embed, search, preprocess_query

REQUIRED_FILES = ["embeddings.npy", "gallery_images.npy", "config.json"]
DEFAULT_CONFIG = {"threshold": 0.8, "top_k": 25}
HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="DoodleSearch", layout="wide")


def candidate_dirs():
    """Places the index folder might be, in order of preference."""
    env = os.environ.get("DOODLE_INDEX_DIR")
    paths = [Path(env)] if env else []
    paths += [HERE.parent / "index", HERE / "index", Path.cwd() / "index", Path.cwd().parent / "index"]
    return paths


def find_index_dir():
    """Returns (folder with all required files or None, list of what was missing where)."""
    problems = []
    for folder in candidate_dirs():
        missing = [n for n in REQUIRED_FILES if not (folder / n).exists()]
        if not missing:
            return folder, problems
        problems.append((folder, missing))
    return None, problems


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def load_index(folder):
    """Only called when the files exist, so a failure is never cached."""
    config = {**DEFAULT_CONFIG, **json.loads((Path(folder) / "config.json").read_text())}
    return (np.load(Path(folder) / "embeddings.npy"),
            np.load(Path(folder) / "gallery_images.npy"),
            config)


index_dir, problems = find_index_dir()
if index_dir is None:
    st.error("Index files not found. Add them to the repo, or run prepare_data.py, "
             "build_index.py and evaluate.py.")
    st.write("Looked in these places:")
    for folder, missing in problems:
        st.code(f"{folder}\n  missing: {', '.join(missing)}")
    st.caption("Tip: set the DOODLE_INDEX_DIR environment variable to use a custom folder.")
    st.stop()

embeddings, gallery, config = load_index(str(index_dir))
model = get_model()

# Sidebar controls, so the behaviour adapts without editing code
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, float(config["threshold"]), 0.01,
                          help="Results below this score are treated as 'nothing similar'.")
    max_k = max(1, min(50, len(embeddings)))
    top_k = st.slider("Number of results", 1, max_k, min(int(config["top_k"]), max_k))
    n_cols = st.slider("Results per row", 2, 8, 5)
    st.caption(f"Gallery size: {len(embeddings):,} doodles")


def show(raw_image):
    """Preprocess one query, search the index, render the results."""
    query_28 = preprocess_query(raw_image)
    if query_28 is None:
        st.info("Draw something first.")
        return

    left, right = st.columns(2)
    left.image(raw_image, caption="Your input", width=200)
    right.image(query_28, caption="What the model sees", width=200, clamp=True)

    indices, scores = search(embed(model, query_28[None])[0], embeddings, top_k)

    if scores[0] < threshold:
        st.warning(f"Nothing similar found (best score {scores[0]:.2f}, "
                   f"threshold {threshold:.2f}). Try lowering the threshold in the sidebar.")
        return

    st.subheader("Doodles that look like yours")
    for start in range(0, len(indices), n_cols):
        columns = st.columns(n_cols)
        for column, n in zip(columns, range(start, min(start + n_cols, len(indices)))):
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
