import os
import shutil
import numpy as np
import torch
import clip
from PIL import Image
import streamlit as st
TOP_N = 100
# --- Load CLIP model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
st.set_page_config(layout="wide")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@st.cache_resource 
def load_embeddings(image_dir):
    """Precompute embeddings for all images in folder."""
    embeddings = {}
    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)
        try:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img)
                emb /= emb.norm(dim=-1, keepdim=True)
            embeddings[fname] = emb.cpu().numpy().flatten()
        except Exception as e:
            print("Skipping", fname, e)
    return embeddings


def streamlit_app(image_dir: str):
    IMAGE_DIR = image_dir
    TRASH_DIR = os.path.join(IMAGE_DIR, "trash")
    os.makedirs(TRASH_DIR, exist_ok=True)

    st.title("🖼️ CLIP Image Cleaner")

    embeddings = load_embeddings(IMAGE_DIR)
    image_files = list(embeddings.keys())

    if not image_files:
        st.warning(f"No images found in {IMAGE_DIR} folder.")
        st.stop()

    # Use session_state for reference image
    if "ref_file" not in st.session_state or st.session_state.ref_file not in image_files:
        st.session_state.ref_file = image_files[0]

    # Pick a reference image (dropdown)
    ref_file = st.selectbox("Select a reference image", image_files, index=image_files.index(st.session_state.ref_file))
    st.session_state.ref_file = ref_file
    ref_path = os.path.join(IMAGE_DIR, ref_file)
    st.image(ref_path, caption="Reference", width=250)

    # Compute similarities
    ref_emb = embeddings[ref_file]
    sims = {f: cosine_similarity(ref_emb, emb) for f, emb in embeddings.items() if f != ref_file}
    sorted_files = sorted(sims, key=sims.get, reverse=True)[:TOP_N]

    # Make the reference image the first in the grid
    page_files = [ref_file] + [f for f in sorted_files if f != ref_file]

    # Pagination logic (adjust for reference image at start)
    PAGE_SIZE = 100
    total_pages = (len(page_files) - 1) // PAGE_SIZE + 1

    # Initialize state
    if "images" not in st.session_state:
        st.session_state.images = images.copy()
    if "selected" not in st.session_state:
        st.session_state.selected = set()
        if "page_num" not in st.session_state:
            st.session_state.page_num = 0

    col1, col2, col3, col4 = st.columns([1,2,1,1])
    with col1:
        if st.button("Previous Page") and st.session_state.page_num > 0:
            st.session_state.page_num -= 1
    with col3:
        if st.button("Next Page") and st.session_state.page_num < total_pages - 1:
            st.session_state.page_num += 1
    with col4:
        if st.button("Go to End"):
            st.session_state.page_num = total_pages - 1

    with col2:
        st.markdown(f"<div style='text-align:center;'>Page {st.session_state.page_num+1} of {total_pages}</div>", unsafe_allow_html=True)

    # Slice the files for the current page
    start = st.session_state.page_num * PAGE_SIZE
    end = start + PAGE_SIZE
    page_files = page_files[start:end]

    st.subheader("Similar Images")
    to_delete = []
    cols = st.columns(20)

    for i, fname in enumerate(page_files):
        with cols[i % 20]:
            img_path = os.path.join(IMAGE_DIR, fname)
            # Make image clickable to set as reference or mark for deletion
            if st.image(img_path, caption=fname):
                if st.session_state.get("ctrl_pressed", False):
                    st.session_state.ref_file = fname
                    st.experimental_rerun()
                else:
                    if fname not in to_delete:
                        to_delete.append(fname)

    if st.button("Delete Selected"):
        for fname in to_delete:
            src = os.path.join(IMAGE_DIR, fname)
            dst = os.path.join(TRASH_DIR, fname)
            shutil.move(src, dst)
        st.success(f"Moved {len(to_delete)} files to {TRASH_DIR}. Please refresh.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory containing images to sort")
    args = parser.parse_args()
    streamlit_app(args.image_dir)
