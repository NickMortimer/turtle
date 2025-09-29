import os
from pathlib import Path
import shutil

import clip
import torch
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

# -------- CONFIG --------
image_folder = Path("/media/mor582/Timor2/surveys/yolo_finds/clip_turtle")      # Input folder with images
output_folder = Path("/media/mor582/Timor2/surveys/yolo_finds/clip_clusters")  # Where to save clustered images
num_clusters = 5                        # Set number of clusters
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- SETUP --------
model, preprocess = clip.load("ViT-B/32", device=device)
output_folder.mkdir(exist_ok=True)

# -------- LOAD IMAGES & EMBED --------
image_paths = list(image_folder.glob("*.JPG"))  # jpg and png
embeddings = []

print(f"Encoding {len(image_paths)} images...")

with torch.no_grad():
    for img_path in tqdm(image_paths):
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        emb = model.encode_image(image).float().cpu().squeeze(0)
        embeddings.append(emb.numpy())

# -------- CLUSTERING --------
print(f"Clustering into {num_clusters} clusters...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# -------- SAVE IMAGES INTO CLUSTER FOLDERS --------
for img_path, label in zip(image_paths, labels):
    cluster_dir = output_folder / f"cluster_{label}"
    cluster_dir.mkdir(exist_ok=True)
    shutil.copy(img_path, cluster_dir / img_path.name)

print("Done! Images saved by cluster.")