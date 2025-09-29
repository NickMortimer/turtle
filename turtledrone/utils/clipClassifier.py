import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
from pathlib import Path


class CLIPClassifier:
    def __init__(self, reference_dir: Path, device=None):
        """
        ref_images: dict mapping label -> image path for reference examples
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.prototypes ={}
        # Cache reference image features
        self.ref_features = {}
        with torch.no_grad():
            for class_dir in reference_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                embeddings = []
                for img_path in class_dir.glob("*.JPG"):
                    img = self.preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    feat = self.model.encode_image(img).float().cpu()
                    embeddings.append(feat)
                if embeddings:
                    prototype = torch.mean(torch.stack(embeddings), dim=0).squeeze(0)
                    self.prototypes[class_dir.name] = prototype / prototype.norm() 

    def classify_batch(self, image_paths):
        """
        Classify a batch of images (list of file paths).
        Returns list of tuples: (image_path, predicted_label, similarities_dict)
        """
        images = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            images.append(self.preprocess(img))
        
        batch = torch.stack(images).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats /= feats.norm(dim=-1, keepdim=True)

        results = []
        for i, feat in enumerate(feats):
            similarities = {
                label: (feat @ ref_feat.T).item()
                for label, ref_feat in self.ref_features.items()
            }
            best_label = max(similarities, key=similarities.get)
            results.append((image_paths[i], best_label, similarities))

        return results

    def classify_directory(self, image_dir, batch_size=32):
        """
        Classify all images in a directory in batches.
        """
        def chunked(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i:i + size]

        all_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        all_results = []
        for batch in chunked(all_files, batch_size):
            batch_results = self.classify_batch(batch)
            all_results.extend(batch_results)

        return all_results