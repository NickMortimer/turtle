import typer
from pathlib import Path
import numpy as np
import torch
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False
from PIL import Image
from tqdm import tqdm

app = typer.Typer()

def load_clip_model(device='cuda'):
    """Load CLIP model and preprocessing function."""
    if not CLIP_AVAILABLE:
        raise ImportError("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def load_existing_embeddings(npz_path):
    """Helper function to load existing embeddings from npz file"""
    existing_embeddings = {}
    npz_path = Path(npz_path)
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'embeddings' in data and 'filenames' in data:
                existing_embeddings = dict(zip(data['filenames'], data['embeddings']))
                print(f"Loaded {len(existing_embeddings)} existing embeddings from {npz_path}")
        except Exception as e:
            print(f"Error loading existing embeddings: {e}")
    return existing_embeddings

def encode_images_in_folder(
    folder_path: Path,
    output_path: Path = None,
    crop_size: int = 0,
    batch_size: int = 32,
    device: str = "auto"
):
    """
    Encode all images in a folder using CLIP embeddings.
    
    Args:
        folder_path: Path to folder containing images
        output_path: Path for output NPZ file (defaults to folder_name_clip.npz)
        crop_size: Size to crop images to (0 = no crop)
        batch_size: Batch size for processing
        device: Device to use ('cuda', 'cpu', or 'auto')
    """
    if not CLIP_AVAILABLE:
        raise ImportError("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Set up output path
    if output_path is None:
        output_path = folder_path.parent / f"{folder_path.name}_clip.npz"
    else:
        output_path = Path(output_path)
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    print(f"Loading CLIP model on {device}...")
    model, preprocess = load_clip_model(device)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    image_files.sort()
    print(f"Found {len(image_files)} images in {folder_path}")
    
    if not image_files:
        print("No images found!")
        return
    
    def load_and_crop(image_path):
        """Load and optionally crop image"""
        try:
            data = Image.open(image_path).convert("RGB")
            if crop_size > 0:
                w, h = data.size
                side = min(w, h, crop_size)
                left = max((w - side) // 2, 0)
                top = max((h - side) // 2, 0)
                right = left + side
                bottom = top + side
                data = data.crop((left, top, right, bottom))
                if side != crop_size:
                    data = data.resize((crop_size, crop_size))
            return data
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def encode_batch(image_paths):
        """Encode a batch of images"""
        raw_images = []
        valid_paths = []
        
        for path in image_paths:
            img = load_and_crop(path)
            if img is not None:
                raw_images.append(img)
                valid_paths.append(path)
        
        if not raw_images:
            return np.array([]), []
        
        images = [preprocess(img) for img in raw_images]
        batch = torch.stack(images).to(device)
        
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats /= feats.norm(dim=-1, keepdim=True)
        
        feats = feats.to(dtype=torch.float32)
        return feats.cpu().numpy(), valid_paths
    
    # Process images in batches
    all_embeddings = []
    all_filenames = []
    
    def chunked(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]
    
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    for batch in tqdm(chunked(image_files, batch_size), total=num_batches, desc="Encoding batches"):
        batch_embeddings, valid_paths = encode_batch(batch)
        if len(batch_embeddings) > 0:
            all_embeddings.append(batch_embeddings)
            all_filenames.extend([p.name for p in valid_paths])
    
    if not all_embeddings:
        print("No images were successfully processed!")
        return
    
    # Combine all embeddings
    embeddings_array = np.vstack(all_embeddings)
    filenames_array = np.array(all_filenames)
    
    # Save to NPZ file
    print(f"Saving {len(embeddings_array)} embeddings to {output_path}")
    np.savez_compressed(
        output_path,
        embeddings=embeddings_array,
        filenames=filenames_array
    )
    
    print(f"Successfully encoded {len(embeddings_array)} images")
    return output_path

@app.command()
def encode(
    folder: Path = typer.Argument(..., help="Folder containing images to encode"),
    output: Path = typer.Option(None, help="Output NPZ file path"),
    crop_size: int = typer.Option(0, help="Crop size for images (0 = no crop)"),
    batch_size: int = typer.Option(32, help="Batch size for processing"),
    device: str = typer.Option("auto", help="Device to use (cuda/cpu/auto)")
):
    """
    Encode images in a folder using CLIP embeddings.
    """
    try:
        output_path = encode_images_in_folder(
            folder_path=folder,
            output_path=output,
            crop_size=crop_size,
            batch_size=batch_size,
            device=device
        )
        typer.echo(f"✅ Successfully encoded images to {output_path}")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def inspect(
    npz_file: Path = typer.Argument(..., help="Path to .npz file to inspect")
):
    """
    Inspect the contents of an embeddings .npz file.
    """
    if not npz_file.exists():
        typer.echo(f"Error: File {npz_file} does not exist")
        raise typer.Exit(1)
    
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        typer.echo(f"Contents of {npz_file}:")
        typer.echo(f"Keys: {list(data.keys())}")
        
        if 'embeddings' in data:
            embeddings = data['embeddings']
            typer.echo(f"Embeddings shape: {embeddings.shape}")
            typer.echo(f"Embeddings dtype: {embeddings.dtype}")
        
        if 'filenames' in data:
            filenames = data['filenames']
            typer.echo(f"Number of files: {len(filenames)}")
            typer.echo(f"First 5 filenames: {filenames[:5]}")
        
        if 'hidden_flags' in data:
            hidden_flags = data['hidden_flags'].item()
            if isinstance(hidden_flags, dict):
                hidden_count = sum(1 for hidden in hidden_flags.values() if hidden)
                typer.echo(f"Hidden flags: {len(hidden_flags)} total, {hidden_count} hidden")
        
        if 'categories' in data:
            categories = data['categories'].item()
            if isinstance(categories, dict):
                typer.echo(f"Categories: {len(categories)} images classified")
        
    except Exception as e:
        typer.echo(f"Error reading file: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
