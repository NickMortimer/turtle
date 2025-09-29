import typer
from pathlib import Path
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
from doit.doit_cmd import DoitMain
from doit.cmd_base import ModuleTaskLoader
from doit.tools import run_once
import pandas as pd
from turtledrone.config import cfg
from doit.tools import check_timestamp_unchanged

app = typer.Typer()

DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}

def load_clip_model(device='cuda'):
    """Load CLIP model and preprocessing function."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def load_existing_embeddings(npz_path):
    """Helper function to load existing embeddings from npz file"""
    existing_embeddings = {}
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'embeddings' in data and 'filenames' in data:
                existing_embeddings = dict(zip(data['filenames'], data['embeddings']))
                print(f"Loaded {len(existing_embeddings)} existing embeddings from {npz_path}")
        except Exception as e:
            print(f"Error loading existing embeddings: {e}")
    return existing_embeddings

def task_encode_folders():
    """Doit task generator for encoding image folders."""
    
    def image_to_pickle_name(image_path,crop_size):
        return image_path.parent.parent.parent / f'{image_path.parent.parent.parent.name}_{image_path.parent.name}_clip_{crop_size:03d}.npz'

    def encode_folder_action(dependencies, targets):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        dependencies.sort()

        def load_and_crop(image):
            data = Image.open(image).convert("RGB")
            if crop_size>0:
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
        
        def encode_batch(image_paths):
            """
            Encode a batch of images (list of file paths).
            Returns numpy array of embeddings
            """
            raw_images = [ load_and_crop(p) for p in image_paths]
            images = [preprocess(img) for img in raw_images]
            batch = torch.stack(images).to(device)

            with torch.no_grad():
                feats = model.encode_image(batch)
                feats /= feats.norm(dim=-1, keepdim=True)
            feats = feats.to(dtype=torch.float32)
            return feats.cpu().numpy()

        def save_folder_embeddings(npz_path, existing_embeddings):
            """Helper function to save folder embeddings"""
            if not existing_embeddings:
                return

            np.savez_compressed(
                npz_path,
                embeddings=np.array([v for v in existing_embeddings.values()]),
                filenames=np.array([k for k in existing_embeddings.keys()]),
                )
            print(f"Saved {len(existing_embeddings)} total embeddings to {npz_path}")


        # Prepare the batches
        def chunked(dependencies, size):
            for i in range(0, len(dependencies), size):
                yield dependencies[i:i + size]
        
        # Process all images in large batches with progress bar
        all_embeddings = []
        batch_size = cfg.get('encode_batch_size')
        num_batches = (len(dependencies) + batch_size - 1) // batch_size
        for batch in tqdm(chunked(dependencies, batch_size), total=num_batches, desc="Encoding batches"):
            batch_embeddings = encode_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Now organize by folder and save to npz files
        current_folder = None
        current_npz_path = None
        existing_embeddings = {}
        folder_embeddings = []
        folder_filenames = []

        for image_path, embedding in tqdm(zip(dependencies, all_embeddings), total=len(dependencies), desc="Saving embeddings"):
            image_path = Path(image_path)
            npz_path = image_to_pickle_name(image_path,crop_size)
            relative_name = image_path.relative_to(npz_path.parent)
            # If we've moved to a new folder, save the previous one
            if npz_path != current_npz_path:
                if current_npz_path is not None:
                    # Save previous folder's data
                    save_folder_embeddings(current_npz_path, existing_embeddings)

                # Load existing embeddings for new folder
                current_npz_path = npz_path
                existing_embeddings = load_existing_embeddings(current_npz_path)
            
            # Add current image's embedding if it's not already in existing
            existing_embeddings[relative_name] = embedding
        # Save the last folder
        if current_npz_path is not None:
            save_folder_embeddings(current_npz_path, existing_embeddings)
    crop_size = cfg.get('encode_crop_size', 0)
    input_dir = cfg.get_url('encode_input_dir')
    pattern = cfg.get('encode_pattern')
    # get all jpgs in subfolders
    all_files = list(input_dir.rglob(f'**/{pattern.lower()}')) + list(input_dir.rglob(f'**/{pattern.upper()}'))
    
    # Group files by their npz path first
    files_by_npz = {}
    for file_path in all_files:
        folder = file_path.parent
        npz_path = image_to_pickle_name(file_path,crop_size)
        if npz_path not in files_by_npz:
            files_by_npz[npz_path] = []
        files_by_npz[npz_path].append(file_path)
    
    # Filter to only include files that need processing
    files_to_process = []
    for npz_path, file_paths in files_by_npz.items():
        # Load existing embeddings once per npz file
        existing_filenames = set()
        if npz_path.exists():
            try:
                data = np.load(npz_path, allow_pickle=True)
                if 'embeddings' in data and 'filenames' in data:
                    existing_filenames = set(str(f) for f in data['filenames'])
            except Exception:
                pass  # If we can't load, assume all files need processing
        
        # Check each file in this group
        for file_path in file_paths:
            relative_name = file_path.relative_to(npz_path.parent)
            if str(relative_name) not in existing_filenames:
                files_to_process.append(file_path)
    
    # Only create targets for folders that have files to process
    targets = list(set([image_to_pickle_name(f,crop_size) for f in files_to_process]))

    return {
            'file_dep' : files_to_process,
            'targets' : targets,
            'actions':[encode_folder_action],
            'clean':True,

        
    }

def task_cluster_images():
    def cluster_images(dependencies, targets):
        from sklearn.cluster import KMeans
        import pandas as pd
        import numpy as np

        if not dependencies:
            print("No .npz files found to cluster.")
            return
        npz_path = Path(dependencies[0])
        try:
            data = np.load(npz_path, allow_pickle=True)
            embeddings = data['embeddings']
            filenames = data['filenames']
            if len(embeddings) > 2:
                num_clusters = min(5, len(embeddings) // 2)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings)

                df = pd.DataFrame({
                    'filename': filenames,
                    'cluster': labels
                })
                output_csv = targets[0]

                print(f"Clustered {len(embeddings)} embeddings into {num_clusters} clusters. Saved to {output_csv}")
            else:
                df = pd.DataFrame(columns=['filename', 'cluster'])                
            df.to_csv(output_csv, index=False)                

        except Exception as e:
            print(f"Error processing {npz_path}: {e}")


    input_dir = cfg.get_url('encode_input_dir')
    output_dir = cfg.get_url('encode_output_dir')
    recursive = cfg.get('encode_recursive')
    pattern = cfg.get('encode_pattern')
    # get all jpgs in subfolders
    files = list(input_dir.rglob(f'**/*.npz')) 
    for file in files:
        targets = file.parent / f"{file.stem}_clustered.csv"
        yield {
                    'name' : targets,
                    'file_dep' : [file],
                    'targets' : [targets],
                    'actions':[cluster_images],
                    'clean':True,
                    'uptodate': [True],

                }
    
# def task_plot_dendrograms():
#     def plot_dendrogram(dependencies, targets):
#         import numpy as np
#         from sklearn.metrics.pairwise import cosine_similarity
#         from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
#         import matplotlib.pyplot as plt
        
#         npz_path = Path(dependencies[0])
        
#         try:
#             data = np.load(npz_path, allow_pickle=True)
#             embeddings = data['embeddings']
#             filenames = data['filenames']
            
#             if len(embeddings) < 2:
#                 print(f"Skipping {npz_path.stem}: too few samples ({len(embeddings)})")
#                 # Create empty marker file
#                 with open(targets[0], 'w') as f:
#                     f.write(f"Skipped {npz_path.stem}: too few samples")
#                 return
            
#             # Step 1: Compute cosine distance (1 - similarity)
#             cos_sim = cosine_similarity(embeddings)
#             cos_dist = 1 - cos_sim
            
#             # Step 2: Hierarchical clustering
#             Z = linkage(cos_dist, method="average")  # "average" works well for cosine
            
#             # Step 3: Plot dendrogram
#             plt.figure(figsize=(12, 6))
#             dendrogram(Z, labels=np.arange(len(embeddings)))
#             plt.title(f"Hierarchical Clustering Dendrogram - {npz_path.stem}")
#             plt.xlabel("Image index")
#             plt.ylabel("Cosine distance")
            
#             # Save plot
#             plot_path = targets[0]
#             plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#             plt.close()  # Close figure to free memory
            
#             # Step 4: Optionally cut into flat clusters and save info
#             labels = fcluster(Z, t=1.5, criterion="distance")
            
#             # Save cluster info to text file
#             info_path = Path(plot_path).with_suffix('.txt')
#             with open(info_path, 'w') as f:
#                 f.write(f"Hierarchical clustering results for {npz_path.stem}\n")
#                 f.write(f"Number of images: {len(embeddings)}\n")
#                 f.write(f"Number of clusters (distance < 0.3): {len(np.unique(labels))}\n")
#                 f.write(f"Cluster labels: {labels.tolist()}\n")
#                 f.write("\nFilename to cluster mapping:\n")
#                 for filename, label in zip(filenames, labels):
#                     f.write(f"{filename}: cluster_{label}\n")
            
#             print(f"Created dendrogram for {npz_path.stem}: {len(embeddings)} images, {len(np.unique(labels))} clusters")
            
#         except Exception as e:
#             print(f"Error creating dendrogram for {npz_path}: {e}")
#             # Create error marker file
#             with open(targets[0], 'w') as f:
#                 f.write(f"Error processing {npz_path.stem}: {e}")

#     input_dir = cfg.get_url('encode_input_dir')
#     output_dir = cfg.get_url('encode_output_dir')
    
#     # Find all .npz files
#     npz_files = list(input_dir.rglob('**/*.npz'))
    
#     for npz_file in npz_files:
#         # Create dendrogram task for each npz file
#         survey_name = npz_file.stem
#         plot_name = f"{survey_name}_dendrogram.png"
#         plot_path = Path(output_dir) / plot_name
#         plot_path.parent.mkdir(exist_ok=True, parents=True)
        
#         yield {
#             'name': f"dendrogram_{survey_name}",
#             'file_dep': [npz_file],
#             'targets': [plot_path],
#             'actions': [plot_dendrogram],
#             'clean': True,
#         }

# def task_organize_clusters():
#     def organize_cluster_images(dependencies, targets):
#         import pandas as pd
#         import shutil
        
#         csv_path = Path(dependencies[0])
        
#         try:
#             # Load cluster CSV
#             df = pd.read_csv(csv_path)
            
#             # Get the base directory (where the images are located)
#             base_dir = csv_path.parent
            
#             # Get output directory from config
#             output_dir = Path(cfg.get('encode_output_dir'))
            
#             # Create cluster directories for this survey/folder
#             survey_name = csv_path.stem.replace('_clustered', '')
#             cluster_base_dir = output_dir / f"{survey_name}_clusters"
#             cluster_base_dir.mkdir(exist_ok=True)
            
#             # Group by cluster
#             clusters = df.groupby('cluster')
            
#             for cluster_id, cluster_df in clusters:
#                 # Create cluster directory
#                 cluster_dir = cluster_base_dir / f"cluster_{cluster_id}"
#                 cluster_dir.mkdir(exist_ok=True)
                
#                 # Copy images to cluster directory
#                 copied_count = 0
#                 for filename in cluster_df['filename']:
#                     # Construct source path
#                     src_path = base_dir / filename
                    
#                     if src_path.exists():
#                         # Create destination path (keep original filename)
#                         dst_path = cluster_dir / src_path.name
                        
#                         # Copy file if it doesn't exist or is newer
#                         if not dst_path.exists() or src_path.stat().st_mtime > dst_path.stat().st_mtime:
#                             shutil.copy2(src_path, dst_path)
#                             copied_count += 1
                
#                 print(f"Cluster {cluster_id}: copied {copied_count} images to {cluster_dir}")
            
#             # Create completion marker
#             marker_file = targets[0]
#             with open(marker_file, 'w') as f:
#                 f.write(f"Organized clusters for {survey_name} at {cluster_base_dir}")
                
#         except Exception as e:
#             print(f"Error organizing clusters from {csv_path}: {e}")

#     input_dir = cfg.get_url('encode_input_dir')
#     output_dir = cfg.get_url('encode_output_dir')
    
#     # Find all cluster CSV files
#     cluster_files = list(input_dir.rglob('**/*_clustered.csv'))
    
#     for csv_file in cluster_files:
#         # Create organize task for each cluster CSV
#         survey_name = csv_file.stem.replace('_clustered', '')
#         marker_name = f"{survey_name}_clusters_organized.txt"
#         marker_path = Path(output_dir) / marker_name
#         marker_path.parent.mkdir(exist_ok=True, parents=True)
        
#         yield {
#             'name': f"organize_clusters_{survey_name}",
#             'file_dep': [csv_file],
#             'targets': [marker_path],
#             'actions': [organize_cluster_images],
#             'clean': True,
#         }




@app.command('encode')
def doit_encode(
    ctx: typer.Context,
    input_dir: Path = typer.Option(..., help="Directory containing image folders to process"),
    output_dir: Path = typer.Option(..., help="Directory to save .npz embedding files"),
    recursive: bool = typer.Option(False, help="Process subdirectories recursively"),
    pattern: str = typer.Option("*.jpg", help="Image pattern to match"),
    doit_db: str = typer.Option(".doit-db", help="Path to doit database file"),
    batch_size: int = typer.Option(128, help="Batch size for processing images"),
    crop_size: int = typer.Option(0, help="Crop size for images to encode"),
):
    """
    Use doit to encode images in folders (for dependency tracking and incremental builds).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    doit_db = input_dir / doit_db 
    
    # Run doit with custom database location
    cfg.set('encode_input_dir', str(input_dir))
    cfg.set('encode_output_dir', str(output_dir))
    cfg.set('encode_recursive', recursive)
    cfg.set('encode_pattern', pattern)
    cfg.set('encode_batch_size', batch_size)
    cfg.set('encode_crop_size', crop_size)
  
    DoitMain(ModuleTaskLoader(globals())).run(ctx.args + ['--db-file', str(doit_db),'encode_folders'])



@app.command()
def inspect(
    npz_file: Path = typer.Option(..., help="Path to .npz file to inspect")
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
        
        if 'folder_name' in data:
            typer.echo(f"Folder name: {data['folder_name']}")
        
        if 'folder_path' in data:
            typer.echo(f"Folder path: {data['folder_path']}")
            
    except Exception as e:
        typer.echo(f"Error reading {npz_file}: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
