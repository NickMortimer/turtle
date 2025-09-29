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
from doit import create_after

app = typer.Typer()

DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}

from clip_encode import load_existing_embeddings

def task_classify_images():
    def classify_images(dependencies, targets,batch_size=32):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        crop_size = cfg.get('class_crop_size', 0)
        def classify_batch( filenames_batch, embeddings_batch):
            """
            Classify a batch of embeddings (list of (filename, embedding) tuples).
            Returns list of tuples: (filename, predicted_label, similarities_dict, width, height)
            Width and height will be set to None (unless you have this info elsewhere).
            """

            feats = torch.stack([torch.tensor(item, device=device, dtype=torch.float32) for item in embeddings_batch])
            feats /= feats.norm(dim=-1, keepdim=True)
            sims = feats @ proto_matrix.T  # (N, C)
            best_indices = sims.argmax(dim=1)

            results = []
            for i, idx in enumerate(best_indices):
                sim_dict = {label_list[j]: sims[i, j].item() for j in range(len(label_list))}
                # width, height unknown from embedding alone
                results.append((filenames_batch[i], label_list[idx], sim_dict, None, None))
            return results
        prototypes = np.load(f"{cfg.get_url('class_pickle').with_suffix('')}_{crop_size:03d}.plk", allow_pickle=True)
        label_list = list(prototypes.keys())
        proto_matrix = torch.stack([
            torch.tensor(prototypes[label], device=device, dtype=torch.float32)
            for label in label_list
        ])

        # Normalize just in case (they should be normalized already, but no harm)
        proto_matrix /= proto_matrix.norm(dim=1, keepdim=True)


        #prepare the batches
        def chunked(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i:i + size]
        all_results = []
        for survey_npz in dependencies:
            existing_embeddings = load_existing_embeddings(Path(survey_npz))
            items = list(existing_embeddings.items())
            for batch in tqdm(list(chunked(items, batch_size)), desc=f"Classifying {Path(survey_npz).name}"):
                files = [Path(survey_npz).parent / f[0] for f in batch]
                embeddings = [e[1] for e in batch]
                batch_results = classify_batch(files, embeddings)
                all_results.extend(batch_results)

        df = pd.DataFrame(all_results,columns=['SourceFile','Label','Scores','Width','Height'])
        df['BestScore'] = df.apply(lambda x: x.Scores[x.Label],axis=1)
        df = pd.concat([df,df['Scores'].apply(pd.Series)],axis=1).drop('Scores',axis=1)
        df.columns = df.columns.str.capitalize()
        df.sort_values(['Label','Bestscore']).to_csv(targets[0],index=False)


        #_crops_metadata.csv
    input_dir = cfg.get_url('classify_input_dir')
    #file_dep = list(input_dir.rglob('.csv'))
    crop_size = cfg.get('class_crop_size', 0)
    file_dep = list(input_dir.rglob(f'*clip_{crop_size:03d}.npz'))
    # file_dep = list(set([f.parent / f'{f.parent.name}_224.npz' for f in file_dep]))
    # file_dep = list(filter(lambda x: x.exists(), file_dep))
    return {
        'file_dep' :file_dep,
        'targets' : [Path(cfg.get_url('classify_output_dir')) / f'all_thumbs_classified_{crop_size:03d}.csv'],
        'actions':[classify_images],
        'clean':True,
        'uptodate':[False],
    }

@create_after(executed='classify_images')   
def task_file_clip():
    """
    Doit task for processing files with CLIP.
    """
    crop_size = cfg.get('class_crop_size', 0)
    def process_file_clip(dependencies, targets):
        import os
        from tqdm import tqdm
        data = pd.read_csv(dependencies[0])
        output_dir = cfg.get_url('classify_output_dir')  
        for index, row in tqdm(data.iterrows(), total=len(data), desc="Linking sorted thumbs"):
            #if (row.Width<110) and (row.Height <110):
            source = Path(row.Sourcefile)            
            output = output_dir / 'sorted_thumbs'
            output.mkdir(exist_ok=True, parents=True)
            Image_dest = output / row.Label / f'{int(row.Bestscore*1000):03d}-{source.name}'
            Image_dest.parent.mkdir(exist_ok=True, parents=True)
            if not Image_dest.exists():                                                                                                                                                                                                                                                                                                                                                      
                os.link(source.absolute(), Image_dest)
    file_dep = cfg.get_url('classify_output_dir')  / f'all_thumbs_classified_{crop_size:03d}.csv'             
    return {
            'file_dep' : [file_dep],
            'actions':[process_file_clip],
            'clean':True,                                                                                                                                                                                
            'uptodate':[False],
            
        } 




@app.command('classify')
def doit_classify(
    ctx: typer.Context,
    input_dir: Path = typer.Option(..., help="Directory containing npz files to process"),
    output_dir: Path = typer.Option(..., help="Directory to save .npz embedding files"),
    class_pickle: Path = typer.Option(..., help="Directory to save class npz files"),
    recursive: bool = typer.Option(False, help="Process subdirectories recursively"),
    pattern: str = typer.Option("*.jpg", help="Image pattern to match"),
    doit_db: str = typer.Option(".doit-db", help="Path to doit database file"),
    batch_size: int = typer.Option(128, help="Batch size for processing images"),
    crop_size: int = typer.Option(0, help="Size to which to crop/resize images")
):
    """
    Use doit to encode images in folders (for dependency tracking and incremental builds).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    doit_db = input_dir / doit_db 
    
    # Run doit with custom database location
    cfg.set('classify_input_dir', str(input_dir))
    cfg.set('classify_output_dir', str(output_dir))
    cfg.set('classify_recursive', recursive)
    cfg.set('classify_pattern', pattern)
    cfg.set('encode_batch_size', batch_size)
    cfg.set('class_pickle', str(class_pickle))
    cfg.set('class_crop_size', crop_size)
  
    DoitMain(ModuleTaskLoader(globals())).run(ctx.args + ['--db-file', str(doit_db)])

@app.command('make-prototypes')
def make_prototypes(
    reference_dir: Path = typer.Option(..., help="Directory with reference images (subfolders per class)"),
    output_pickle: Path = typer.Option(..., help="Path to save output pickle file"),
    crop_size: int = typer.Option(0, help="Size to which to crop/resize images")
):
    """
    Encode reference images in subfolders and save class prototypes as a pickle file.
    """
    import pickle
    import torch
    import clip
    import numpy as np
    from PIL import Image
    reference_dir = Path(reference_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    prototypes = {}
    for class_dir in reference_dir.iterdir():
        if not class_dir.is_dir():
            continue
        image_feats = []
        for image_path in class_dir.glob("*.jpg"):
            data = Image.open(image_path).convert("RGB")
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
            image = preprocess(data).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image)
                feat /= feat.norm(dim=-1, keepdim=True)
            image_feats.append(feat.cpu().numpy())
        if image_feats:
            prototype = np.mean(np.vstack(image_feats), axis=0)
            prototype /= np.linalg.norm(prototype)
            prototypes[class_dir.name] = prototype
    pickle_name = output_pickle.with_name(f'{output_pickle.stem}_{crop_size:03d}.pkl')
    with open(pickle_name, "wb") as f:
        pickle.dump(prototypes, f)
    print(f"Saved prototypes for {len(prototypes)} classes to {pickle_name}")

if __name__ == "__main__":
    app()