"""
Command-line interface for SymSorter
"""
import typer
from pathlib import Path
import sys
from .clip_encode import app as encode_app, encode_images_in_folder

app = typer.Typer(
    name="symsorter",
    help="SymSorter - A CLIP-based image classification and similarity tool",
    no_args_is_help=True
)

# Add encoding commands as subcommand
app.add_typer(encode_app, name="encode", help="CLIP encoding commands")

@app.command()
def gui(
    embeddings_file: Path = typer.Option(None, help="NPZ file with CLIP embeddings to load"),
    class_file: Path = typer.Option(None, help="Text file with class names (one per line)")
):
    """
    Launch the SymSorter GUI for image classification and sorting.
    """
    try:
        from PySide6.QtWidgets import QApplication
        from .image_browser import ImageBrowser
        
        # Create QApplication
        if not QApplication.instance():
            app_qt = QApplication(sys.argv)
        else:
            app_qt = QApplication.instance()
        
        # Create and show the image browser
        browser = ImageBrowser(class_file=str(class_file) if class_file else None)
        
        # Load embeddings if provided
        if embeddings_file and embeddings_file.exists():
            browser.load_folder_from_path(str(embeddings_file))
        
        browser.show()
        
        # Run the application
        sys.exit(app_qt.exec())
        
    except ImportError as e:
        typer.echo("❌ GUI dependencies not available. Install with:", err=True)
        typer.echo("pip install 'symsorter[gui]'", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error launching GUI: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def similarity(
    embeddings_file: Path = typer.Argument(..., help="NPZ file with CLIP embeddings"),
    query_image: Path = typer.Argument(..., help="Query image to find similar images for"),
    top_k: int = typer.Option(10, help="Number of most similar images to return"),
    output_dir: Path = typer.Option(None, help="Directory to copy similar images to")
):
    """
    Find images most similar to a query image using CLIP embeddings.
    """
    import numpy as np
    import shutil
    from .clip_encode import load_existing_embeddings, load_clip_model, CLIP_AVAILABLE
    
    if not CLIP_AVAILABLE:
        typer.echo("❌ CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git", err=True)
        raise typer.Exit(1)
    
    try:
        # Load embeddings
        embeddings = load_existing_embeddings(embeddings_file)
        if not embeddings:
            typer.echo(f"❌ No embeddings found in {embeddings_file}")
            raise typer.Exit(1)
        
        # Load CLIP model
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = load_clip_model(device)
        
        # Process query image
        from PIL import Image
        query_img = Image.open(query_image).convert("RGB")
        query_tensor = preprocess(query_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_features = model.encode_image(query_tensor)
            query_features /= query_features.norm(dim=-1, keepdim=True)
        
        query_embedding = query_features.cpu().numpy().flatten()
        
        # Calculate similarities
        similarities = []
        for filename, embedding in embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((filename, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        typer.echo(f"🔍 Top {top_k} most similar images to {query_image.name}:")
        typer.echo()
        
        image_dir = embeddings_file.parent
        for i, (filename, sim) in enumerate(similarities[:top_k], 1):
            typer.echo(f"{i:2d}. {filename:<30} (similarity: {sim:.4f})")
            
            # Copy to output directory if specified
            if output_dir:
                output_dir.mkdir(exist_ok=True)
                src = image_dir / filename
                dst = output_dir / f"{i:02d}_{filename}"
                if src.exists():
                    shutil.copy2(src, dst)
        
        if output_dir:
            typer.echo(f"\n📁 Similar images copied to {output_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)

def main():
    """Main entry point for symsorter CLI"""
    app()

if __name__ == "__main__":
    main()
