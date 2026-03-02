#!/usr/bin/env python3
"""
Basic usage example for SymSorter

This script demonstrates how to:
1. Encode images with CLIP
2. Load embeddings
3. Find similar images
4. Launch the GUI
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

def create_sample_images():
    """Create some sample images for testing"""
    try:
        from PIL import Image, ImageDraw
        import random
    except ImportError:
        print("PIL not available. Install with: pip install Pillow")
        return None
    
    # Create temporary directory for sample images
    temp_dir = Path(tempfile.mkdtemp(prefix="symsorter_example_"))
    print(f"📁 Creating sample images in: {temp_dir}")
    
    # Generate different colored rectangles as sample images
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green  
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 128, 128), # Gray
        (255, 165, 0),  # Orange
    ]
    
    for i, color in enumerate(colors):
        # Create image with colored rectangle and some noise
        img = Image.new('RGB', (224, 224), color)
        draw = ImageDraw.Draw(img)
        
        # Add some variation to make images different
        for _ in range(10):
            x = random.randint(0, 200)
            y = random.randint(0, 200)
            draw.rectangle([x, y, x+20, y+20], fill=(
                random.randint(0, 255),
                random.randint(0, 255), 
                random.randint(0, 255)
            ))
        
        # Save image
        img_path = temp_dir / f"sample_{i:02d}_{['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray', 'orange'][i]}.jpg"
        img.save(img_path)
    
    print(f"✅ Created {len(colors)} sample images")
    return temp_dir

def example_encoding():
    """Example of encoding images with CLIP"""
    print("\n🧠 Example 1: Encoding Images with CLIP")
    print("=" * 50)
    
    # Create sample images
    image_dir = create_sample_images()
    if not image_dir:
        return
    
    try:
        from symsorter.clip_encode import encode_images_in_folder
        
        print("🔄 Encoding images...")
        embeddings_path = encode_images_in_folder(
            folder_path=image_dir,
            batch_size=4,
            device="auto"
        )
        
        print(f"✅ Embeddings saved to: {embeddings_path}")
        return image_dir, embeddings_path
        
    except ImportError:
        print("❌ CLIP not installed - cannot encode images")
        print("   Install with: pip install git+https://github.com/openai/CLIP.git")
        return image_dir, None
    except Exception as e:
        print(f"❌ Error encoding images: {e}")
        return image_dir, None

def example_similarity(image_dir, embeddings_path):
    """Example of finding similar images"""
    if not embeddings_path or not embeddings_path.exists():
        print("\n⚠️  Skipping similarity example - no embeddings available")
        return
    
    print("\n🔍 Example 2: Finding Similar Images")
    print("=" * 50)
    
    try:
        from symsorter.clip_encode import load_existing_embeddings
        import numpy as np
        
        # Load embeddings
        embeddings = load_existing_embeddings(embeddings_path)
        print(f"📚 Loaded {len(embeddings)} image embeddings")
        
        # Pick first image as query
        query_filename = list(embeddings.keys())[0]
        query_embedding = embeddings[query_filename]
        
        print(f"🎯 Using '{query_filename}' as query image")
        
        # Calculate similarities
        similarities = []
        for filename, embedding in embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((filename, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("\n📊 Most similar images:")
        for i, (filename, sim) in enumerate(similarities[:5], 1):
            print(f"  {i}. {filename:<25} (similarity: {sim:.4f})")
        
    except Exception as e:
        print(f"❌ Error in similarity calculation: {e}")

def example_gui(embeddings_path):
    """Example of launching the GUI"""
    print("\n🖼️  Example 3: Launching GUI")
    print("=" * 50)
    
    try:
        from PySide6.QtWidgets import QApplication
        from symsorter.image_browser import ImageBrowser
        
        print("🚀 Launching SymSorter GUI...")
        print("   - Use File > Load Embeddings to load your NPZ file")
        print("   - Double-click images to sort by similarity")
        print("   - Use Ctrl+/- to resize thumbnails")
        print("   - Close the window to continue with the example")
        
        # Create QApplication
        if not QApplication.instance():
            app = QApplication([])
        else:
            app = QApplication.instance()
        
        # Create and show browser
        browser = ImageBrowser()
        
        # Load embeddings if available
        if embeddings_path and embeddings_path.exists():
            browser.load_folder_from_path(str(embeddings_path))
        
        browser.show()
        
        print("✅ GUI launched successfully!")
        print("   Close the GUI window to continue...")
        
        # Run the application
        app.exec()
        
    except ImportError:
        print("❌ PySide6 not available - cannot launch GUI")
        print("   Install with: pip install PySide6")
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")

def main():
    """Run all examples"""
    print("🎯 SymSorter Usage Examples")
    print("=" * 50)
    print("This script demonstrates basic SymSorter functionality")
    
    # Example 1: Encoding
    result = example_encoding()
    if result:
        image_dir, embeddings_path = result
    else:
        return 1
    
    # Example 2: Similarity
    example_similarity(image_dir, embeddings_path)
    
    # Example 3: GUI
    if input("\n🤔 Launch GUI example? (y/n): ").lower().startswith('y'):
        example_gui(embeddings_path)
    
    # Cleanup
    print("\n🧹 Cleaning up temporary files...")
    if image_dir and image_dir.exists():
        shutil.rmtree(image_dir)
        print("✅ Temporary files cleaned up")
    
    print("\n🎉 Examples completed!")
    print("\n📖 Next steps:")
    print("   - Try with your own images: symsorter encode /path/to/images")
    print("   - Launch GUI: symsorter gui")
    print("   - Read the documentation: https://github.com/NickMortimer/symsorter")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
