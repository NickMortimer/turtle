# SymSorter

A powerful CLIP-based image classification and similarity tool for intelligent image sorting and research workflows.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyPI](https://img.shields.io/pypi/v/symsorter)
![Python](https://img.shields.io/pypi/pyversions/symsorter)

## Features

SymSorter provides state-of-the-art image analysis capabilities using OpenAI's CLIP (Contrastive Language-Image Pre-training) model:

### 🖼️ **Smart Image Browser**
- **Grid-based interface** with customizable thumbnail sizes (Small, Medium, Large, Extra Large)
- **Intelligent caching** system with LRU eviction for optimal performance
- **Multi-threaded loading** with background processing
- **Crop functionality** (64px, 128px, or full image view)
- **Similarity sorting** by double-clicking any image

### 🧠 **CLIP-Powered Analysis**
- **Semantic embeddings** for true image understanding
- **Cosine similarity** calculations for finding related images  
- **Batch processing** for efficient encoding of large collections
- **NPZ storage format** for fast loading and persistence

### 🏷️ **Classification Workflow**
- **YOLO-compatible** class management
- **Keyboard shortcuts** (Shift+F1-F12) for rapid classification
- **Auto-hiding** of classified images to focus on unprocessed content
- **Class filtering** to view specific categories
- **Export functionality** for YOLO training datasets

### 💾 **Data Management**
- **Persistent storage** of classifications and hidden flags
- **Unsaved changes warnings** to protect your work
- **Import/Export** capabilities for different workflows
- **Command-line tools** for automation and batch processing

## Installation

### Basic Installation

```bash
pip install symsorter
```

### With CLIP Support

```bash
pip install symsorter
pip install git+https://github.com/openai/CLIP.git
```

### Development Installation

```bash
git clone https://github.com/NickMortimer/symsorter.git
cd symsorter
pip install -e .
pip install git+https://github.com/openai/CLIP.git
```

## Quick Start

### 1. Encode Images with CLIP

First, generate CLIP embeddings for your image collection:

```bash
# Encode all images in a folder
symsorter encode /path/to/your/images

# With custom output path
symsorter encode /path/to/images --output /path/to/embeddings.npz

# With center cropping
symsorter encode /path/to/images --crop-size 224
```

### 2. Launch the GUI

```bash
# Launch with file dialogs
symsorter gui

# Load embeddings and classes automatically  
symsorter gui --embeddings /path/to/embeddings.npz --class-file /path/to/classes.txt
```

### 3. Find Similar Images

```bash
# Find the 10 most similar images to a query
symsorter similarity embeddings.npz query_image.jpg

# Copy results to output directory
symsorter similarity embeddings.npz query_image.jpg --output-dir similar_images/
```

## Usage Guide

### GUI Interface

#### Loading Data
1. **File > Load Embeddings...** - Select your `.npz` file with CLIP embeddings
2. **File > Load Class File...** - Load a text file with class names (one per line)

#### Navigation & Viewing
- **Mouse wheel** or **scroll bar** - Navigate through image collection
- **Ctrl + Mouse wheel** - Zoom thumbnails in/out  
- **Double-click** image - Sort collection by similarity to clicked image
- **R** - Reset to original order

#### Classification Workflow
- **Select images** (click, Ctrl+click, or drag selection)
- **Shift+F1 to F12** - Assign selected images to classes 1-12
- **Enter** - Assign to last used class (quick repeat)
- Classified images auto-hide from "Unallocated" view

#### Filtering & Organization
- **Filter dropdown** - View "All Images", "Unallocated", or specific classes
- **View > Show Classified Images** - Temporarily show all classified images
- **File > Save Classifications** - Save progress to NPZ file

#### Advanced Features
- **Crop controls** (Shift+Ctrl+/-) - View center crops at 64px, 128px, or full size
- **Thumbnail sizes** (Ctrl+/-) - Adjust display size for optimal workflow
- **Export** - Generate YOLO-format annotations for training

### Command Line Tools

#### Encoding Images

```bash
# Basic encoding
symsorter encode /images/folder

# Advanced options
symsorter encode /images/folder \
    --output custom_embeddings.npz \
    --crop-size 224 \
    --batch-size 64 \
    --device cuda
```

#### Similarity Search

```bash
# Find similar images
symsorter similarity embeddings.npz query.jpg --top-k 20

# Save results
symsorter similarity embeddings.npz query.jpg \
    --top-k 10 \
    --output-dir ./similar_images
```

#### Inspect Embeddings

```bash
# View NPZ file contents
symsorter encode inspect embeddings.npz
```

## File Formats

### NPZ Embeddings File
- **embeddings**: CLIP feature vectors (float32 array)
- **filenames**: Image filenames (string array)  
- **hidden_flags**: Dict of hidden status per image
- **categories**: Dict of classification assignments

### Class File Format
Plain text file with one class name per line:
```
turtle
background
rocks
vegetation
water
```

## Configuration

### Performance Tuning

**Cache Settings**: The default cache holds 3000 processed thumbnails and 50 raw images. Adjust based on your system:

- **More RAM**: Increase cache for better performance with large collections
- **Less RAM**: Decrease cache to prevent memory issues
- **SSD Storage**: Caching is less critical with fast storage

**Threading**: Uses CPU count × 2 threads by default, capped at 8 for optimal balance.

### GPU Acceleration

CLIP encoding automatically uses CUDA if available:
```bash
# Force CPU usage
symsorter encode /images --device cpu

# Force GPU usage  
symsorter encode /images --device cuda
```

## Integration

### Python API

```python
from symsorter import ImageBrowser, load_existing_embeddings
from symsorter.clip_encode import encode_images_in_folder

# Encode images programmatically
embeddings_path = encode_images_in_folder(
    folder_path="./images",
    crop_size=224,
    batch_size=32
)

# Load embeddings
embeddings = load_existing_embeddings(embeddings_path)

# Launch GUI programmatically
app = QApplication([])
browser = ImageBrowser(class_file="classes.txt")
browser.load_folder_from_path(embeddings_path)
browser.show()
app.exec()
```

### Research Workflows

**Turtle Research Example**:
1. Collect drone imagery of turtle populations
2. Use YOLO to detect potential turtles → crop images
3. Encode crops with SymSorter: `symsorter encode turtle_crops/`
4. Classify in GUI: Load embeddings + turtle/background classes
5. Export training data: **File > Export YOLO Annotations**
6. Train improved YOLO model with classified data

## Requirements

- **Python 3.9+**
- **PySide6** (GUI framework)
- **PyTorch** (deep learning backend)
- **CLIP** (vision-language model)
- **NumPy, Pillow, tqdm** (supporting libraries)

### System Requirements
- **RAM**: 4GB minimum, 8GB+ recommended for large collections
- **GPU**: Optional but recommended for encoding (CUDA support)
- **Storage**: Fast SSD recommended for large image collections

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/NickMortimer/symsorter.git
cd symsorter
pip install -e ".[dev]"
pip install git+https://github.com/openai/CLIP.git

# Run tests
pytest

# Format code
black symsorter/
isort symsorter/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use SymSorter in your research, please cite:

```bibtex
@software{symsorter2024,
  title={SymSorter: A CLIP-based Image Classification and Similarity Tool},
  author={Mortimer, Nick},
  year={2024},
  url={https://github.com/NickMortimer/symsorter}
}
```

## Acknowledgments

- **OpenAI CLIP** - The foundation model powering semantic understanding
- **Qt/PySide6** - Cross-platform GUI framework
- **PyTorch** - Deep learning infrastructure

## Support

- **Issues**: [GitHub Issues](https://github.com/NickMortimer/symsorter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NickMortimer/symsorter/discussions)
- **Email**: nick.mortimer@csiro.au

---

**SymSorter** - Making image classification intelligent, efficient, and enjoyable.
