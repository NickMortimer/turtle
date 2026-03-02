# TurtleDrone Image Classification Tool

A powerful PySide6-based image browser with CLIP embeddings integration for efficient image classification and management of turtle survey images.

## Features

- **Smart Image Loading**: Multi-threaded lazy loading with intelligent caching
- **CLIP Integration**: Uses CLIP embeddings for semantic image similarity
- **Visual Classification**: Assign images to classes using keyboard shortcuts
- **Crop & Zoom**: Multiple thumbnail sizes and crop options for detailed inspection
- **Similarity Sorting**: Double-click any image to sort by visual similarity
- **Filter & Search**: Filter images by classification status or class
- **Professional UI**: Modern interface with menus, toolbars, and keyboard shortcuts
- **Data Persistence**: Save classifications and hidden flags to NPZ files
- **YOLO Export**: Export classifications as YOLO format annotations
- **Unsaved Changes Warning**: Prevents data loss with exit warnings

## Requirements

- Python 3.8+
- PySide6
- NumPy
- Pillow
- CLIP (clip-by-openai)

## Installation

```bash
pip install PySide6 numpy pillow
pip install git+https://github.com/openai/CLIP.git
```

## Usage

### Command Line

```bash
# Basic usage
python -m turtledrone.labelme.image_qt

# With class file and embeddings
python -m turtledrone.labelme.image_qt -c classes.txt -e embeddings.npz
```

### Creating Embeddings

First, generate CLIP embeddings for your images:

```python
# Example using the included clip_encode module
python -m turtledrone.labelme.clip_encode /path/to/your/images/
```

This will create an `embeddings.npz` file in the image directory.

### Basic Workflow

1. **Load Data**:
   - `File > Load Embeddings...` (Ctrl+O) - Load NPZ embeddings file
   - `File > Load Class File...` (Ctrl+Shift+O) - Load class definitions

2. **Classify Images**:
   - Select images in the grid
   - Press `Shift+F1` through `Shift+F12` to assign to classes 1-12
   - Use `Enter` to repeat the last classification
   - Images auto-hide after classification when in "Unallocated" filter

3. **Navigation**:
   - Double-click any image to sort by visual similarity
   - Use thumbnail size controls (`Ctrl+/-`) 
   - Use crop controls (`Shift+Ctrl+/-`) to zoom into image centers
   - Filter by "All Images", "Unallocated", or specific classes

4. **Save & Export**:
   - `Ctrl+S` to save classifications (prevents data loss)
   - `Ctrl+E` to export YOLO format annotations

## Interface Guide

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Load embeddings file |
| `Ctrl+Shift+O` | Load class file |
| `Ctrl+S` | Save classifications |
| `Ctrl+E` | Export YOLO annotations |
| `Shift+F1-F12` | Assign to class 1-12 |
| `Enter` | Assign to last used class |
| `Ctrl+/-` | Thumbnail size |
| `Shift+Ctrl+/-` | Crop zoom |
| `R` | Reset order |
| `Ctrl+Shift+C` | Show classified images |

### Mouse Controls

- **Click**: Select images
- **Ctrl+Click**: Multi-select
- **Double-click**: Sort by similarity
- **Scroll**: Navigate collection

### Thumbnail Options

- **Sizes**: Small (80px), Medium (120px), Large (320px), Extra Large (640px)
- **Crop**: 64px, 128px, or none (shows center crop at full thumbnail size)

### Filters

- **All Images**: Shows all non-hidden images
- **Unallocated**: Shows only unclassified images
- **Class Names**: Shows only images assigned to that class

## Performance Features

### Smart Caching System
- **Processed Thumbnails**: Caches up to 200 processed images with different size/crop combinations
- **Raw Images**: Caches up to 50 raw images (≤2K resolution) for fast reprocessing
- **LRU Eviction**: Automatically manages memory usage
- **Background Loading**: Continues loading while you work

### Memory Optimization
- Progressive image loading as you scroll
- Thread pool optimization (up to 8 threads)
- Efficient Qt model-view architecture
- Smart cache invalidation on setting changes

## File Formats

### Input Files
- **Embeddings**: NumPy NPZ format containing CLIP embeddings
- **Classes**: Plain text file, one class name per line

### Output Files
- **Classifications**: Saved to original NPZ file with metadata
- **YOLO Annotations**: Standard YOLO format in `labels/` directory
- **Classes Export**: `classes.txt` file for YOLO training

## Advanced Features

### Similarity Search
Double-click any image to reorder the entire collection by visual similarity. This uses cosine similarity on CLIP embeddings to find the most visually similar images.

### Crop Functionality
The crop feature allows you to focus on the center of images:
- **64px**: Shows center 64x64 pixels, scaled to thumbnail size
- **128px**: Shows center 128x128 pixels, scaled to thumbnail size
- **none**: Shows full image scaled to fit thumbnail

This is useful for focusing on turtle features while maintaining consistent thumbnail sizes.

### Classification Persistence
All classifications and hidden flags are automatically saved to the NPZ file:
```python
# NPZ file structure
{
    'image1.jpg': embedding_array,
    'image2.jpg': embedding_array,
    # ... more embeddings
    'categories': {
        'image1.jpg': {'class_id': 0, 'class_name': 'turtle'},
        # ... more classifications
    },
    'hidden_flags': {
        'bad_image.jpg': True,
        # ... more hidden flags
    }
}
```

### Unsaved Changes Detection
The application tracks when you make classifications or hide images:
- Window title shows `*` when there are unsaved changes
- Exit warning dialog prevents accidental data loss
- Status bar shows current save state

## Troubleshooting

### Common Issues

1. **No images appear**:
   - Verify images exist in same directory as NPZ file
   - Check console for error messages
   - Ensure NPZ file contains valid embeddings

2. **Slow performance**:
   - Reduce thumbnail size
   - Use smaller image collections for testing
   - Check available system memory

3. **Shortcuts not working**:
   - Ensure application window has focus
   - Load class file for Shift+F shortcuts
   - Check for conflicting system shortcuts

4. **Can't save classifications**:
   - Check file permissions on NPZ file
   - Verify NPZ file path is writable
   - Look for error messages in console

### Debug Output

Enable verbose logging:
```bash
python -m turtledrone.labelme.image_qt --verbose
```

## Example Class File

Create a `classes.txt` file:
```
green_turtle
hawksbill_turtle
loggerhead_turtle
leatherback_turtle
other_species
debris
empty_water
```

## Development

The image classification tool consists of several key components:

- `image_qt.py`: Main application with Qt interface
- `clip_encode.py`: CLIP embedding generation utilities
- Smart caching system with LRU eviction
- Multi-threaded image loading pipeline
- Professional Qt-based user interface

### Architecture

```
ImageBrowser (QMainWindow)
├── ImageLoadWorker (QRunnable) - Multi-threaded loading
├── Smart Cache System - Memory management
├── CLIP Integration - Similarity calculations
└── Qt Model/View - Efficient UI updates
```

## License

Part of the TurtleDrone package - see main repository for license details.
