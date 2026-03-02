# Changelog

All notable changes to SymSorter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-09-30

### Added
- Initial release of SymSorter
- CLIP-based image similarity and classification
- Professional Qt-based GUI with grid view
- Smart caching system with LRU eviction (3000 image cache)
- Multi-threaded image loading with background processing
- Crop functionality (64px, 128px, none)
- YOLO-compatible class management with keyboard shortcuts
- Similarity sorting by double-clicking images
- Class filtering and auto-hiding of classified images
- Export functionality for YOLO training datasets
- Command-line tools for encoding and similarity search
- Comprehensive documentation and examples
- Unsaved changes protection
- Persistent storage of classifications in NPZ format

### Features
- **GUI Application**: Full-featured image classification interface
- **CLI Tools**: `symsorter encode`, `symsorter similarity`, `symsorter gui`
- **Python API**: Programmatic access to all functionality
- **Performance**: Optimized for large image collections
- **Flexibility**: Supports various image formats and workflows

### Technical Details
- Python 3.9+ support
- PySide6-based GUI
- PyTorch and CLIP integration
- NPZ file format for efficient storage
- Multi-threaded architecture
- Smart memory management
