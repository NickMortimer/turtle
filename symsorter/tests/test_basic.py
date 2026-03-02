"""
Tests for SymSorter package
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

def test_package_import():
    """Test that the package can be imported"""
    import symsorter
    assert symsorter.__version__ == "0.1.0"

def test_clip_encode_import():
    """Test that clip_encode module can be imported"""
    from symsorter.clip_encode import load_existing_embeddings
    assert callable(load_existing_embeddings)

def test_load_existing_embeddings():
    """Test loading embeddings from NPZ file"""
    from symsorter.clip_encode import load_existing_embeddings
    
    # Create temporary NPZ file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_path = f.name
    
    try:
        # Create test data
        embeddings = np.random.rand(5, 512).astype(np.float32)
        filenames = np.array(['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg'])
        
        # Save test data
        np.savez_compressed(temp_path, embeddings=embeddings, filenames=filenames)
        
        # Test loading
        loaded_embeddings = load_existing_embeddings(Path(temp_path))
        
        assert len(loaded_embeddings) == 5
        assert 'img1.jpg' in loaded_embeddings
        assert loaded_embeddings['img1.jpg'].shape == (512,)
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_load_nonexistent_embeddings():
    """Test loading from non-existent file"""
    from symsorter.clip_encode import load_existing_embeddings
    
    non_existent_path = Path("/tmp/does_not_exist.npz")
    embeddings = load_existing_embeddings(non_existent_path)
    
    assert embeddings == {}

@pytest.mark.skipif(not os.environ.get('GUI_TESTS'), reason="GUI tests disabled")
def test_image_browser_import():
    """Test that ImageBrowser can be imported (only if GUI tests enabled)"""
    from symsorter.image_browser import ImageBrowser
    assert ImageBrowser is not None

def test_cli_import():
    """Test that CLI module can be imported"""
    from symsorter.cli import main
    assert callable(main)
