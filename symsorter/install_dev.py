#!/usr/bin/env python3
"""
Build and install SymSorter package locally for testing
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n📦 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        return False
    else:
        print(f"✅ Success: {description}")
        if result.stdout.strip():
            print(result.stdout.strip())
        return True

def main():
    """Build and install the package"""
    print("🚀 Building SymSorter package...")
    
    # Change to package directory
    package_dir = Path(__file__).parent
    original_dir = Path.cwd()
    
    try:
        os.chdir(package_dir)
        
        # Clean previous builds
        if not run_command("rm -rf build/ dist/ *.egg-info/", "Cleaning previous builds"):
            return 1
        
        # Build the package
        if not run_command("python -m build", "Building package"):
            print("⚠️  If build failed, try: pip install build")
            return 1
        
        # Install in development mode
        if not run_command("pip install -e .", "Installing in development mode"):
            return 1
        
        # Install CLIP (optional)
        print("\n🧠 Installing CLIP (optional, may take a few minutes)...")
        clip_result = subprocess.run(
            "pip install git+https://github.com/openai/CLIP.git", 
            shell=True, capture_output=True, text=True
        )
        if clip_result.returncode == 0:
            print("✅ CLIP installed successfully")
        else:
            print("⚠️  CLIP installation failed (will limit functionality)")
            print("   You can install it later with:")
            print("   pip install git+https://github.com/openai/CLIP.git")
        
        # Test import
        if not run_command("python -c 'import symsorter; print(f\"SymSorter {symsorter.__version__} imported successfully\")'", "Testing import"):
            return 1
        
        # Test CLI
        if not run_command("symsorter --help", "Testing CLI"):
            return 1
        
        print("\n🎉 SymSorter installed successfully!")
        print("\n📖 Quick start:")
        print("   # Encode images:")
        print("   symsorter encode /path/to/images")
        print("   # Launch GUI:")
        print("   symsorter gui")
        print("   # Find similar images:")
        print("   symsorter similarity embeddings.npz query.jpg")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    import os
    sys.exit(main())
