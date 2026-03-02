"""
GUI launcher for SymSorter
"""
import sys
from pathlib import Path

def main():
    """Main entry point for SymSorter GUI"""
    try:
        from PySide6.QtWidgets import QApplication
        from .image_browser import ImageBrowser
        
        # Create QApplication
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()
        
        # Create and show the image browser
        browser = ImageBrowser()
        browser.show()
        
        # Run the application
        sys.exit(app.exec())
        
    except ImportError as e:
        print("❌ GUI dependencies not available. Install with:")
        print("pip install 'symsorter[gui]'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
