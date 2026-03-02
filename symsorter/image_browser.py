from PySide6.QtWidgets import (
    QApplication, QWidget, QListView, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QAbstractItemView, QSlider, QLabel, QComboBox,
    QToolBar, QMenuBar, QMainWindow, QMessageBox
)
from PySide6.QtGui import QPixmap, QIcon, QStandardItemModel, QStandardItem, QKeySequence, QShortcut, QAction
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer, QThreadPool, QRunnable, QObject
import sys
import os
import math
import numpy as np
import argparse
from pathlib import Path
from .clip_encode import load_existing_embeddings

class WorkerSignals(QObject):
    """Signals for the worker thread"""
    imageLoaded = Signal(int, QIcon)
    error = Signal(str)

class ImageLoadWorker(QRunnable):
    """Worker for loading a single image in thread pool"""
    def __init__(self, index, file_path, icon_size, crop_size, cache_manager=None):
        super().__init__()
        self.index = index
        self.file_path = file_path
        self.icon_size = icon_size
        self.crop_size = crop_size  # None for no crop, or pixel size for center crop
        self.cache_manager = cache_manager  # Reference to main window for cache access
        self.should_stop = False
        self.signals = WorkerSignals()
    
    def stop(self):
        """Set stop flag for graceful shutdown"""
        self.should_stop = True
    
    def run(self):
        """Load and process image"""
        if self.should_stop:
            return
            
        try:
            filename = os.path.basename(self.file_path)
            
            # Check cache first
            if self.cache_manager:
                cached_icon = self.cache_manager.get_cached_icon(filename, self.icon_size, self.crop_size)
                if cached_icon:
                    self.signals.imageLoaded.emit(self.index, cached_icon)
                    return
            
            if self.should_stop:
                return
                
            # Load image
            if not os.path.exists(self.file_path):
                self.signals.error.emit(f"File not found: {self.file_path}")
                return
            
            # Try to get cached raw image first
            raw_pixmap = None
            if self.cache_manager:
                raw_pixmap = self.cache_manager.get_cached_raw_image(filename)
            
            if raw_pixmap is None:
                # Load from disk
                raw_pixmap = QPixmap(self.file_path)
                if raw_pixmap.isNull():
                    self.signals.error.emit(f"Failed to load image: {self.file_path}")
                    return
                
                # Cache the raw image if it's reasonably sized
                if self.cache_manager:
                    self.cache_manager.cache_raw_image(filename, raw_pixmap)
            
            if self.should_stop:
                return
            
            # Process the image (crop if needed, then scale)
            processed_pixmap = raw_pixmap
            
            # Apply crop if specified
            if self.crop_size is not None and self.crop_size > 0:
                # Calculate center crop
                orig_width = processed_pixmap.width()
                orig_height = processed_pixmap.height()
                
                # Calculate crop rectangle (center crop)
                crop_x = max(0, (orig_width - self.crop_size) // 2)
                crop_y = max(0, (orig_height - self.crop_size) // 2)
                crop_width = min(self.crop_size, orig_width)
                crop_height = min(self.crop_size, orig_height)
                
                processed_pixmap = processed_pixmap.copy(crop_x, crop_y, crop_width, crop_height)
            
            # Scale to thumbnail size
            scaled_pixmap = processed_pixmap.scaled(
                QSize(self.icon_size, self.icon_size),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            if self.should_stop:
                return
            
            # Create icon
            icon = QIcon(scaled_pixmap)
            
            # Cache the processed icon
            if self.cache_manager:
                self.cache_manager.cache_icon(filename, self.icon_size, self.crop_size, icon)
            
            # Emit the result
            self.signals.imageLoaded.emit(self.index, icon)
            
        except Exception as e:
            self.signals.error.emit(f"Error loading image {self.file_path}: {str(e)}")

class LazyImageLoader(QThread):
    """Thread for lazy loading images as needed"""
    imageLoaded = Signal(int, QIcon)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.images_to_load = []
        self.folder = ""
        self.icon_size = 120
        self.crop_size = None  # No crop by default
        
    def run(self):
        for index, filename in self.images_to_load:
            file_path = os.path.join(self.folder, filename)
            
            # Load and scale image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Apply crop if specified
                if self.crop_size is not None and self.crop_size > 0:
                    # Calculate center crop
                    orig_width = pixmap.width()
                    orig_height = pixmap.height()
                    
                    # Calculate crop rectangle (center crop)
                    crop_x = max(0, (orig_width - self.crop_size) // 2)
                    crop_y = max(0, (orig_height - self.crop_size) // 2)
                    crop_width = min(self.crop_size, orig_width)
                    crop_height = min(self.crop_size, orig_height)
                    
                    pixmap = pixmap.copy(crop_x, crop_y, crop_width, crop_height)
                
                # Scale to thumbnail size
                scaled_pixmap = pixmap.scaled(
                    QSize(self.icon_size, self.icon_size),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                icon = QIcon(scaled_pixmap)
                self.imageLoaded.emit(index, icon)

class ImageBrowser(QMainWindow):
    def __init__(self, class_file=None):
        super().__init__()
        self.resize(1000, 700)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Image grid view
        self.view = QListView()
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setViewMode(QListView.IconMode)
        self.view.setResizeMode(QListView.Adjust)
        self.view.setUniformItemSizes(True)
        self.view.setSpacing(2)
        layout.addWidget(self.view)

        # Model
        self.model = QStandardItemModel()
        self.view.setModel(self.model)
        
        # Store folder path and image files
        self.folder = ""
        self.image_files = []
        self.embeddings = {}  # Store embeddings for each image
        self.hidden_flags = {}  # Store hidden status for each image
        self.loaded_images = set()  # Track which individual images have been loaded
        self.original_order = []  # Store original order for reset
        self.npz_file_path = None  # Store path to npz file for saving
        
        # Smart image caching system
        self.image_cache = {}  # Cache for processed thumbnails: {(filename, icon_size, crop_size): QIcon}
        self.raw_image_cache = {}  # Cache for raw QPixmap images: {filename: QPixmap}
        self.max_cache_size = 3000  # Maximum number of images to keep in cache
        self.cache_access_order = []  # Track access order for LRU eviction
        
        # Track unsaved changes
        self.has_unsaved_changes = False
        
        # YOLO class management
        self.classes = []  # List of class names
        self.image_categories = {}  # Store category assignments for each image
        self.class_file_path = class_file
        self.last_used_class_idx = None  # Track last used class for Enter key
        
        # Icon and crop settings
        self.icon_sizes = {"Small": 80, "Medium": 120, "Large": 320, "Extra Large": 640}
        self.current_size_name = "Medium"
        self.icon_size = self.icon_sizes[self.current_size_name]  # Current icon size
        self.crop_sizes = {"64": 64, "128": 128, "none": None}
        self.current_crop_name = "none"
        self.crop_size = self.crop_sizes[self.current_crop_name]  # Current crop size (None = no crop)

        # Create a simple placeholder icon (will be updated with zoom)
        self.update_placeholder_icon()
        
        # Thread pool for faster image loading
        self.thread_pool = QThreadPool()
        # Set max threads to CPU count * 2 (good for I/O bound tasks like image loading)
        max_threads = min(8, QThreadPool.globalInstance().maxThreadCount() * 2)
        self.thread_pool.setMaxThreadCount(max_threads)
        self.active_workers = []  # Keep references to active workers
        self.background_load_timer = QTimer()  # Timer for background loading
        self.background_load_timer.timeout.connect(self.load_next_batch_background)
        print(f"Image loading thread pool initialized with {max_threads} threads")
        
        # Connect scroll event for lazy loading and double-click for sorting
        self.view.verticalScrollBar().valueChanged.connect(self.on_scroll)
        self.view.doubleClicked.connect(self.on_double_click)
        
        # Connect resize event to load more images when window grows
        self.view.resizeEvent = self.on_view_resize
        
        # Load class file if provided
        if self.class_file_path:
            self.load_class_file(self.class_file_path)
        
        # Set up menus and toolbar
        self.setup_menus_and_toolbar()
        
        # Set up keyboard shortcuts
        self.setup_shortcuts()

        # Class filter layout
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter by class:")
        filter_layout.addWidget(filter_label)
        
        # Class filter combobox
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItem("All Images")
        self.class_filter_combo.addItem("Unallocated")
        self.class_filter_combo.currentTextChanged.connect(self.on_class_filter_changed)
        filter_layout.addWidget(self.class_filter_combo)
        filter_layout.addStretch()  # Push everything to the left
        layout.addLayout(filter_layout)
        
        # Class info label
        self.class_info_label = QLabel("No classes loaded")
        layout.addWidget(self.class_info_label)
        
        # Set initial window title
        self.update_window_title()
    
    def setup_menus_and_toolbar(self):
        """Setup menus and toolbar"""
        # Create menu bar
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Load embeddings action
        load_action = QAction('&Load Embeddings...', self)
        load_action.setShortcut(QKeySequence('Ctrl+O'))
        load_action.setStatusTip('Load embeddings from NPZ file')
        load_action.triggered.connect(self.load_folder)
        file_menu.addAction(load_action)
        
        # Load class file action
        load_classes_action = QAction('Load &Class File...', self)
        load_classes_action.setShortcut(QKeySequence('Ctrl+Shift+O'))
        load_classes_action.setStatusTip('Load YOLO class names file')
        load_classes_action.triggered.connect(self.load_class_file_dialog)
        file_menu.addAction(load_classes_action)
        
        file_menu.addSeparator()
        
        # Save action
        save_action = QAction('&Save Classifications', self)
        save_action.setShortcut(QKeySequence('Ctrl+S'))
        save_action.setStatusTip('Save classifications and hidden flags')
        save_action.triggered.connect(self.save_hidden_flags)
        file_menu.addAction(save_action)
        
        # Export action
        export_action = QAction('&Export YOLO Annotations...', self)
        export_action.setShortcut(QKeySequence('Ctrl+E'))
        export_action.setStatusTip('Export classifications as YOLO annotations')
        export_action.triggered.connect(self.export_yolo_annotations)
        file_menu.addAction(export_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Reset order action
        reset_action = QAction('&Reset Order', self)
        reset_action.setShortcut(QKeySequence('R'))
        reset_action.setStatusTip('Reset images to original order')
        reset_action.triggered.connect(self.reset_order)
        view_menu.addAction(reset_action)
        
        # Show classified action
        show_classified_action = QAction('Show &Classified Images', self)
        show_classified_action.setShortcut(QKeySequence('Ctrl+Shift+C'))
        show_classified_action.setStatusTip('Show all classified images')
        show_classified_action.triggered.connect(self.show_classified_images)
        view_menu.addAction(show_classified_action)
        
        view_menu.addSeparator()
        
        # Thumbnail size controls
        increase_size_action = QAction('&Increase Thumbnail Size', self)
        increase_size_action.setShortcut(QKeySequence('Ctrl+='))
        increase_size_action.setStatusTip('Increase thumbnail size (Ctrl++)')
        increase_size_action.triggered.connect(self.increase_thumbnail_size)
        view_menu.addAction(increase_size_action)
        
        decrease_size_action = QAction('&Decrease Thumbnail Size', self)
        decrease_size_action.setShortcut(QKeySequence('Ctrl+-'))
        decrease_size_action.setStatusTip('Decrease thumbnail size (Ctrl+-)')
        decrease_size_action.triggered.connect(self.decrease_thumbnail_size)
        view_menu.addAction(decrease_size_action)
        
        view_menu.addSeparator()
        
        # Crop controls
        increase_crop_action = QAction('Increase &Crop Zoom', self)
        increase_crop_action.setShortcut(QKeySequence('Shift+Ctrl+='))
        increase_crop_action.setStatusTip('Increase crop zoom (Shift+Ctrl++)')
        increase_crop_action.triggered.connect(self.increase_crop_zoom)
        view_menu.addAction(increase_crop_action)
        
        decrease_crop_action = QAction('Decrease C&rop Zoom', self)
        decrease_crop_action.setShortcut(QKeySequence('Shift+Ctrl+-'))
        decrease_crop_action.setStatusTip('Decrease crop zoom (Shift+Ctrl+-)')
        decrease_crop_action.triggered.connect(self.decrease_crop_zoom)
        view_menu.addAction(decrease_crop_action)
        
        view_menu.addSeparator()
        
        # Thumbnail size submenu
        size_menu = view_menu.addMenu('&Thumbnail Size')
        self.size_actions = []
        for size_name in self.icon_sizes.keys():
            action = QAction(f'&{size_name}', self)
            action.setCheckable(True)
            action.setStatusTip(f'Set thumbnail size to {size_name} ({self.icon_sizes[size_name]}px)')
            # Fix closure issue by using proper closure
            action.triggered.connect(self.create_size_handler(size_name))
            if size_name == self.current_size_name:
                action.setChecked(True)
            size_menu.addAction(action)
            self.size_actions.append(action)
        
        # Crop submenu
        crop_menu = view_menu.addMenu('&Crop')
        self.crop_actions = []
        for crop_name in self.crop_sizes.keys():
            action = QAction(f'&{crop_name}', self)
            action.setCheckable(True)
            if crop_name == "none":
                action.setStatusTip('No cropping - show full image')
            else:
                action.setStatusTip(f'Crop to {crop_name}x{crop_name} pixels from center')
            # Fix closure issue by using proper closure
            action.triggered.connect(self.create_crop_handler(crop_name))
            if crop_name == self.current_crop_name:
                action.setChecked(True)
            crop_menu.addAction(action)
            self.crop_actions.append(action)
        
        # Classes menu (will be populated when classes are loaded)
        self.classes_menu = menubar.addMenu('&Classes')
        self.classes_menu.setEnabled(False)  # Disabled until classes are loaded
        self.update_classes_menu()
        
        # Create toolbar
        toolbar = QToolBar()
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)
        
        # Add actions to toolbar
        toolbar.addAction(load_action)
        toolbar.addAction(load_classes_action)
        toolbar.addSeparator()
        toolbar.addAction(save_action)
        toolbar.addAction(export_action)
        toolbar.addSeparator()
        toolbar.addAction(reset_action)
        toolbar.addAction(show_classified_action)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts for class assignment"""        
        # Keyboard shortcuts are now handled by QAction shortcuts in the Classes menu
        # This avoids duplicate shortcut registration which causes "Ambiguous shortcut overload"
        self.function_key_shortcuts = []
        print("DEBUG: Keyboard shortcuts will be created by Classes menu actions")
    
    def create_class_assignment_handler(self, class_idx):
        """Create a proper closure for class assignment"""
        def handler():
            print(f"DEBUG: Shortcut activated for class index {class_idx}")
            self.assign_class_to_selected(class_idx)
        return handler
    
    def create_size_handler(self, size_name):
        """Create a proper closure for thumbnail size selection"""
        def handler():
            self.set_thumbnail_size(size_name)
        return handler
    
    def create_crop_handler(self, crop_name):
        """Create a proper closure for crop size selection"""
        def handler():
            self.set_crop_size(crop_name)
        return handler
    
    def set_thumbnail_size(self, size_name):
        """Set thumbnail size from menu action"""
        if size_name == self.current_size_name:
            return  # Already this size
            
        # Update menu checkmarks
        for action in self.size_actions:
            action.setChecked(action.text().replace('&', '') == size_name)
        
        # Apply the size change directly
        self.on_thumbnail_size_changed(size_name)
    
    def increase_thumbnail_size(self):
        """Increase thumbnail size using Ctrl+Plus"""
        size_names = list(self.icon_sizes.keys())
        current_index = size_names.index(self.current_size_name)
        
        if current_index < len(size_names) - 1:
            new_size_name = size_names[current_index + 1]
            self.set_thumbnail_size(new_size_name)
            print(f"Increased thumbnail size to {new_size_name}")
        else:
            print("Already at maximum thumbnail size")
    
    def decrease_thumbnail_size(self):
        """Decrease thumbnail size using Ctrl+Minus"""
        size_names = list(self.icon_sizes.keys())
        current_index = size_names.index(self.current_size_name)
        
        if current_index > 0:
            new_size_name = size_names[current_index - 1]
            self.set_thumbnail_size(new_size_name)
            print(f"Decreased thumbnail size to {new_size_name}")
        else:
            print("Already at minimum thumbnail size")
    
    def assign_to_last_used_class(self):
        """Assign selected images to the last used class (Enter key)"""
        if self.last_used_class_idx is not None:
            self.assign_class_to_selected(self.last_used_class_idx)
        else:
            print("No class has been used yet. Use Shift+F1-F12 to assign a class first.")
    
    def increase_crop_zoom(self):
        """Increase crop zoom using Shift+Ctrl+Plus"""
        crop_names = list(self.crop_sizes.keys())
        current_index = crop_names.index(self.current_crop_name)
        
        if current_index < len(crop_names) - 1:
            new_crop_name = crop_names[current_index + 1]
            self.set_crop_size(new_crop_name)
            print(f"Increased crop zoom to {new_crop_name}")
        else:
            print("Already at maximum crop zoom")
    
    def decrease_crop_zoom(self):
        """Decrease crop zoom using Shift+Ctrl+Minus"""
        crop_names = list(self.crop_sizes.keys())
        current_index = crop_names.index(self.current_crop_name)
        
        if current_index > 0:
            new_crop_name = crop_names[current_index - 1]
            self.set_crop_size(new_crop_name)
            print(f"Decreased crop zoom to {new_crop_name}")
        else:
            print("Already at minimum crop zoom")
    
    def set_crop_size(self, crop_name):
        """Set crop size"""
        if crop_name == self.current_crop_name:
            return  # Already this crop size
            
        old_crop = self.current_crop_name
        self.current_crop_name = crop_name
        self.crop_size = self.crop_sizes[crop_name]
        
        print(f"Changing crop from {old_crop} to {crop_name}")
        
        # Update menu checkmarks if they exist
        if hasattr(self, 'crop_actions'):
            for action in self.crop_actions:
                action.setChecked(action.text().replace('&', '') == crop_name)
        
        # Reload all images with new crop size
        if self.model.rowCount() > 0:
            self.reload_images_with_new_crop()
        
        # Update the info label
        self.update_class_info_label()
    
    def reload_images_with_new_crop(self):
        """Reload all images with new crop size"""
        if self.model:
            # Stop background loading
            self.stop_background_loading()
            
            # Clear processed thumbnail cache (but keep raw image cache)
            self.image_cache.clear()
            # Remove processed thumbnails from access order, keep raw images
            self.cache_access_order = [key for key in self.cache_access_order if isinstance(key, str)]
            
            # Clear current icons but preserve item structure
            for row in range(self.model.rowCount()):
                item = self.model.item(row)
                if item:
                    item.setIcon(self.placeholder_icon)  # Reset to placeholder
            
            # Clear loaded images set so everything gets reloaded with new crop
            self.loaded_images.clear()
            
            # Stop any existing workers gracefully
            for worker in self.active_workers:
                worker.stop()
            
            # Clear the workers list
            self.active_workers.clear()
            
            # Clear the thread pool (this will wait for current tasks to finish)
            self.thread_pool.clear()
            
            # Reload visible images with new crop size
            self.load_visible_images()
    
    def update_placeholder_icon(self):
        """Update placeholder icon with current icon size"""
        placeholder_pixmap = QPixmap(self.icon_size, self.icon_size)
        placeholder_pixmap.fill(Qt.lightGray)
        self.placeholder_icon = QIcon(placeholder_pixmap)
    
    def get_cache_key(self, filename, icon_size, crop_size):
        """Generate cache key for processed thumbnail"""
        return (filename, icon_size, crop_size)
    
    def get_cached_icon(self, filename, icon_size, crop_size):
        """Get cached processed icon if available"""
        cache_key = self.get_cache_key(filename, icon_size, crop_size)
        if cache_key in self.image_cache:
            # Update access order (move to end)
            if cache_key in self.cache_access_order:
                self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)
            return self.image_cache[cache_key]
        return None
    
    def cache_icon(self, filename, icon_size, crop_size, icon):
        """Cache a processed icon with LRU eviction"""
        cache_key = self.get_cache_key(filename, icon_size, crop_size)
        
        # Add to cache
        self.image_cache[cache_key] = icon
        
        # Update access order
        if cache_key in self.cache_access_order:
            self.cache_access_order.remove(cache_key)
        self.cache_access_order.append(cache_key)
        
        # Evict old entries if cache is too large
        while len(self.image_cache) > self.max_cache_size:
            oldest_key = self.cache_access_order.pop(0)
            if oldest_key in self.image_cache:
                del self.image_cache[oldest_key]
    
    def get_cached_raw_image(self, filename):
        """Get cached raw QPixmap if available"""
        if filename in self.raw_image_cache:
            # Update access order for raw images too
            if filename in self.cache_access_order:
                self.cache_access_order.remove(filename)
            self.cache_access_order.append(filename)
            return self.raw_image_cache[filename]
        return None
    
    def cache_raw_image(self, filename, pixmap):
        """Cache a raw QPixmap with size limits"""
        # Only cache if image is reasonably sized (< 2K resolution)
        if pixmap.width() <= 2048 and pixmap.height() <= 2048:
            self.raw_image_cache[filename] = pixmap
            
            # Update access order
            if filename in self.cache_access_order:
                self.cache_access_order.remove(filename)
            self.cache_access_order.append(filename)
            
            # Evict old raw images if we have too many
            max_raw_cache = 50  # Keep fewer raw images due to memory usage
            while len(self.raw_image_cache) > max_raw_cache:
                # Find oldest raw image in access order
                for old_filename in list(self.cache_access_order):
                    if old_filename in self.raw_image_cache:
                        del self.raw_image_cache[old_filename]
                        self.cache_access_order.remove(old_filename)
                        break
    
    def clear_caches(self):
        """Clear all caches"""
        self.image_cache.clear()
        self.raw_image_cache.clear()
        self.cache_access_order.clear()
    
    def update_window_title(self):
        """Update window title to show unsaved changes"""
        base_title = "SymSorter - Image Classification Tool"
        if self.npz_file_path:
            base_title += f" - {self.npz_file_path.name}"
        
        if self.has_unsaved_changes:
            base_title += " *"
        
        self.setWindowTitle(base_title)

    # ... (rest of the methods would continue with the same adaptations)
    # For brevity, I'll include the key methods that need adaptation


    def on_thumbnail_size_changed(self, size_name):
        """Handle thumbnail size changes"""
        if size_name not in self.icon_sizes:
            return
            
        old_size = self.icon_size
        self.current_size_name = size_name
        self.icon_size = self.icon_sizes[size_name]
        
        print(f"Changing thumbnail size from {old_size}px to {self.icon_size}px ({size_name})")
        
        # Update placeholder icon with new size
        self.update_placeholder_icon()
        
        # Update grid and icon sizes for the view
        if self.model.rowCount() > 0:
            padding = 5
            grid_size = self.icon_size + padding
            self.view.setGridSize(QSize(grid_size, grid_size))
            self.view.setIconSize(QSize(self.icon_size, self.icon_size))
            
            # Reload all images with new thumbnail size
            self.reload_images_with_new_size()
        
        # Update the info label to show current size
        self.update_class_info_label()

    def reload_images_with_new_size(self):
        """Reload all images with new thumbnail size"""
        if self.model:
            # Stop background loading
            self.stop_background_loading()
            
            # Clear processed thumbnail cache (but keep raw image cache)
            self.image_cache.clear()
            # Remove processed thumbnails from access order, keep raw images
            self.cache_access_order = [key for key in self.cache_access_order if isinstance(key, str)]
            
            # Clear current icons but preserve item structure
            for row in range(self.model.rowCount()):
                item = self.model.item(row)
                if item:
                    item.setIcon(self.placeholder_icon)  # Reset to placeholder with new size
            
            # Clear loaded images set so everything gets reloaded with new size
            self.loaded_images.clear()
            
            # Stop any existing workers gracefully
            for worker in self.active_workers:
                worker.stop()
            
            # Clear the workers list
            self.active_workers.clear()
            
            # Clear the thread pool (this will wait for current tasks to finish)
            self.thread_pool.clear()
            
            # Reload visible images with new thumbnail size
            self.load_visible_images()

    def load_class_file_dialog(self):
        """Open dialog to load class file"""
        class_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Class File",
            "",
            "Text files (*.txt);;Names files (*.names);;All files (*)"
        )
        if class_file:
            self.load_class_file(class_file)
    
    def load_class_file(self, class_file_path):
        """Load YOLO class names from file"""
        try:
            if isinstance(class_file_path, str):
                class_file_path = Path(class_file_path)
            
            if not class_file_path.exists():
                print(f"Class file not found: {class_file_path}")
                return
            
            with open(class_file_path, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f.readlines() if line.strip()]
            
            self.class_file_path = class_file_path
            print(f"Loaded {len(self.classes)} classes from {class_file_path}")
            
            # Update UI
            self.update_class_info_label()
            
        except Exception as e:
            print(f"Error loading class file: {e}")
    
    def update_class_info_label(self):
        """Update the class info label"""
        # Cache statistics
        cache_text = f"Cache: {len(self.image_cache)} thumbnails, {len(self.raw_image_cache)} raw images"
        
        if self.classes:
            class_text = f"Classes loaded: {len(self.classes)} (Shift+F1-F{min(len(self.classes), 12)}, Enter=repeat last)"
            size_text = f"Thumbnail size: {self.current_size_name} (Ctrl+/-)"
            crop_text = f"Crop: {self.current_crop_name} (Shift+Ctrl+/-)"
            if len(self.classes) > 12:
                class_text += f" [showing first 12 of {len(self.classes)}]"
            self.class_info_label.setText(f"{class_text}\n{size_text} | {crop_text} | {cache_text}")
            
            # Update combobox with class names
            self.update_class_filter_combo()
            
            # Update classes menu
            self.update_classes_menu()
        else:
            size_text = f"Thumbnail size: {self.current_size_name} (Ctrl+/-)"
            crop_text = f"Crop: {self.current_crop_name} (Shift+Ctrl+/-)"
            self.class_info_label.setText(f"No classes loaded\n{size_text} | {crop_text} | {cache_text}")
    
    def update_class_filter_combo(self):
        """Update the class filter combobox with loaded classes"""
        # Store current selection
        current_text = self.class_filter_combo.currentText()
        
        # Clear and repopulate
        self.class_filter_combo.clear()
        self.class_filter_combo.addItem("All Images")
        self.class_filter_combo.addItem("Unallocated")
        
        # Add each class
        for class_name in self.classes:
            self.class_filter_combo.addItem(class_name)
        
        # Restore selection if it still exists
        index = self.class_filter_combo.findText(current_text)
        if index >= 0:
            self.class_filter_combo.setCurrentIndex(index)
    
    def update_classes_menu(self):
        """Update the classes menu with loaded classes and their shortcuts"""
        # Clear existing actions
        self.classes_menu.clear()
        
        if not self.classes:
            # Add disabled placeholder
            no_classes_action = QAction('No classes loaded', self)
            no_classes_action.setEnabled(False)
            self.classes_menu.addAction(no_classes_action)
            self.classes_menu.setEnabled(False)
            return
        
        # Enable the menu
        self.classes_menu.setEnabled(True)
        
        # Add action for each class (up to 12 classes with Shift+F-key shortcuts)
        for i, class_name in enumerate(self.classes):
            if i < 12:  # Only first 12 classes get function key shortcuts
                f_key = f"Shift+F{i+1}"
                action_text = f"&{class_name}"
                action = QAction(action_text, self)
                action.setShortcut(QKeySequence(f_key))
                action.setStatusTip(f'Assign selected images to class "{class_name}" ({f_key})')
                
                # Fix closure issue by using proper closure
                action.triggered.connect(self.create_class_assignment_handler(i))
                
                self.classes_menu.addAction(action)
                print(f"DEBUG: Created menu action with shortcut {f_key} for class '{class_name}' (index {i})")
            else:
                # Classes beyond F12 don't get shortcuts but are still in the menu
                action_text = f"{class_name}"
                action = QAction(action_text, self)
                action.setStatusTip(f'Assign selected images to class "{class_name}" (no shortcut)')
                
                # Fix closure issue by using proper closure
                action.triggered.connect(self.create_class_assignment_handler(i))
                
                self.classes_menu.addAction(action)
        
        # Add separator and Enter key action
        self.classes_menu.addSeparator()
        
        # Add Enter key action
        enter_action = QAction('Assign to &Last Used Class', self)
        enter_action.setShortcut(QKeySequence('Return'))
        enter_action.setStatusTip('Assign selected images to last used class (Enter)')
        enter_action.triggered.connect(self.assign_to_last_used_class)
        self.classes_menu.addAction(enter_action)
        
        # Add info action for classes beyond F12
        if len(self.classes) > 12:
            info_action = QAction(f'Classes {13}-{len(self.classes)} have no shortcuts', self)
            info_action.setEnabled(False)
            self.classes_menu.addAction(info_action)
    
    def on_class_filter_changed(self, filter_text):
        """Handle class filter combobox changes"""
        print(f"Filter changed to: {filter_text}")
        self.apply_class_filter(filter_text)
    
    def apply_class_filter(self, filter_text):
        """Apply class filter to the image list"""
        if not self.embeddings:
            return
        
        folder_path = Path(self.folder)
        
        # Determine which images should be visible based on filter
        if filter_text == "All Images":
            # Show all images (including classified ones) but exclude hidden
            visible_files = [filename for filename in self.embeddings.keys() 
                            if (folder_path / filename).exists() 
                            and not self.hidden_flags.get(filename, False)]
        elif filter_text == "Unallocated":
            # Show only unclassified images
            visible_files = [filename for filename in self.embeddings.keys() 
                            if (folder_path / filename).exists() 
                            and not self.hidden_flags.get(filename, False)
                            and filename not in self.image_categories]
        else:
            # Show only images with specific class
            visible_files = [filename for filename in self.embeddings.keys() 
                            if (folder_path / filename).exists() 
                            and not self.hidden_flags.get(filename, False)
                            and filename in self.image_categories
                            and self.image_categories[filename]['class_name'] == filter_text]
        
        # Check if we need to add new images that aren't in current model
        current_files_set = set(self.image_files)
        visible_files_set = set(visible_files)
        
        # If the visible set is completely different, do a full rebuild
        if len(visible_files_set - current_files_set) > len(visible_files_set) * 0.5:
            self.image_files = visible_files
            self.rebuild_model_after_filter()
        else:
            # Just update the current list efficiently
            self.image_files = visible_files
            self.apply_filter_to_existing_model(visible_files_set)
        
        print(f"Filtered to {len(self.image_files)} images for '{filter_text}'")
    

    def assign_class_to_selected(self, class_idx):
        """Assign class to selected images using function keys"""
        print(f"DEBUG: assign_class_to_selected called with class_idx={class_idx}")
        print(f"DEBUG: Have {len(self.classes)} classes loaded")
        print(f"DEBUG: Selected indexes count: {len(self.view.selectedIndexes())}")
        
        if not self.classes or class_idx >= len(self.classes):
            print(f"Class index {class_idx} not available (have {len(self.classes)} classes)")
            return
        
        indexes = self.view.selectedIndexes()
        if not indexes:
            print("No images selected")
            return
        
        # Update last used class
        self.last_used_class_idx = class_idx
        
        class_name = self.classes[class_idx]
        assigned_count = 0
        
        for index in indexes:
            filename = index.data(Qt.UserRole)
            if filename:
                self.image_categories[filename] = {
                    'class_id': class_idx,
                    'class_name': class_name
                }
                assigned_count += 1
        
        # Mark as having unsaved changes
        if assigned_count > 0:
            self.has_unsaved_changes = True
            self.update_window_title()
        
        print(f"Assigned {assigned_count} images to class {class_idx}: '{class_name}'")
        
        # Auto-hide classified images by removing them from view
        self.hide_classified_images(indexes)
        
        # Update the model items to show category (optional visual feedback)
        self.update_model_with_categories()
    
    def hide_classified_images(self, indexes):
        """Hide images that have been classified by removing them from view"""
        hidden_count = 0
        current_filter = self.class_filter_combo.currentText()
        
        # Sort indexes in reverse order to avoid index shifting issues
        for index in sorted(indexes, key=lambda x: x.row(), reverse=True):
            filename = index.data(Qt.UserRole)
            if filename and filename in self.image_categories:
                # Only remove if we're in "Unallocated" mode
                # In other modes, the image might still be relevant to show
                if current_filter == "Unallocated":
                    # Remove from current image_files list
                    if filename in self.image_files:
                        self.image_files.remove(filename)
                    # Remove from model
                    self.model.removeRow(index.row())
                    hidden_count += 1
                else:
                    # In other filter modes, just update the tooltip but don't remove
                    item = self.model.item(index.row())
                    if item:
                        category_info = self.image_categories[filename]
                        tooltip = f"{filename}\nClass: {category_info['class_name']} (ID: {category_info['class_id']})"
                        item.setToolTip(tooltip)
        
        if hidden_count > 0:
            print(f"Auto-hidden {hidden_count} classified images from view")
            # Update loaded_images set to account for removed indices
            self.loaded_images = set(i for i in self.loaded_images if i < len(self.image_files))
    
    def update_model_with_categories(self):
        """Update model items to show assigned categories"""
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            if item:
                filename = item.data(Qt.UserRole)
                if filename in self.image_categories:
                    category_info = self.image_categories[filename]
                    tooltip = f"{filename}\nClass: {category_info['class_name']} (ID: {category_info['class_id']})"
                    item.setToolTip(tooltip)
                else:
                    item.setToolTip(filename)

    def load_folder(self):
        npz_file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Embeddings File", 
            "", 
            "NPZ files (*.npz);;All files (*)"
        )
        if not npz_file:
            return
        
        self.load_folder_from_path(npz_file)
    
    def load_folder_from_path(self, npz_file):
        """Load folder from NPZ file path (used by both dialog and command line)"""
        npz_path = Path(npz_file)
        if not npz_path.exists():
            print(f"NPZ file not found: {npz_path}")
            return
            
        self.folder = str(npz_path.parent)
        self.npz_file_path = npz_path
        
        print(f"Loading embeddings from {npz_path}")
        self.embeddings = load_existing_embeddings(npz_path)
        
        # Load hidden flags and categories from npz file if they exist
        self.hidden_flags = {}
        self.image_categories = {}
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'hidden_flags' in data:
                # Convert numpy array back to dict
                hidden_array = data['hidden_flags'].item()
                if isinstance(hidden_array, dict):
                    self.hidden_flags = hidden_array
                    print(f"Loaded {len(self.hidden_flags)} hidden flags")
            
            if 'categories' in data:
                # Load category assignments
                categories_array = data['categories'].item()
                if isinstance(categories_array, dict):
                    self.image_categories = categories_array
                    print(f"Loaded {len(self.image_categories)} category assignments")
        except Exception as e:
            print(f"No additional data found or error loading: {e}")
        
        # Clear unsaved changes flag when loading new data
        self.has_unsaved_changes = False
        self.update_window_title()
        
        if not self.embeddings:
            print("No embeddings loaded!")
            return
        
        self.model.clear()
        self.loaded_images.clear()
        # Stop any active workers
        for worker in self.active_workers:
            worker.stop()
        self.active_workers.clear()
        self.thread_pool.clear()
        
        # Set initial filter to "Unallocated" to maintain original behavior
        self.class_filter_combo.setCurrentText("Unallocated")
        
        # Get image files that have embeddings, are not hidden, and are not classified
        folder_path = Path(self.folder)
        self.image_files = [filename for filename in self.embeddings.keys() 
                           if (folder_path / filename).exists() 
                           and not self.hidden_flags.get(filename, False)
                           and filename not in self.image_categories]
        self.original_order = self.image_files.copy()
        
        print(f"Found {len(self.image_files)} images with embeddings")
        
        # Create placeholder items for all images
        for file in self.image_files:
            item = QStandardItem()
            item.setIcon(self.placeholder_icon)
            item.setEditable(False)
            item.setData(file, Qt.UserRole)  # Store filename in user data
            self.model.appendRow(item)
        
        print(f"Added {self.model.rowCount()} items to model")
        
        # Set initial grid and icon sizes (fixed size, doesn't change with zoom)
        padding = 5
        grid_size = self.icon_size + padding
        self.view.setGridSize(QSize(grid_size, grid_size))
        self.view.setIconSize(QSize(self.icon_size, self.icon_size))
        
        # Load first batch immediately
        if self.image_files:
            # Load first 30 images immediately (larger batch for initial load)
            initial_batch_size = min(30, len(self.image_files))
            print(f"Loading initial batch of {initial_batch_size} images")
            self.load_image_batch(0, initial_batch_size)
            
            # Update model with category information
            self.update_model_with_categories()
            
            # Also try to load visible images after a short delay
            QTimer.singleShot(100, self.load_visible_images)

    # Essential missing methods for core functionality
    def calculate_visible_range(self):
        """Calculate which images should be visible on screen"""
        if not self.image_files:
            return 0, 0
            
        # Calculate grid dimensions
        view_width = self.view.viewport().width()
        view_height = self.view.viewport().height()
        
        # If view dimensions are too small (initial state), use default values
        if view_width < 100 or view_height < 100:
            view_width = 800  # Default window width
            view_height = 600  # Default window height
        
        grid_width = self.view.gridSize().width()
        grid_height = self.view.gridSize().height()
        
        cols = max(1, view_width // grid_width)
        rows_visible = max(1, view_height // grid_height) + 2  # Add buffer
        
        # Get scroll position
        scroll_value = self.view.verticalScrollBar().value()
        max_scroll = self.view.verticalScrollBar().maximum()
        
        if max_scroll > 0:
            scroll_ratio = scroll_value / max_scroll
        else:
            scroll_ratio = 0
            
        total_rows = math.ceil(len(self.image_files) / cols)
        start_row = int(scroll_ratio * (total_rows - rows_visible))
        start_row = max(0, start_row)
        
        start_idx = start_row * cols
        end_idx = min(start_idx + (rows_visible * cols), len(self.image_files))
        
        return start_idx, end_idx
    
    def load_image_batch(self, start_idx, end_idx):
        """Load a batch of images using thread pool"""
        print(f"load_image_batch called with range {start_idx} to {end_idx}")
        
        # Filter out already loaded images
        images_to_load = []
        for i in range(start_idx, end_idx):
            if i not in self.loaded_images and i < len(self.image_files):
                images_to_load.append(i)
        
        if not images_to_load:
            print(f"No new images to load in range {start_idx}-{end_idx}")
            return  # Nothing new to load
            
        print(f"Loading {len(images_to_load)} new images: indices {images_to_load}")
        
        # Mark these images as being loaded
        for i in images_to_load:
            self.loaded_images.add(i)
        
        # Create individual workers for each image and submit to thread pool
        for i in images_to_load:
            if i < len(self.image_files):
                file_path = os.path.join(self.folder, self.image_files[i])
                worker = ImageLoadWorker(i, file_path, self.icon_size, self.crop_size, cache_manager=self)
                # Connect signals
                worker.signals.imageLoaded.connect(self.update_image)
                worker.signals.error.connect(self.handle_worker_error)
                self.active_workers.append(worker)
                self.thread_pool.start(worker)
    
    def handle_worker_error(self, error_msg):
        """Handle errors from worker threads"""
        print(error_msg)
    
    def load_visible_images(self):
        """Load images that are currently visible"""
        start_idx, end_idx = self.calculate_visible_range()
        
        if start_idx < end_idx:
            self.load_image_batch(start_idx, end_idx)
            
        # Start background loading after visible images
        self.start_background_loading()
    
    def start_background_loading(self):
        """Start background loading of remaining images"""
        if not self.background_load_timer.isActive():
            self.background_load_timer.start(500)  # Load batch every 500ms
    
    def stop_background_loading(self):
        """Stop background loading"""
        self.background_load_timer.stop()
    
    def load_next_batch_background(self):
        """Load next batch of images in background"""
        if not self.image_files:
            self.stop_background_loading()
            return
            
        # Find the next unloaded batch
        batch_size = 10  # Smaller batches for background loading
        start_idx = None
        
        for i in range(0, len(self.image_files), batch_size):
            # Check if any images in this batch are unloaded
            batch_needs_loading = False
            for j in range(i, min(i + batch_size, len(self.image_files))):
                if j not in self.loaded_images:
                    batch_needs_loading = True
                    break
            
            if batch_needs_loading:
                start_idx = i
                break
        
        if start_idx is not None:
            end_idx = min(start_idx + batch_size, len(self.image_files))
            self.load_image_batch(start_idx, end_idx)
        else:
            # All images loaded, stop background loading
            self.stop_background_loading()
            print("Background loading complete - all images loaded")
    
    def update_image(self, index, icon):
        """Update the model with the loaded image"""
        if index >= self.model.rowCount():
            # This can happen during filtering or model rebuilds - just ignore
            return
            
        item = self.model.item(index)
        if not item:
            return
        
        # Always update the icon - Qt's model-view handles visibility efficiently
        # and we need icons ready for when items become visible through scrolling
        item.setIcon(icon)
    
    def on_scroll(self):
        """Handle scroll events for lazy loading"""
        self.load_visible_images()
    
    def on_view_resize(self, event):
        """Handle view resize events to load more images when window grows"""
        # Call the original resize event first
        QListView.resizeEvent(self.view, event)
        
        # Load visible images after a short delay to avoid excessive calls during resize
        if hasattr(self, '_resize_timer'):
            self._resize_timer.stop()
        
        self._resize_timer = QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self.load_visible_images)
        self._resize_timer.start(100)  # Wait 100ms after resize finishes
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def on_double_click(self, index):
        """Handle double-click on an image to sort by similarity"""
        if not self.embeddings:
            return
            
        # Get the filename of the clicked image
        filename = index.data(Qt.UserRole)
        if filename not in self.embeddings:
            return
            
        print(f"Sorting by similarity to: {filename}")
        
        # Get the embedding of the clicked image
        reference_embedding = self.embeddings[filename]
        
        # Calculate similarities to all images
        similarities = []
        for img_file in self.image_files:
            if img_file in self.embeddings:
                sim = self.cosine_similarity(reference_embedding, self.embeddings[img_file])
                similarities.append((img_file, sim))
        
        # Sort by similarity (descending - most similar first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Reorder the image files list
        self.image_files = [item[0] for item in similarities]
        
        # Clear and rebuild the model with new order
        self.rebuild_model()
    
    def rebuild_model(self):
        """Rebuild the model with current image order, preserving loaded icons"""
        # Store existing icons before clearing
        existing_icons = {}
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            if item:
                filename = item.data(Qt.UserRole)
                icon = item.icon()
                # Only store non-placeholder icons
                if icon.pixmap(100, 100).toImage() != self.placeholder_icon.pixmap(100, 100).toImage():
                    existing_icons[filename] = icon
        
        self.model.clear()
        # Don't clear loaded_images - we want to keep track of what's already loaded
        
        # Create items for all images in new order, using existing icons where available
        for file in self.image_files:
            item = QStandardItem()
            if file in existing_icons:
                item.setIcon(existing_icons[file])
            else:
                item.setIcon(self.placeholder_icon)
            item.setEditable(False)
            item.setData(file, Qt.UserRole)
            self.model.appendRow(item)
        
        # Update loaded_images set to reflect new order
        new_loaded_images = set()
        for i, filename in enumerate(self.image_files):
            if filename in existing_icons:
                new_loaded_images.add(i)
        self.loaded_images = new_loaded_images
        
        # Load any visible images that aren't already loaded
        self.load_visible_images()
    
    def reset_order(self):
        """Reset images to original order"""
        if not self.original_order:
            return
            
        self.image_files = self.original_order.copy()
        self.rebuild_model()

    def show_classified_images(self):
        """Show all classified images by temporarily including them in the view"""
        if not self.image_categories:
            print("No classified images to show")
            return
            
        # Count currently classified images
        classified_count = len(self.image_categories)
        
        if classified_count == 0:
            print("No classified images to show")
            return
        
        # Temporarily clear image categories to show them
        temp_categories = self.image_categories.copy()
        self.image_categories.clear()
        
        # Reload the folder to show all images including classified ones
        self.reload_current_folder()
        
        # Restore the categories but don't auto-hide them this time
        self.image_categories = temp_categories
        
        # Update tooltips to show classifications
        self.update_model_with_categories()
        
        print(f"Showed {classified_count} classified images (classifications preserved)")
    
    def reload_current_folder(self):
        """Reload the current folder with updated hidden flags"""
        if not self.npz_file_path:
            return
            
        # Get all image files that have embeddings, are not hidden, and are not classified
        folder_path = Path(self.folder)
        self.image_files = [filename for filename in self.embeddings.keys() 
                           if (folder_path / filename).exists() 
                           and not self.hidden_flags.get(filename, False)
                           and filename not in self.image_categories]
        self.original_order = self.image_files.copy()
        
        # Clear and rebuild the model
        self.model.clear()  
        self.loaded_images.clear()
        
        # Create placeholder items for all visible images
        for file in self.image_files:
            item = QStandardItem()
            item.setIcon(self.placeholder_icon)
            item.setEditable(False)
            item.setData(file, Qt.UserRole)
            self.model.appendRow(item)
        
        print(f"Reloaded: Found {len(self.image_files)} visible images")
        
        # Load first batch immediately
        if self.image_files:
            self.load_image_batch(0, min(20, len(self.image_files)))
    
    def save_hidden_flags(self):
        """Save hidden flags and categories back to the npz file"""
        if not self.npz_file_path:
            print("No npz file loaded")
            return
            
        try:
            # Load existing data
            existing_data = dict(np.load(self.npz_file_path, allow_pickle=True))
            
            # Add or update hidden flags and categories
            existing_data['hidden_flags'] = self.hidden_flags
            existing_data['categories'] = self.image_categories
            
            # Save back to file
            np.savez_compressed(self.npz_file_path, **existing_data)
            
            hidden_count = sum(1 for hidden in self.hidden_flags.values() if hidden)
            category_count = len(self.image_categories)
            print(f"Saved to {self.npz_file_path}:")
            print(f"  - {len(self.hidden_flags)} hidden flags ({hidden_count} hidden)")
            print(f"  - {category_count} category assignments")
            
            # Clear unsaved changes flag
            self.has_unsaved_changes = False
            self.update_window_title()
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def export_yolo_annotations(self):
        """Export category assignments as YOLO format annotations"""
        if not self.image_categories or not self.classes:
            print("No categories assigned or classes loaded")
            return
        
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        
        export_path = Path(export_dir)
        
        # Create labels directory
        labels_dir = export_path / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        # Export each categorized image
        exported_count = 0
        for filename, category_info in self.image_categories.items():
            # Create corresponding txt file for YOLO annotations
            txt_filename = Path(filename).stem + ".txt"
            txt_path = labels_dir / txt_filename
            
            # For now, create a simple classification annotation
            # In a full implementation, you'd need bounding box coordinates
            class_id = category_info['class_id']
            
            # Write minimal YOLO format (class_id x_center y_center width height)
            # Using full image (0.5, 0.5, 1.0, 1.0) as placeholder
            with open(txt_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            
            exported_count += 1
        
        # Also export classes file
        classes_file = export_path / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
        
        print(f"Exported {exported_count} YOLO annotations to {export_path}")
        print(f"Classes file saved to {classes_file}")

    # Placeholder methods for missing functionality
    def apply_filter_to_existing_model(self, visible_files_set):
        """Efficiently apply filter by hiding/showing existing model items"""
        # Simplified implementation - rebuild for now
        self.rebuild_model_after_filter()
    
    def rebuild_model_after_filter(self):
        """Rebuild model after applying filter (used for major changes)"""
        # Stop background loading
        self.stop_background_loading()
        
        # Clear everything
        self.model.clear()
        self.loaded_images.clear()
        
        # Stop any active workers
        for worker in self.active_workers:
            worker.stop()
        self.active_workers.clear()
        self.thread_pool.clear()
        
        # Create placeholder items for filtered images
        for file in self.image_files:
            item = QStandardItem()
            item.setIcon(self.placeholder_icon)
            item.setEditable(False)
            item.setData(file, Qt.UserRole)
            self.model.appendRow(item)
        
        # Update model with category information
        self.update_model_with_categories()
        
        # Load first batch immediately
        if self.image_files:
            self.load_image_batch(0, min(20, len(self.image_files)))
            QTimer.singleShot(100, self.load_visible_images)

    def resizeEvent(self, event):
        """Handle main window resize events"""
        super().resizeEvent(event)
        
        # Load visible images after a short delay to avoid excessive calls during resize
        if hasattr(self, '_main_resize_timer'):
            self._main_resize_timer.stop()
        
        self._main_resize_timer = QTimer()
        self._main_resize_timer.setSingleShot(True)
        self._main_resize_timer.timeout.connect(self.load_visible_images)
        self._main_resize_timer.start(150)  # Wait 150ms after resize finishes
    
    def keyPressEvent(self, event):
        """Override to handle custom key press events"""
        key_code = event.key()
        modifiers = event.modifiers()
        
        # Handle Ctrl+Plus for increasing thumbnail size
        if modifiers & Qt.ControlModifier and key_code in (Qt.Key_Plus, Qt.Key_Equal):
            self.increase_thumbnail_size()
            event.accept()
            return
            
        # Handle Ctrl+Minus for decreasing thumbnail size
        if modifiers & Qt.ControlModifier and key_code == Qt.Key_Minus:
            self.decrease_thumbnail_size()
            event.accept()
            return
            
        # Handle Shift+Ctrl+Plus for increasing crop zoom
        if modifiers & (Qt.ShiftModifier | Qt.ControlModifier) and key_code in (Qt.Key_Plus, Qt.Key_Equal):
            self.increase_crop_zoom()
            event.accept()
            return
            
        # Handle Shift+Ctrl+Minus for decreasing crop zoom
        if modifiers & (Qt.ShiftModifier | Qt.ControlModifier) and key_code == Qt.Key_Minus:
            self.decrease_crop_zoom()
            event.accept()
            return
            
        # Handle Enter key to assign to last used class
        if key_code in (Qt.Key_Return, Qt.Key_Enter):
            if self.last_used_class_idx is not None:
                self.assign_class_to_selected(self.last_used_class_idx)
            else:
                print("No class has been used yet. Use Shift+F1-F12 to assign a class first.")
            event.accept()
            return
        
        # Call parent implementation
        super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        # Check for unsaved changes
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self,
                'Unsaved Changes',
                'You have unsaved classifications and hidden images.\n'
                'Do you want to save your work before exiting?\n\n'
                'Click "Save" to save your work,\n'
                '"Discard" to exit without saving,\n'
                'or "Cancel" to continue working.',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save  # Default button
            )
            
            if reply == QMessageBox.Save:
                # Save the work
                self.save_hidden_flags()
                # If save was successful (no exception), continue with closing
            elif reply == QMessageBox.Cancel:
                # Cancel the close operation
                event.ignore()
                return
            # If Discard, just continue with closing
        
        # Stop background loading
        self.stop_background_loading()
        
        # Stop all workers
        for worker in self.active_workers:
            worker.stop()
        
        # Wait for thread pool to finish (with timeout)
        if not self.thread_pool.waitForDone(3000):  # 3 second timeout
            print("Warning: Some image loading threads did not finish cleanly")
        
        # Clear caches to free memory
        self.clear_caches()
        
        event.accept()


def main():
    """Main function with command line argument parsing"""
    import argparse
    parser = argparse.ArgumentParser(description='SymSorter - Image Browser with CLIP embeddings and YOLO class assignment')
    parser.add_argument('--classes', '-c', type=str, help='Path to YOLO classes file (.txt or .names)')
    parser.add_argument('--embeddings', '-e', type=str, help='Path to embeddings NPZ file')
    
    args = parser.parse_args()
    
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = ImageBrowser(class_file=args.classes)
    
    # Auto-load embeddings if provided
    if args.embeddings:
        window.npz_file_path = Path(args.embeddings)
        window.folder = str(window.npz_file_path.parent)
        QTimer.singleShot(100, lambda: window.load_folder_from_path(args.embeddings))
    
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
