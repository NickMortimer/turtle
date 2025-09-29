from PySide6.QtWidgets import (
    QApplication, QWidget, QListView, QVBoxLayout,
    QPushButton, QFileDialog, QAbstractItemView
)
from PySide6.QtGui import QPixmap, QIcon, QStandardItemModel, QStandardItem
import sys
import os

class ImageBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Browser")
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        # Image list view
        self.view = QListView()
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.view)

        # Model
        self.model = QStandardItemModel()
        self.view.setModel(self.model)

        # Delete button
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected)
        layout.addWidget(self.delete_btn)

        # Load images button
        self.load_btn = QPushButton("Load Folder")
        self.load_btn.clicked.connect(self.load_folder)
        layout.addWidget(self.load_btn)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self.model.clear()
        for file in os.listdir(folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                pixmap = QPixmap(os.path.join(folder, file)).scaled(100, 100)
                item = QStandardItem(QIcon(pixmap), file)
                item.setEditable(False)
                self.model.appendRow(item)

    def delete_selected(self):
        indexes = self.view.selectedIndexes()
        for index in sorted(indexes, reverse=True):
            file_path = os.path.join(self.folder, index.data())
            os.remove(file_path)
            self.model.removeRow(index.row())

app = QApplication(sys.argv)
window = ImageBrowser()
window.show()
sys.exit(app.exec())