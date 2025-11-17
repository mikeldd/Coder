"""
PDF upload component for the Medical CPT Coding Application.
Provides drag-and-drop interface for medical record PDFs with validation and preview.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QProgressBar, QMessageBox,
    QFrame, QScrollArea, QSplitter, QTextEdit, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData, QThread, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QIcon, QFont

from ..utils.config import Config
from ..core.file_manager import FileManager

logger = logging.getLogger(__name__)

class FileValidationThread(QThread):
    """Background thread for file validation."""
    validation_completed = pyqtSignal(list)

    def __init__(self, file_paths: List[str], file_manager: FileManager):
        super().__init__()
        self.file_paths = file_paths
        self.file_manager = file_manager

    def run(self):
        """Validate files in background thread."""
        validated_files = []

        for file_path in self.file_paths:
            try:
                is_valid, validation_result = self.file_manager.validate_pdf_file(file_path)
                validated_files.append({
                    'path': file_path,
                    'valid': is_valid,
                    'validation_result': validation_result
                })
            except Exception as e:
                validated_files.append({
                    'path': file_path,
                    'valid': False,
                    'validation_result': {
                        'valid': False,
                        'errors': [str(e)],
                        'warnings': [],
                        'file_info': {}
                    }
                })

        self.validation_completed.emit(validated_files)

class PDFUploader(QWidget):
    """PDF upload component with drag-and-drop and file validation."""

    # Signals
    files_selected = pyqtSignal(list)
    process_files = pyqtSignal(list)

    def __init__(self, config: Config):
        """Initialize PDF uploader."""
        super().__init__()
        self.config = config
        self.file_manager = FileManager(config)
        self.uploaded_files = []

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title and instructions
        title_label = QLabel("📁 Upload Medical Records")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)

        instructions = QLabel("Drag and drop PDF files below or click to browse. Maximum file size: 50MB")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; margin: 5px 0;")
        layout.addWidget(instructions)

        # Create splitter for upload area and file list
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side - Upload area
        upload_widget = self.create_upload_area()
        splitter.addWidget(upload_widget)

        # Right side - File list and preview
        list_widget = self.create_file_list_area()
        splitter.addWidget(list_widget)

        splitter.setSizes([400, 600])  # Initial sizes

        # Bottom buttons
        button_widget = self.create_button_area()
        layout.addWidget(button_widget)

    def create_upload_area(self) -> QWidget:
        """Create the drag-and-drop upload area."""
        upload_widget = QWidget()
        upload_layout = QVBoxLayout(upload_widget)

        # Drag and drop area
        self.drop_zone = QFrame()
        self.drop_zone.setFrameStyle(QFrame.Shape.Box)
        self.drop_zone.setStyleSheet("""
            QFrame {
                border: 3px dashed #ccc;
                border-radius: 10px;
                background-color: #f9f9f9;
                min-height: 200px;
            }
            QFrame:hover {
                border-color: #007acc;
                background-color: #f0f8ff;
            }
        """)

        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Upload icon and text
        upload_icon = QLabel("📤")
        upload_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_icon.setStyleSheet("font-size: 48px; margin: 10px;")
        drop_layout.addWidget(upload_icon)

        drop_text = QLabel("Drop PDF files here")
        drop_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_text.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        drop_layout.addWidget(drop_text)

        subtext = QLabel("or click to browse")
        subtext.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtext.setStyleSheet("color: #666; margin: 5px 0;")
        drop_layout.addWidget(subtext)

        # Make drop zone accept drops
        self.drop_zone.setAcceptDrops(True)

        # Browse button
        browse_btn = QPushButton("📂 Browse Files")
        browse_btn.clicked.connect(self.browse_files)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        drop_layout.addWidget(browse_btn)

        upload_layout.addWidget(self.drop_zone)

        # Statistics
        self.stats_label = QLabel("Ready to receive files")
        self.stats_label.setStyleSheet("color: #666; font-style: italic;")
        upload_layout.addWidget(self.stats_label)

        return upload_widget

    def create_file_list_area(self) -> QWidget:
        """Create the file list and preview area."""
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)

        # File list header
        header_label = QLabel("📋 Uploaded Files")
        header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        list_layout.addWidget(header_label)

        # File list
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                margin: 2px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #007acc;
                color: white;
            }
        """)
        list_layout.addWidget(self.file_list)

        # File details area
        details_label = QLabel("📄 File Details")
        details_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        list_layout.addWidget(details_label)

        self.file_details = QTextEdit()
        self.file_details.setReadOnly(True)
        self.file_details.setMaximumHeight(150)
        self.file_details.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 8px;
                background-color: #f9f9f9;
            }
        """)
        list_layout.addWidget(self.file_details)

        return list_widget

    def create_button_area(self) -> QWidget:
        """Create the action buttons area."""
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)

        # Clear all button
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self.clear_uploads)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        button_layout.addWidget(clear_btn)

        button_layout.addStretch()

        # Process button
        self.process_btn = QPushButton("🚀 Process All Files")
        self.process_btn.clicked.connect(self.process_uploaded_files)
        self.process_btn.setEnabled(False)  # Disabled until files are uploaded
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        button_layout.addWidget(self.process_btn)

        return button_widget

    def connect_signals(self):
        """Connect signals and event handlers."""
        # File list selection
        self.file_list.itemSelectionChanged.connect(self.on_file_selection_changed)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            # Check if all files are PDFs
            urls = event.mimeData().urls()
            pdf_files = [url for url in urls if url.isLocalFile() and url.toLocalFile().lower().endswith('.pdf')]

            if pdf_files:
                event.acceptProposedAction()
                self.drop_zone.setStyleSheet("""
                    QFrame {
                        border: 3px dashed #007acc;
                        border-radius: 10px;
                        background-color: #e6f3ff;
                        min-height: 200px;
                    }
                """)
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.drop_zone.setStyleSheet("""
            QFrame {
                border: 3px dashed #ccc;
                border-radius: 10px;
                background-color: #f9f9f9;
                min-height: 200px;
            }
            QFrame:hover {
                border-color: #007acc;
                background-color: #f0f8ff;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        """Handle file drop event."""
        urls = event.mimeData().urls()
        file_paths = [url.toLocalFile() for url in urls if url.isLocalFile()]

        # Filter only PDF files
        pdf_files = [path for path in file_paths if path.lower().endswith('.pdf')]

        if pdf_files:
            self.handle_file_selection(pdf_files)

        # Reset drop zone style
        self.dragLeaveEvent(event)

    def browse_files(self):
        """Open file dialog to select PDF files."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Medical Record PDFs (*.pdf)")
        file_dialog.setViewMode(QFileDialog.ViewMode.List)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            self.handle_file_selection(selected_files)

    def handle_file_selection(self, file_paths: List[str]):
        """Handle selected files with validation."""
        if not file_paths:
            return

        # Show validation progress
        self.stats_label.setText(f"Validating {len(file_paths)} file(s)...")

        # Start validation in background thread
        self.validation_thread = FileValidationThread(file_paths, self.file_manager)
        self.validation_thread.validation_completed.connect(self.on_validation_completed)
        self.validation_thread.start()

    def on_validation_completed(self, validated_files: List[Dict[str, Any]]):
        """Handle completed file validation."""
        valid_files = []
        invalid_files = []

        for file_data in validated_files:
            if file_data['valid']:
                valid_files.append(file_data['path'])
            else:
                invalid_files.append({
                    'path': file_data['path'],
                    'errors': file_data['validation_result']['errors']
                })

        # Add valid files to upload list
        for file_path in valid_files:
            self.add_uploaded_file(file_path)

        # Show errors for invalid files
        if invalid_files:
            error_messages = []
            for file_data in invalid_files:
                filename = os.path.basename(file_data['path'])
                errors = ', '.join(file_data['errors'])
                error_messages.append(f"• {filename}: {errors}")

            QMessageBox.warning(self, "Invalid Files",
                              f"The following files could not be uploaded:\n\n" +
                              "\n".join(error_messages))

        # Update statistics
        self.update_stats()

        # Emit files selected signal
        if valid_files:
            self.files_selected.emit(valid_files)

    def add_uploaded_file(self, file_path: str):
        """Add a file to the uploaded files list."""
        if file_path not in self.uploaded_files:
            self.uploaded_files.append(file_path)

            # Create list item
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            size_mb = round(file_size / (1024 * 1024), 2)

            item = QListWidgetItem(f"📄 {file_name} ({size_mb} MB)")
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            self.file_list.addItem(item)

            # Enable process button
            self.process_btn.setEnabled(True)

            logger.info(f"Added file to upload queue: {file_name}")

    def on_file_selection_changed(self):
        """Handle file selection in list."""
        selected_items = self.file_list.selectedItems()
        if selected_items:
            file_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
            self.show_file_details(file_path)
        else:
            self.file_details.clear()

    def show_file_details(self, file_path: str):
        """Show details for selected file."""
        try:
            # Validate file to get details
            is_valid, validation_result = self.file_manager.validate_pdf_file(file_path)

            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            details = f"""
<b>File Information:</b><br>
Name: {file_name}<br>
Size: {round(file_size / (1024 * 1024), 2)} MB<br>
Path: {file_path}<br>
Valid: {'✅ Yes' if is_valid else '❌ No'}
"""
            if validation_result['file_info'].get('hash'):
                details += f"<br>SHA-256: {validation_result['file_info']['hash'][:16]}..."

            if validation_result['warnings']:
                details += "<br><br><b>Warnings:</b><br>"
                details += "<br>".join(f"• {w}" for w in validation_result['warnings'])

            self.file_details.setHtml(details)

        except Exception as e:
            self.file_details.setHtml(f"<b>Error:</b><br>{str(e)}")

    def update_stats(self):
        """Update upload statistics."""
        total_files = len(self.uploaded_files)
        if total_files == 0:
            self.stats_label.setText("Ready to receive files")
        else:
            total_size = sum(os.path.getsize(f) for f in self.uploaded_files)
            size_mb = round(total_size / (1024 * 1024), 1)
            self.stats_label.setText(f"✅ {total_files} file(s) uploaded ({size_mb} MB total)")

    def clear_uploads(self):
        """Clear all uploaded files."""
        if self.uploaded_files:
            reply = QMessageBox.question(self, "Clear Uploads",
                                       f"Remove all {len(self.uploaded_files)} uploaded files?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                self.uploaded_files.clear()
                self.file_list.clear()
                self.file_details.clear()
                self.process_btn.setEnabled(False)
                self.update_stats()
                logger.info("Cleared all uploaded files")

    def process_uploaded_files(self):
        """Start processing uploaded files."""
        if self.uploaded_files:
            self.process_files.emit(self.uploaded_files)

    def get_uploaded_files(self) -> List[str]:
        """Get list of uploaded file paths."""
        return self.uploaded_files.copy()

    def remove_file(self, file_path: str):
        """Remove a specific file from the upload list."""
        if file_path in self.uploaded_files:
            self.uploaded_files.remove(file_path)

            # Remove from list widget
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == file_path:
                    self.file_list.takeItem(i)
                    break

            # Update UI
            if not self.uploaded_files:
                self.process_btn.setEnabled(False)

            self.update_stats()