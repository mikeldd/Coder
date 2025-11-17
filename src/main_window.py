"""
Main application window for the Medical CPT Coding Application.
Provides tabbed interface for upload, processing, and results viewing.
"""

import sys
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QMenuBar, QStatusBar, QFileDialog, QMessageBox, QSplitter,
    QApplication, QToolBar, QLabel, QPushButton, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QKeySequence

from .components.pdf_uploader import PDFUploader
from .components.results_display import ResultsDisplay
from .components.processing_status import ProcessingStatus
from .core.database import Database
from .core.file_manager import FileManager
from .utils.config import Config

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""

    # Signals
    files_selected = pyqtSignal(list)
    processing_started = pyqtSignal(str)
    processing_completed = pyqtSignal(str)

    def __init__(self, config: Config):
        """Initialize main window."""
        super().__init__()
        self.config = config
        self.db = Database(config)
        self.file_manager = FileManager(config)

        # Set window properties
        self.setWindowTitle("Medical CPT Coder - HIPAA Compliant")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)

        # Initialize UI components
        self.init_ui()
        self.init_menu_bar()
        self.init_toolbar()
        self.init_status_bar()

        # Connect signals
        self.connect_signals()

        # Auto-delete timer
        self.auto_delete_timer = QTimer()
        self.auto_delete_timer.timeout.connect(self.check_auto_delete)
        self.auto_delete_timer.start(3600000)  # Check every hour

        logger.info("Main window initialized successfully")

    def init_ui(self):
        """Initialize the main UI layout."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create tabbed interface
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Upload tab
        self.uploader = PDFUploader(self.config)
        self.tabs.addTab(self.uploader, "📁 Upload & Process")

        # Results tab
        self.results_display = ResultsDisplay(self.config, self.db)
        self.tabs.addTab(self.results_display, "📊 Results")

        # Processing status tab
        self.processing_status = ProcessingStatus(self.config)
        self.tabs.addTab(self.processing_status, "⚙️ Processing Status")

        # Add tabs to layout
        main_layout.addWidget(self.tabs)

        # Statistics bar (below tabs)
        stats_widget = self.create_stats_widget()
        main_layout.addWidget(stats_widget)

    def create_stats_widget(self):
        """Create statistics display widget."""
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 5, 0, 0)

        # Processing statistics
        self.stats_label = QLabel("Ready to process medical records")
        stats_layout.addWidget(self.stats_label)

        stats_layout.addStretch()

        # Theme toggle button
        self.theme_button = QPushButton("🌙 Dark Mode")
        self.theme_button.clicked.connect(self.toggle_theme)
        stats_layout.addWidget(self.theme_button)

        return stats_widget

    def init_menu_bar(self):
        """Create menu bar with file operations and settings."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Open files action
        open_action = QAction("📂 Open Medical Records...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_files)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        # Export results action
        export_action = QAction("💾 Export Results...", self)
        export_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("🚪 Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        # Clear temp files action
        clear_temp_action = QAction("🗑️ Clear Temporary Files", self)
        clear_temp_action.triggered.connect(self.clear_temp_files)
        edit_menu.addAction(clear_temp_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")

        # Configure processing action
        config_action = QAction("⚙️ Processing Settings", self)
        config_action.triggered.connect(self.show_settings)
        settings_menu.addAction(config_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        # About action
        about_action = QAction("ℹ️ About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def init_toolbar(self):
        """Create main toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Open files button
        open_btn = QPushButton("📂 Open Files")
        open_btn.clicked.connect(self.open_files)
        toolbar.addWidget(open_btn)

        toolbar.addSeparator()

        # Process all button
        process_btn = QPushButton("🚀 Process All")
        process_btn.clicked.connect(self.process_all_files)
        toolbar.addWidget(process_btn)

        toolbar.addSeparator()

        # Export results button
        export_btn = QPushButton("💾 Export")
        export_btn.clicked.connect(self.export_results)
        toolbar.addWidget(export_btn)

        # Progress bar
        toolbar.addSeparator()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        toolbar.addWidget(self.progress_bar)

    def init_status_bar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - HIPAA Compliant Processing")

    def connect_signals(self):
        """Connect signals between components."""
        # File upload signals
        self.uploader.files_selected.connect(self.on_files_selected)
        self.uploader.process_files.connect(self.on_process_files)

        # Processing status signals
        self.processing_status.status_updated.connect(self.update_status)
        self.processing_status.processing_completed.connect(self.on_processing_completed)

        # Results display signals
        self.results_display.export_requested.connect(self.export_results)

    def open_files(self):
        """Open file dialog to select medical record PDFs."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Medical Record PDFs (*.pdf)")
        file_dialog.setViewMode(QFileDialog.ViewMode.List)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            self.on_files_selected(selected_files)

    def on_files_selected(self, files):
        """Handle file selection."""
        if files:
            self.files_selected.emit(files)
            self.status_bar.showMessage(f"Selected {len(files)} medical record(s)")
            logger.info(f"User selected {len(files)} files for processing")

    def on_process_files(self, files):
        """Handle file processing request."""
        if files:
            self.processing_started.emit("batch")
            self.tabs.setCurrentIndex(2)  # Switch to processing status tab
            self.process_files(files)

    def process_files(self, files):
        """Process selected files."""
        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(files))
            self.progress_bar.setValue(0)

            # Start processing
            self.processing_status.process_files(files)

        except Exception as e:
            logger.error(f"Error starting file processing: {e}")
            QMessageBox.critical(self, "Processing Error",
                               f"Error starting processing: {str(e)}")
            self.progress_bar.setVisible(False)

    def process_all_files(self):
        """Process all uploaded files."""
        files = self.uploader.get_uploaded_files()
        if files:
            self.on_process_files(files)
        else:
            QMessageBox.information(self, "No Files",
                                  "Please upload medical records first.")

    def update_status(self, message, progress=None):
        """Update status bar and progress."""
        self.status_bar.showMessage(message)
        if progress is not None:
            self.progress_bar.setValue(progress)

    def on_processing_completed(self, job_id):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.tabs.setCurrentIndex(1)  # Switch to results tab
        self.results_display.refresh_results()
        self.status_bar.showMessage("Processing completed successfully")
        logger.info(f"Processing completed for job: {job_id}")

    def export_results(self):
        """Export processing results."""
        self.results_display.export_results()

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current_theme = self.config.get_theme()
        new_theme = "dark" if current_theme == "light" else "light"
        self.config.set_theme(new_theme)

        # Update button text
        self.theme_button.setText("☀️ Light Mode" if new_theme == "dark" else "🌙 Dark Mode")

        # Apply new theme (requires restart)
        QMessageBox.information(self, "Theme Changed",
                              f"Theme changed to {new_theme} mode.\n\nPlease restart the application for full effect.")

    def clear_temp_files(self):
        """Clear temporary files and data."""
        reply = QMessageBox.question(self, "Clear Temporary Files",
                                   "This will delete all temporary files and clear the processing queue.\n\nContinue?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.file_manager.clear_temp_files()
                self.uploader.clear_uploads()
                self.results_display.clear_results()
                self.status_bar.showMessage("Temporary files cleared")
                logger.info("User cleared temporary files")
            except Exception as e:
                logger.error(f"Error clearing temp files: {e}")
                QMessageBox.critical(self, "Error",
                                   f"Error clearing temporary files: {str(e)}")

    def check_auto_delete(self):
        """Check and perform automatic file deletion."""
        if self.config.should_auto_delete():
            try:
                deleted_count = self.file_manager.auto_delete_old_files()
                if deleted_count > 0:
                    logger.info(f"Auto-deleted {deleted_count} old files")
            except Exception as e:
                logger.error(f"Error in auto-deletion: {e}")

    def show_settings(self):
        """Show settings dialog."""
        QMessageBox.information(self, "Settings",
                              "Settings dialog will be implemented in a future version.")

    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h3>Medical CPT Coder</h3>
        <p>Version 1.0.0</p>
        <p>HIPAA-Compliant Medical Record Processing System</p>
        <p>Features:</p>
        <ul>
            <li>Automated CPT code suggestions using ensemble AI/ML models</li>
            <li>Offline processing for maximum privacy</li>
            <li>Automatic 30-day data deletion</li>
            <li>SHAP-based explanations for code recommendations</li>
        </ul>
        <p><em>Not a substitute for professional medical coding review.</em></p>
        """
        QMessageBox.about(self, "About Medical CPT Coder", about_text)

    def closeEvent(self, event):
        """Handle application close event."""
        # Stop any running processes
        try:
            self.processing_status.stop_all_processing()
        except:
            pass

        # Close database connection
        try:
            self.db.close()
        except:
            pass

        logger.info("Application closing")
        event.accept()