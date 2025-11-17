"""
Processing status monitor component for the Medical CPT Coding Application.
Provides real-time updates on CPT coding processing using QThread background processing.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QListWidget, QListWidgetItem, QPushButton, QTextEdit,
    QFrame, QScrollArea, QMessageBox, QSplitter, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QMutexLocker
from PyQt6.QtGui import QTextCursor, QFont, QColor

from ..core.processing_engine import ProcessingEngine
from ..core.file_manager import FileManager
from ..utils.config import Config

logger = logging.getLogger(__name__)

class ProcessingWorker(QThread):
    """Background worker thread for processing medical records."""

    # Signals
    status_updated = pyqtSignal(str, str)  # job_id, status_message
    progress_updated = pyqtSignal(str, int)  # job_id, progress_percentage
    completed = pyqtSignal(str, dict)  # job_id, results
    error_occurred = pyqtSignal(str, str)  # job_id, error_message

    def __init__(self, processing_engine: ProcessingEngine):
        super().__init__()
        self.processing_engine = processing_engine
        self.jobs_queue = []
        self.is_running = False
        self.mutex = QMutex()

        # Set up callbacks
        self.processing_engine.set_callbacks(
            status_callback=self.on_status_update,
            progress_callback=self.on_progress_update,
            completion_callback=self.on_completion
        )

    def add_job(self, job_data: Dict[str, Any]):
        """Add a job to the processing queue."""
        with QMutexLocker(self.mutex):
            self.jobs_queue.append(job_data)
            if not self.is_running:
                self.start()

    def stop_processing(self):
        """Stop all processing."""
        with QMutexLocker(self.mutex):
            self.is_running = False
            self.jobs_queue.clear()
            self.processing_engine.shutdown()

    def run(self):
        """Main processing loop."""
        self.is_running = True
        logger.info("Processing worker started")

        while self.is_running:
            with QMutexLocker(self.mutex):
                if not self.jobs_queue:
                    self.msleep(100)  # Wait for jobs
                    continue

                job_data = self.jobs_queue.pop(0)

            try:
                # Process the job
                result = self.processing_engine.process_medical_record(
                    job_data['secure_path'],
                    job_data['filename']
                )

                if result.success:
                    self.completed.emit(job_data['job_id'], result.__dict__)
                else:
                    self.error_occurred.emit(job_data['job_id'], result.error_message or "Processing failed")

            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
                self.error_occurred.emit(job_data['job_id'], str(e))

    def on_status_update(self, job_id: str, status: str):
        """Handle status updates from processing engine."""
        self.status_updated.emit(job_id, status)

    def on_progress_update(self, job_id: str, progress: int):
        """Handle progress updates from processing engine."""
        self.progress_updated.emit(job_id, progress)

    def on_completion(self, job_id: str):
        """Handle job completion."""
        # Completion is handled by the main processing loop
        pass

class ProcessingStatusWidget(QWidget):
    """Widget for displaying processing status and progress."""

    # Signals
    processing_completed = pyqtSignal(str)  # job_id

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.processing_engine = ProcessingEngine(config)
        self.file_manager = FileManager(config)

        # Initialize processing worker
        self.worker = ProcessingWorker(self.processing_engine)
        self.worker.status_updated.connect(self.on_status_updated)
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.completed.connect(self.on_processing_completed)
        self.worker.error_occurred.connect(self.on_processing_error)

        # Active jobs tracking
        self.active_jobs = {}
        self.completed_jobs = []

        self.init_ui()

        # Set up update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_statistics)
        self.update_timer.start(1000)  # Update every second

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title_label = QLabel("⚙️ Processing Status")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)

        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # Upper section - Active jobs
        upper_widget = self.create_active_jobs_widget()
        splitter.addWidget(upper_widget)

        # Lower section - Statistics and logs
        lower_widget = self.create_statistics_widget()
        splitter.addWidget(lower_widget)

        splitter.setSizes([400, 300])

        # Control buttons
        button_widget = self.create_control_buttons()
        layout.addWidget(button_widget)

    def create_active_jobs_widget(self) -> QWidget:
        """Create widget for displaying active processing jobs."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Active jobs label
        active_label = QLabel("📋 Active Processing Jobs")
        active_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(active_label)

        # Jobs list
        self.jobs_list = QListWidget()
        self.jobs_list.setMaximumHeight(300)
        self.jobs_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                margin: 2px;
                border-radius: 3px;
                border: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #007acc;
                color: white;
            }
        """)
        layout.addWidget(self.jobs_list)

        return widget

    def create_statistics_widget(self) -> QWidget:
        """Create widget for displaying processing statistics."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left side - Statistics
        stats_group = QGroupBox("📊 Processing Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_label = QLabel("No processing activity")
        self.stats_label.setStyleSheet("color: #666; font-style: italic;")
        stats_layout.addWidget(self.stats_label)

        # Queue status
        self.queue_label = QLabel("Queue: Empty")
        self.queue_label.setStyleSheet("color: #666;")
        stats_layout.addWidget(self.queue_label)

        layout.addWidget(stats_group)

        # Right side - Processing log
        log_group = QGroupBox("📝 Processing Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: #f9f9f9;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        log_layout.addWidget(self.log_text)

        # Clear log button
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.clear_log)
        clear_log_btn.setMaximumWidth(100)
        log_layout.addWidget(clear_log_btn)

        layout.addWidget(log_group)

        return widget

    def create_control_buttons(self) -> QWidget:
        """Create control buttons for processing management."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Stop all processing button
        self.stop_btn = QPushButton("⏹️ Stop All Processing")
        self.stop_btn.clicked.connect(self.stop_all_processing)
        self.stop_btn.setStyleSheet("""
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
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        # Clear completed button
        clear_btn = QPushButton("🗑️ Clear Completed")
        clear_btn.clicked.connect(self.clear_completed_jobs)
        layout.addWidget(clear_btn)

        # Refresh statistics button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.update_statistics)
        layout.addWidget(refresh_btn)

        layout.addStretch()

        # Status indicator
        self.status_indicator = QLabel("🟢 Ready")
        self.status_indicator.setStyleSheet("color: #28a745; font-weight: bold;")
        layout.addWidget(self.status_indicator)

        return widget

    def process_files(self, file_paths: List[str]):
        """Process a list of files."""
        try:
            # Create processing jobs for each file
            jobs = self.file_manager.batch_file_processing(file_paths)

            if not jobs:
                QMessageBox.warning(self, "No Valid Files",
                                  "No valid medical record files were found.")
                return

            # Add jobs to processing queue
            for job in jobs:
                job_data = {
                    'job_id': job['job_id'],
                    'secure_path': job['secure_path'],
                    'filename': job['file_name'],
                    'original_path': job['original_path']
                }

                self.worker.add_job(job_data)

                # Add to active jobs tracking
                self.active_jobs[job['job_id']] = {
                    'filename': job['file_name'],
                    'status': 'Queued',
                    'progress': 0,
                    'start_time': datetime.now(),
                    'item': None
                }

            # Update UI
            self.update_jobs_list()
            self.stop_btn.setEnabled(True)
            self.status_indicator.setText("🟡 Processing")
            self.status_indicator.setStyleSheet("color: #ffc107; font-weight: bold;")

            self.add_log_message(f"Started processing {len(jobs)} medical record(s)")

        except Exception as e:
            logger.error(f"Error starting file processing: {e}")
            QMessageBox.critical(self, "Processing Error",
                               f"Error starting processing: {str(e)}")

    def on_status_updated(self, job_id: str, status: str):
        """Handle status updates from processing worker."""
        if job_id in self.active_jobs:
            self.active_jobs[job_id]['status'] = status
            self.update_jobs_list()
            self.add_log_message(f"{job_id[:8]}: {status}")

    def on_progress_updated(self, job_id: str, progress: int):
        """Handle progress updates from processing worker."""
        if job_id in self.active_jobs:
            self.active_jobs[job_id]['progress'] = progress
            self.update_jobs_list()

    def on_processing_completed(self, job_id: str, result: dict):
        """Handle processing completion."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job['status'] = 'Completed'
            job['progress'] = 100

            # Move to completed jobs
            self.completed_jobs.append({
                'job_id': job_id,
                'filename': job['filename'],
                'completion_time': datetime.now(),
                'success': result.get('success', False),
                'num_codes': len(result.get('cpt_recommendations', []))
            })

            self.add_log_message(f"{job_id[:8]}: Processing completed - {job['filename']}")
            self.update_jobs_list()

            # Emit completion signal
            self.processing_completed.emit(job_id)

            # Check if all jobs are completed
            self.check_all_completed()

    def on_processing_error(self, job_id: str, error_message: str):
        """Handle processing errors."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job['status'] = f'Error: {error_message[:50]}...'

            self.add_log_message(f"❌ {job_id[:8]}: {error_message}")
            self.update_jobs_list()
            self.check_all_completed()

    def update_jobs_list(self):
        """Update the jobs list display."""
        self.jobs_list.clear()

        for job_id, job_data in self.active_jobs.items():
            # Create list item
            filename = job_data['filename']
            status = job_data['status']
            progress = job_data['progress']

            # Determine status icon
            if status == 'Completed':
                icon = "✅"
            elif 'Error' in status:
                icon = "❌"
            elif progress > 0:
                icon = "🔄"
            else:
                icon = "⏳"

            # Create progress bar widget
            progress_widget = QWidget()
            progress_layout = QHBoxLayout(progress_widget)
            progress_layout.setContentsMargins(0, 0, 0, 0)

            # Status label
            status_label = QLabel(f"{icon} {filename}")
            status_label.setMinimumWidth(200)
            progress_layout.addWidget(status_label)

            # Progress bar
            progress_bar = QProgressBar()
            progress_bar.setMaximum(100)
            progress_bar.setValue(progress)
            progress_bar.setMinimumWidth(150)
            progress_layout.addWidget(progress_bar)

            # Status text
            status_text = QLabel(f"{progress}% - {status}")
            status_text.setMinimumWidth(200)
            progress_layout.addWidget(status_text)

            # Add to list
            item = QListWidgetItem()
            item.setSizeHint(progress_widget.sizeHint())
            self.jobs_list.addItem(item)
            self.jobs_list.setItemWidget(item, progress_widget)

            job_data['item'] = item

    def update_statistics(self):
        """Update processing statistics display."""
        try:
            # Get queue status from processing engine
            queue_status = self.processing_engine.get_queue_status()

            # Calculate statistics
            total_active = len(self.active_jobs)
            total_completed = len(self.completed_jobs)
            avg_processing_time = 0

            if self.completed_jobs:
                total_time = sum([
                    (job['completion_time'] - job.get('start_time', job['completion_time'])).total_seconds()
                    for job in self.completed_jobs
                ])
                avg_processing_time = total_time / len(self.completed_jobs)

            # Update statistics label
            stats_text = f"""Active Jobs: {total_active}
Completed Jobs: {total_completed}
Average Processing Time: {avg_processing_time:.1f}s
Total CPT Codes Generated: {sum(job.get('num_codes', 0) for job in self.completed_jobs)}"""

            self.stats_label.setText(stats_text)

            # Update queue status
            queue_text = f"Queue: {queue_status['queued']} waiting, {queue_status['active']} processing"
            self.queue_label.setText(queue_text)

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def add_log_message(self, message: str):
        """Add a message to the processing log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        self.log_text.append(log_entry)

        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

        # Limit log size
        if self.log_text.document().blockCount() > 100:
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()

    def clear_log(self):
        """Clear the processing log."""
        self.log_text.clear()

    def clear_completed_jobs(self):
        """Clear completed jobs from the display."""
        self.completed_jobs.clear()
        self.update_statistics()
        self.add_log_message("Cleared completed jobs from display")

    def stop_all_processing(self):
        """Stop all active processing."""
        reply = QMessageBox.question(self, "Stop Processing",
                                   "Stop all active processing jobs?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.worker.stop_processing()
                self.add_log_message("Stopped all processing jobs")
                self.status_indicator.setText("🔴 Stopped")
                self.status_indicator.setStyleSheet("color: #dc3545; font-weight: bold;")
                self.stop_btn.setEnabled(False)
            except Exception as e:
                logger.error(f"Error stopping processing: {e}")
                QMessageBox.critical(self, "Error",
                                   f"Error stopping processing: {str(e)}")

    def check_all_completed(self):
        """Check if all jobs are completed and update UI accordingly."""
        if all(job['status'] in ['Completed', 'Error'] for job in self.active_jobs.values()):
            self.status_indicator.setText("🟢 All Completed")
            self.status_indicator.setStyleSheet("color: #28a745; font-weight: bold;")
            self.stop_btn.setEnabled(False)

    def stop_all_processing(self):
        """Stop all processing and cleanup."""
        try:
            self.worker.stop_processing()
            self.processing_engine.shutdown()
            self.update_timer.stop()
            logger.info("Processing status monitor stopped")
        except Exception as e:
            logger.error(f"Error stopping processing status monitor: {e}")