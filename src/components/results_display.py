"""
Results display component for CPT coding recommendations.
Provides sortable tables, detailed views, and export functionality.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem,
    QTextEdit, QSplitter, QGroupBox, QMessageBox, QFileDialog,
    QHeaderView, QProgressBar, QFrame, QCheckBox, QLineEdit,
    QComboBox, QTabWidget, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCursor

from ..utils.config import Config
from ..core.database import Database

logger = logging.getLogger(__name__)

class ResultsTableWidget(QTableWidget):
    """Custom table widget for displaying CPT results with sorting."""

    def __init__(self):
        super().__init__()
        self.setup_table()
        self.sort_column = -1
        self.sort_order = Qt.SortOrder.AscendingOrder

    def setup_table(self):
        """Set up the table structure."""
        # Set up columns
        headers = [
            "CPT Code", "Description", "Confidence", "Category",
            "Requires Review", "Processing Date", "File Name"
        ]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)

        # Configure table properties
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSortingEnabled(True)

        # Set column widths
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # CPT Code
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Description
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Confidence
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Category
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Review
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # Date
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)  # File Name

        # Enable sorting on click
        header.sectionClicked.connect(self.on_header_clicked)

        # Style
        self.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                gridline-color: #f0f0f0;
                selection-background-color: #007acc;
                selection-color: white;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #f0f0f0;
            }
            QTableWidget::item:selected {
                background-color: #007acc;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                border: 1px solid #ddd;
                font-weight: bold;
            }
        """)

    def on_header_clicked(self, column):
        """Handle header click for sorting."""
        if self.sort_column == column:
            # Toggle sort order
            self.sort_order = Qt.SortOrder.DescendingOrder if self.sort_order == Qt.SortOrder.AscendingOrder else Qt.SortOrder.AscendingOrder
        else:
            self.sort_column = column
            self.sort_order = Qt.SortOrder.AscendingOrder

        self.sortItems(column, self.sort_order)

    def add_result(self, result_data: Dict[str, Any]):
        """Add a result row to the table."""
        row = self.rowCount()
        self.insertRow(row)

        # CPT Code
        code_item = QTableWidgetItem(result_data.get('cpt_code', ''))
        code_item.setFont(QFont("Courier", 10, QFont.Weight.Bold))
        self.setItem(row, 0, code_item)

        # Description
        desc_item = QTableWidgetItem(result_data.get('description', ''))
        self.setItem(row, 1, desc_item)

        # Confidence (with color coding)
        confidence = result_data.get('confidence_score', 0.0)
        conf_item = QTableWidgetItem(f"{confidence:.1%}")
        conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        # Color code based on confidence
        if confidence >= 0.9:
            conf_item.setBackground(QColor(200, 255, 200))  # Green
        elif confidence >= 0.7:
            conf_item.setBackground(QColor(255, 255, 200))  # Yellow
        else:
            conf_item.setBackground(QColor(255, 200, 200))  # Red

        self.setItem(row, 2, conf_item)

        # Category
        category_item = QTableWidgetItem(result_data.get('category', 'Unknown'))
        self.setItem(row, 3, category_item)

        # Requires Review
        requires_review = result_data.get('requires_review', False)
        review_item = QTableWidgetItem("Yes" if requires_review else "No")
        review_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if requires_review:
            review_item.setBackground(QColor(255, 200, 200))
            review_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        self.setItem(row, 4, review_item)

        # Processing Date
        date_str = result_data.get('created_at', '')
        if date_str:
            try:
                # Format date
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                date_item = QTableWidgetItem(date_obj.strftime("%Y-%m-%d %H:%M"))
            except:
                date_item = QTableWidgetItem(date_str[:10])  # Fallback to date part
        else:
            date_item = QTableWidgetItem("")
        self.setItem(row, 5, date_item)

        # File Name
        file_item = QTableWidgetItem(result_data.get('file_name', ''))
        self.setItem(row, 6, file_item)

    def clear_results(self):
        """Clear all results from the table."""
        self.setRowCount(0)

class ResultsDisplay(QWidget):
    """Main results display component with tabs and export functionality."""

    # Signals
    export_requested = pyqtSignal()

    def __init__(self, config: Config, database: Database):
        super().__init__()
        self.config = config
        self.db = database
        self.current_results = []

        self.init_ui()

        # Set up refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_results)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title_label = QLabel("📊 CPT Coding Results")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)

        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)

        # Tabbed interface
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Results table tab
        table_tab = self.create_results_table_tab()
        self.tabs.addTab(table_tab, "📋 Results Table")

        # Detailed view tab
        detail_tab = self.create_detailed_view_tab()
        self.tabs.addTab(detail_tab, "🔍 Detailed View")

        # Statistics tab
        stats_tab = self.create_statistics_tab()
        self.tabs.addTab(stats_tab, "📈 Statistics")

        # Export buttons
        export_panel = self.create_export_panel()
        layout.addWidget(export_panel)

    def create_control_panel(self) -> QWidget:
        """Create control panel for filtering and searching."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Search box
        search_label = QLabel("Search:")
        layout.addWidget(search_label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search CPT codes, descriptions, or filenames...")
        self.search_box.textChanged.connect(self.filter_results)
        layout.addWidget(self.search_box)

        # Category filter
        cat_label = QLabel("Category:")
        layout.addWidget(cat_label)

        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories")
        self.category_filter.currentTextChanged.connect(self.filter_results)
        layout.addWidget(self.category_filter)

        # Confidence filter
        conf_label = QLabel("Min Confidence:")
        layout.addWidget(conf_label)

        self.confidence_filter = QComboBox()
        self.confidence_filter.addItems(["Any", "70%+", "80%+", "90%+"])
        self.confidence_filter.currentTextChanged.connect(self.filter_results)
        layout.addWidget(self.confidence_filter)

        # Show review required checkbox
        self.show_review_only = QCheckBox("Review Required Only")
        self.show_review_only.stateChanged.connect(self.filter_results)
        layout.addWidget(self.show_review_only)

        layout.addStretch()

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_results)
        layout.addWidget(refresh_btn)

        # Clear button
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self.clear_results)
        layout.addWidget(clear_btn)

        return widget

    def create_results_table_tab(self) -> QWidget:
        """Create the results table tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Results table
        self.results_table = ResultsTableWidget()
        layout.addWidget(self.results_table)

        # Selection info
        self.selection_info = QLabel("Select a row to see details")
        self.selection_info.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.selection_info)

        # Connect selection signal
        self.results_table.itemSelectionChanged.connect(self.on_table_selection_changed)

        return widget

    def create_detailed_view_tab(self) -> QWidget:
        """Create the detailed view tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Detail view splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side - Results list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(QLabel("📋 CPT Recommendations"))

        self.detail_tree = QTreeWidget()
        self.detail_tree.setHeaderLabels(["CPT Code", "Description", "Confidence"])
        self.detail_tree.setAlternatingRowColors(True)
        self.detail_tree.setMaximumWidth(400)

        # Configure columns
        self.detail_tree.setColumnWidth(0, 80)   # CPT Code
        self.detail_tree.setColumnWidth(1, 200)  # Description
        self.detail_tree.setColumnWidth(2, 80)   # Confidence

        left_layout.addWidget(self.detail_tree)
        splitter.addWidget(left_widget)

        # Right side - Detailed information
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(QLabel("📄 Detailed Information"))

        # Detail text area
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(400)
        right_layout.addWidget(self.detail_text)

        # Explanation area
        right_layout.addWidget(QLabel("💡 Explanation"))

        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        right_layout.addWidget(self.explanation_text)

        splitter.addWidget(right_widget)

        splitter.setSizes([300, 500])

        # Connect tree selection
        self.detail_tree.itemSelectionChanged.connect(self.on_tree_selection_changed)

        return widget

    def create_statistics_tab(self) -> QWidget:
        """Create the statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Statistics scroll area
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Processing statistics
        self.stats_group = QGroupBox("📈 Processing Statistics")
        stats_layout = QVBoxLayout(self.stats_group)
        self.stats_label = QLabel("Loading statistics...")
        stats_layout.addWidget(self.stats_label)
        scroll_layout.addWidget(self.stats_group)

        # CPT code usage statistics
        self.usage_group = QGroupBox("🏥 CPT Code Usage")
        usage_layout = QVBoxLayout(self.usage_group)
        self.usage_label = QLabel("Loading usage statistics...")
        usage_layout.addWidget(self.usage_label)
        scroll_layout.addWidget(self.usage_group)

        # Quality metrics
        self.quality_group = QGroupBox("✅ Quality Metrics")
        quality_layout = QVBoxLayout(self.quality_group)
        self.quality_label = QLabel("Loading quality metrics...")
        quality_layout.addWidget(self.quality_label)
        scroll_layout.addWidget(self.quality_group)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        return widget

    def create_export_panel(self) -> QWidget:
        """Create export panel."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Export CSV button
        csv_btn = QPushButton("📊 Export CSV")
        csv_btn.clicked.connect(self.export_csv)
        csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        layout.addWidget(csv_btn)

        # Export JSON button
        json_btn = QPushButton("📄 Export JSON")
        json_btn.clicked.connect(self.export_json)
        json_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        layout.addWidget(json_btn)

        # Generate report button
        report_btn = QPushButton("📋 Generate Report")
        report_btn.clicked.connect(self.generate_report)
        report_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5a32a3;
            }
        """)
        layout.addWidget(report_btn)

        layout.addStretch()

        # Results count
        self.results_count = QLabel("No results")
        self.results_count.setStyleSheet("color: #666; font-weight: bold;")
        layout.addWidget(self.results_count)

        return widget

    def refresh_results(self):
        """Refresh results from database."""
        try:
            # Get recent processing jobs with results
            jobs = self.db.get_processing_jobs(limit=100)
            self.current_results = []

            for job in jobs:
                job_results = self.db.get_job_results(job['id'])
                if job_results and job_results['results']:
                    for result in job_results['results']:
                        result_data = {
                            'job_id': job['id'],
                            'cpt_code': result['cpt_code'],
                            'description': result['description'],
                            'confidence_score': result['confidence_score'],
                            'requires_review': result['requires_review'],
                            'category': result.get('category', 'Unknown'),
                            'created_at': result.get('created_at', job['created_at']),
                            'file_name': job['file_name'],
                            'explanation': result.get('explanation', ''),
                            'model_used': result.get('model_used', '')
                        }
                        self.current_results.append(result_data)

            # Update displays
            self.update_table()
            self.update_detail_tree()
            self.update_statistics()
            self.update_filters()

            # Update count
            self.results_count.setText(f"{len(self.current_results)} results")

        except Exception as e:
            logger.error(f"Error refreshing results: {e}")

    def update_table(self):
        """Update the results table."""
        self.results_table.clear_results()

        for result in self.current_results:
            self.results_table.add_result(result)

    def update_detail_tree(self):
        """Update the detail tree view."""
        self.detail_tree.clear()

        # Group by job_id
        jobs = {}
        for result in self.current_results:
            job_id = result['job_id']
            if job_id not in jobs:
                jobs[job_id] = {
                    'file_name': result['file_name'],
                    'results': []
                }
            jobs[job_id]['results'].append(result)

        # Create tree items
        for job_id, job_data in jobs.items():
            # Job item
            job_item = QTreeWidgetItem(self.detail_tree)
            job_item.setText(0, f"Job: {job_id[:8]}")
            job_item.setText(1, job_data['file_name'])
            job_item.setText(2, f"{len(job_data['results'])} codes")

            # Add result items
            for result in job_data['results']:
                result_item = QTreeWidgetItem(job_item)
                result_item.setText(0, result['cpt_code'])
                result_item.setText(1, result['description'][:50] + "..." if len(result['description']) > 50 else result['description'])
                result_item.setText(2, f"{result['confidence_score']:.1%}")
                result_item.setData(0, Qt.ItemDataRole.UserRole, result)

        job_item.setExpanded(True)

    def update_statistics(self):
        """Update statistics display."""
        try:
            if not self.current_results:
                self.stats_label.setText("No results to analyze")
                self.usage_label.setText("No CPT usage data")
                self.quality_label.setText("No quality data")
                return

            # Processing statistics
            total_jobs = len(set(r['job_id'] for r in self.current_results))
            total_codes = len(self.current_results)
            avg_confidence = sum(r['confidence_score'] for r in self.current_results) / total_codes
            review_required = sum(1 for r in self.current_results if r['requires_review'])

            stats_text = f"""Total Jobs Processed: {total_jobs}
Total CPT Codes Generated: {total_codes}
Average Confidence: {avg_confidence:.1%}
Requiring Review: {review_required} ({review_required/total_codes:.1%})"""

            self.stats_label.setText(stats_text)

            # CPT usage statistics
            code_usage = {}
            for result in self.current_results:
                code = result['cpt_code']
                if code not in code_usage:
                    code_usage[code] = {
                        'description': result['description'],
                        'count': 0,
                        'avg_confidence': 0
                    }
                code_usage[code]['count'] += 1
                code_usage[code]['avg_confidence'] += result['confidence_score']

            # Calculate average confidence
            for code in code_usage:
                code_usage[code]['avg_confidence'] /= code_usage[code]['count']

            # Sort by frequency
            sorted_codes = sorted(code_usage.items(), key=lambda x: x[1]['count'], reverse=True)[:10]

            usage_text = "Top CPT Codes by Usage:\n\n"
            for code, data in sorted_codes:
                usage_text += f"{code}: {data['count']} times (avg confidence: {data['avg_confidence']:.1%})\n"
                usage_text += f"  {data['description'][:60]}...\n\n"

            self.usage_label.setText(usage_text)

            # Quality metrics
            high_confidence = sum(1 for r in self.current_results if r['confidence_score'] >= 0.9)
            medium_confidence = sum(1 for r in self.current_results if 0.7 <= r['confidence_score'] < 0.9)
            low_confidence = sum(1 for r in self.current_results if r['confidence_score'] < 0.7)

            quality_text = f"""Confidence Distribution:
High (≥90%): {high_confidence} ({high_confidence/total_codes:.1%})
Medium (70-89%): {medium_confidence} ({medium_confidence/total_codes:.1%})
Low (<70%): {low_confidence} ({low_confidence/total_codes:.1%})

Categories by Frequency:
"""

            # Category distribution
            category_counts = {}
            for result in self.current_results:
                cat = result.get('category', 'Unknown')
                category_counts[cat] = category_counts.get(cat, 0) + 1

            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            for cat, count in sorted_categories:
                quality_text += f"  {cat}: {count}\n"

            self.quality_label.setText(quality_text)

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def update_filters(self):
        """Update filter options."""
        # Update category filter
        current_cat = self.category_filter.currentText()
        self.category_filter.clear()
        self.category_filter.addItem("All Categories")

        categories = sorted(set(r.get('category', 'Unknown') for r in self.current_results))
        for cat in categories:
            self.category_filter.addItem(cat)

        # Restore selection
        index = self.category_filter.findText(current_cat)
        if index >= 0:
            self.category_filter.setCurrentIndex(index)

    def filter_results(self):
        """Apply filters to results."""
        search_text = self.search_box.text().lower()
        category_filter = self.category_filter.currentText()
        confidence_filter = self.confidence_filter.currentText()
        review_only = self.show_review_only.isChecked()

        filtered_results = []

        for result in self.current_results:
            # Search filter
            if search_text:
                if (search_text not in result['cpt_code'].lower() and
                    search_text not in result['description'].lower() and
                    search_text not in result['file_name'].lower()):
                    continue

            # Category filter
            if category_filter != "All Categories":
                if result.get('category', 'Unknown') != category_filter:
                    continue

            # Confidence filter
            if confidence_filter != "Any":
                min_conf = {'70%+': 0.7, '80%+': 0.8, '90%+': 0.9}.get(confidence_filter, 0)
                if result['confidence_score'] < min_conf:
                    continue

            # Review required filter
            if review_only and not result['requires_review']:
                continue

            filtered_results.append(result)

        # Update table with filtered results
        self.results_table.clear_results()
        for result in filtered_results:
            self.results_table.add_result(result)

        # Update count
        self.results_count.setText(f"{len(filtered_results)} of {len(self.current_results)} results")

    def on_table_selection_changed(self):
        """Handle table row selection."""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            code_item = self.results_table.item(current_row, 0)
            code = code_item.text() if code_item else ""

            desc_item = self.results_table.item(current_row, 1)
            description = desc_item.text() if desc_item else ""

            conf_item = self.results_table.item(current_row, 2)
            confidence = conf_item.text() if conf_item else ""

            file_item = self.results_table.item(current_row, 6)
            filename = file_item.text() if file_item else ""

            self.selection_info.setText(f"Selected: {code} - {description} ({confidence}) from {filename}")
        else:
            self.selection_info.setText("Select a row to see details")

    def on_tree_selection_changed(self):
        """Handle tree selection for detailed view."""
        current_item = self.detail_tree.currentItem()
        if current_item and current_item.parent():  # Result item (not job item)
            result_data = current_item.data(0, Qt.ItemDataRole.UserRole)
            if result_data:
                self.show_detailed_result(result_data)

    def show_detailed_result(self, result_data: Dict[str, Any]):
        """Show detailed information for a result."""
        # Detail information
        detail_text = f"""CPT Code: {result_data['cpt_code']}
Description: {result_data['description']}
Category: {result_data.get('category', 'Unknown')}
Confidence Score: {result_data['confidence_score']:.1%}
Requires Review: {'Yes' if result_data['requires_review'] else 'No'}
Processing Date: {result_data.get('created_at', 'Unknown')}
Source File: {result_data['file_name']}
Model Used: {result_data.get('model_used', 'Ensemble')}

Job ID: {result_data['job_id']}
"""

        self.detail_text.setPlainText(detail_text)

        # Explanation
        explanation = result_data.get('explanation', 'No detailed explanation available.')
        self.explanation_text.setPlainText(explanation)

    def export_csv(self):
        """Export results to CSV file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "", "CSV Files (*.csv)"
            )

            if file_path:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = [
                        'cpt_code', 'description', 'confidence_score', 'category',
                        'requires_review', 'created_at', 'file_name', 'job_id'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for result in self.current_results:
                        writer.writerow({
                            'cpt_code': result['cpt_code'],
                            'description': result['description'],
                            'confidence_score': f"{result['confidence_score']:.4f}",
                            'category': result.get('category', 'Unknown'),
                            'requires_review': result['requires_review'],
                            'created_at': result.get('created_at', ''),
                            'file_name': result['file_name'],
                            'job_id': result['job_id']
                        })

                QMessageBox.information(self, "Export Complete",
                                     f"Results exported to {file_path}")
                self.export_requested.emit()

        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            QMessageBox.critical(self, "Export Error", f"Error exporting CSV: {str(e)}")

    def export_json(self):
        """Export results to JSON file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export JSON", "", "JSON Files (*.json)"
            )

            if file_path:
                export_data = {
                    'export_date': datetime.now().isoformat(),
                    'total_results': len(self.current_results),
                    'results': self.current_results
                }

                with open(file_path, 'w', encoding='utf-8') as jsonfile:
                    json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)

                QMessageBox.information(self, "Export Complete",
                                     f"Results exported to {file_path}")
                self.export_requested.emit()

        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            QMessageBox.critical(self, "Export Error", f"Error exporting JSON: {str(e)}")

    def generate_report(self):
        """Generate a comprehensive report."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Generate Report", "", "Text Files (*.txt)"
            )

            if file_path:
                report = self.generate_report_text()

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                QMessageBox.information(self, "Report Generated",
                                     f"Report saved to {file_path}")

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            QMessageBox.critical(self, "Report Error", f"Error generating report: {str(e)}")

    def generate_report_text(self) -> str:
        """Generate comprehensive report text."""
        report_lines = [
            "MEDICAL CPT CODING ANALYSIS REPORT",
            "=" * 50,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Results: {len(self.current_results)}",
            "",
            "EXECUTIVE SUMMARY:",
            "-" * 20
        ]

        if self.current_results:
            # Calculate summary statistics
            total_jobs = len(set(r['job_id'] for r in self.current_results))
            avg_confidence = sum(r['confidence_score'] for r in self.current_results) / len(self.current_results)
            high_conf_count = sum(1 for r in self.current_results if r['confidence_score'] >= 0.9)
            review_count = sum(1 for r in self.current_results if r['requires_review'])

            report_lines.extend([
                f"- Medical records processed: {total_jobs}",
                f"- CPT codes generated: {len(self.current_results)}",
                f"- Average confidence: {avg_confidence:.1%}",
                f"- High confidence codes (≥90%): {high_conf_count} ({high_conf_count/len(self.current_results):.1%})",
                f"- Codes requiring review: {review_count} ({review_count/len(self.current_results):.1%})",
                ""
            ])

            # Top CPT codes
            code_counts = {}
            for result in self.current_results:
                code_counts[result['cpt_code']] = code_counts.get(result['cpt_code'], 0) + 1

            top_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            report_lines.extend([
                "TOP CPT CODES BY FREQUENCY:",
                "-" * 30
            ])

            for code, count in top_codes:
                result = next(r for r in self.current_results if r['cpt_code'] == code)
                report_lines.append(f"{code}: {count} times - {result['description']}")

            report_lines.extend(["", "DETAILED RESULTS:", "-" * 20])

            # Group by job
            jobs = {}
            for result in self.current_results:
                job_id = result['job_id']
                if job_id not in jobs:
                    jobs[job_id] = []
                jobs[job_id].append(result)

            for job_id, job_results in jobs.items():
                report_lines.extend([
                    f"\nJob ID: {job_id}",
                    f"File: {job_results[0]['file_name']}",
                    f"CPT Codes Generated: {len(job_results)}",
                    ""
                ])

                for result in job_results:
                    report_lines.extend([
                        f"  • {result['cpt_code']} - {result['confidence_score']:.1%} confidence",
                        f"    {result['description']}",
                        f"    {'⚠️ Requires professional review' if result['requires_review'] else '✅ Ready for billing'}",
                        ""
                    ])

        else:
            report_lines.append("No processing results available.")

        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "-" * 20,
            "• Review all codes marked as requiring professional review",
            "• Verify documentation supports all generated CPT codes",
            "• Consider additional training for low-confidence predictions",
            "• Ensure HIPAA compliance and data retention policies",
            "",
            "NOTICE:",
            "-" * 10,
            "This report was generated by an AI-assisted medical coding system.",
            "All CPT codes should be reviewed by qualified medical coders",
            "before submission for billing. This system is not a substitute",
            "for professional medical coding expertise.",
            "",
            "=" * 50
        ])

        return "\n".join(report_lines)

    def clear_results(self):
        """Clear all results."""
        reply = QMessageBox.question(self, "Clear Results",
                                   "Clear all displayed results? This will not delete data from the database.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.current_results.clear()
            self.results_table.clear_results()
            self.detail_tree.clear()
            self.detail_text.clear()
            self.explanation_text.clear()
            self.results_count.setText("No results")

    def get_selected_results(self) -> List[Dict[str, Any]]:
        """Get currently selected results."""
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            return []

        selected_results = []
        for item in selected_items:
            row = item.row()
            code_item = self.results_table.item(row, 0)
            if code_item:
                code = code_item.text()
                # Find corresponding result data
                for result in self.current_results:
                    if result['cpt_code'] == code:
                        selected_results.append(result)
                        break

        return selected_results