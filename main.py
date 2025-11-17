#!/usr/bin/env python3
"""
Medical CPT Coding Application
Main entry point for the desktop application that processes medical records
and provides CPT code suggestions using ensemble AI/ML models.
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from qdarkstyle.lightstyle import LightStyle
from qdarkstyle.darkstyle import DarkStyle

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main_window import MainWindow
from src.utils.config import Config
from src.utils.logger import setup_logging

def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Medical CPT Coder")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Medical AI Solutions")

    # Set high DPI support
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Load configuration
    config = Config()

    # Apply theme based on user preference
    if config.get_theme() == 'dark':
        app.setStyleSheet(DarkStyle().stylesheet())
    else:
        app.setStyleSheet(LightStyle().stylesheet())

    # Create and show main window
    window = MainWindow(config)
    window.show()

    # Start the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()