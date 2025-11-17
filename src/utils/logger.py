"""
Logging configuration for the Medical CPT Coding Application.
HIPAA-compliant logging that avoids PHI in logs while maintaining audit trails.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from .config import Config

class PHIFormatter(logging.Formatter):
    """Custom formatter that ensures no PHI is logged."""

    def __init__(self):
        super().__init__()
        # Common PHI patterns to redact
        self.phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{10}\b',  # Phone numbers
            r'\b[A-Z]\d{7}\b',  # Medical record numbers
            r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b',  # Dates (partial)
            r'\b[A-Za-z]+,\s*[A-Z]\.\s*[A-Za-z]+\b',  # Names (simple pattern)
        ]

    def format(self, record):
        """Format log record while redacting potential PHI."""
        import re

        # Format the record first
        formatted = super().format(record)

        # Redact potential PHI
        for pattern in self.phi_patterns:
            formatted = re.sub(pattern, '[REDACTED]', formatted, flags=re.IGNORECASE)

        return formatted

class AuditLogger:
    """HIPAA-compliant audit logger for tracking system activities."""

    def __init__(self, config: Config):
        self.config = config
        self.audit_dir = Path(config.get_temp_directory()) / "audit"
        self.audit_dir.mkdir(exist_ok=True)

        # Set up audit file handler
        audit_file = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        self.handler = logging.handlers.RotatingFileHandler(
            audit_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=30
        )

        self.handler.setFormatter(PHIFormatter())
        self.logger = logging.getLogger('audit')
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def log_action(self, action: str, details: Optional[Dict[str, Any]] = None,
                   user_id: Optional[str] = None, job_id: Optional[str] = None):
        """Log an audit action with minimal PHI."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details or {},
            'user_id': user_id or 'system',
            'job_id': job_id
        }

        # Log structured audit entry
        self.logger.info(json.dumps(audit_entry, separators=(',', ':')))

    def log_file_processing(self, file_name: str, file_size: int, job_id: str):
        """Log file processing start."""
        self.log_action(
            'file_processing_started',
            {
                'file_name': os.path.basename(file_name),  # No full path
                'file_size': file_size,
                'file_type': 'medical_record'
            },
            job_id=job_id
        )

    def log_cpt_suggestions(self, job_id: str, num_codes: int, avg_confidence: float):
        """Log CPT code generation results."""
        self.log_action(
            'cpt_codes_generated',
            {
                'num_codes': num_codes,
                'avg_confidence': round(avg_confidence, 3),
                'processing_complete': True
            },
            job_id=job_id
        )

    def log_error(self, error_type: str, error_message: str, job_id: Optional[str] = None):
        """Log error events."""
        self.log_action(
            'error_occurred',
            {
                'error_type': error_type,
                'error_message': error_message[:200]  # Truncate long messages
            },
            job_id=job_id
        )

    def log_data_deletion(self, file_count: int, job_ids: list):
        """Log data deletion for audit purposes."""
        self.log_action(
            'data_deleted',
            {
                'files_deleted': file_count,
                'jobs_deleted': len(job_ids),
                'deletion_type': 'auto_cleanup'
            }
        )

def setup_logging():
    """Set up logging configuration for the application."""
    config = Config()

    # Create logs directory
    logs_dir = Path(config.get_temp_directory()) / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = PHIFormatter()
    console_formatter.format = lambda record: f"{datetime.now().strftime('%H:%M:%S')} - {record.name} - {record.levelname} - {console_formatter.format(record)}"
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler for application logs
    log_file = logs_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=7
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = PHIFormatter()
    file_formatter.format = lambda record: f"{datetime.now().isoformat()} - {record.name} - {record.levelname} - {file_formatter.format(record)}"
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    # Initialize audit logger
    audit_logger = AuditLogger(config)

    logging.info("Medical CPT Coder logging system initialized")

    return audit_logger