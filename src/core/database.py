"""
Database manager for the Medical CPT Coding Application.
Handles SQLite database operations for audit trails, processing jobs, and CPT results.
"""

import sqlite3
import uuid
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager

from ..utils.config import Config

logger = logging.getLogger(__name__)

class Database:
    """SQLite database manager for HIPAA-compliant audit trails and results storage."""

    def __init__(self, config: Config):
        """Initialize database connection and create schema."""
        self.config = config
        self.db_path = config.get_database_path()

        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.init_database()
        logger.info(f"Database initialized: {self.db_path}")

    def init_database(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Processing jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    processing_time_seconds INTEGER,
                    text_length INTEGER,
                    quality_score REAL,
                    metadata TEXT
                )
            ''')

            # CPT results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cpt_results (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    cpt_code TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    explanation TEXT,
                    model_used TEXT,
                    requires_review BOOLEAN DEFAULT FALSE,
                    category TEXT,
                    is_primary BOOLEAN DEFAULT FALSE,
                    supporting_text TEXT,
                    feature_importance TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES processing_jobs (id)
                )
            ''')

            # Audit logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    job_id TEXT,
                    action TEXT NOT NULL,
                    details TEXT,
                    user_id TEXT DEFAULT 'system',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES processing_jobs (id)
                )
            ''')

            # File cleanup tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_cleanup (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    cleanup_reason TEXT NOT NULL,
                    cleanup_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER
                )
            ''')

            # Processing statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_stats (
                    id TEXT PRIMARY KEY,
                    date DATE NOT NULL,
                    total_files_processed INTEGER DEFAULT 0,
                    total_cpt_codes_generated INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    avg_processing_time REAL DEFAULT 0.0,
                    errors_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_created ON processing_jobs (created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_job ON cpt_results (job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_code ON cpt_results (cpt_code)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_job ON audit_logs (job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cleanup_date ON file_cleanup (cleanup_at)')

            conn.commit()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dictionary-like access
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def create_processing_job(self, file_name: str, file_path: str,
                            file_size: int, metadata: Optional[Dict] = None) -> str:
        """Create a new processing job."""
        job_id = str(uuid.uuid4())

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO processing_jobs
                (id, file_name, file_path, file_size, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (job_id, file_name, file_path, file_size, 'pending',
                  json.dumps(metadata) if metadata else None))

            # Log job creation
            self.create_audit_log(job_id, 'job_created', {
                'file_name': file_name,
                'file_size': file_size
            })

            conn.commit()

        logger.info(f"Created processing job: {job_id} for file: {file_name}")
        return job_id

    def update_job_status(self, job_id: str, status: str,
                         error_message: Optional[str] = None,
                         processing_time: Optional[int] = None,
                         text_length: Optional[int] = None,
                         quality_score: Optional[float] = None):
        """Update processing job status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Build update query based on provided parameters
            update_fields = ['status = ?']
            update_values = [status]

            if error_message is not None:
                update_fields.append('error_message = ?')
                update_values.append(error_message)

            if processing_time is not None:
                update_fields.append('processing_time_seconds = ?')
                update_values.append(processing_time)

            if text_length is not None:
                update_fields.append('text_length = ?')
                update_values.append(text_length)

            if quality_score is not None:
                update_fields.append('quality_score = ?')
                update_values.append(quality_score)

            # Add timestamps based on status
            if status == 'processing':
                update_fields.append('started_at = CURRENT_TIMESTAMP')
            elif status in ['completed', 'failed']:
                update_fields.append('completed_at = CURRENT_TIMESTAMP')

            update_values.append(job_id)

            cursor.execute(f'''
                UPDATE processing_jobs
                SET {', '.join(update_fields)}
                WHERE id = ?
            ''', update_values)

            # Log status change
            self.create_audit_log(job_id, 'status_updated', {
                'new_status': status,
                'error_message': error_message
            })

            conn.commit()

    def save_cpt_results(self, job_id: str, cpt_codes: List[Dict[str, Any]]):
        """Save CPT code results for a processing job."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            for i, code_data in enumerate(cpt_codes):
                result_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO cpt_results
                    (id, job_id, cpt_code, description, confidence_score,
                     explanation, model_used, requires_review, category,
                     is_primary, supporting_text, feature_importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id,
                    job_id,
                    code_data['cpt_code'],
                    code_data['description'],
                    code_data['confidence_score'],
                    code_data.get('explanation'),
                    code_data.get('model_used'),
                    code_data.get('requires_review', False),
                    code_data.get('category'),
                    code_data.get('is_primary', i == 0),  # First code is primary
                    json.dumps(code_data.get('supporting_text', [])),
                    json.dumps(code_data.get('feature_importance', {}))
                ))

            # Log results saved
            self.create_audit_log(job_id, 'cpt_results_saved', {
                'num_codes': len(cpt_codes),
                'avg_confidence': sum(c['confidence_score'] for c in cpt_codes) / len(cpt_codes)
            })

            conn.commit()

    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get processing job and associated CPT results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get job details
            cursor.execute('SELECT * FROM processing_jobs WHERE id = ?', (job_id,))
            job = cursor.fetchone()

            if not job:
                return None

            # Get CPT results
            cursor.execute('''
                SELECT * FROM cpt_results
                WHERE job_id = ?
                ORDER BY is_primary DESC, confidence_score DESC
            ''', (job_id,))
            results = cursor.fetchall()

            return {
                'job': dict(job),
                'results': [dict(result) for result in results]
            }

    def get_processing_jobs(self, status: Optional[str] = None,
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get processing jobs with optional status filter."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = 'SELECT * FROM processing_jobs'
            params = []

            if status:
                query += ' WHERE status = ?'
                params.append(status)

            query += ' ORDER BY created_at DESC'

            if limit:
                query += ' LIMIT ?'
                params.append(limit)

            cursor.execute(query, params)
            return [dict(job) for job in cursor.fetchall()]

    def get_cpt_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get processing statistics for the last N days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=days)

            # Job statistics
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM processing_jobs
                WHERE created_at >= ?
                GROUP BY status
            ''', (cutoff_date.isoformat(),))
            job_stats = dict(cursor.fetchall())

            # CPT code statistics
            cursor.execute('''
                SELECT COUNT(*) as total_codes,
                       AVG(confidence_score) as avg_confidence,
                       MAX(confidence_score) as max_confidence
                FROM cpt_results cr
                JOIN processing_jobs pj ON cr.job_id = pj.id
                WHERE pj.created_at >= ?
            ''', (cutoff_date.isoformat(),))
            cpt_stats = dict(cursor.fetchone())

            # Average processing time
            cursor.execute('''
                SELECT AVG(processing_time_seconds) as avg_time
                FROM processing_jobs
                WHERE created_at >= ? AND processing_time_seconds IS NOT NULL
            ''', (cutoff_date.isoformat(),))
            time_stats = dict(cursor.fetchone())

            return {
                'job_stats': job_stats,
                'cpt_stats': cpt_stats,
                'avg_processing_time': time_stats.get('avg_time', 0)
            }

    def create_audit_log(self, job_id: Optional[str], action: str,
                        details: Optional[Dict[str, Any]] = None,
                        user_id: str = 'system'):
        """Create an audit log entry."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            log_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO audit_logs (id, job_id, action, details, user_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                log_id,
                job_id,
                action,
                json.dumps(details) if details else None,
                user_id
            ))

            conn.commit()

    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Clean up data older than specified days for HIPAA compliance."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff_date.isoformat()

        deleted_counts = {}

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get old jobs to delete
            cursor.execute('''
                SELECT id, file_path FROM processing_jobs
                WHERE created_at < ? AND status IN ('completed', 'failed')
            ''', (cutoff_iso,))
            old_jobs = cursor.fetchall()

            if old_jobs:
                job_ids = [job[0] for job in old_jobs]
                file_paths = [job[1] for job in old_jobs]

                # Delete CPT results
                cursor.execute(f'''
                    DELETE FROM cpt_results
                    WHERE job_id IN ({','.join(['?'] * len(job_ids))})
                ''', job_ids)
                deleted_counts['cpt_results'] = cursor.rowcount

                # Delete audit logs
                cursor.execute(f'''
                    DELETE FROM audit_logs
                    WHERE job_id IN ({','.join(['?'] * len(job_ids))})
                ''', job_ids)
                deleted_counts['audit_logs'] = cursor.rowcount

                # Delete jobs
                cursor.execute(f'''
                    DELETE FROM processing_jobs
                    WHERE id IN ({','.join(['?'] * len(job_ids))})
                ''', job_ids)
                deleted_counts['processing_jobs'] = cursor.rowcount

                # Log file cleanup
                for file_path in file_paths:
                    cleanup_id = str(uuid.uuid4())
                    cursor.execute('''
                        INSERT INTO file_cleanup (id, file_path, cleanup_reason, file_size)
                        VALUES (?, ?, ?, ?)
                    ''', (cleanup_id, file_path, 'auto_deletion', 0))

            # Clean old audit logs (keep only system logs)
            cursor.execute('''
                DELETE FROM audit_logs
                WHERE timestamp < ? AND job_id IS NOT NULL
            ''', (cutoff_iso,))
            deleted_counts['old_audit_logs'] = cursor.rowcount

            conn.commit()

        logger.info(f"Cleaned up old data: {deleted_counts}")
        return deleted_counts

    def close(self):
        """Close database connection."""
        # SQLite connections are closed automatically in get_connection context
        logger.info("Database connections closed")

    def backup_database(self, backup_path: str):
        """Create a backup of the database."""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")