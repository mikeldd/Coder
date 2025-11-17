"""
File manager for the Medical CPT Coding Application.
Handles file validation, secure storage, and HIPAA-compliant auto-deletion.
"""

import os
import hashlib
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
from cryptography.fernet import Fernet

from ..utils.config import Config
from .database import Database

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file operations with HIPAA compliance and security."""

    def __init__(self, config: Config):
        """Initialize file manager."""
        self.config = config
        self.db = Database(config)
        self.temp_dir = Path(config.get_temp_directory())
        self.secure_temp_dir = self.temp_dir / "secure"

        # Ensure directories exist
        self.temp_dir.mkdir(exist_ok=True)
        self.secure_temp_dir.mkdir(exist_ok=True, mode=0o700)  # Restricted permissions

        # Initialize encryption if enabled
        self.cipher_suite = None
        if config.get("hipaa.enable_encryption", True) and hasattr(config, 'cipher_suite'):
            self.cipher_suite = config.cipher_suite

        logger.info("File manager initialized with secure storage")

    def validate_pdf_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate PDF file for medical record processing."""
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }

        try:
            path = Path(file_path)

            # Check if file exists
            if not path.exists():
                validation_result['errors'].append("File does not exist")
                return False, validation_result

            # Get file info
            file_stat = path.stat()
            file_size = file_stat.st_size
            validation_result['file_info'] = {
                'size': file_size,
                'name': path.name,
                'extension': path.suffix.lower()
            }

            # Check file extension
            if path.suffix.lower() != '.pdf':
                validation_result['errors'].append("File must be a PDF")
                return False, validation_result

            # Check file size (max 50MB as per config)
            max_size = self.config.get("app.max_file_size_mb", 50) * 1024 * 1024
            if file_size > max_size:
                validation_result['errors'].append(f"File size exceeds {max_size // (1024*1024)}MB limit")
                return False, validation_result

            # Check minimum file size
            if file_size < 1024:  # 1KB minimum
                validation_result['errors'].append("File is too small to be a valid PDF")
                return False, validation_result

            # Validate PDF header
            try:
                with open(path, 'rb') as f:
                    header = f.read(5)
                    if not header.startswith(b'%PDF-'):
                        validation_result['errors'].append("Invalid PDF format - missing PDF header")
                        return False, validation_result
            except Exception as e:
                validation_result['errors'].append(f"Cannot read file: {str(e)}")
                return False, validation_result

            # Calculate file hash for integrity
            validation_result['file_info']['hash'] = self._calculate_file_hash(path)

            validation_result['valid'] = True
            logger.info(f"File validation passed: {path.name}")
            return True, validation_result

        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"File validation error for {file_path}: {e}")
            return False, validation_result

    def secure_file_storage(self, source_path: str) -> Tuple[str, str]:
        """Store file securely with encryption and generate secure filename."""
        try:
            source = Path(source_path)
            secure_filename = self._generate_secure_filename(source.name)

            secure_path = self.secure_temp_dir / secure_filename

            if self.cipher_suite:
                # Encrypt the file
                with open(source, 'rb') as src_file:
                    encrypted_data = self.cipher_suite.encrypt(src_file.read())
                with open(secure_path, 'wb') as dest_file:
                    dest_file.write(encrypted_data)
            else:
                # Copy without encryption
                shutil.copy2(source, secure_path)

            # Set secure permissions
            os.chmod(secure_path, 0o600)  # Read/write for owner only

            logger.info(f"File stored securely: {secure_filename}")
            return str(secure_path), secure_filename

        except Exception as e:
            logger.error(f"Error storing file securely: {e}")
            raise

    def _generate_secure_filename(self, original_name: str) -> str:
        """Generate a secure filename to prevent path traversal."""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        extension = Path(original_name).suffix.lower()
        return f"{timestamp}_{unique_id}{extension}"

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for integrity verification."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""

    def batch_file_processing(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files and return processing jobs."""
        jobs = []

        for file_path in file_paths:
            try:
                # Validate file
                is_valid, validation_result = self.validate_pdf_file(file_path)
                if not is_valid:
                    logger.warning(f"Skipping invalid file: {file_path} - {validation_result['errors']}")
                    continue

                # Store file securely
                secure_path, secure_name = self.secure_file_storage(file_path)

                # Create processing job
                job_id = self.db.create_processing_job(
                    file_name=validation_result['file_info']['name'],
                    file_path=secure_path,
                    file_size=validation_result['file_info']['size'],
                    metadata={
                        'original_path': file_path,
                        'secure_name': secure_name,
                        'file_hash': validation_result['file_info'].get('hash'),
                        'validation_result': validation_result
                    }
                )

                jobs.append({
                    'job_id': job_id,
                    'original_path': file_path,
                    'secure_path': secure_path,
                    'file_name': validation_result['file_info']['name'],
                    'file_size': validation_result['file_info']['size']
                })

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

        logger.info(f"Created {len(jobs)} processing jobs")
        return jobs

    def get_file_content(self, secure_path: str) -> bytes:
        """Retrieve and decrypt file content."""
        try:
            path = Path(secure_path)
            if not path.exists():
                raise FileNotFoundError(f"Secure file not found: {secure_path}")

            with open(path, 'rb') as f:
                encrypted_data = f.read()

            if self.cipher_suite:
                return self.cipher_suite.decrypt(encrypted_data)
            else:
                return encrypted_data

        except Exception as e:
            logger.error(f"Error retrieving file content: {e}")
            raise

    def delete_file(self, secure_path: str, job_id: Optional[str] = None) -> bool:
        """Securely delete a file with multiple passes."""
        try:
            path = Path(secure_path)
            if not path.exists():
                return True  # File already doesn't exist

            # Secure deletion with multiple passes (for sensitive data)
            if self.config.get("hipaa.enable_encryption", True):
                file_size = path.stat().st_size
                passes = 3  # Triple overwrite for secure deletion

                with open(path, 'r+b') as f:
                    for _ in range(passes):
                        # Overwrite with random data
                        random_data = os.urandom(file_size)
                        f.seek(0)
                        f.write(random_data)
                        f.flush()
                        os.fsync(f.fileno())

            # Remove the file
            path.unlink()

            # Log deletion
            if job_id:
                self.db.create_audit_log(job_id, 'file_deleted', {
                    'file_path': secure_path,
                    'deletion_method': 'secure_multi_pass'
                })

            logger.info(f"Securely deleted file: {secure_path}")
            return True

        except Exception as e:
            logger.error(f"Error deleting file {secure_path}: {e}")
            return False

    def auto_delete_old_files(self) -> int:
        """Automatically delete files older than threshold."""
        if not self.config.should_auto_delete():
            return 0

        try:
            # Get old jobs from database
            old_days = self.config.get_auto_delete_days()
            deleted_count = 0

            # Use database cleanup method
            deleted_data = self.db.cleanup_old_data(old_days)

            # Clean up any orphaned files in temp directory
            cutoff_date = datetime.now() - timedelta(days=old_days)
            for file_path in self.secure_temp_dir.iterdir():
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        if self.delete_file(str(file_path)):
                            deleted_count += 1

            # Clean old temp files
            self._cleanup_temp_files()

            total_deleted = deleted_count + sum(deleted_data.values())
            logger.info(f"Auto-deleted {total_deleted} items (files + database records)")
            return total_deleted

        except Exception as e:
            logger.error(f"Error in auto-deletion: {e}")
            return 0

    def _cleanup_temp_files(self):
        """Clean up temporary files and directories."""
        try:
            # Clean old temp files older than 1 day
            cutoff_date = datetime.now() - timedelta(days=1)

            for temp_file in self.temp_dir.rglob('*'):
                if temp_file.is_file() and temp_file.parent.name != 'secure':
                    file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        try:
                            temp_file.unlink()
                        except:
                            pass

        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")

    def clear_temp_files(self) -> int:
        """Clear all temporary files (manual cleanup)."""
        deleted_count = 0

        try:
            # Delete all files in secure temp directory
            for file_path in self.secure_temp_dir.iterdir():
                if file_path.is_file():
                    if self.delete_file(str(file_path)):
                        deleted_count += 1

            # Clean other temp files
            self._cleanup_temp_files()

            logger.info(f"Manually cleared {deleted_count} temporary files")
            return deleted_count

        except Exception as e:
            logger.error(f"Error clearing temp files: {e}")
            return 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            total_size = 0
            file_count = 0

            for file_path in self.secure_temp_dir.iterdir():
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

            return {
                'total_files': file_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'temp_directory': str(self.temp_dir),
                'auto_delete_days': self.config.get_auto_delete_days(),
                'auto_delete_enabled': self.config.should_auto_delete()
            }

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}

    def create_backup(self, backup_path: str) -> bool:
        """Create encrypted backup of secure files."""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy secure files to backup
            for file_path in self.secure_temp_dir.iterdir():
                if file_path.is_file():
                    backup_file = backup_dir / file_path.name
                    shutil.copy2(file_path, backup_file)

            # Backup database
            db_backup = backup_dir / "database_backup.db"
            self.db.backup_database(str(db_backup))

            logger.info(f"Backup created at: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False