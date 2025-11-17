"""
Configuration management for the Medical CPT Coding Application.
Handles application settings, user preferences, and system configuration.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the application."""

    def __init__(self):
        """Initialize configuration with default values."""
        self.app_dir = Path.home() / '.medical_cpt_coder'
        self.config_file = self.app_dir / 'config.json'
        self.encryption_key_file = self.app_dir / 'encryption.key'

        # Ensure app directory exists
        self.app_dir.mkdir(exist_ok=True)

        # Load or create configuration
        self.config = self._load_config()

        # Initialize encryption for HIPAA compliance
        self._init_encryption()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        default_config = {
            "app": {
                "theme": "light",
                "auto_delete_days": 30,
                "max_file_size_mb": 50,
                "batch_processing": True,
                "confidence_threshold": 0.7
            },
            "processing": {
                "use_ocr_fallback": True,
                "ocr_language": "eng",
                "max_concurrent_jobs": 2,
                "timeout_seconds": 300
            },
            "models": {
                "ensemble_weights": {
                    "random_forest": 0.3,
                    "svm": 0.25,
                    "gradient_boosting": 0.25,
                    "lstm": 0.2
                },
                "confidence_threshold": 0.7,
                "top_k_codes": 5
            },
            "hipaa": {
                "enable_encryption": True,
                "auto_delete_enabled": True,
                "audit_logging": True,
                "minimal_phi_logging": True
            },
            "paths": {
                "temp_dir": str(self.app_dir / "temp"),
                "database": str(self.app_dir / "cpt_coder.db"),
                "models_dir": str(Path(__file__).parent.parent.parent / "data" / "models"),
                "cpt_codes_file": str(Path(__file__).parent.parent.parent / "data" / "codes" / "cpt_2021.json")
            }
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to handle new config options
                    return self._merge_configs(default_config, loaded_config)
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                logger.info("Using default configuration")
                return default_config
        else:
            # Create default config file
            self._save_config(default_config)
            return default_config

    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge loaded config with defaults."""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def _save_config(self, config: Optional[Dict[str, Any]] = None):
        """Save configuration to file."""
        if config is None:
            config = self.config

        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config file: {e}")

    def _init_encryption(self):
        """Initialize encryption for HIPAA compliance."""
        if self.config["hipaa"]["enable_encryption"]:
            if self.encryption_key_file.exists():
                with open(self.encryption_key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                self.encryption_key = Fernet.generate_key()
                with open(self.encryption_key_file, 'wb') as f:
                    f.write(self.encryption_key)

            self.cipher_suite = Fernet(self.encryption_key)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'app.theme')."""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        self._save_config()

    def get_theme(self) -> str:
        """Get current theme setting."""
        return self.get("app.theme", "light")

    def set_theme(self, theme: str):
        """Set theme (light or dark)."""
        if theme in ["light", "dark"]:
            self.set("app.theme", theme)

    def get_database_path(self) -> str:
        """Get database file path."""
        return self.get("paths.database")

    def get_models_directory(self) -> str:
        """Get models directory path."""
        return self.get("paths.models_dir")

    def get_cpt_codes_path(self) -> str:
        """Get CPT codes file path."""
        return self.get("paths.cpt_codes_file")

    def get_temp_directory(self) -> str:
        """Get temporary directory path."""
        temp_dir = self.get("paths.temp_dir")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data for HIPAA compliance."""
        if self.config["hipaa"]["enable_encryption"] and hasattr(self, 'cipher_suite'):
            return self.cipher_suite.encrypt(data)
        return data

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data for HIPAA compliance."""
        if self.config["hipaa"]["enable_encryption"] and hasattr(self, 'cipher_suite'):
            return self.cipher_suite.decrypt(encrypted_data)
        return encrypted_data

    def should_auto_delete(self) -> bool:
        """Check if auto-deletion is enabled."""
        return self.get("hipaa.auto_delete_enabled", True)

    def get_auto_delete_days(self) -> int:
        """Get auto-deletion threshold in days."""
        return self.get("app.auto_delete_days", 30)