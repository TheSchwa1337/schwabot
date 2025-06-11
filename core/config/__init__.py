"""
Configuration Management
======================

Centralized configuration loading and validation for the core system.
Handles YAML file loading with proper path resolution and default config generation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Generic
import logging
import os
from dataclasses import dataclass
import json
from .defaults import DEFAULT_CONFIGS

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ConfigError(Exception):
    """Base exception for configuration errors"""
    pass

class ConfigNotFoundError(ConfigError):
    """Raised when a required config file is not found"""
    pass

class ConfigValidationError(ConfigError):
    """Raised when config validation fails"""
    pass

@dataclass
class ConfigSchema(Generic[T]):
    """Schema for validating configuration"""
    required_fields: Dict[str, type]
    default_values: Dict[str, Any]
    validator: Optional[callable] = None

class ConfigLoader:
    """Centralized configuration loader with repository-relative paths."""
    _instance = None
    _config_dir: Optional[Path] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            if cls._config_dir is None:
                # Assumes this file is in core/config/
                cls.repo_root = Path(__file__).resolve().parent.parent
                cls._config_dir = cls.repo_root / 'config'
                cls._config_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance

    def __init__(self):
        """Initialize loader and ensure default configs exist."""
        if self._initialized:
            return
        try:
            from .defaults import ensure_configs_exist
            ensure_configs_exist(self)
        except Exception as e:
            logger.error(f"Failed to ensure default configs: {e}")
        self._initialized = True

    @property
    def config_dir(self) -> Path:
        """Get the configuration directory path"""
        return self.__class__._config_dir

    def load_yaml(self, filename: str, create_default: bool = True) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            filename: Name of the configuration file
            create_default: Whether to create default config if not found
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            ConfigNotFoundError: If the configuration file is not found
            ConfigError: If there is an error parsing the YAML
        """
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if config_data is None:
                    logger.warning(f"Config file {filename} is empty. Returning empty dict.")
                    return {}
                return config_data
        except FileNotFoundError:
            if create_default:
                logger.warning(f"Config '{filename}' not found, attempting to create default.")
                default_config = DEFAULT_CONFIGS.get(filename)
                if default_config:
                    self.save_yaml(filename, default_config)
                    return default_config
            raise ConfigNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filename}: {e}")
            raise ConfigError(f"Malformed YAML file: {config_path}") from e

    def save_yaml(self, filename: str, config: Dict[str, Any]) -> None:
        """
        Save a configuration to YAML file.
        
        Args:
            filename: Name of the configuration file
            config: Configuration dictionary to save
            
        Raises:
            ConfigError: If there is an error saving the file
        """
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving YAML file {filename}: {e}")
            raise ConfigError(f"Failed to save configuration: {config_path}") from e

    def validate_config(self, config: Dict[str, Any], schema: ConfigSchema) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: Schema to validate against
            
        Returns:
            True if validation passes
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Check required fields
        for field, field_type in schema.required_fields.items():
            if field not in config:
                raise ConfigValidationError(f"Missing required field: {field}")
            if not isinstance(config[field], field_type):
                raise ConfigValidationError(f"Invalid type for field {field}: expected {field_type}")
                
        # Run custom validator if provided
        if schema.validator:
            schema.validator(config)
            
        return True

def load_yaml_config(config_name: str, schema: Optional[ConfigSchema] = None,
                    create_default: bool = True) -> Dict[str, Any]:
    """
    Load and validate a YAML configuration file.
    
    Args:
        config_name: Name of the configuration file
        schema: Optional schema for validation
        create_default: Whether to create default config if not found
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        ConfigNotFoundError: If the configuration file is not found
        ConfigValidationError: If the configuration is invalid
    """
    try:
        config_path = Path(__file__).parent.parent.parent / 'config' / config_name
        
        if not config_path.exists():
            if create_default:
                default_config = DEFAULT_CONFIGS.get(config_name)
                if default_config is None:
                    raise ConfigNotFoundError(f"No default config for {config_name}")
                    
                if schema:
                    validate_config(default_config, schema)
                    
                # Create config directory if it doesn't exist
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write default config
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                    
                return default_config
            else:
                raise ConfigNotFoundError(f"Config file {config_name} not found")
                
        # Load existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate if schema provided
        if schema:
            validate_config(config, schema)
            
        return config
        
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing {config_name}: {str(e)}")

def validate_config(config: Dict[str, Any], schema: ConfigSchema) -> bool:
    """
    Validate a configuration dictionary against a schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema to validate against
        
    Returns:
        True if validation passes
        
    Raises:
        ConfigValidationError: If validation fails
    """
    # Check required fields
    for field, field_type in schema.required_fields.items():
        if field not in config:
            raise ConfigValidationError(f"Missing required field: {field}")
        if not isinstance(config[field], field_type):
            raise ConfigValidationError(f"Invalid type for field {field}: expected {field_type}")
            
    # Run custom validator if provided
    if schema.validator:
        schema.validator(config)
        
    return True

# Define schemas for known config files
MATRIX_RESPONSE_SCHEMA = ConfigSchema(
    required_fields={
        'safe': str,
        'warn': str,
        'fail': str,
        'ZPE-risk': str
    },
    default_values={
        'safe': 'hold',
        'warn': 'delay_entry',
        'fail': 'matrix_realign',
        'ZPE-risk': 'cooldown_abort'
    }
)

VALIDATION_CONFIG_SCHEMA = ConfigSchema(
    required_fields={
        'validation': dict,
        'logging': dict
    },
    default_values={
        'validation': {
            'enabled': True,
            'max_retries': 3,
            'timeout': 30
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
)

# Schema for Schwafit strategies configuration
STRATEGIES_CONFIG_SCHEMA = ConfigSchema(
    required_fields={
        'meta_tag': str,
        'fallback_matrix': str,
        'scoring': dict
    },
    default_values={
        'meta_tag': 'default',
        'fallback_matrix': 'default_fallback',
        'scoring': {
            'hash_weight': 0.3,
            'volume_weight': 0.2,
            'drift_weight': 0.4,
            'error_weight': 0.1
        }
    }
)

# Initialize a default logger for this module if not already set up by a main logging config
if not logging.getLogger(__name__).handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 