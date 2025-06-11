"""
Centralized Configuration System

Provides a unified interface for loading and validating YAML configurations
across the Schwabot system.
"""

import yaml
from pathlib import Path
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("config")

class ConfigError(Exception):
    """Base exception for configuration errors"""
    pass

class ConfigNotFoundError(ConfigError):
    """Raised when a configuration file is not found"""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails"""
    pass

def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """
    Load and validate a YAML configuration file.
    
    Args:
        config_name: Name of the configuration file (e.g., 'oracle_config.yaml')
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        ConfigNotFoundError: If the configuration file is not found
        ConfigValidationError: If the configuration is invalid
    """
    try:
        # Construct path to config file
        config_path = Path(__file__).parent.parent / "config" / config_name
        
        if not config_path.exists():
            raise ConfigNotFoundError(f"Configuration file not found: {config_path}")
            
        # Load YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if not isinstance(config, dict):
            raise ConfigValidationError(f"Invalid configuration format in {config_name}")
            
        logger.info(f"Successfully loaded configuration: {config_name}")
        return config
        
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"YAML parsing error in {config_name}: {str(e)}")
    except Exception as e:
        raise ConfigError(f"Failed to load configuration {config_name}: {str(e)}")

def validate_config(config: Dict[str, Any], required_sections: Optional[list] = None) -> bool:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_sections: List of required top-level sections
        
    Returns:
        True if validation passes
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ConfigValidationError("Configuration must be a dictionary")
        
    if required_sections:
        missing = [section for section in required_sections if section not in config]
        if missing:
            raise ConfigValidationError(f"Missing required sections: {missing}")
            
    return True

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Safely retrieve a value from a nested configuration dictionary.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., 'oracle.min_coherence')
        default: Default value if the key is not found
        
    Returns:
        The configuration value or default
    """
    try:
        value = config
        for key in key_path.split('.'):
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default 