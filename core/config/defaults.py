"""
Default configurations for Schwabot system.
Contains default values for all configuration files.
"""

from typing import Dict, Any

DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'matrix_response_paths.yaml': {
        'safe': 'hold',
        'warn': 'delay_entry',
        'fail': 'matrix_realign',
        'ZPE-risk': 'cooldown_abort'
    },
    'validation_config.yaml': {
        'validation': {
            'enabled': True,
            'max_retries': 3,
            'timeout': 30
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'strategies.yaml': {
        'meta_tag': 'default',
        'fallback_matrix': 'default_fallback',
        'scoring': {
            'hash_weight': 0.3,
            'volume_weight': 0.2,
            'drift_weight': 0.4,
            'error_weight': 0.1
        }
    },
    'default_strategies.yaml': {
        'meta_tag': 'default',
        'fallback_matrix': 'default_fallback',
        'scoring': {
            'hash_weight': 0.3,
            'volume_weight': 0.2,
            'drift_weight': 0.4,
            'error_weight': 0.1
        }
    }
}

def ensure_configs_exist(config_loader):
    """
    Ensure all required configurations exist in the config directory.
    If a file does not exist, a default version will be created.
    
    Args:
        config_loader: An instance of ConfigLoader.
    """
    for filename, default_content in DEFAULT_CONFIGS.items():
        config_path = config_loader.config_dir / filename
        if not config_path.exists():
            try:
                config_loader.save_yaml(filename, default_content)
                logger.info(f"Created default config file: {filename}")
            except Exception as e:
                logger.error(f"Failed to create default config file {filename}: {e}")
        else:
            logger.debug(f"Config file already exists: {filename}") 