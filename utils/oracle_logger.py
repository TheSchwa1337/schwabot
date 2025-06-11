"""
Oracle Logger

Provides unified logging for all Oracle-related components with consistent formatting
and output channels.
"""

import logging
from pathlib import Path
import yaml

# Create logger
logger = logging.getLogger("oracle")
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
formatter = logging.Formatter('[ORACLE] %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Create file handler
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
fh = logging.FileHandler(log_dir / "oracle.log")
fh.setFormatter(formatter)
logger.addHandler(fh)

def load_oracle_config():
    """Load Oracle configuration from YAML file"""
    config_path = Path(__file__).resolve().parent.parent / "config/oracle_config.yaml"
    if not config_path.exists():
        logger.error(f"Missing Oracle config at {config_path}")
        return None
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required sections
        required_sections = ['manifold', 'topology', 'quantum', 'encryption']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section '{section}' in Oracle config")
                return None
                
        return config
    except Exception as e:
        logger.error(f"Failed to load Oracle config: {str(e)}")
        return None 