"""
Test suite for Schwafit core functionality.
Tests the Schwafitting antifragile validation and partitioning system.
"""

import unittest
import numpy as np
from typing import List, Dict, Any
import tempfile
import os
import yaml
from pathlib import Path

from ..schwafit_core import SchwafitManager
from ..config import (
    load_yaml_config,
    validate_config,
    STRATEGIES_CONFIG_SCHEMA,
    ConfigValidationError
)

class TestSchwafitCore(unittest.TestCase):
    """Test cases for Schwafit core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test configs
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.test_dir.name) / 'config'
        self.config_dir.mkdir(exist_ok=True)
        
        # Create a test config file
        self.test_config = {
            'meta_tag': 'test',
            'fallback_matrix': 'test_fallback',
            'scoring': {
                'hash_weight': 0.3,
                'volume_weight': 0.2,
                'drift_weight': 0.4,
                'error_weight': 0.1
            }
        }
        self.config_path = self.config_dir / 'test_strategies.yaml'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)

        # Initialize SchwafitManager with test parameters
        self.manager = SchwafitManager(
            min_ratio=0.1,
            max_ratio=0.5,
            cycle_period=100,
            noise_scale=0.01
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.test_dir.cleanup()

    def test_config_loading(self):
        """Test YAML configuration loading."""
        config = load_yaml_config('test_strategies.yaml', schema=STRATEGIES_CONFIG_SCHEMA)
        self.assertEqual(config['meta_tag'], 'test')
        self.assertEqual(config['fallback_matrix'], 'test_fallback')
        self.assertIn('scoring', config)

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        validate_config(self.test_config, STRATEGIES_CONFIG_SCHEMA)
        
        # Test invalid config
        invalid_config = self.test_config.copy()
        del invalid_config['meta_tag']
        with self.assertRaises(ConfigValidationError):
            validate_config(invalid_config, STRATEGIES_CONFIG_SCHEMA)

    def test_default_config_generation(self):
        """Test default configuration generation."""
        default_config = load_yaml_config(
            'default_strategies.yaml',
            schema=STRATEGIES_CONFIG_SCHEMA
        )
        self.assertEqual(default_config['meta_tag'], 'default')
        self.assertIn('scoring', default_config)

    def test_dynamic_holdout_ratio(self):
        """Test dynamic holdout ratio calculation."""
        ratios = []
        for _ in range(100):
            ratio = self.manager.dynamic_holdout_ratio()
            ratios.append(ratio)
            self.assertGreaterEqual(ratio, self.manager.min_ratio)
            self.assertLessEqual(ratio, self.manager.max_ratio)
        
        # Test that ratios vary (not constant)
        self.assertGreater(np.std(ratios), 0)

    def test_data_splitting(self):
        """Test data partitioning functionality."""
        data = list(range(100))
        visible, holdout = self.manager.split_data(data)
        
        # Test basic properties
        self.assertEqual(len(visible) + len(holdout), len(data))
        self.assertGreater(len(visible), 0)
        self.assertGreater(len(holdout), 0)
        
        # Test with shell states
        shell_states = [f"shell_{i}" for i in range(100)]
        (visible_data, visible_shells), (holdout_data, holdout_shells) = self.manager.split_data(data, shell_states)
        
        # Test shell state alignment
        self.assertEqual(len(visible_data), len(visible_shells))
        self.assertEqual(len(holdout_data), len(holdout_shells))
        
        # Test data integrity
        all_data = visible_data + holdout_data
        all_shells = visible_shells + holdout_shells
        self.assertEqual(set(all_data), set(data))
        self.assertEqual(set(all_shells), set(shell_states)) 