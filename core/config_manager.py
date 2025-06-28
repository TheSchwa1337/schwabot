# -*- coding: utf-8 -*-
"""
Configuration Manager for Mathematical Trading System

This module provides comprehensive configuration management for the Schwabot
mathematical trading system. It includes parameter validation, optimization,
and mathematical constraint handling.

MATHEMATICAL PRESERVATION NOTES:
- Risk tolerance calculations (0.0-1.0 range with mathematical constraints)
- Position size optimization algorithms
- Precision control for mathematical operations
- Iteration limits for convergence algorithms
- Validation scoring with weighted mathematical metrics
- Performance optimization using mathematical scaling factors
"""

import copy
import hashlib
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
import time
import os
import yaml
import json
import logging
from dual_unicore_handler import DualUnicoreHandler

from core.unified_math_system import unified_math
from utils.safe_print import safe_print, info, warn, error, success, debug
import numpy as np
from numpy.typing import NDArray

# Initialize Unicode handler
unicore = DualUnicoreHandler()

# Configuration format constants
JSON = "json"
YAML = "yaml"
ENV = "env"
INI = "ini"
TOML = "toml"


class ValidationLevel(Enum):
    """Validation levels for mathematical parameter validation."""
    STRICT = "strict"    # Mathematical precision validation
    NORMAL = "normal"    # Standard validation
    RELAXED = "relaxed"  # Loose validation for testing


@dataclass
class ConfigParameter:
    """
    Configuration parameter with mathematical validation support.

    MATHEMATICAL PRESERVATION: This structure contains mathematical constraints
    including min/max values, precision requirements, and validation rules
    that are essential for trading algorithm accuracy.
    """
    parameter_id: str
    value: Any
    data_type: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigUpdate:
    """Configuration update tracking with mathematical change validation."""
    update_id: str
    parameter_id: str
    old_value: Any
    new_value: Any
    update_reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigValidation:
    """
    Configuration validation result with mathematical scoring.

    MATHEMATICAL PRESERVATION: The validation_score is calculated using
    weighted mathematical metrics that determine configuration quality
    for trading system performance.
    """
    validation_id: str
    config_hash: str
    validation_score: float  # Mathematical score 0.0-1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """
    Mathematical Configuration Manager for Trading System

    Handles configuration management with mathematical validation,
    optimization algorithms, and performance tuning for trading operations.

    MATHEMATICAL PRESERVATION: This class contains critical mathematical
    logic for:
    - Risk tolerance optimization (0.0-1.0 scaling)
    - Position size calculations with safety constraints
    - API rate limiting with mathematical scaling
    - Precision control for mathematical operations
    - Iteration limits for convergence algorithms
    - Validation scoring with weighted metrics
    """

    def __init__(self, config_path: str = "./config", default_config: str = "schwabot_config.json"):
        """Initialize the configuration manager with mathematical defaults."""
        self.config_path = config_path
        self.default_config_file = os.path.join(config_path, default_config)
        self.logger = logging.getLogger(__name__)

        # Configuration storage
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self.parameters: Dict[str, ConfigParameter] = {}
        self.updates: Dict[str, ConfigUpdate] = {}
        self.validations: Dict[str, ConfigValidation] = {}

        # History tracking
        self.config_history: List[ConfigUpdate] = []
        self.validation_history: List[ConfigValidation] = []

        # Mathematical validation rules
        self.validation_rules: Dict[str, Dict[str, Any]] = {}

        # Initialize the manager
        self._initialize_manager()
        self._load_configuration()
        self.logger.info("Configuration Manager initialized")

    def _load_configuration(self) -> None:
        """Load configuration with mathematical parameter validation."""
        try:
            if os.path.exists(self.default_config_file):
                with open(self.default_config_file, 'r') as f:
                    self.configurations['default'] = json.load(f)
                self.logger.info(f"Loaded default configuration from {self.default_config_file}")
            else:
                self._create_default_configuration()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()

    def _create_default_configuration(self) -> None:
        """
        Create default configuration with mathematical trading parameters.

        MATHEMATICAL PRESERVATION: These default values contain critical
        mathematical constants for trading system operation:
        - Risk tolerance: 0.5 (50% risk level)
        - Position size: 0.1 (10% of portfolio)
        - Stop loss: 0.2 (20% loss threshold)
        - Precision: 0.1 (mathematical precision control)
        - Max iterations: 1000 (convergence limit)
        """
        default_config = {
            "system": {
                "name": "Schwabot",
                "version": "1.0.0",
                "environment": "development",
                "debug_mode": True
            },
            "trading": {
                "default_strategy": "conservative",
                "risk_tolerance": 0.5,  # Mathematical: 0.0-1.0 risk scale
                "max_position_size": 0.1,  # Mathematical: 0.0-1.0 position scale
                "stop_loss_percentage": 0.2  # Mathematical: loss threshold
            },
            "api": {
                "coinmarketcap_key": "",
                "coingecko_enabled": True,
                "rate_limit": 100,  # Mathematical: API call frequency limit
                "timeout": 30
            },
            "database": {
                "type": "sqlite",
                "path": "./data/schwabot.db",
                "backup_interval": 3600
            },
            "logging": {
                "level": "INFO",
                "file_path": "./logs/schwabot.log",
                "max_size": 10485760,
                "backup_count": 5
            },
            "mathematical": {
                "sfsss_enabled": True,
                "ufs_enabled": True,
                "precision": 0.1,  # Mathematical: calculation precision
                "max_iterations": 1000  # Mathematical: convergence limit
            }
        }

        self.configurations['default'] = default_config

        # Save default configuration
        try:
            os.makedirs(self.config_path, exist_ok=True)
            with open(self.default_config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Created default configuration at {self.default_config_file}")
        except Exception as e:
            self.logger.error(f"Error saving default configuration: {e}")

    def _initialize_manager(self) -> None:
        """Initialize manager components with mathematical validation setup."""
        self._initialize_parameter_registry()
        self._initialize_validation_rules()
        self._load_environment_configs()
        self.logger.info("Configuration manager initialized successfully")

    def _initialize_parameter_registry(self) -> None:
        """
        Initialize parameter registry with mathematical constraints.

        MATHEMATICAL PRESERVATION: These parameters define mathematical
        constraints and validation rules essential for trading algorithm
        accuracy and safety.
        """
        try:
            # Register system parameters
            self._register_parameter("system.name", "Schwabot", "string", {
                "required": True,
                "min_length": 1,
                "max_length": 50
            })

            self._register_parameter("system.version", "1.0.0", "string", {
                "required": True,
                "pattern": r"^\d+\.\d+\.\d+$"
            })

            # Mathematical trading parameters
            self._register_parameter("trading.risk_tolerance", 0.5, "float", {
                "required": True,
                "min_value": 0.0,  # Mathematical: minimum risk
                "max_value": 1.0,  # Mathematical: maximum risk
                "weight": 2.0  # Mathematical: importance weight
            })

            self._register_parameter("trading.max_position_size", 0.1, "float", {
                "required": True,
                "min_value": 0.1,  # Mathematical: minimum position
                "max_value": 1.0,  # Mathematical: maximum position
                "weight": 2.0  # Mathematical: importance weight
            })

            # API mathematical constraints
            self._register_parameter("api.rate_limit", 100, "integer", {
                "required": True,
                "min_value": 1,  # Mathematical: minimum rate
                "max_value": 10000,  # Mathematical: maximum rate
                "weight": 1.5  # Mathematical: importance weight
            })

            self.logger.info(f"Initialized parameter registry with {len(self.parameters)} parameters")
        except Exception as e:
            self.logger.error(f"Error initializing parameter registry: {e}")

    def _initialize_validation_rules(self) -> None:
        """
        Initialize validation rules with mathematical constraints.

        MATHEMATICAL PRESERVATION: These rules define mathematical validation
        logic for different data types, including precision requirements,
        range constraints, and pattern matching.
        """
        try:
            self.validation_rules = {
                "string": {
                    "required": True,
                    "min_length": 0,
                    "max_length": 1000,
                    "pattern": None
                },
                "integer": {
                    "required": True,
                    "min_value": None,
                    "max_value": None
                },
                "float": {
                    "required": True,
                    "min_value": None,
                    "max_value": None,
                    "precision": 6  # Mathematical: decimal precision
                },
                "boolean": {
                    "required": True
                },
                "array": {
                    "required": True,
                    "min_length": 0,
                    "max_length": 1000,
                    "element_type": None
                },
                "object": {
                    "required": True,
                    "required_fields": [],
                    "optional_fields": []
                }
            }
            self.logger.info("Validation rules initialized")
        except Exception as e:
            self.logger.error(f"Error initializing validation rules: {e}")

    def _load_environment_configs(self) -> None:
        """Load environment-specific configurations."""
        try:
            environment = os.getenv("SCHWABOT_ENV", "development")

            # Load environment-specific config
            env_config_file = os.path.join(self.config_path, f"schwabot_{environment}.json")
            if os.path.exists(env_config_file):
                with open(env_config_file, 'r') as f:
                    self.configurations[environment] = json.load(f)
                self.logger.info(f"Loaded {environment} configuration")

            # Load local overrides
            local_config_file = os.path.join(self.config_path, "schwabot_local.json")
            if os.path.exists(local_config_file):
                with open(local_config_file, 'r') as f:
                    self.configurations['local'] = json.load(f)
                self.logger.info("Loaded local configuration overrides")
        except Exception as e:
            self.logger.error(f"Error loading environment configs: {e}")

    def _register_parameter(self, parameter_id: str, default_value: Any, data_type: str,
                            validation_rules: Dict[str, Any]) -> None:
        """
        Register a parameter with mathematical validation rules.

        MATHEMATICAL PRESERVATION: This method creates parameter objects
        with mathematical constraints that are essential for trading
        algorithm validation and optimization.
        """
        try:
            parameter = ConfigParameter(
                parameter_id=parameter_id,
                value=default_value,
                data_type=data_type,
                description=f"Configuration parameter: {parameter_id}",
                timestamp=datetime.now(),
                validation_rules=validation_rules,
                metadata={
                    "registered": True,
                    "validation_level": ValidationLevel.NORMAL.value
                }
            )
            self.parameters[parameter_id] = parameter
        except Exception as e:
            self.logger.error(f"Error registering parameter {parameter_id}: {e}")

    def get_config(self, config_name: str = "default") -> Dict[str, Any]:
        """Get configuration with mathematical parameter resolution."""
        try:
            return self.configurations.get(config_name, {})
        except Exception as e:
            self.logger.error(f"Error getting config {config_name}: {e}")
            return {}

    def get_parameter(self, parameter_id: str) -> Optional[ConfigParameter]:
        """Get parameter with mathematical validation metadata."""
        try:
            return self.parameters.get(parameter_id)
        except Exception as e:
            self.logger.error(f"Error getting parameter {parameter_id}: {e}")
            return None

    def set_parameter(self, parameter_id: str, value: Any, update_reason: str = "manual") -> bool:
        """
        Set parameter with mathematical validation.

        MATHEMATICAL PRESERVATION: This method validates parameter changes
        using mathematical constraints and records the change for audit
        and rollback purposes.
        """
        try:
            if parameter_id not in self.parameters:
                self.logger.warning(f"Parameter {parameter_id} not registered")
                return False

            parameter = self.parameters[parameter_id]
            old_value = parameter.value

            # Validate the new value
            validation_result = self._validate_parameter_value(parameter_id, value)
            if not validation_result["valid"]:
                self.logger.error(f"Parameter validation failed: {validation_result['errors']}")
                return False

            # Update parameter value
            parameter.value = value
            parameter.timestamp = datetime.now()

            # Record update
            update_id = f"update_{int(time.time())}"
            config_update = ConfigUpdate(
                update_id=update_id,
                parameter_id=parameter_id,
                old_value=old_value,
                new_value=value,
                update_reason=update_reason,
                timestamp=datetime.now(),
                metadata={
                    "validation_result": validation_result
                }
            )

            self.updates[update_id] = config_update
            self.config_history.append(config_update)

            self.logger.info(f"Parameter {parameter_id} updated: {old_value} -> {value}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting parameter {parameter_id}: {e}")
            return False

    def _validate_parameter_value(self, parameter_id: str, value: Any) -> Dict[str, Any]:
        """
        Validate parameter value using mathematical constraints.

        MATHEMATICAL PRESERVATION: This method implements mathematical
        validation logic including range checking, precision validation,
        and type-specific mathematical constraints.
        """
        try:
            parameter = self.parameters.get(parameter_id)
            if not parameter:
                return {"valid": False, "errors": ["Parameter not registered"]}

            errors = []
            warnings = []

            # Check required
            if parameter.validation_rules.get("required", False) and value is None:
                errors.append("Parameter is required but value is None")

            # Type-specific validation
            if parameter.data_type == "string":
                validation_result = self._validate_string(value, parameter.validation_rules)
            elif parameter.data_type == "integer":
                validation_result = self._validate_integer(value, parameter.validation_rules)
            elif parameter.data_type == "float":
                validation_result = self._validate_float(value, parameter.validation_rules)
            elif parameter.data_type == "boolean":
                validation_result = self._validate_boolean(value, parameter.validation_rules)
            elif parameter.data_type == "array":
                validation_result = self._validate_array(value, parameter.validation_rules)
            elif parameter.data_type == "object":
                validation_result = self._validate_object(value, parameter.validation_rules)
            else:
                validation_result = {"valid": True, "errors": [], "warnings": []}

            errors.extend(validation_result.get("errors", []))
            warnings.extend(validation_result.get("warnings", []))

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
        except Exception as e:
            self.logger.error(f"Error validating parameter {parameter_id}: {e}")
            return {"valid": False, "errors": [str(e)]}

    def _validate_string(self, value: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate string with mathematical length constraints."""
        errors = []
        warnings = []

        if not isinstance(value, str):
            errors.append("Value must be a string")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check length constraints
        min_length = rules.get("min_length", 0)
        max_length = rules.get("max_length", 1000)

        if len(value) < min_length:
            errors.append(f"String length {len(value)} is less than minimum {min_length}")

        if len(value) > max_length:
            errors.append(f"String length {len(value)} exceeds maximum {max_length}")

        # Check pattern
        pattern = rules.get("pattern")
        if pattern:
            import re
            if not re.match(pattern, value):
                errors.append(f"String does not match pattern {pattern}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_integer(self, value: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate integer with mathematical range constraints."""
        errors = []
        warnings = []

        try:
            int_value = int(value)
        except (ValueError, TypeError):
            errors.append("Value must be an integer")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check range constraints
        min_value = rules.get("min_value")
        max_value = rules.get("max_value")

        if min_value is not None and int_value < min_value:
            errors.append(f"Value {int_value} is less than minimum {min_value}")

        if max_value is not None and int_value > max_value:
            errors.append(f"Value {int_value} exceeds maximum {max_value}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_float(self, value: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate float with mathematical precision and range constraints.

        MATHEMATICAL PRESERVATION: This method validates floating-point
        values with precision requirements essential for trading calculations.
        """
        errors = []
        warnings = []

        try:
            float_value = float(value)
        except (ValueError, TypeError):
            errors.append("Value must be a float")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check range constraints
        min_value = rules.get("min_value")
        max_value = rules.get("max_value")

        if min_value is not None and float_value < min_value:
            errors.append(f"Value {float_value} is less than minimum {min_value}")

        if max_value is not None and float_value > max_value:
            errors.append(f"Value {float_value} exceeds maximum {max_value}")

        # Check precision
        precision = rules.get("precision", 6)
        if len(str(float_value).split('.')[-1]) > precision:
            warnings.append(f"Value precision exceeds {precision} decimal places")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_boolean(self, value: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate boolean value."""
        errors = []
        warnings = []

        if not isinstance(value, bool):
            errors.append("Value must be a boolean")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_array(self, value: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate array with mathematical length constraints."""
        errors = []
        warnings = []

        if not isinstance(value, (list, tuple)):
            errors.append("Value must be an array")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check length constraints
        min_length = rules.get("min_length", 0)
        max_length = rules.get("max_length", 1000)

        if len(value) < min_length:
            errors.append(f"Array length {len(value)} is less than minimum {min_length}")

        if len(value) > max_length:
            errors.append(f"Array length {len(value)} exceeds maximum {max_length}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_object(self, value: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate object with required field constraints."""
        errors = []
        warnings = []

        if not isinstance(value, dict):
            errors.append("Value must be an object")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check required fields
        required_fields = rules.get("required_fields", [])
        for field in required_fields:
            if field not in value:
                errors.append(f"Required field '{field}' is missing")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def validate_configuration(self, config_name: str = "default") -> Optional[ConfigValidation]:
        """
        Validate configuration with mathematical scoring.

        MATHEMATICAL PRESERVATION: This method calculates a validation score
        using weighted mathematical metrics that determine configuration
        quality for trading system performance.
        """
        try:
            validation_id = f"validation_{int(time.time())}"
            config = self.configurations.get(config_name, {})

            # Calculate configuration hash
            config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

            total_score = 0.0
            total_weight = 0.0
            errors = []
            warnings = []

            # Validate each parameter
            for parameter_id, parameter in self.parameters.items():
                # Get parameter value from config
                value = self._get_nested_value(config, parameter_id)

                # Validate parameter
                validation_result = self._validate_parameter_value(parameter_id, value)

                # Calculate weight based on parameter importance
                weight = parameter.validation_rules.get("weight", 1.0)
                score = 1.0 if validation_result["valid"] else 0.0

                total_score += weight * score
                total_weight += weight

                # Collect errors and warnings
                errors.extend([f"{parameter_id}: {error}" for error in validation_result.get("errors", [])])
                warnings.extend([f"{parameter_id}: {warning}" for warning in validation_result.get("warnings", [])])

            # Calculate final validation score
            validation_score = total_score / total_weight if total_weight > 0 else 0.0

            # Create validation object
            validation = ConfigValidation(
                validation_id=validation_id,
                config_hash=config_hash,
                validation_score=validation_score,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now(),
                metadata={
                    "config_name": config_name,
                    "total_parameters": len(self.parameters),
                    "valid_parameters": sum(1 for p in self.parameters.values()
                                            if self._validate_parameter_value(p.parameter_id, p.value)["valid"])
                }
            )

            self.validations[validation_id] = validation
            self.validation_history.append(validation)

            self.logger.info(f"Configuration validation completed: score {validation_score:.3f}")
            return validation
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return None

    def _get_nested_value(self, config: Dict[str, Any], parameter_id: str) -> Any:
        """Get nested value from configuration using dot notation."""
        try:
            keys = parameter_id.split('.')
            value = config
            for key in keys:
                value = value.get(key, None)
                if value is None:
                    break
            return value
        except Exception as e:
            self.logger.error(f"Error getting nested value for {parameter_id}: {e}")
            return None

    def optimize_parameters(self, optimization_target: str = "performance") -> Dict[str, Any]:
        """
        Optimize parameters using mathematical algorithms.

        MATHEMATICAL PRESERVATION: This method implements mathematical
        optimization algorithms for different targets:
        - Performance: Increases risk tolerance and rate limits
        - Safety: Decreases risk tolerance and position sizes
        - Efficiency: Adjusts precision and iteration limits
        """
        try:
            optimization_results = {}

            for parameter_id, parameter in self.parameters.items():
                if parameter.data_type in ["float", "integer"]:
                    # Apply optimization based on target
                    if optimization_target == "performance":
                        optimized_value = self._optimize_for_performance(parameter)
                    elif optimization_target == "safety":
                        optimized_value = self._optimize_for_safety(parameter)
                    elif optimization_target == "efficiency":
                        optimized_value = self._optimize_for_efficiency(parameter)
                    else:
                        optimized_value = parameter.value

                    if optimized_value != parameter.value:
                        optimization_results[parameter_id] = {
                            "old_value": parameter.value,
                            "new_value": optimized_value,
                            "improvement": self._calculate_improvement(parameter.value, optimized_value)
                        }

            self.logger.info(f"Parameter optimization completed: {len(optimization_results)} parameters optimized")
            return optimization_results
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            return {}

    def _optimize_for_performance(self, parameter: ConfigParameter) -> Any:
        """
        Optimize parameter for performance using mathematical scaling.

        MATHEMATICAL PRESERVATION: This method applies mathematical scaling
        factors to increase performance while respecting maximum constraints.
        """
        try:
            if parameter.parameter_id == "trading.risk_tolerance":
                # Increase risk tolerance for better performance
                return unified_math.min(parameter.value * 1.1, 1.0)
            elif parameter.parameter_id == "api.rate_limit":
                # Increase rate limit for better performance
                return unified_math.min(parameter.value * 1.2, 10000)
            elif parameter.parameter_id == "mathematical.max_iterations":
                # Increase iterations for better accuracy
                return unified_math.min(parameter.value * 1.5, 10000)
            else:
                return parameter.value
        except Exception as e:
            self.logger.error(f"Error optimizing for performance: {e}")
            return parameter.value

    def _optimize_for_safety(self, parameter: ConfigParameter) -> Any:
        """
        Optimize parameter for safety using mathematical scaling.

        MATHEMATICAL PRESERVATION: This method applies mathematical scaling
        factors to decrease risk while respecting minimum constraints.
        """
        try:
            if parameter.parameter_id == "trading.risk_tolerance":
                # Decrease risk tolerance for safety
                return unified_math.max(parameter.value * 0.8, 0.1)
            elif parameter.parameter_id == "trading.max_position_size":
                # Decrease position size for safety
                return unified_math.max(parameter.value * 0.7, 0.1)
            elif parameter.parameter_id == "api.rate_limit":
                # Decrease rate limit for safety
                return unified_math.max(parameter.value * 0.8, 10)
            else:
                return parameter.value
        except Exception as e:
            self.logger.error(f"Error optimizing for safety: {e}")
            return parameter.value

    def _optimize_for_efficiency(self, parameter: ConfigParameter) -> Any:
        """
        Optimize parameter for efficiency using mathematical scaling.

        MATHEMATICAL PRESERVATION: This method applies mathematical scaling
        factors to balance accuracy and computational efficiency.
        """
        try:
            if parameter.parameter_id == "mathematical.precision":
                # Adjust precision for efficiency
                return unified_math.max(parameter.value * 0.5, 0.1)
            elif parameter.parameter_id == "mathematical.max_iterations":
                # Decrease iterations for efficiency
                return unified_math.max(parameter.value * 0.7, 100)
            else:
                return parameter.value
        except Exception as e:
            self.logger.error(f"Error optimizing for efficiency: {e}")
            return parameter.value

    def _calculate_improvement(self, old_value: Any, new_value: Any) -> float:
        """
        Calculate mathematical improvement percentage.

        MATHEMATICAL PRESERVATION: This method calculates the percentage
        improvement between old and new values for optimization tracking.
        """
        try:
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if old_value != 0:
                    return ((new_value - old_value) / old_value) * 100
                else:
                    return 0.0
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating improvement: {e}")
            return 0.0

    def save_configuration(self, config_name: str = "default") -> bool:
        """Save configuration with mathematical parameter preservation."""
        try:
            config_file = os.path.join(self.config_path, f"schwabot_{config_name}.json")

            # Convert parameters to configuration format
            config_data = {}
            for parameter_id, parameter in self.parameters.items():
                self._set_nested_value(config_data, parameter_id, parameter.value)

            # Save to file
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"Configuration saved to {config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def _set_nested_value(self, config: Dict[str, Any], parameter_id: str, value: Any) -> None:
        """Set nested value in configuration using dot notation."""
        try:
            keys = parameter_id.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        except Exception as e:
            self.logger.error(f"Error setting nested value for {parameter_id}: {e}")

    def get_manager_statistics(self) -> Dict[str, Any]:
        """
        Get manager statistics with mathematical metrics.

        MATHEMATICAL PRESERVATION: This method provides mathematical
        statistics about configuration management performance and
        validation success rates.
        """
        try:
            total_parameters = len(self.parameters)
            total_configs = len(self.configurations)
            total_validations = len(self.validations)
            total_updates = len(self.updates)

            # Calculate validation success rate
            if total_validations > 0:
                successful_validations = sum(1 for v in self.validations.values()
                                             if v.validation_score > 0.8)
                validation_success_rate = successful_validations / total_validations
            else:
                validation_success_rate = 0.0

            # Calculate parameter type distribution
            type_distribution = defaultdict(int)
            for param in self.parameters.values():
                type_distribution[param.data_type] += 1

            return {
                "total_parameters": total_parameters,
                "total_configs": total_configs,
                "total_validations": total_validations,
                "total_updates": total_updates,
                "validation_success_rate": validation_success_rate,
                "parameter_type_distribution": dict(type_distribution),
                "config_history_size": len(self.config_history),
                "validation_history_size": len(self.validation_history)
            }
        except Exception as e:
            self.logger.error(f"Error getting manager statistics: {e}")
            return {}


def main() -> None:
    """Main function for testing configuration manager."""
    try:
        config_manager = ConfigManager("./test_config")

        # Test parameter setting
        config_manager.set_parameter("trading.risk_tolerance", 0.6, "testing")
        config_manager.set_parameter("api.rate_limit", 150, "testing")

        # Test configuration validation
        validation = config_manager.validate_configuration()
        if validation:
            safe_print(f"Configuration validation score: {validation.validation_score:.3f}")

        # Test parameter optimization
        optimization_results = config_manager.optimize_parameters("performance")
        safe_print(f"Optimized {len(optimization_results)} parameters")

        # Get statistics
        stats = config_manager.get_manager_statistics()
        safe_print(f"Manager Statistics: {stats}")

    except Exception as e:
        safe_print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
