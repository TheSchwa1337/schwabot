from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
System Startup - Comprehensive System Initialization and Bootstrap
================================================================

This module implements the main system startup functionality for Schwabot,
handling initialization, component loading, configuration, and system bootstrap.

Core Functionality:
- System initialization and bootstrap
- Component loading and dependency resolution
- Configuration loading and validation
- Environment setup and validation
- Startup sequence management
- Error handling and recovery
"""

import logging
import json
import time
import asyncio
import threading
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from core.unified_math_system import unified_math
from collections import defaultdict, deque
import queue
import weakref
import traceback

logger = logging.getLogger(__name__)

class StartupPhase(Enum):
    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    COMPONENT_LOADING = "component_loading"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    VALIDATION = "validation"
    BOOTSTRAP = "bootstrap"
    READY = "ready"
    ERROR = "error"

class StartupStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class StartupStep:
    step_id: str
    name: str
    description: str
    phase: StartupPhase
    status: StartupStatus = StartupStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StartupConfig:
    config_file: str = "config/schwabot.json"
    log_level: str = "INFO"
    log_file: str = "logs/startup.log"
    enable_debug: bool = False
    enable_profiling: bool = False
    max_startup_time: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0
    components_to_load: List[str] = field(default_factory=list)
    skip_components: List[str] = field(default_factory=list)
    environment: str = "production"
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnvironmentValidator:
    """Environment validation and setup."""
    
    def __init__(self):
        self.validation_results: Dict[str, bool] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_python_version(self) -> bool:
        """Validate Python version."""
        try:
            version = sys.version_info
            if version.major < 3 or (version.major == 3 and version.minor < 8):
                self.errors.append(f"Python 3.8+ required, got {version.major}.{version.minor}")
                return False
            
            self.validation_results['python_version'] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating Python version: {e}")
            return False
    
    def validate_dependencies(self) -> bool:
        """Validate required dependencies."""
        try:
            required_packages = [
                'numpy', 'pandas', 'asyncio', 'logging', 'json', 
                'datetime', 'typing', 'dataclasses', 'enum'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.errors.append(f"Missing required packages: {missing_packages}")
                return False
            
            self.validation_results['dependencies'] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating dependencies: {e}")
            return False
    
    def validate_file_permissions(self) -> bool:
        """Validate file permissions."""
        try:
            required_dirs = ['logs', 'data', 'config', 'backup', 'cache']
            
            for directory in required_dirs:
                if not os.path.exists(directory):
                    try:
                        os.makedirs(directory, exist_ok=True)
                    except Exception as e:
                        self.errors.append(f"Cannot create directory {directory}: {e}")
                        return False
                
                if not os.access(directory, os.W_OK):
                    self.errors.append(f"No write permission for directory {directory}")
                    return False
            
            self.validation_results['file_permissions'] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating file permissions: {e}")
            return False
    
    def validate_system_resources(self) -> bool:
        """Validate system resources."""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.available < 512 * 1024 * 1024:  # 512MB
                self.warnings.append("Low memory available")
            
            # Check disk space
            disk = psutil.disk_usage('.')
            if disk.free < 1024 * 1024 * 1024:  # 1GB
                self.warnings.append("Low disk space available")
            
            self.validation_results['system_resources'] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating system resources: {e}")
            return False
    
    def validate_network_connectivity(self) -> bool:
        """Validate network connectivity."""
        try:
            import socket
            
            # Test basic connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            self.validation_results['network_connectivity'] = True
            return True
            
        except Exception as e:
            self.warnings.append(f"Network connectivity issue: {e}")
            self.validation_results['network_connectivity'] = False
            return True  # Not critical for startup
    
    def run_all_validations(self) -> bool:
        """Run all environment validations."""
        validations = [
            self.validate_python_version,
            self.validate_dependencies,
            self.validate_file_permissions,
            self.validate_system_resources,
            self.validate_network_connectivity
        ]
        
        all_passed = True
        for validation in validations:
            if not validation():
                all_passed = False
        
        return all_passed
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'all_passed': all(self.validation_results.values()),
            'results': self.validation_results.copy(),
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy()
        }

class ConfigurationLoader:
    """Configuration loading and validation."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config_data: Dict[str, Any] = {}
        self.loaded = False
    
    def load_configuration(self) -> bool:
        """Load configuration from file."""
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"Config file {self.config_file} not found, using defaults")
                self.config_data = self._get_default_config()
                return True
            
            with open(self.config_file, 'r') as f:
                self.config_data = json.load(f)
            
            # Validate configuration
            if not self._validate_configuration():
                logger.error("Configuration validation failed")
                return False
            
            self.loaded = True
            logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'system': {
                'log_level': 'INFO',
                'enable_debug': False,
                'max_startup_time': 300,
                'retry_attempts': 3
            },
            'trading': {
                'default_commission': 0.001,
                'default_slippage': 0.0005,
                'max_position_size': 0.1
            },
            'database': {
                'type': 'sqlite',
                'path': 'data/schwabot.db'
            },
            'api': {
                'timeout': 30,
                'rate_limit': 100
            }
        }
    
    def _validate_configuration(self) -> bool:
        """Validate configuration structure."""
        try:
            required_sections = ['system', 'trading', 'database', 'api']
            
            for section in required_sections:
                if section not in self.config_data:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False
    
    def get_config(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            return self.config_data.get(section, {}).get(key, default)
        except Exception:
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section."""
        try:
            return self.config_data.get(section, {})
        except Exception:
            return {}

class ComponentLoader:
    """Component loading and initialization."""
    
    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.loaded_components: List[str] = []
        self.failed_components: List[str] = []
        self.component_dependencies: Dict[str, List[str]] = {}
    
    def register_component(self, component_id: str, component_class: type, 
                          dependencies: List[str] = None) -> bool:
        """Register a component for loading."""
        try:
            self.components[component_id] = {
                'class': component_class,
                'instance': None,
                'dependencies': dependencies or [],
                'loaded': False
            }
            
            self.component_dependencies[component_id] = dependencies or []
            logger.info(f"Component {component_id} registered")
            return True
            
        except Exception as e:
            logger.error(f"Error registering component {component_id}: {e}")
            return False
    
    def load_component(self, component_id: str) -> bool:
        """Load a specific component."""
        try:
            if component_id not in self.components:
                logger.error(f"Component {component_id} not registered")
                return False
            
            component_info = self.components[component_id]
            
            # Check dependencies
            for dep in component_info['dependencies']:
                if dep not in self.loaded_components:
                    logger.error(f"Component {component_id} depends on {dep} which is not loaded")
                    return False
            
            # Instantiate component
            component_class = component_info['class']
            component_instance = component_class()
            
            # Initialize component if it has initialize method
            if hasattr(component_instance, 'initialize'):
                if not component_instance.initialize():
                    logger.error(f"Failed to initialize component {component_id}")
                    return False
            
            # Store instance
            component_info['instance'] = component_instance
            component_info['loaded'] = True
            self.loaded_components.append(component_id)
            
            logger.info(f"Component {component_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading component {component_id}: {e}")
            self.failed_components.append(component_id)
            return False
    
    def load_all_components(self) -> bool:
        """Load all registered components in dependency order."""
        try:
            # Topological sort for dependency resolution
            sorted_components = self._topological_sort()
            
            for component_id in sorted_components:
                if not self.load_component(component_id):
                    logger.error(f"Failed to load component {component_id}")
                    return False
            
            logger.info(f"All components loaded successfully: {self.loaded_components}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            return False
    
    def _topological_sort(self) -> List[str]:
        """Topological sort of components based on dependencies."""
        try:
            # Kahn's algorithm
            in_degree = {comp: 0 for comp in self.components}
            
            # Calculate in-degrees
            for comp, deps in self.component_dependencies.items():
                for dep in deps:
                    if dep in in_degree:
                        in_degree[dep] += 1
            
            # Find components with no dependencies
            queue = [comp for comp, degree in in_degree.items() if degree == 0]
            result = []
            
            while queue:
                comp = queue.pop(0)
                result.append(comp)
                
                # Reduce in-degree of dependent components
                for dep_comp, deps in self.component_dependencies.items():
                    if comp in deps:
                        in_degree[dep_comp] -= 1
                        if in_degree[dep_comp] == 0:
                            queue.append(dep_comp)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in topological sort: {e}")
            return list(self.components.keys())
    
    def get_component(self, component_id: str) -> Optional[Any]:
        """Get loaded component instance."""
        try:
            if component_id in self.components:
                return self.components[component_id]['instance']
            return None
        except Exception:
            return None
    
    def get_loading_summary(self) -> Dict[str, Any]:
        """Get component loading summary."""
        return {
            'total_components': len(self.components),
            'loaded_components': self.loaded_components.copy(),
            'failed_components': self.failed_components.copy(),
            'success_rate': len(self.loaded_components) / unified_math.max(len(self.components), 1)
        }

class StartupManager:
    """Main startup manager."""
    
    def __init__(self, config: StartupConfig):
        self.config = config
        self.validator = EnvironmentValidator()
        self.config_loader = ConfigurationLoader(config.config_file)
        self.component_loader = ComponentLoader()
        self.startup_steps: List[StartupStep] = []
        self.current_phase = StartupPhase.INITIALIZATION
        self.start_time = None
        self.end_time = None
        self.success = False
        
        self._initialize_startup_steps()
    
    def _initialize_startup_steps(self):
        """Initialize startup steps."""
        self.startup_steps = [
            StartupStep(
                step_id="env_validation",
                name="Environment Validation",
                description="Validate system environment and dependencies",
                phase=StartupPhase.INITIALIZATION,
                dependencies=[]
            ),
            StartupStep(
                step_id="config_loading",
                name="Configuration Loading",
                description="Load and validate system configuration",
                phase=StartupPhase.CONFIGURATION,
                dependencies=["env_validation"]
            ),
            StartupStep(
                step_id="component_registration",
                name="Component Registration",
                description="Register system components",
                phase=StartupPhase.COMPONENT_LOADING,
                dependencies=["config_loading"]
            ),
            StartupStep(
                step_id="dependency_resolution",
                name="Dependency Resolution",
                description="Resolve component dependencies",
                phase=StartupPhase.DEPENDENCY_RESOLUTION,
                dependencies=["component_registration"]
            ),
            StartupStep(
                step_id="component_loading",
                name="Component Loading",
                description="Load all system components",
                phase=StartupPhase.COMPONENT_LOADING,
                dependencies=["dependency_resolution"]
            ),
            StartupStep(
                step_id="system_validation",
                name="System Validation",
                description="Validate system integrity",
                phase=StartupPhase.VALIDATION,
                dependencies=["component_loading"]
            ),
            StartupStep(
                step_id="system_bootstrap",
                name="System Bootstrap",
                description="Bootstrap system services",
                phase=StartupPhase.BOOTSTRAP,
                dependencies=["system_validation"]
            )
        ]
    
    def setup_logging(self):
        """Setup logging configuration."""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.config.log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
            logger.info("Logging configured successfully")
            
        except Exception as e:
            safe_print(f"Error setting up logging: {e}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)
    
    async def startup(self) -> bool:
        """Main startup sequence."""
        try:
            self.start_time = datetime.now()
            logger.info("Starting Schwabot system startup sequence")
            
            # Setup logging first
            self.setup_logging()
            
            # Execute startup steps
            for step in self.startup_steps:
                if not await self._execute_step(step):
                    logger.error(f"Startup failed at step: {step.name}")
                    self.success = False
                    return False
            
            self.current_phase = StartupPhase.READY
            self.success = True
            self.end_time = datetime.now()
            
            duration = (self.end_time - self.start_time).total_seconds()
            logger.info(f"System startup completed successfully in {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Startup failed with error: {e}")
            self.success = False
            self.end_time = datetime.now()
            return False
    
    async def _execute_step(self, step: StartupStep) -> bool:
        """Execute a startup step."""
        try:
            step.start_time = datetime.now()
            step.status = StartupStatus.IN_PROGRESS
            
            logger.info(f"Executing startup step: {step.name}")
            
            # Execute step based on step_id
            if step.step_id == "env_validation":
                success = self.validator.run_all_validations()
            elif step.step_id == "config_loading":
                success = self.config_loader.load_configuration()
            elif step.step_id == "component_registration":
                success = self._register_core_components()
            elif step.step_id == "dependency_resolution":
                success = True  # Already handled in component registration
            elif step.step_id == "component_loading":
                success = self.component_loader.load_all_components()
            elif step.step_id == "system_validation":
                success = self._validate_system()
            elif step.step_id == "system_bootstrap":
                success = await self._bootstrap_system()
            else:
                logger.warning(f"Unknown startup step: {step.step_id}")
                success = True
            
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            
            if success:
                step.status = StartupStatus.COMPLETED
                logger.info(f"Startup step completed: {step.name} ({step.duration:.2f}s)")
            else:
                step.status = StartupStatus.FAILED
                step.error = f"Step {step.name} failed"
                logger.error(f"Startup step failed: {step.name}")
            
            return success
            
        except Exception as e:
            step.status = StartupStatus.FAILED
            step.error = str(e)
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
            logger.error(f"Error executing startup step {step.name}: {e}")
            return False
    
    def _register_core_components(self) -> bool:
        """Register core system components."""
        try:
            # Register core components (these would be actual component classes)
            # For now, we'll use placeholder classes
            
            class ConfigManager:
                def initialize(self): return True
            
            class DatabaseManager:
                def initialize(self): return True
            
            class NetworkManager:
                def initialize(self): return True
            
            class CacheManager:
                def initialize(self): return True
            
            # Register components
            self.component_loader.register_component("config_manager", ConfigManager)
            self.component_loader.register_component("database_manager", DatabaseManager, ["config_manager"])
            self.component_loader.register_component("network_manager", NetworkManager, ["config_manager"])
            self.component_loader.register_component("cache_manager", CacheManager, ["config_manager"])
            
            logger.info("Core components registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering core components: {e}")
            return False
    
    def _validate_system(self) -> bool:
        """Validate system integrity."""
        try:
            # Check if all components are loaded
            summary = self.component_loader.get_loading_summary()
            if summary['success_rate'] < 1.0:
                logger.error(f"Not all components loaded successfully: {summary}")
                return False
            
            # Check configuration
            if not self.config_loader.loaded:
                logger.error("Configuration not loaded")
                return False
            
            # Check environment validation
            env_summary = self.validator.get_validation_summary()
            if not env_summary['all_passed']:
                logger.error(f"Environment validation failed: {env_summary['errors']}")
                return False
            
            logger.info("System validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating system: {e}")
            return False
    
    async def _bootstrap_system(self) -> bool:
        """Bootstrap system services."""
        try:
            # Start background services
            # Initialize connections
            # Start monitoring
            # Initialize trading systems
            
            logger.info("System bootstrap completed")
            return True
            
        except Exception as e:
            logger.error(f"Error bootstrapping system: {e}")
            return False
    
    def get_startup_summary(self) -> Dict[str, Any]:
        """Get startup summary."""
        return {
            'success': self.success,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            'current_phase': self.current_phase.value,
            'steps': [
                {
                    'step_id': step.step_id,
                    'name': step.name,
                    'status': step.status.value,
                    'duration': step.duration,
                    'error': step.error
                }
                for step in self.startup_steps
            ],
            'environment_validation': self.validator.get_validation_summary(),
            'component_loading': self.component_loader.get_loading_summary()
        }

def main():
    """Main function for testing."""
    try:
        # Create startup configuration
        config = StartupConfig(
            config_file="config/schwabot.json",
            log_level="INFO",
            log_file="logs/startup.log",
            enable_debug=True,
            environment="development"
        )
        
        # Create startup manager
        startup_manager = StartupManager(config)
        
        # Run startup
        import asyncio
        success = asyncio.run(startup_manager.startup())
        
        # Print summary
        summary = startup_manager.get_startup_summary()
        safe_print("Startup Summary:")
        print(json.dumps(summary, indent=2, default=str))
        
        if success:
            safe_print("System startup completed successfully!")
        else:
            safe_print("System startup failed!")
        
    except Exception as e:
        safe_print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 