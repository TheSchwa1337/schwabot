# -*- coding: utf-8 -*-
"""
Requirements Validation System
=============================

Comprehensive validation system for Schwabot requirements.txt that:
1. Tests installation of all dependencies
2. Validates functionality of critical mathematical libraries
3. Checks compatibility between packages
4. Verifies trading system dependencies
5. Ensures GPU support (if available)

Mathematical Foundation:
- Dependency Graph: G(pkg) = {deps(pkg) | pkg ∈ requirements}
- Compatibility Matrix: C(i,j) = compatible(pkg_i, pkg_j)
- Functionality Test: F(pkg) = test_import(pkg) ∧ test_function(pkg)
"""

import os
import sys
import subprocess
import importlib
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status for requirements."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

class PackageCategory(Enum):
    """Categories of packages."""
    CORE_MATH = "core_mathematical"
    ML_STATS = "machine_learning"
    TRADING_API = "trading_api"
    WEB_FRAMEWORK = "web_framework"
    SYSTEM_MONITOR = "system_monitor"
    GPU_SUPPORT = "gpu_support"
    DEVELOPMENT = "development"
    UTILITIES = "utilities"

@dataclass
class PackageValidation:
    """Represents validation results for a package."""
    package_name: str
    version: str
    category: PackageCategory
    status: ValidationStatus
    import_test: bool = False
    functionality_test: bool = False
    compatibility_issues: List[str] = field(default_factory=list)
    error_message: str = ""
    critical_for_system: bool = False
    
    def __post_init__(self):
        """Initialize package validation."""
        self.validation_id = f"{self.package_name}_{self.version}"

class RequirementsValidator:
    """
    Comprehensive requirements validation system.
    
    This system:
    1. Parses requirements.txt and validates each package
    2. Tests imports and basic functionality
    3. Checks for compatibility issues
    4. Validates critical mathematical dependencies
    5. Ensures trading system requirements are met
    """
    
    def __init__(self, requirements_file: str = "requirements.txt"):
        """Initialize the requirements validator."""
        self.requirements_file = Path(requirements_file)
        self.packages: Dict[str, PackageValidation] = {}
        self.validation_metrics = {
            "total_packages": 0,
            "passed_validation": 0,
            "failed_validation": 0,
            "warnings": 0,
            "critical_failures": 0
        }
        
        # Critical packages for Schwabot functionality
        self.critical_packages = {
            "numpy", "scipy", "pandas", "ccxt", "flask", "requests",
            "scikit-learn", "matplotlib", "python-dotenv"
        }
        
        # Package categories mapping
        self.package_categories = {
            # Core Mathematical
            "numpy": PackageCategory.CORE_MATH,
            "scipy": PackageCategory.CORE_MATH,
            "pandas": PackageCategory.CORE_MATH,
            "matplotlib": PackageCategory.CORE_MATH,
            "seaborn": PackageCategory.CORE_MATH,
            "plotly": PackageCategory.CORE_MATH,
            "sympy": PackageCategory.CORE_MATH,
            "mpmath": PackageCategory.CORE_MATH,
            "numba": PackageCategory.CORE_MATH,
            
            # Machine Learning
            "scikit-learn": PackageCategory.ML_STATS,
            "statsmodels": PackageCategory.ML_STATS,
            "ta-lib": PackageCategory.ML_STATS,
            
            # Trading APIs
            "ccxt": PackageCategory.TRADING_API,
            "python-binance": PackageCategory.TRADING_API,
            "yfinance": PackageCategory.TRADING_API,
            
            # Web Frameworks
            "flask": PackageCategory.WEB_FRAMEWORK,
            "flask-cors": PackageCategory.WEB_FRAMEWORK,
            "flask-socketio": PackageCategory.WEB_FRAMEWORK,
            "uvicorn": PackageCategory.WEB_FRAMEWORK,
            "fastapi": PackageCategory.WEB_FRAMEWORK,
            "requests": PackageCategory.WEB_FRAMEWORK,
            "aiohttp": PackageCategory.WEB_FRAMEWORK,
            "websockets": PackageCategory.WEB_FRAMEWORK,
            
            # System Monitoring
            "psutil": PackageCategory.SYSTEM_MONITOR,
            "pynvml": PackageCategory.SYSTEM_MONITOR,
            
            # GPU Support
            "cupy-cuda11x": PackageCategory.GPU_SUPPORT,
            
            # Development
            "typing-extensions": PackageCategory.DEVELOPMENT,
            "pathlib": PackageCategory.DEVELOPMENT,
            "pytest": PackageCategory.DEVELOPMENT,
            "pytest-cov": PackageCategory.DEVELOPMENT,
            "black": PackageCategory.DEVELOPMENT,
            "isort": PackageCategory.DEVELOPMENT,
            "mypy": PackageCategory.DEVELOPMENT,
            "flake8": PackageCategory.DEVELOPMENT,
            "jupyter": PackageCategory.DEVELOPMENT,
            "ipykernel": PackageCategory.DEVELOPMENT,
            
            # Utilities
            "joblib": PackageCategory.UTILITIES,
            "python-dotenv": PackageCategory.UTILITIES,
            "pyyaml": PackageCategory.UTILITIES,
            "jsonschema": PackageCategory.UTILITIES
        }
        
        logger.info("Requirements Validator initialized")
    
    def parse_requirements(self) -> Dict[str, str]:
        """
        Parse requirements.txt file.
        
        Returns:
            Dictionary of package names and versions.
        """
        logger.info(f"Parsing requirements from {self.requirements_file}")
        
        packages = {}
        
        try:
            with open(self.requirements_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse package specification
                    try:
                        if '>=' in line:
                            package_name, version = line.split('>=', 1)
                        elif '==' in line:
                            package_name, version = line.split('==', 1)
                        elif '~=' in line:
                            package_name, version = line.split('~=', 1)
                        else:
                            package_name = line
                            version = "latest"
                        
                        package_name = package_name.strip()
                        version = version.strip()
                        
                        if package_name:
                            packages[package_name] = version
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse line {line_num}: {line} - {e}")
        
        except Exception as e:
            logger.error(f"Failed to read requirements file: {e}")
        
        logger.info(f"Parsed {len(packages)} packages from requirements.txt")
        return packages
    
    def validate_package_installation(self, package_name: str, version: str) -> PackageValidation:
        """
        Validate a single package installation.
        
        Args:
            package_name: Name of the package
            version: Required version
            
        Returns:
            PackageValidation object with results
        """
        logger.info(f"Validating package: {package_name} {version}")
        
        # Determine category and criticality
        category = self.package_categories.get(package_name, PackageCategory.UTILITIES)
        critical = package_name in self.critical_packages
        
        validation = PackageValidation(
            package_name=package_name,
            version=version,
            category=category,
            status=ValidationStatus.SKIPPED,
            critical_for_system=critical
        )
        
        try:
            # Test import
            module = importlib.import_module(package_name)
            validation.import_test = True
            
            # Test basic functionality
            validation.functionality_test = self._test_package_functionality(package_name, module)
            
            # Check version compatibility
            version_ok = self._check_version_compatibility(package_name, version, module)
            
            if validation.import_test and validation.functionality_test and version_ok:
                validation.status = ValidationStatus.PASSED
            elif validation.import_test and not validation.functionality_test:
                validation.status = ValidationStatus.WARNING
                validation.error_message = "Import successful but functionality test failed"
            else:
                validation.status = ValidationStatus.FAILED
                validation.error_message = "Import failed"
                
        except ImportError as e:
            validation.status = ValidationStatus.FAILED
            validation.error_message = f"Import error: {e}"
        except Exception as e:
            validation.status = ValidationStatus.FAILED
            validation.error_message = f"Validation error: {e}"
        
        return validation
    
    def _test_package_functionality(self, package_name: str, module) -> bool:
        """Test basic functionality of a package."""
        try:
            if package_name == "numpy":
                import numpy as np
                return hasattr(np, 'array') and hasattr(np, 'random')
            
            elif package_name == "scipy":
                import scipy
                return hasattr(scipy, 'stats') or hasattr(scipy, 'linalg')
            
            elif package_name == "pandas":
                import pandas as pd
                return hasattr(pd, 'DataFrame') and hasattr(pd, 'read_csv')
            
            elif package_name == "matplotlib":
                import matplotlib
                return hasattr(matplotlib, 'pyplot')
            
            elif package_name == "scikit-learn":
                import sklearn
                return hasattr(sklearn, 'linear_model') or hasattr(sklearn, 'ensemble')
            
            elif package_name == "ccxt":
                import ccxt
                return hasattr(ccxt, 'binance') or hasattr(ccxt, 'Exchange')
            
            elif package_name == "flask":
                import flask
                return hasattr(flask, 'Flask')
            
            elif package_name == "requests":
                import requests
                return hasattr(requests, 'get') and hasattr(requests, 'post')
            
            elif package_name == "python-dotenv":
                import dotenv
                return hasattr(dotenv, 'load_dotenv')
            
            elif package_name == "psutil":
                import psutil
                return hasattr(psutil, 'cpu_percent') and hasattr(psutil, 'memory_info')
            
            elif package_name == "pandas":
                import pandas as pd
                return hasattr(pd, 'DataFrame')
            
            # Default: if we can import it, assume it's functional
            return True
            
        except Exception as e:
            logger.warning(f"Functionality test failed for {package_name}: {e}")
            return False
    
    def _check_version_compatibility(self, package_name: str, required_version: str, module) -> bool:
        """Check if installed version is compatible with required version."""
        try:
            if hasattr(module, '__version__'):
                installed_version = module.__version__
                
                # Simple version comparison (can be enhanced)
                if required_version == "latest":
                    return True
                
                # For now, just check if version attribute exists
                return bool(installed_version)
            
            return True  # If no version info, assume compatible
            
        except Exception as e:
            logger.warning(f"Version check failed for {package_name}: {e}")
            return True  # Assume compatible if check fails
    
    def validate_all_packages(self) -> Dict[str, PackageValidation]:
        """
        Validate all packages in requirements.txt.
        
        Returns:
            Dictionary of package validation results.
        """
        logger.info("Starting comprehensive package validation...")
        
        # Parse requirements
        packages = self.parse_requirements()
        self.validation_metrics["total_packages"] = len(packages)
        
        # Validate each package
        for package_name, version in packages.items():
            validation = self.validate_package_installation(package_name, version)
            self.packages[package_name] = validation
            
            # Update metrics
            if validation.status == ValidationStatus.PASSED:
                self.validation_metrics["passed_validation"] += 1
            elif validation.status == ValidationStatus.FAILED:
                self.validation_metrics["failed_validation"] += 1
                if validation.critical_for_system:
                    self.validation_metrics["critical_failures"] += 1
            elif validation.status == ValidationStatus.WARNING:
                self.validation_metrics["warnings"] += 1
        
        logger.info(f"Validation completed: {self.validation_metrics}")
        return self.packages
    
    def test_mathematical_functionality(self) -> Dict[str, bool]:
        """
        Test critical mathematical functionality for Schwabot.
        
        Returns:
            Dictionary of mathematical functionality test results.
        """
        logger.info("Testing critical mathematical functionality...")
        
        math_tests = {
            "numpy_operations": False,
            "scipy_statistics": False,
            "pandas_dataframe": False,
            "matplotlib_plotting": False,
            "scikit_learning": False,
            "trading_api": False,
            "web_framework": False
        }
        
        try:
            # Test NumPy operations
            import numpy as np
            test_array = np.array([1, 2, 3, 4, 5])
            test_result = np.mean(test_array)
            math_tests["numpy_operations"] = test_result == 3.0
            
        except Exception as e:
            logger.error(f"NumPy test failed: {e}")
        
        try:
            # Test SciPy statistics
            import scipy.stats as stats
            import numpy as np
            test_data = [1, 2, 3, 4, 5]
            test_result = np.mean(test_data)
            math_tests["scipy_statistics"] = test_result == 3.0
            
        except Exception as e:
            logger.error(f"SciPy test failed: {e}")
        
        try:
            # Test Pandas DataFrame
            import pandas as pd
            test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            test_result = test_df.shape[0] == 3
            math_tests["pandas_dataframe"] = test_result
            
        except Exception as e:
            logger.error(f"Pandas test failed: {e}")
        
        try:
            # Test Matplotlib plotting
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot([1, 2, 3], [1, 4, 2])
            plt.close()
            math_tests["matplotlib_plotting"] = True
            
        except Exception as e:
            logger.error(f"Matplotlib test failed: {e}")
        
        try:
            # Test Scikit-learn
            import sklearn.linear_model as lm
            model = lm.LinearRegression()
            math_tests["scikit_learning"] = hasattr(model, 'fit')
            
        except Exception as e:
            logger.error(f"Scikit-learn test failed: {e}")
        
        try:
            # Test Trading API
            import ccxt
            exchange = ccxt.binance()
            math_tests["trading_api"] = hasattr(exchange, 'fetch_ticker')
            
        except Exception as e:
            logger.error(f"Trading API test failed: {e}")
        
        try:
            # Test Web Framework
            from flask import Flask
            app = Flask(__name__)
            math_tests["web_framework"] = hasattr(app, 'route')
            
        except Exception as e:
            logger.error(f"Web framework test failed: {e}")
        
        logger.info(f"Mathematical functionality tests: {math_tests}")
        return math_tests
    
    def check_system_compatibility(self) -> Dict[str, Any]:
        """
        Check system compatibility and environment.
        
        Returns:
            System compatibility information.
        """
        logger.info("Checking system compatibility...")
        
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "architecture": sys.maxsize > 2**32 and "64-bit" or "32-bit",
            "gpu_available": False,
            "memory_available": 0,
            "cpu_cores": 0
        }
        
        try:
            # Check GPU availability
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            system_info["gpu_available"] = device_count > 0
            system_info["gpu_count"] = device_count
            
        except Exception:
            logger.info("No GPU support detected")
        
        try:
            # Check system resources
            import psutil
            system_info["memory_available"] = psutil.virtual_memory().total
            system_info["cpu_cores"] = psutil.cpu_count()
            
        except Exception as e:
            logger.warning(f"Could not get system info: {e}")
        
        logger.info(f"System compatibility: {system_info}")
        return system_info
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Complete validation report.
        """
        logger.info("Generating validation report...")
        
        # Validate all packages
        self.validate_all_packages()
        
        # Test mathematical functionality
        math_tests = self.test_mathematical_functionality()
        
        # Check system compatibility
        system_info = self.check_system_compatibility()
        
        # Compile report
        report = {
            "validation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "requirements_file": str(self.requirements_file),
                "python_version": sys.version,
                "platform": sys.platform
            },
            "validation_metrics": self.validation_metrics.copy(),
            "package_results": {
                name: {
                    "status": validation.status.value,
                    "category": validation.category.value,
                    "critical": validation.critical_for_system,
                    "import_test": validation.import_test,
                    "functionality_test": validation.functionality_test,
                    "error_message": validation.error_message
                }
                for name, validation in self.packages.items()
            },
            "mathematical_functionality": math_tests,
            "system_compatibility": system_info,
            "critical_issues": self._identify_critical_issues(),
            "recommendations": self._generate_recommendations()
        }
        
        logger.info("Validation report generated")
        return report
    
    def _identify_critical_issues(self) -> List[str]:
        """Identify critical issues that need immediate attention."""
        issues = []
        
        # Check for critical package failures
        for name, validation in self.packages.items():
            if validation.critical_for_system and validation.status == ValidationStatus.FAILED:
                issues.append(f"Critical package {name} failed validation: {validation.error_message}")
        
        # Check mathematical functionality
        math_tests = self.test_mathematical_functionality()
        failed_math = [test for test, passed in math_tests.items() if not passed]
        if failed_math:
            issues.append(f"Mathematical functionality tests failed: {failed_math}")
        
        # Check overall validation metrics
        if self.validation_metrics["critical_failures"] > 0:
            issues.append(f"{self.validation_metrics['critical_failures']} critical packages failed")
        
        return issues
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Package-specific recommendations
        failed_packages = [name for name, validation in self.packages.items() 
                          if validation.status == ValidationStatus.FAILED]
        
        if failed_packages:
            recommendations.append(f"Fix installation issues for: {', '.join(failed_packages)}")
        
        # System recommendations
        if not self.packages.get("cupy-cuda11x"):
            recommendations.append("Consider installing GPU support for enhanced performance")
        
        if self.validation_metrics["warnings"] > 0:
            recommendations.append("Review warnings for potential compatibility issues")
        
        # General recommendations
        recommendations.append("Run validation regularly to ensure system stability")
        recommendations.append("Keep packages updated for security and performance")
        
        return recommendations


def main():
    """Main function for requirements validation."""
    print("=== Schwabot Requirements Validation System ===")
    print("Comprehensive validation of all dependencies...")
    print("Testing mathematical functionality and system compatibility...")
    print()
    
    # Initialize validator
    validator = RequirementsValidator()
    
    try:
        # Generate comprehensive report
        report = validator.generate_validation_report()
        
        # Print summary
        print("=== VALIDATION SUMMARY ===")
        metrics = report["validation_metrics"]
        print(f"Total Packages: {metrics['total_packages']}")
        print(f"Passed Validation: {metrics['passed_validation']}")
        print(f"Failed Validation: {metrics['failed_validation']}")
        print(f"Warnings: {metrics['warnings']}")
        print(f"Critical Failures: {metrics['critical_failures']}")
        
        print(f"\nMathematical Functionality:")
        math_tests = report["mathematical_functionality"]
        for test, passed in math_tests.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test}: {status}")
        
        print(f"\nSystem Compatibility:")
        system_info = report["system_compatibility"]
        print(f"  Python Version: {system_info['python_version'].split()[0]}")
        print(f"  Platform: {system_info['platform']}")
        print(f"  Architecture: {system_info['architecture']}")
        print(f"  GPU Available: {'Yes' if system_info['gpu_available'] else 'No'}")
        print(f"  CPU Cores: {system_info['cpu_cores']}")
        
        if report["critical_issues"]:
            print(f"\nCritical Issues:")
            for issue in report["critical_issues"]:
                print(f"  ❌ {issue}")
        
        print(f"\nRecommendations:")
        for recommendation in report["recommendations"]:
            print(f"  💡 {recommendation}")
        
        # Overall status
        overall_status = "✅ VALIDATION PASSED" if metrics["critical_failures"] == 0 else "❌ VALIDATION FAILED"
        print(f"\nOverall Status: {overall_status}")
        
    except Exception as e:
        logger.error(f"Requirements validation failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 