# !/usr/bin/env python3
"""
Dependency Installation Script for Unified BTC-to-Profit Trading System
======================================================================

This script installs all necessary dependencies and verifies the installation
for the unified BTC-to-profit trading system.

Usage:
    python install_dependencies.py [--prod] [--verify]
"""

import subprocess
import sys
import os
import importlib
from typing import List, Dict, Tuple


def run_command(command: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def install_dependencies(requirements_file: str) -> bool:
    """Install dependencies from requirements file."""
    return run_command(
        [sys.executable, "-m", "pip", "install", "-r", requirements_file],
        f"Installing dependencies from {requirements_file}"
    )


def verify_imports() -> Dict[str, bool]:
    """Verify that all critical imports work."""
    print("🔍 Verifying critical imports...")

    critical_imports = {
        'numpy': 'Core mathematical operations',
        'pandas': 'Data analysis and manipulation',
        'scipy': 'Scientific computing',
        'sklearn': 'Machine learning',
        'flask': 'Web framework',
        'requests': 'HTTP client',
        'aiohttp': 'Async HTTP client',
        'websockets': 'WebSocket support',
        'matplotlib': 'Data visualization',
        'seaborn': 'Statistical visualization',
        'plotly': 'Interactive plotting',
        'uvicorn': 'ASGI server',
        'fastapi': 'Modern web framework',
        'psutil': 'System monitoring',
        'joblib': 'Parallel processing',
        'statsmodels': 'Statistical analysis',
        'ta-lib': 'Technical analysis',
    }

    results = {}
    for module, description in critical_imports.items():
        try:
            importlib.import_module(module)
            print(f"✅ {module} - {description}")
            results[module] = True
        except ImportError as e:
            print(f"❌ {module} - {description} (Error: {e})")
            results[module] = False

    return results


def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements."""
    print("🔍 Checking system requirements...")

    checks = {
        'python_version': sys.version_info >= (3, 8),
        'pip_available': True,  # We're running pip commands
        'write_permissions': os.access('.', os.W_OK),
    }

    # Check Python version
    if checks['python_version']:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    else:
        print(
            f"❌ Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} (requires 3.8+)")

    # Check write permissions
    if checks['write_permissions']:
        print("✅ Write permissions available")
    else:
        print("❌ No write permissions in current directory")

    return checks


def main():
    """Main installation function."""
    print("🚀 Unified BTC-to-Profit Trading System - Dependency Installer")
    print("=" * 60)

    # Parse command line arguments
    is_production = '--prod' in sys.argv
    verify_only = '--verify' in sys.argv

    # Check system requirements
    system_ok = check_system_requirements()
    if not all(system_ok.values()):
        print("❌ System requirements not met. Please fix issues above.")
        return 1

    if verify_only:
        print("\n🔍 Verification mode - checking imports only...")
        import_results = verify_imports()
        success_count = sum(import_results.values())
        total_count = len(import_results)

        print(f"\n📊 Verification Results: {success_count}/{total_count} imports successful")

        if success_count == total_count:
            print("✅ All dependencies verified successfully!")
            return 0
        else:
            print("❌ Some dependencies failed verification. Please install missing packages.")
            return 1

    # Install dependencies
    requirements_file = "requirements-prod.txt" if is_production else "requirements.txt"

    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file {requirements_file} not found!")
        return 1

    print(f"\n📦 Installing dependencies from {requirements_file}...")
    if not install_dependencies(requirements_file):
        print("❌ Dependency installation failed!")
        return 1

    # Verify installation
    print("\n🔍 Verifying installation...")
    import_results = verify_imports()
    success_count = sum(import_results.values())
    total_count = len(import_results)

    print(f"\n📊 Installation Results: {success_count}/{total_count} dependencies installed successfully")

    if success_count == total_count:
        print("✅ All dependencies installed and verified successfully!")
        print("\n🚀 Your unified BTC-to-profit trading system is ready for deployment!")
        return 0
    else:
        print("⚠️ Some dependencies may need manual installation.")
        print("Please check the failed imports above and install them manually.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
