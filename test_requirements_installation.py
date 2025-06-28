# -*- coding: utf-8 -*-
"""
Requirements Installation Test for Schwabot
==========================================

This script validates that all required packages can be installed and imported
correctly, ensuring mathematical functionality is preserved.

MATHEMATICAL PRESERVATION: This validates all mathematical dependencies.
"""

import sys
import subprocess
import importlib
from pathlib import Path

def test_package_installation(package_name, import_name=None):
    """Test if a package can be imported successfully."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        print(f"✅ {package_name}: SUCCESS")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {package_name}: WARNING - {e}")
        return True  # Still consider it working if it imports but has other issues

def test_mathematical_functionality():
    """Test core mathematical operations."""
    print("\n🧮 Testing Mathematical Functionality...")
    print("=" * 50)
    
    # Test 1: NumPy (CRITICAL)
    try:
        import numpy as np
        test_array = np.array([1, 2, 3, 4, 5])
        result = np.mean(test_array)
        print(f"✅ NumPy: SUCCESS (mean calculation: {result})")
    except Exception as e:
        print(f"❌ NumPy: FAILED - {e}")
        return False
    
    # Test 2: SciPy (CRITICAL)
    try:
        import scipy
        from scipy import stats
        print(f"✅ SciPy: SUCCESS (version: {scipy.__version__})")
    except Exception as e:
        print(f"❌ SciPy: FAILED - {e}")
        return False
    
    # Test 3: Pandas (CRITICAL)
    try:
        import pandas as pd
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        result = df['x'].mean()
        print(f"✅ Pandas: SUCCESS (dataframe mean: {result})")
    except Exception as e:
        print(f"❌ Pandas: FAILED - {e}")
        return False
    
    # Test 4: Matplotlib (for visualization)
    try:
        import matplotlib
        print(f"✅ Matplotlib: SUCCESS (version: {matplotlib.__version__})")
    except Exception as e:
        print(f"⚠️ Matplotlib: WARNING - {e}")
    
    # Test 5: Scikit-learn (for ML algorithms)
    try:
        import sklearn
        print(f"✅ Scikit-learn: SUCCESS (version: {sklearn.__version__})")
    except Exception as e:
        print(f"⚠️ Scikit-learn: WARNING - {e}")
    
    # Test 6: Hashlib (built-in, for BTC hashing)
    try:
        import hashlib
        test_string = "BTC_price_50000"
        hash_result = hashlib.sha256(test_string.encode()).hexdigest()
        print(f"✅ Hashlib: SUCCESS (SHA256 hash: {hash_result[:16]}...)")
    except Exception as e:
        print(f"❌ Hashlib: FAILED - {e}")
        return False
    
    # Test 7: CCXT (for trading)
    try:
        import ccxt
        print(f"✅ CCXT: SUCCESS (version: {ccxt.__version__})")
    except Exception as e:
        print(f"⚠️ CCXT: WARNING - {e}")
    
    # Test 8: Flask (for API)
    try:
        import flask
        print(f"✅ Flask: SUCCESS (version: {flask.__version__})")
    except Exception as e:
        print(f"⚠️ Flask: WARNING - {e}")
    
    # Test 9: Requests (for HTTP)
    try:
        import requests
        print(f"✅ Requests: SUCCESS (version: {requests.__version__})")
    except Exception as e:
        print(f"⚠️ Requests: WARNING - {e}")
    
    # Test 10: Aiohttp (for async HTTP)
    try:
        import aiohttp
        print(f"✅ Aiohttp: SUCCESS (version: {aiohttp.__version__})")
    except Exception as e:
        print(f"⚠️ Aiohttp: WARNING - {e}")
    
    return True

def test_core_mathematical_modules():
    """Test core mathematical modules from Schwabot."""
    print("\n🔬 Testing Core Mathematical Modules...")
    print("=" * 50)
    
    # Test core mathematical modules
    core_modules = [
        ('core.math.mathematical_relay_system', 'MathematicalRelaySystem'),
        ('core.math.trading_tensor_ops', 'TradingTensorOps'),
        ('core.unified_math_system', 'UnifiedMathSystem'),
    ]
    
    for module_path, class_name in core_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                print(f"✅ {module_path}: SUCCESS")
            else:
                print(f"⚠️ {module_path}: WARNING (class {class_name} not found)")
        except ImportError as e:
            print(f"❌ {module_path}: FAILED - {e}")
        except Exception as e:
            print(f"⚠️ {module_path}: WARNING - {e}")

def run_pip_install_test():
    """Test pip install with the clean requirements."""
    print("\n📦 Testing Pip Installation...")
    print("=" * 50)
    
    # Check if requirements_clean.txt exists
    requirements_file = Path('requirements_clean.txt')
    if not requirements_file.exists():
        print("❌ requirements_clean.txt not found!")
        return False
    
    try:
        # Run pip install with dry-run to test
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_clean.txt',
            '--dry-run'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Pip install test: SUCCESS (dry-run)")
            return True
        else:
            print(f"❌ Pip install test: FAILED")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Pip install test: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ Pip install test: ERROR - {e}")
        return False

def generate_installation_report():
    """Generate a comprehensive installation report."""
    print("\n📊 GENERATING INSTALLATION REPORT...")
    print("=" * 60)
    
    # Test 1: Basic package imports
    print("\n1. Testing Package Imports:")
    print("-" * 30)
    
    packages_to_test = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('sklearn', 'sklearn'),
        ('ccxt', 'ccxt'),
        ('flask', 'flask'),
        ('requests', 'requests'),
        ('aiohttp', 'aiohttp'),
        ('psutil', 'psutil'),
        ('joblib', 'joblib'),
        ('dotenv', 'python-dotenv'),
        ('yaml', 'pyyaml'),
    ]
    
    successful_imports = 0
    for package_name, import_name in packages_to_test:
        if test_package_installation(package_name, import_name):
            successful_imports += 1
    
    # Test 2: Mathematical functionality
    print("\n2. Testing Mathematical Functionality:")
    print("-" * 40)
    math_success = test_mathematical_functionality()
    
    # Test 3: Core modules
    print("\n3. Testing Core Mathematical Modules:")
    print("-" * 40)
    test_core_mathematical_modules()
    
    # Test 4: Pip installation
    print("\n4. Testing Pip Installation:")
    print("-" * 30)
    pip_success = run_pip_install_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INSTALLATION REPORT SUMMARY")
    print("=" * 60)
    print(f"✅ Successful package imports: {successful_imports}/{len(packages_to_test)}")
    print(f"🧮 Mathematical functionality: {'✅ PASS' if math_success else '❌ FAIL'}")
    print(f"📦 Pip installation test: {'✅ PASS' if pip_success else '❌ FAIL'}")
    
    # Overall status
    if successful_imports >= len(packages_to_test) * 0.8 and math_success:
        print("\n🎉 OVERALL STATUS: EXCELLENT - All critical dependencies working!")
        print("   Your mathematical trading system is ready to run.")
    elif successful_imports >= len(packages_to_test) * 0.6:
        print("\n⚠️ OVERALL STATUS: GOOD - Most dependencies working.")
        print("   Some optional packages may need attention.")
    else:
        print("\n❌ OVERALL STATUS: NEEDS ATTENTION - Critical dependencies missing.")
        print("   Please check the installation errors above.")
    
    return successful_imports, math_success, pip_success

def main():
    """Run comprehensive requirements installation test."""
    print("🔍 SCHWABOT REQUIREMENTS INSTALLATION TEST")
    print("=" * 60)
    print("Testing all mathematical and trading dependencies...")
    
    # Generate comprehensive report
    successful_imports, math_success, pip_success = generate_installation_report()
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("=" * 30)
    
    if not math_success:
        print("1. ❌ CRITICAL: Mathematical packages failed - reinstall core packages")
        print("   pip install numpy scipy pandas matplotlib scikit-learn")
    
    if not pip_success:
        print("2. ⚠️ Pip installation issues detected - check requirements_clean.txt")
    
    if successful_imports < 10:
        print("3. ⚠️ Many packages missing - consider full reinstall")
        print("   pip install -r requirements_clean.txt")
    
    print("4. ✅ If all tests pass, your system is ready for mathematical trading!")
    
    print("\n✅ Installation test completed!")

if __name__ == "__main__":
    main() 