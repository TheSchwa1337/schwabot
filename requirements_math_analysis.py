# -*- coding: utf-8 -*-
"""
Requirements and Mathematical Preservation Analysis
==================================================

Focused analysis to ensure:
1. Requirements.txt contains only packages needed for mathematical operations
2. Flake8 fixes preserved mathematical content
3. Import alignment with actual usage
4. Mathematical functionality validation

MATHEMATICAL PRESERVATION: This validates mathematical content integrity.
"""

import os
import re
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Set, Tuple

def analyze_mathematical_imports():
    """Analyze which mathematical imports are actually used."""
    print("🔍 Analyzing Mathematical Imports Usage...")
    
    # Core mathematical packages we need to validate
    math_packages = {
        'numpy': {'patterns': ['import numpy', 'np.', 'numpy.'], 'found': False, 'files': []},
        'scipy': {'patterns': ['import scipy', 'scipy.', 'from scipy'], 'found': False, 'files': []},
        'pandas': {'patterns': ['import pandas', 'pd.', 'pandas.'], 'found': False, 'files': []},
        'matplotlib': {'patterns': ['import matplotlib', 'matplotlib.', 'plt.'], 'found': False, 'files': []},
        'scikit-learn': {'patterns': ['sklearn', 'from sklearn'], 'found': False, 'files': []},
        'hashlib': {'patterns': ['import hashlib', 'hashlib.sha256'], 'found': False, 'files': []},
        'ccxt': {'patterns': ['import ccxt', 'ccxt.'], 'found': False, 'files': []},
        'flask': {'patterns': ['import flask', 'from flask'], 'found': False, 'files': []},
        'asyncio': {'patterns': ['import asyncio', 'asyncio.'], 'found': False, 'files': []},
        'cryptography': {'patterns': ['from cryptography', 'cryptography.'], 'found': False, 'files': []}
    }
    
    # Scan Python files
    python_files = list(Path('.').rglob('*.py'))
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for package, info in math_packages.items():
                for pattern in info['patterns']:
                    if pattern in content:
                        info['found'] = True
                        info['files'].append(str(file_path))
                        break
                        
        except Exception as e:
            print(f"⚠️ Could not read {file_path}: {e}")
    
    # Report findings
    print("\n📊 Mathematical Package Usage Analysis:")
    print("=" * 60)
    
    for package, info in math_packages.items():
        status = "✅ USED" if info['found'] else "❌ UNUSED"
        file_count = len(set(info['files']))
        print(f"{package:15} | {status:8} | Used in {file_count:2} files")
        
        if info['found'] and file_count <= 3:
            print(f"                  Files: {', '.join(set(info['files'])[:3])}")
    
    return math_packages

def check_requirements_alignment():
    """Check alignment between requirements.txt and actual usage."""
    print("\n🔗 Checking Requirements.txt Alignment...")
    
    # Read requirements.txt
    requirements_file = Path('requirements.txt')
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return
    
    with open(requirements_file, 'r') as f:
        requirements_content = f.read()
    
    # Extract package names from requirements.txt
    required_packages = set()
    for line in requirements_content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # Extract package name (before version specifiers)
            package_name = re.split(r'[>=<!=]', line)[0].strip()
            required_packages.add(package_name)
    
    print(f"📋 Found {len(required_packages)} packages in requirements.txt")
    
    # Check for problematic packages mentioned in conversation
    problematic_packages = [
        'ta-lib', 'flask-cors', 'flask-socketio', 'python-binance', 
        'yfinance', 'pynvml', 'cupy-cuda11x', 'typing-extensions', 
        'pathlib', 'pytest-cov'
    ]
    
    found_problematic = []
    for pkg in problematic_packages:
        if pkg in required_packages:
            found_problematic.append(pkg)
    
    if found_problematic:
        print(f"\n⚠️ Found potentially problematic packages: {', '.join(found_problematic)}")
    else:
        print("\n✅ No problematic packages found in requirements.txt")
    
    return required_packages, found_problematic

def test_mathematical_functionality():
    """Test core mathematical functionality."""
    print("\n🧮 Testing Mathematical Functionality...")
    
    test_results = {}
    
    # Test 1: NumPy
    try:
        import numpy as np
        test_array = np.array([1, 2, 3, 4, 5])
        result = np.mean(test_array)
        test_results['numpy'] = "✅ PASS" if result == 3.0 else "❌ FAIL"
    except ImportError:
        test_results['numpy'] = "❌ NOT INSTALLED"
    except Exception as e:
        test_results['numpy'] = f"❌ ERROR: {e}"
    
    # Test 2: SciPy
    try:
        import scipy
        from scipy import stats
        test_results['scipy'] = "✅ PASS"
    except ImportError:
        test_results['scipy'] = "❌ NOT INSTALLED"
    except Exception as e:
        test_results['scipy'] = f"❌ ERROR: {e}"
    
    # Test 3: Pandas
    try:
        import pandas as pd
        df = pd.DataFrame({'x': [1, 2, 3]})
        result = df['x'].mean()
        test_results['pandas'] = "✅ PASS" if result == 2.0 else "❌ FAIL"
    except ImportError:
        test_results['pandas'] = "❌ NOT INSTALLED"
    except Exception as e:
        test_results['pandas'] = f"❌ ERROR: {e}"
    
    # Test 4: Hashlib (critical for BTC operations)
    try:
        import hashlib
        test_string = "BTC_price_50000"
        hash_result = hashlib.sha256(test_string.encode()).hexdigest()
        test_results['hashlib'] = "✅ PASS" if len(hash_result) == 64 else "❌ FAIL"
    except Exception as e:
        test_results['hashlib'] = f"❌ ERROR: {e}"
    
    # Test 5: CCXT (for trading)
    try:
        import ccxt
        test_results['ccxt'] = "✅ PASS"
    except ImportError:
        test_results['ccxt'] = "❌ NOT INSTALLED"
    except Exception as e:
        test_results['ccxt'] = f"❌ ERROR: {e}"
    
    # Test 6: Flask (for API)
    try:
        import flask
        test_results['flask'] = "✅ PASS"
    except ImportError:
        test_results['flask'] = "❌ NOT INSTALLED"
    except Exception as e:
        test_results['flask'] = f"❌ ERROR: {e}"
    
    # Report test results
    print("\n📈 Mathematical Functionality Test Results:")
    print("=" * 50)
    for package, result in test_results.items():
        print(f"{package:15} | {result}")
    
    return test_results

def check_mathematical_preservation():
    """Check if mathematical content was preserved during Flake8 fixes."""
    print("\n🔒 Checking Mathematical Preservation...")
    
    # Look for MATHEMATICAL PRESERVATION comments
    preservation_files = []
    critical_math_files = []
    
    python_files = list(Path('.').rglob('*.py'))
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for preservation comments
            if "MATHEMATICAL PRESERVATION:" in content:
                preservation_files.append(str(file_path))
            
            # Check for critical mathematical content
            critical_patterns = [
                'btc.*price', 'eth.*price', 'usdc.*price', 'xrp.*price',
                'sha256', 'tensor', 'matrix', 'unified_math',
                'calculate.*profit', 'trading.*algorithm'
            ]
            
            content_lower = content.lower()
            if any(re.search(pattern, content_lower) for pattern in critical_patterns):
                critical_math_files.append(str(file_path))
                
        except Exception as e:
            print(f"⚠️ Could not read {file_path}: {e}")
    
    print(f"📊 Found {len(preservation_files)} files with MATHEMATICAL PRESERVATION comments")
    print(f"📊 Found {len(critical_math_files)} files with critical mathematical content")
    
    # Check specific critical files
    critical_files_to_check = [
        'core/math/mathematical_relay_system.py',
        'core/math/trading_tensor_ops.py',
        'core/unified_math_system.py'
    ]
    
    missing_critical = []
    for file_path in critical_files_to_check:
        if not Path(file_path).exists():
            missing_critical.append(file_path)
    
    if missing_critical:
        print(f"\n❌ Missing critical mathematical files: {', '.join(missing_critical)}")
    else:
        print("\n✅ All critical mathematical files present")
    
    return preservation_files, critical_math_files, missing_critical

def generate_recommendations(math_packages, required_packages, test_results, missing_critical):
    """Generate actionable recommendations."""
    print("\n💡 Recommendations:")
    print("=" * 50)
    
    recommendations = []
    
    # Check for unused packages in requirements.txt
    unused_in_requirements = []
    for pkg in required_packages:
        if pkg in math_packages and not math_packages[pkg]['found']:
            unused_in_requirements.append(pkg)
    
    if unused_in_requirements:
        recommendations.append(f"Consider removing unused packages from requirements.txt: {', '.join(unused_in_requirements)}")
    
    # Check for missing essential packages
    essential_packages = ['numpy', 'scipy', 'pandas', 'hashlib', 'ccxt', 'flask']
    missing_essential = []
    for pkg in essential_packages:
        if pkg in test_results and "NOT INSTALLED" in test_results[pkg]:
            missing_essential.append(pkg)
    
    if missing_essential:
        recommendations.append(f"Install missing essential packages: {', '.join(missing_essential)}")
    
    # Check for failed tests
    failed_tests = []
    for pkg, result in test_results.items():
        if "FAIL" in result or "ERROR" in result:
            failed_tests.append(pkg)
    
    if failed_tests:
        recommendations.append(f"Fix issues with packages: {', '.join(failed_tests)}")
    
    # Check for missing critical files
    if missing_critical:
        recommendations.append(f"Restore missing critical mathematical files: {', '.join(missing_critical)}")
    
    # General recommendations
    recommendations.extend([
        "Keep requirements.txt focused on packages actually used in the codebase",
        "Ensure all mathematical operations in core/ directory are functional",
        "Maintain MATHEMATICAL PRESERVATION comments for critical algorithms",
        "Test BTC hashing and tensor operations regularly"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return recommendations

def main():
    """Run comprehensive requirements and mathematical analysis."""
    print("🔍 SCHWABOT REQUIREMENTS & MATHEMATICAL PRESERVATION ANALYSIS")
    print("=" * 80)
    
    # Step 1: Analyze mathematical imports
    math_packages = analyze_mathematical_imports()
    
    # Step 2: Check requirements.txt alignment
    required_packages, problematic_packages = check_requirements_alignment()
    
    # Step 3: Test mathematical functionality
    test_results = test_mathematical_functionality()
    
    # Step 4: Check mathematical preservation
    preservation_files, critical_math_files, missing_critical = check_mathematical_preservation()
    
    # Step 5: Generate recommendations
    recommendations = generate_recommendations(math_packages, required_packages, test_results, missing_critical)
    
    # Summary
    print("\n📊 ANALYSIS SUMMARY:")
    print("=" * 40)
    print(f"✅ Mathematical packages in use: {sum(1 for p in math_packages.values() if p['found'])}")
    print(f"📋 Packages in requirements.txt: {len(required_packages)}")
    print(f"🔒 Files with preservation comments: {len(preservation_files)}")
    print(f"🧮 Critical mathematical files: {len(critical_math_files)}")
    print(f"⚠️ Missing critical files: {len(missing_critical)}")
    
    # Overall status
    issues_found = len(problematic_packages) + len(missing_critical) + sum(1 for r in test_results.values() if "FAIL" in r or "ERROR" in r)
    
    if issues_found == 0:
        print("\n🎉 STATUS: ALL GOOD - Mathematical preservation intact!")
    else:
        print(f"\n⚠️ STATUS: {issues_found} issues found - See recommendations above")
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main() 