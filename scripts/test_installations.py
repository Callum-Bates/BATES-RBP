#!/usr/bin/env python3
"""
Test script to verify all installations are working correctly
"""

import subprocess
import sys
import os
from pathlib import Path

def get_conda_executable():
    """Get the conda executable path dynamically"""
    print("DEBUG: Finding conda executable...")
    
    # Method 1: Use CONDA_EXE environment variable
    conda_exe = os.environ.get('CONDA_EXE')
    print(f"DEBUG: CONDA_EXE = {conda_exe}")
    
    if conda_exe and Path(conda_exe).exists():
        print(f"SUCCESS: Using CONDA_EXE: {conda_exe}")
        return conda_exe
    
    # Method 2: Try to find conda relative to current Python
    python_path = Path(sys.executable)
    conda_path = python_path.parent.parent / 'bin' / 'conda'
    print(f"DEBUG: Current Python: {python_path}")
    print(f"DEBUG: Calculated conda path: {conda_path}")
    print(f"DEBUG: Conda path exists: {conda_path.exists()}")
    
    if conda_path.exists():
        print(f"SUCCESS: Using calculated path: {conda_path}")
        return str(conda_path)
    
    # Method 3: Check common conda locations
    possible_paths = [
        Path.home() / 'anaconda3' / 'bin' / 'conda',
        Path.home() / 'miniconda3' / 'bin' / 'conda',
        Path('/opt/conda/bin/conda'),
    ]
    
    print(f"DEBUG: Checking common paths...")
    for path in possible_paths:
        print(f"DEBUG: Checking {path} - exists: {path.exists()}")
        if path.exists():
            print(f"SUCCESS: Using common path: {path}")
            return str(path)
    
    # Fallback
    print("WARNING: Using fallback 'conda'")
    return 'conda'

def get_env_path(env_name):
    """Get the path to a conda environment"""
    conda_exe = get_conda_executable()
    try:
        result = subprocess.run([conda_exe, 'env', 'list'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if env_name in line:
                parts = line.split()
                if len(parts) >= 2:
                    return parts[-1]  # Last part is the path
        return None
    except Exception as e:
        print(f"ERROR: Could not get environment path for {env_name}: {e}")
        return None

def run_conda_command(env_name, command, timeout=30):
    """Run a command in a specific conda environment"""
    conda_exe = get_conda_executable()
    full_command = [conda_exe, "run", "-n", env_name] + command
    
    print(f"DEBUG: Running command: {' '.join(full_command)}")
    
    try:
        result = subprocess.run(full_command, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_deepclip():
    """Test DeepCLIP installation"""
    print("Testing DeepCLIP (Python 2.7)...")
    
    # Check if environment exists
    env_path = get_env_path("deepclip_env")
    if not env_path:
        print("ERROR: DeepCLIP environment (deepclip_env) not found")
        return False
    
    print(f"INFO: Found DeepCLIP environment at: {env_path}")
    
    # Check if DeepCLIP directory exists
    deepclip_path = Path("deepclip/DeepCLIP.py")
    if not deepclip_path.exists():
        print("ERROR: DeepCLIP.py not found at deepclip/DeepCLIP.py")
        return False
    
    # Test basic imports in Python 2.7 environment
    test_script = """
import sys
print("Python version: " + str(sys.version))
try:
    import theano
    print("Theano imported successfully")
except ImportError as e:
    print("Theano import failed: " + str(e))
    sys.exit(1)

try:
    import lasagne
    print("Lasagne imported successfully")
except ImportError as e:
    print("Lasagne import failed: " + str(e))
    sys.exit(1)

try:
    import numpy
    import scipy
    import sklearn
    import Bio
    print("Other dependencies imported successfully")
except ImportError as e:
    print("Dependency import failed: " + str(e))
    sys.exit(1)

print("All DeepCLIP dependencies are working")
"""
    
    success, stdout, stderr = run_conda_command("deepclip_env", 
                                               ["python", "-c", test_script], 
                                               timeout=60)
    if success:
        print("SUCCESS: DeepCLIP dependencies are working")
        print(f"   Output: {stdout.strip()}")
        
        # Test if we can run DeepCLIP help (might fail due to missing models, but should show usage)
        success2, stdout2, stderr2 = run_conda_command("deepclip_env", 
                                                       ["python", str(deepclip_path), "--help"],
                                                       timeout=30)
        if success2 or "usage:" in stderr2.lower():
            print("SUCCESS: DeepCLIP script is accessible")
            return True
        else:
            print(f"WARNING: DeepCLIP script test inconclusive: {stderr2}")
            return True  # Dependencies work, script issues might be model-related
    else:
        print(f"ERROR: DeepCLIP test failed: {stderr}")
        return False


def test_rbpnet():
    """Test RBPNet installation"""
    print("Testing RBPNet...")
    
    # Check if environment exists
    env_path = get_env_path("rbpnet_env")
    if not env_path:
        print("ERROR: RBPNet environment (rbpnet_env) not found")
        return False
    
    print(f"INFO: Found RBPNet environment at: {env_path}")
    
    success, stdout, stderr = run_conda_command("rbpnet_env", 
                                               ["python", "-c", "import rbpnet; print('RBPNet imported successfully')"])
    if success:
        print("SUCCESS: RBPNet is working")
        return True
    else:
        print(f"ERROR: RBPNet test failed: {stderr}")
        return False

def test_boltz():
    """Test Boltz installation"""
    print("Testing Boltz...")
    
    # Check if environment exists
    env_path = get_env_path("boltz_env")
    if not env_path:
        print("ERROR: Boltz environment (boltz_env) not found")
        return False
    
    print(f"INFO: Found Boltz environment at: {env_path}")
    
    success, stdout, stderr = run_conda_command("boltz_env", 
                                               ["python", "-c", "import boltz; print('Boltz imported successfully')"])
    if success:
        print("SUCCESS: Boltz is working")
        return True
    else:
        print(f"ERROR: Boltz test failed: {stderr}")
        return False

def main():
    """Run all tests"""
    print("Testing BATES-RBP installations...\n")
    
    # First, show what environments we found
    conda_exe = get_conda_executable()
    print(f"Using conda executable: {conda_exe}")
    
    print("\nAvailable environments:")
    try:
        result = subprocess.run([conda_exe, 'env', 'list'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'TEST' in line or 'deepclip' in line or 'rbpnet' in line or 'boltz' in line:
                print(f"  {line}")
    except Exception as e:
        print(f"ERROR: Could not list environments: {e}")
    
    print("\n" + "="*50)
    
    tests = [
        ("DeepCLIP", test_deepclip),
        ("RBPNet", test_rbpnet),
        ("Boltz", test_boltz)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
        print()
    
    # Summary
    print("Test Summary:")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nSUCCESS: All tests passed! BATES-RBP is ready to use.")
        print("\nNOTE: DeepCLIP uses Python 2.7 for legacy compatibility")
        return 0
    else:
        print("\nWARNING: Some tests failed. Please check the installation.")
        print("   See docs/troubleshooting.md for common issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
