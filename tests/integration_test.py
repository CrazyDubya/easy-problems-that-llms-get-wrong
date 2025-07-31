#!/usr/bin/env python3
"""
Integration test to validate the complete LLM Testing Harness system
"""

import subprocess
import tempfile
import json
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_cli_functionality():
    """Test CLI functionality end-to-end"""
    print("🧪 Testing CLI Functionality")
    print("-" * 40)
    
    base_dir = Path(__file__).parent
    
    # Test 1: Create configuration
    print("1. Testing configuration creation...")
    success, stdout, stderr = run_command(
        "python cli.py create-config --type minimal --output test_integration.yaml",
        cwd=base_dir
    )
    
    if not success:
        print(f"   ❌ Failed: {stderr}")
        return False
    
    config_file = base_dir / "test_integration.yaml"
    if not config_file.exists():
        print("   ❌ Configuration file not created")
        return False
    
    print("   ✅ Configuration created successfully")
    
    # Test 2: Validate configuration
    print("2. Testing configuration validation...")
    success, stdout, stderr = run_command(
        "python cli.py validate --config test_integration.yaml",
        cwd=base_dir
    )
    
    if not success:
        print(f"   ❌ Validation failed: {stderr}")
        return False
    
    if "valid" not in stdout.lower():
        print(f"   ❌ Unexpected validation output: {stdout}")
        return False
    
    print("   ✅ Configuration validation passed")
    
    # Test 3: List models
    print("3. Testing model listing...")
    success, stdout, stderr = run_command(
        "python cli.py list-models",
        cwd=base_dir
    )
    
    if not success:
        print(f"   ❌ List models failed: {stderr}")
        return False
    
    if "OpenAI" not in stdout or "Anthropic" not in stdout:
        print(f"   ❌ Expected model providers not found in output")
        return False
    
    print("   ✅ Model listing works")
    
    # Test 4: Help functionality
    print("4. Testing help functionality...")
    success, stdout, stderr = run_command("python cli.py --help", cwd=base_dir)
    
    if not success:
        print(f"   ❌ Help failed: {stderr}")
        return False
    
    expected_commands = ["create-config", "run", "validate", "list-models", "analyze"]
    for cmd in expected_commands:
        if cmd not in stdout:
            print(f"   ❌ Expected command '{cmd}' not found in help")
            return False
    
    print("   ✅ Help functionality works")
    
    # Cleanup
    try:
        config_file.unlink()
    except:
        pass
    
    return True

def test_configuration_system():
    """Test the configuration system thoroughly"""
    print("\n🧪 Testing Configuration System")
    print("-" * 40)
    
    base_dir = Path(__file__).parent
    
    # Test configuration creation and loading
    print("1. Testing configuration serialization...")
    
    success, stdout, stderr = run_command(
        "python -c \"from config import create_sample_configs; configs = create_sample_configs(); [c.save(f'test_{n}.yaml') for n, c in configs.items()]\"",
        cwd=base_dir
    )
    
    if not success:
        print(f"   ❌ Failed to create sample configs: {stderr}")
        return False
    
    # Verify files were created
    expected_files = ["test_quick_test.yaml", "test_full_evaluation.yaml", "test_default.yaml"]
    for filename in expected_files:
        filepath = base_dir / filename
        if not filepath.exists():
            print(f"   ❌ Config file {filename} not created")
            return False
    
    print("   ✅ Sample configurations created")
    
    # Test loading each configuration
    print("2. Testing configuration loading...")
    for filename in expected_files:
        success, stdout, stderr = run_command(
            f"python cli.py validate --config {filename}",
            cwd=base_dir
        )
        
        if not success:
            print(f"   ❌ Failed to validate {filename}: {stderr}")
            return False
    
    print("   ✅ All configurations load successfully")
    
    # Cleanup
    for filename in expected_files:
        try:
            (base_dir / filename).unlink()
        except:
            pass
    
    return True

def test_unit_tests():
    """Run the unit test suite"""
    print("\n🧪 Running Unit Tests")
    print("-" * 40)
    
    base_dir = Path(__file__).parent
    
    success, stdout, stderr = run_command(
        "python tests/simple_test.py",
        cwd=base_dir
    )
    
    if not success:
        print(f"   ❌ Unit tests failed: {stderr}")
        print(f"   Output: {stdout}")
        return False
    
    if "OK" not in stdout or "✅" not in stdout:
        print(f"   ❌ Unexpected test output: {stdout}")
        return False
    
    print("   ✅ All unit tests passed")
    return True

def test_demo_script():
    """Test the demo script"""
    print("\n🧪 Testing Demo Script")
    print("-" * 40)
    
    base_dir = Path(__file__).parent
    
    success, stdout, stderr = run_command(
        "python demo.py",
        cwd=base_dir
    )
    
    if not success:
        print(f"   ❌ Demo script failed: {stderr}")
        return False
    
    expected_outputs = [
        "Demo completed successfully",
        "Configuration successfully saved and loaded",
        "Generated mock results",
        "🎉 Demo completed!"
    ]
    
    for expected in expected_outputs:
        if expected not in stdout:
            print(f"   ❌ Expected output '{expected}' not found")
            return False
    
    print("   ✅ Demo script runs successfully")
    return True

def test_benchmark_files():
    """Test that benchmark files are valid"""
    print("\n🧪 Testing Benchmark Files")
    print("-" * 40)
    
    base_dir = Path(__file__).parent
    benchmark_files = [
        "linguistic_benchmark.json",
        "linguistic_benchmark_multi_choice.json"
    ]
    
    for filename in benchmark_files:
        filepath = base_dir / filename
        
        if not filepath.exists():
            print(f"   ❌ Benchmark file {filename} not found")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list) or len(data) == 0:
                print(f"   ❌ {filename} is not a valid list or is empty")
                return False
            
            # Check first question structure
            first_q = data[0]
            required_fields = ['index', 'category', 'question', 'human_answer']
            
            for field in required_fields:
                if field not in first_q:
                    print(f"   ❌ {filename} missing required field: {field}")
                    return False
            
            # Additional checks for multiple choice
            if "multi_choice" in filename:
                mc_fields = ['multiple_choice', 'correct_answer']
                for field in mc_fields:
                    if field not in first_q:
                        print(f"   ❌ {filename} missing MC field: {field}")
                        return False
        
        except json.JSONDecodeError:
            print(f"   ❌ {filename} contains invalid JSON")
            return False
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            return False
    
    print("   ✅ All benchmark files are valid")
    return True

def test_dependencies():
    """Test that all required dependencies are available"""
    print("\n🧪 Testing Dependencies")
    print("-" * 40)
    
    required_modules = [
        'yaml', 'pandas', 'numpy', 'json', 'asyncio',
        'pathlib', 'tempfile', 'argparse', 'dataclasses'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} not available")
            return False
    
    # Test that custom modules can be imported
    base_dir = Path(__file__).parent
    
    try:
        sys.path.insert(0, str(base_dir))
        from config import TestHarnessConfig
        print("   ✅ config module")
    except ImportError as e:
        print(f"   ❌ config module: {e}")
        return False
    
    return True

def main():
    """Run all integration tests"""
    print("🔬 LLM Testing Harness - Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Benchmark Files", test_benchmark_files),
        ("Configuration System", test_configuration_system),
        ("CLI Functionality", test_cli_functionality),
        ("Unit Tests", test_unit_tests),
        ("Demo Script", test_demo_script),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🏃 Running {test_name} Tests...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Integration Test Results")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All integration tests passed!")
        print("The LLM Testing Harness is ready for use.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)