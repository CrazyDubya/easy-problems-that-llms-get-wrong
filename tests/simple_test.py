"""
Simple integration test for the test harness system
"""

import unittest
import tempfile
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TestHarnessConfig, ModelConfig, BenchmarkConfig, EvaluationConfig, OutputConfig


class TestConfiguration(unittest.TestCase):
    """Test the configuration system"""
    
    def test_model_config_creation(self):
        """Test ModelConfig creation and validation"""
        model = ModelConfig(
            name="gpt-4o-mini",
            service="litellm",
            max_tokens=500,
            temperature=0.0
        )
        
        self.assertEqual(model.name, "gpt-4o-mini")
        self.assertEqual(model.service, "litellm")
        self.assertEqual(model.max_tokens, 500)
        self.assertEqual(model.temperature, 0.0)
        self.assertEqual(model.num_retries, 2)  # default value
    
    def test_benchmark_config_creation(self):
        """Test BenchmarkConfig creation and validation"""
        benchmark = BenchmarkConfig(
            name="test_benchmark",
            file_path="test.json",
            question_type="open_ended",
            description="Test benchmark"
        )
        
        self.assertEqual(benchmark.name, "test_benchmark")
        self.assertEqual(benchmark.file_path, "test.json")
        self.assertEqual(benchmark.question_type, "open_ended")
        self.assertEqual(benchmark.description, "Test benchmark")
    
    def test_full_config_creation(self):
        """Test creating a complete configuration"""
        config = TestHarnessConfig(
            models=[
                ModelConfig(name="gpt-4o-mini", service="litellm"),
                ModelConfig(name="claude-3-5-sonnet-20240620", service="litellm")
            ],
            benchmarks=[
                BenchmarkConfig(
                    name="test_benchmark",
                    file_path="test.json",
                    question_type="open_ended"
                )
            ],
            evaluation=EvaluationConfig(
                evaluator_model="gpt-4o",
                auto_eval=True
            ),
            output=OutputConfig(
                base_path="./test_outputs"
            ),
            batch_size=1,
            verbose=True
        )
        
        self.assertEqual(len(config.models), 2)
        self.assertEqual(len(config.benchmarks), 1)
        self.assertEqual(config.evaluation.evaluator_model, "gpt-4o")
        self.assertEqual(config.output.base_path, "./test_outputs")
        self.assertEqual(config.batch_size, 1)
        self.assertTrue(config.verbose)
    
    def test_config_serialization(self):
        """Test converting config to/from dict"""
        config = TestHarnessConfig(
            models=[ModelConfig(name="test", service="litellm")],
            benchmarks=[BenchmarkConfig(name="test", file_path="test.json", question_type="open_ended")],
            evaluation=EvaluationConfig(),
            output=OutputConfig()
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('models', config_dict)
        self.assertIn('benchmarks', config_dict)
        self.assertEqual(len(config_dict['models']), 1)
        
        # Test from_dict
        loaded_config = TestHarnessConfig.from_dict(config_dict)
        self.assertEqual(len(loaded_config.models), 1)
        self.assertEqual(loaded_config.models[0].name, "test")
    
    def test_config_file_operations(self):
        """Test saving and loading config files"""
        config = TestHarnessConfig(
            models=[ModelConfig(name="test_model", service="litellm")],
            benchmarks=[BenchmarkConfig(name="test_bench", file_path="test.json", question_type="open_ended")],
            evaluation=EvaluationConfig(),
            output=OutputConfig()
        )
        
        # Test YAML format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            config.save(yaml_path)
            self.assertTrue(Path(yaml_path).exists())
            
            loaded_config = TestHarnessConfig.from_file(yaml_path)
            self.assertEqual(len(loaded_config.models), 1)
            self.assertEqual(loaded_config.models[0].name, "test_model")
            
        finally:
            Path(yaml_path).unlink()
        
        # Test JSON format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            config.save(json_path)
            self.assertTrue(Path(json_path).exists())
            
            loaded_config = TestHarnessConfig.from_file(json_path)
            self.assertEqual(len(loaded_config.models), 1)
            self.assertEqual(loaded_config.models[0].name, "test_model")
            
        finally:
            Path(json_path).unlink()
    
    def test_default_config_creation(self):
        """Test creating default configurations"""
        from config import create_default_config, create_sample_configs
        
        # Test default config
        default_config = create_default_config()
        self.assertIsInstance(default_config, TestHarnessConfig)
        self.assertGreater(len(default_config.models), 0)
        self.assertGreater(len(default_config.benchmarks), 0)
        
        # Test sample configs
        sample_configs = create_sample_configs()
        self.assertIsInstance(sample_configs, dict)
        self.assertIn('quick_test', sample_configs)
        self.assertIn('full_evaluation', sample_configs)
        self.assertIn('default', sample_configs)
        
        # Validate each sample config
        for name, config in sample_configs.items():
            self.assertIsInstance(config, TestHarnessConfig)
            self.assertGreater(len(config.models), 0)
            self.assertGreater(len(config.benchmarks), 0)


class TestBenchmarkData(unittest.TestCase):
    """Test benchmark data handling"""
    
    def test_benchmark_file_validation(self):
        """Test that benchmark files exist and are valid"""
        benchmark_files = [
            "linguistic_benchmark.json",
            "linguistic_benchmark_multi_choice.json"
        ]
        
        for filename in benchmark_files:
            filepath = Path(__file__).parent.parent / filename
            self.assertTrue(filepath.exists(), f"Benchmark file missing: {filename}")
            
            # Test that the file contains valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            
            # Test first question structure
            first_question = data[0]
            self.assertIn('index', first_question)
            self.assertIn('category', first_question)
            self.assertIn('question', first_question)
            self.assertIn('human_answer', first_question)
    
    def test_multiple_choice_structure(self):
        """Test multiple choice benchmark structure"""
        filepath = Path(__file__).parent.parent / "linguistic_benchmark_multi_choice.json"
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        first_mc_question = data[0]
        self.assertIn('multiple_choice', first_mc_question)
        self.assertIn('correct_answer', first_mc_question)
        self.assertIsInstance(first_mc_question['multiple_choice'], list)
        self.assertGreater(len(first_mc_question['multiple_choice']), 1)


class TestSystemIntegration(unittest.TestCase):
    """Test system integration without external API calls"""
    
    def test_cli_module_import(self):
        """Test that CLI module can be imported"""
        try:
            from cli import main
            self.assertTrue(callable(main))
        except ImportError as e:
            self.fail(f"Failed to import CLI module: {e}")
    
    def test_config_creation_command(self):
        """Test config creation functionality"""
        from config import create_sample_configs
        
        configs = create_sample_configs()
        
        # Create temporary configs and validate
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, config in configs.items():
                config_path = Path(temp_dir) / f"test_{name}.yaml"
                config.save(str(config_path))
                
                # Validate the file was created and is readable
                self.assertTrue(config_path.exists())
                
                loaded_config = TestHarnessConfig.from_file(str(config_path))
                self.assertIsInstance(loaded_config, TestHarnessConfig)


def run_simple_tests():
    """Run the simple test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    for test_class in [TestConfiguration, TestBenchmarkData, TestSystemIntegration]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running LLM Test Harness - Simple Integration Tests")
    print("=" * 60)
    
    success = run_simple_tests()
    
    if success:
        print("\n✅ All tests passed! The test harness system is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    exit(0 if success else 1)