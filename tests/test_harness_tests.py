"""
Unit tests for the LLM Testing Harness
"""

import unittest
import tempfile
import json
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TestHarnessConfig, ModelConfig, BenchmarkConfig, EvaluationConfig, OutputConfig
from test_harness import TestHarness, TestResult


class TestConfig(unittest.TestCase):
    """Test configuration management"""
    
    def test_model_config_creation(self):
        """Test ModelConfig creation"""
        model = ModelConfig(
            name="gpt-4o",
            service="litellm",
            max_tokens=500,
            temperature=0.0
        )
        self.assertEqual(model.name, "gpt-4o")
        self.assertEqual(model.service, "litellm")
        self.assertEqual(model.max_tokens, 500)
        self.assertEqual(model.temperature, 0.0)
        self.assertEqual(model.num_retries, 2)  # default value
    
    def test_benchmark_config_creation(self):
        """Test BenchmarkConfig creation"""
        benchmark = BenchmarkConfig(
            name="test_benchmark",
            file_path="test.json",
            question_type="open_ended"
        )
        self.assertEqual(benchmark.name, "test_benchmark")
        self.assertEqual(benchmark.file_path, "test.json")
        self.assertEqual(benchmark.question_type, "open_ended")
    
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = TestHarnessConfig(
            models=[ModelConfig(name="test", service="litellm")],
            benchmarks=[BenchmarkConfig(name="test", file_path="test.json", question_type="open_ended")],
            evaluation=EvaluationConfig(),
            output=OutputConfig()
        )
        
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('models', config_dict)
        self.assertIn('benchmarks', config_dict)
        self.assertEqual(len(config_dict['models']), 1)
    
    def test_config_from_dict(self):
        """Test configuration deserialization"""
        config_dict = {
            'models': [{'name': 'test', 'service': 'litellm'}],
            'benchmarks': [{'name': 'test', 'file_path': 'test.json', 'question_type': 'open_ended'}],
            'evaluation': {},
            'output': {}
        }
        
        config = TestHarnessConfig.from_dict(config_dict)
        self.assertEqual(len(config.models), 1)
        self.assertEqual(config.models[0].name, 'test')
        self.assertEqual(len(config.benchmarks), 1)
        self.assertEqual(config.benchmarks[0].name, 'test')
    
    def test_config_file_operations(self):
        """Test saving and loading configuration files"""
        config = TestHarnessConfig(
            models=[ModelConfig(name="test", service="litellm")],
            benchmarks=[BenchmarkConfig(name="test", file_path="test.json", question_type="open_ended")],
            evaluation=EvaluationConfig(),
            output=OutputConfig()
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            loaded_config = TestHarnessConfig.from_file(temp_path)
            
            self.assertEqual(len(loaded_config.models), 1)
            self.assertEqual(loaded_config.models[0].name, "test")
            
        finally:
            Path(temp_path).unlink()


class TestTestResult(unittest.TestCase):
    """Test TestResult class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TestHarnessConfig(
            models=[ModelConfig(name="test", service="litellm")],
            benchmarks=[BenchmarkConfig(name="test", file_path="test.json", question_type="open_ended")],
            evaluation=EvaluationConfig(),
            output=OutputConfig()
        )
        self.result = TestResult(self.config)
    
    def test_result_initialization(self):
        """Test TestResult initialization"""
        self.assertIsNotNone(self.result.start_time)
        self.assertIsNone(self.result.end_time)
        self.assertEqual(len(self.result.model_results), 0)
        self.assertEqual(len(self.result.errors), 0)
    
    def test_add_model_result(self):
        """Test adding model results"""
        import pandas as pd
        
        df = pd.DataFrame({
            'model_answer': ['answer1', 'answer2'],
            'score': [80, 90]
        })
        
        self.result.add_model_result("test_model", df)
        
        self.assertIn("test_model", self.result.model_results)
        self.assertEqual(self.result.model_results["test_model"]['total_questions'], 2)
        self.assertEqual(self.result.model_results["test_model"]['completed_questions'], 2)
    
    def test_add_error(self):
        """Test adding errors"""
        self.result.add_error("Test error", "test_model")
        
        self.assertEqual(len(self.result.errors), 1)
        self.assertEqual(self.result.errors[0]['error'], "Test error")
        self.assertEqual(self.result.errors[0]['model'], "test_model")
    
    def test_finalize(self):
        """Test result finalization"""
        self.result.finalize()
        
        self.assertIsNotNone(self.result.end_time)
        self.assertIsNotNone(self.result.duration)
        self.assertGreaterEqual(self.result.duration, 0)
    
    def test_to_dict(self):
        """Test result serialization"""
        import pandas as pd
        
        df = pd.DataFrame({
            'model_answer': ['answer1'],
            'score': [80]
        })
        
        self.result.add_model_result("test_model", df, df)
        self.result.finalize()
        
        result_dict = self.result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertIn('metadata', result_dict)
        self.assertIn('model_summary', result_dict)
        self.assertIn('test_model', result_dict['model_summary'])


class TestTestHarness(unittest.TestCase):
    """Test TestHarness class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TestHarnessConfig(
            models=[ModelConfig(name="test_model", service="litellm", max_tokens=100)],
            benchmarks=[BenchmarkConfig(name="test_benchmark", file_path="test.json", question_type="open_ended")],
            evaluation=EvaluationConfig(auto_eval=False),  # Disable auto eval for testing
            output=OutputConfig(base_path=tempfile.mkdtemp(), timestamp_folders=False)
        )
        self.harness = TestHarness(self.config)
    
    def test_harness_initialization(self):
        """Test TestHarness initialization"""
        self.assertIsInstance(self.harness.config, TestHarnessConfig)
        self.assertTrue(self.harness.output_dir.exists())
    
    def test_output_directory_setup(self):
        """Test output directory creation"""
        self.assertTrue(self.harness.output_dir.exists())
        
        # Check subdirectories
        expected_dirs = ['answers', 'evaluations', 'statistics', 'charts', 'logs']
        for dirname in expected_dirs:
            self.assertTrue((self.harness.output_dir / dirname).exists())
    
    def test_load_benchmark_questions(self):
        """Test loading benchmark questions"""
        # Create a temporary benchmark file
        test_questions = [
            {
                "index": 1,
                "category": "Test",
                "question": "What is 2+2?",
                "human_answer": "4"
            }
        ]
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_questions, temp_file)
        temp_file.close()
        
        try:
            # Update config to use temp file
            self.config.benchmarks[0].file_path = temp_file.name
            
            questions = self.harness._load_benchmark_questions()
            
            self.assertIn("test_benchmark", questions)
            self.assertEqual(len(questions["test_benchmark"]), 1)
            self.assertEqual(questions["test_benchmark"][0]["question"], "What is 2+2?")
            
        finally:
            Path(temp_file.name).unlink()
    
    @patch('test_harness.save_answers_as_json')
    @patch('test_harness.runner')
    @patch('test_harness.litellm_service')
    async def test_get_model_answers(self, mock_litellm, mock_runner, mock_save):
        """Test getting model answers"""
        import pandas as pd
        
        # Mock the LLM service and runner
        mock_service = Mock()
        mock_litellm.return_value = mock_service
        mock_runner.return_value = [Mock(), Mock()]  # Mock responses
        
        mock_df = pd.DataFrame({'model_answer': ['answer1', 'answer2']})
        mock_save.return_value = mock_df
        
        questions = [
            {"question": "Test question 1"},
            {"question": "Test question 2"}
        ]
        
        result = await self.harness._get_model_answers(
            self.config.models[0], questions, "test_benchmark", "open_ended"
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        mock_runner.assert_called_once()
        mock_save.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create minimal test data
        self.test_questions = [
            {
                "index": 1,
                "category": "Test",
                "question": "What is 2+2?",
                "human_answer": "4"
            },
            {
                "index": 2,
                "category": "Test",
                "question": "What is 3+3?",
                "human_answer": "6"
            }
        ]
        
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_file = Path(self.temp_dir) / "test_benchmark.json"
        
        with open(self.benchmark_file, 'w') as f:
            json.dump(self.test_questions, f)
    
    def test_mock_benchmark_run(self):
        """Test a complete benchmark run with mocked LLM calls"""
        config = TestHarnessConfig(
            models=[ModelConfig(name="mock_model", service="litellm", max_tokens=50)],
            benchmarks=[BenchmarkConfig(
                name="test_benchmark",
                file_path=str(self.benchmark_file),
                question_type="open_ended"
            )],
            evaluation=EvaluationConfig(auto_eval=False),
            output=OutputConfig(base_path=self.temp_dir, timestamp_folders=False)
        )
        
        harness = TestHarness(config)
        
        # Test that the harness can be initialized and setup properly
        self.assertIsInstance(harness.config, TestHarnessConfig)
        self.assertTrue(harness.output_dir.exists())
        
        # Test loading benchmark questions
        questions = harness._load_benchmark_questions()
        self.assertIn("test_benchmark", questions)
        self.assertEqual(len(questions["test_benchmark"]), 2)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    for test_class in [TestConfig, TestTestResult, TestTestHarness, TestIntegration]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)