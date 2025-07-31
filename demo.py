#!/usr/bin/env python3
"""
Demo script for the LLM Testing Harness

This script demonstrates the key capabilities of the testing harness
without requiring real API calls.
"""

import asyncio
import json
import tempfile
from pathlib import Path
import pandas as pd

from config import TestHarnessConfig, ModelConfig, BenchmarkConfig, EvaluationConfig, OutputConfig
from test_harness import TestHarness, TestResult


def create_demo_benchmark():
    """Create a small demo benchmark for testing"""
    demo_questions = [
        {
            "index": 1,
            "category": "Math",
            "question": "What is 2 + 2?",
            "human_answer": "4"
        },
        {
            "index": 2,
            "category": "Logic",
            "question": "If all cats are animals, and Fluffy is a cat, what is Fluffy?",
            "human_answer": "An animal"
        },
        {
            "index": 3,
            "category": "Language",
            "question": "Complete this sentence: The sky is ___",
            "human_answer": "blue"
        }
    ]
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(demo_questions, temp_file)
    temp_file.close()
    
    return temp_file.name, demo_questions


def create_demo_config(benchmark_file: str) -> TestHarnessConfig:
    """Create a demo configuration"""
    return TestHarnessConfig(
        models=[
            ModelConfig(
                name="demo_model_1",
                service="litellm",
                max_tokens=100,
                temperature=0.0
            ),
            ModelConfig(
                name="demo_model_2", 
                service="litellm",
                max_tokens=100,
                temperature=0.5
            )
        ],
        benchmarks=[
            BenchmarkConfig(
                name="demo_benchmark",
                file_path=benchmark_file,
                question_type="open_ended",
                description="Demo benchmark for testing"
            )
        ],
        evaluation=EvaluationConfig(
            evaluator_model="demo_evaluator",
            auto_eval=False,  # Disable for demo
            bootstrap_samples=100
        ),
        output=OutputConfig(
            base_path=tempfile.mkdtemp(),
            timestamp_folders=False
        ),
        batch_size=1,
        verbose=True
    )


class MockLLMService:
    """Mock LLM service for demonstration"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.responses = {
            "demo_model_1": {
                "What is 2 + 2?": "The answer is 4.",
                "If all cats are animals, and Fluffy is a cat, what is Fluffy?": "Fluffy is an animal.",
                "Complete this sentence: The sky is ___": "The sky is blue."
            },
            "demo_model_2": {
                "What is 2 + 2?": "2 + 2 equals 4",
                "If all cats are animals, and Fluffy is a cat, what is Fluffy?": "Since all cats are animals and Fluffy is a cat, Fluffy must be an animal.",
                "Complete this sentence: The sky is ___": "The sky is typically blue during the day."
            }
        }
    
    def completion(self, messages, model, **kwargs):
        """Mock completion method"""
        question = messages[0]['content']
        response = self.responses.get(model, {}).get(question, f"Mock response from {model}")
        
        return {
            "choices": [{
                "message": {"content": response}
            }]
        }


async def demo_basic_functionality():
    """Demonstrate basic harness functionality"""
    print("🎯 LLM Testing Harness - Demo")
    print("=" * 50)
    
    # Create demo benchmark
    print("📝 Creating demo benchmark...")
    benchmark_file, questions = create_demo_benchmark()
    print(f"   Created benchmark with {len(questions)} questions")
    
    try:
        # Create demo configuration
        print("\n⚙️  Creating demo configuration...")
        config = create_demo_config(benchmark_file)
        print(f"   Models: {[m.name for m in config.models]}")
        print(f"   Benchmarks: {[b.name for b in config.benchmarks]}")
        
        # Test configuration serialization
        print("\n💾 Testing configuration save/load...")
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        config.save(config_file.name)
        loaded_config = TestHarnessConfig.from_file(config_file.name)
        print(f"   ✅ Configuration successfully saved and loaded")
        
        # Create test harness
        print("\n🔧 Creating test harness...")
        harness = TestHarness(config)
        print(f"   Output directory: {harness.output_dir}")
        
        # Test benchmark loading
        print("\n📚 Testing benchmark loading...")
        benchmark_questions = harness._load_benchmark_questions()
        print(f"   ✅ Loaded {len(benchmark_questions['demo_benchmark'])} questions")
        
        # Create mock results
        print("\n🎭 Creating mock test results...")
        result = TestResult(config)
        
        # Simulate model answers
        for model in config.models:
            answers_data = []
            for i, question in enumerate(questions):
                mock_service = MockLLMService(model.name)
                mock_response = mock_service.completion(
                    [{"role": "user", "content": question["question"]}],
                    model.name
                )
                answers_data.append({
                    'index': question['index'],
                    'question': question['question'],
                    'human_answer': question['human_answer'],
                    'model_answer': mock_response['choices'][0]['message']['content'],
                    'score': 80 + (i * 5)  # Mock scores
                })
            
            answers_df = pd.DataFrame(answers_data)
            result.add_model_result(f"{model.name}_demo_benchmark", answers_df, answers_df)
            print(f"   ✅ Generated mock results for {model.name}")
        
        # Finalize results
        result.finalize()
        
        # Display results summary
        print("\n📊 Results Summary:")
        print("-" * 30)
        for model_name, model_result in result.model_results.items():
            completion_rate = (model_result['completed_questions'] / 
                             model_result['total_questions'] * 100)
            if model_result['scores'] is not None and 'score' in model_result['scores'].columns:
                avg_score = model_result['scores']['score'].mean()
                print(f"   {model_name}: {completion_rate:.0f}% completion, avg score: {avg_score:.1f}")
            else:
                print(f"   {model_name}: {completion_rate:.0f}% completion")
        
        print(f"\n   Test duration: {result.duration:.2f} seconds")
        
        # Save results
        results_dict = result.to_dict()
        results_file = harness.output_dir / "demo_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"   💾 Results saved to: {results_file}")
        
    finally:
        # Cleanup
        Path(benchmark_file).unlink()
        Path(config_file.name).unlink()
    
    print("\n✅ Demo completed successfully!")


def demo_cli_interface():
    """Demonstrate CLI interface capabilities"""
    print("\n🖥️  CLI Interface Demo")
    print("=" * 30)
    
    print("The CLI provides these commands:")
    commands = [
        ("create-config", "Create configuration files"),
        ("run", "Run benchmark tests"),
        ("validate", "Validate configuration files"),
        ("list-models", "List available models"),
        ("analyze", "Analyze test results")
    ]
    
    for cmd, desc in commands:
        print(f"   {cmd:15} - {desc}")
    
    print("\nExample usage:")
    examples = [
        "python cli.py create-config --type minimal",
        "python cli.py validate --config config.yaml",
        "python cli.py run --config config.yaml --quick 3",
        "python cli.py list-models",
        "python cli.py analyze ./test_outputs/test_run_*/"
    ]
    
    for example in examples:
        print(f"   {example}")


def demo_configuration_system():
    """Demonstrate configuration system capabilities"""
    print("\n⚙️  Configuration System Demo")
    print("=" * 35)
    
    # Show different configuration types
    from config import create_sample_configs
    
    configs = create_sample_configs()
    
    for name, config in configs.items():
        print(f"\n📄 {name.upper()} Configuration:")
        print(f"   Models: {len(config.models)}")
        for model in config.models:
            print(f"     - {model.name} ({model.service})")
        
        print(f"   Benchmarks: {len(config.benchmarks)}")
        for benchmark in config.benchmarks:
            print(f"     - {benchmark.name} ({benchmark.question_type})")
        
        print(f"   Batch size: {config.batch_size}")
        print(f"   Evaluation: {'Auto' if config.evaluation.auto_eval else 'Manual'}")


async def main():
    """Run all demos"""
    print("🚀 LLM Testing Harness - Complete Demo")
    print("=" * 60)
    
    await demo_basic_functionality()
    demo_cli_interface()
    demo_configuration_system()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed! The testing harness is ready to use.")
    print("\nNext steps:")
    print("1. Set up your API keys in .env file")
    print("2. Create a configuration: python cli.py create-config")
    print("3. Run a quick test: python cli.py run --quick 3")
    print("4. See HARNESS_README.md for detailed documentation")


if __name__ == "__main__":
    asyncio.run(main())