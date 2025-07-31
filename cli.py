"""
Command Line Interface for LLM Testing Harness
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

from config import TestHarnessConfig, create_default_config, create_sample_configs
from test_harness import TestHarness


def create_config_command(args):
    """Create configuration files"""
    if args.type == 'default':
        config = create_default_config()
        config.save(args.output)
        print(f"Created default configuration: {args.output}")
    
    elif args.type == 'samples':
        configs = create_sample_configs()
        for name, config in configs.items():
            filename = f"config_{name}.yaml"
            config.save(filename)
            print(f"Created {name} configuration: {filename}")
    
    elif args.type == 'minimal':
        # Create a minimal config for testing
        from config import ModelConfig, BenchmarkConfig, EvaluationConfig, OutputConfig
        config = TestHarnessConfig(
            models=[
                ModelConfig(name="gpt-4o-mini", service="litellm", max_tokens=200)
            ],
            benchmarks=[
                BenchmarkConfig(
                    name="quick_test",
                    file_path="linguistic_benchmark.json",
                    question_type="open_ended"
                )
            ],
            evaluation=EvaluationConfig(
                evaluator_model="gpt-4o-mini",
                bootstrap_samples=100
            ),
            output=OutputConfig(base_path="./quick_outputs"),
            batch_size=1,
            verbose=True
        )
        config.save(args.output)
        print(f"Created minimal configuration: {args.output}")


async def run_benchmark_command(args):
    """Run benchmark tests"""
    try:
        # Load configuration
        if not Path(args.config).exists():
            print(f"Configuration file not found: {args.config}")
            print("Use 'llm-test create-config' to create one")
            sys.exit(1)
        
        config = TestHarnessConfig.from_file(args.config)
        
        # Apply command line overrides
        if args.quiet:
            config.verbose = False
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.output_dir:
            config.output.base_path = args.output_dir
        
        # Create and run test harness
        harness = TestHarness(config)
        
        if args.quick:
            print(f"Running quick test with {args.quick} questions...")
            result = await harness.quick_test(args.quick, args.models)
        else:
            result = await harness.run_benchmark(args.benchmark, args.models)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Duration: {result.duration:.2f} seconds")
        print(f"Output directory: {harness.output_dir}")
        
        if result.model_results:
            print(f"\nModels tested: {len(result.model_results)}")
            for model_name, model_result in result.model_results.items():
                completion_rate = (model_result['completed_questions'] / 
                                 model_result['total_questions'] * 100)
                print(f"  {model_name}: {model_result['completed_questions']}/{model_result['total_questions']} ({completion_rate:.1f}%)")
        
        if result.statistics:
            print(f"\nTop performing models:")
            sorted_models = sorted(result.statistics.items(), 
                                 key=lambda x: x[1].get('mean_score', 0), 
                                 reverse=True)
            for i, (model, stats) in enumerate(sorted_models[:5], 1):
                score = stats.get('mean_score', 0)
                std = stats.get('std_dev_score', 0)
                print(f"  {i}. {model}: {score:.1f} ± {std:.1f}")
        
        if result.errors:
            print(f"\nErrors encountered: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  - {error['error']}")
            if len(result.errors) > 3:
                print(f"  ... and {len(result.errors) - 3} more errors")
        
        print(f"\nFull results saved to: {harness.output_dir}")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)


def list_models_command(args):
    """List available models"""
    # This would be expanded to query actual available models
    print("Available LLM Models:")
    print("\nOpenAI Models (via litellm):")
    print("  - gpt-4o")
    print("  - gpt-4o-mini")
    print("  - gpt-4-turbo-preview")
    
    print("\nAnthropic Models (via litellm):")
    print("  - claude-3-5-sonnet-20240620")
    print("  - claude-3-opus-20240229")
    
    print("\nGoogle Models (via litellm):")
    print("  - gemini-1.5-pro")
    print("  - gemini-1.0-pro")
    
    print("\nMistral Models (via litellm):")
    print("  - mistral-large-latest")
    print("  - open-mixtral-8x22b")
    
    print("\nNote: Requires appropriate API keys in .env file")


def validate_config_command(args):
    """Validate configuration file"""
    try:
        config = TestHarnessConfig.from_file(args.config)
        print(f"Configuration file '{args.config}' is valid!")
        
        print(f"\nConfiguration Summary:")
        print(f"  Models: {len(config.models)}")
        for model in config.models:
            print(f"    - {model.name} ({model.service})")
        
        print(f"  Benchmarks: {len(config.benchmarks)}")
        for benchmark in config.benchmarks:
            print(f"    - {benchmark.name} ({benchmark.question_type})")
        
        print(f"  Output: {config.output.base_path}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Evaluation model: {config.evaluation.evaluator_model}")
        
        # Check if benchmark files exist
        missing_files = []
        for benchmark in config.benchmarks:
            if not Path(benchmark.file_path).exists():
                missing_files.append(benchmark.file_path)
        
        if missing_files:
            print(f"\nWarning: Missing benchmark files:")
            for file in missing_files:
                print(f"  - {file}")
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)


def analyze_results_command(args):
    """Analyze previous test results"""
    results_path = Path(args.results_dir) / "test_results.json"
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print(f"Test Results Analysis")
        print(f"{'='*50}")
        
        metadata = results.get('metadata', {})
        print(f"Test Date: {metadata.get('timestamp', 'Unknown')}")
        print(f"Duration: {metadata.get('duration_seconds', 0):.2f} seconds")
        
        model_summary = results.get('model_summary', {})
        if model_summary:
            print(f"\nModel Performance:")
            sorted_models = sorted(
                model_summary.items(),
                key=lambda x: x[1].get('mean_score', 0),
                reverse=True
            )
            
            for model, stats in sorted_models:
                if 'mean_score' in stats:
                    print(f"  {model}: {stats['mean_score']:.1f} ± {stats.get('std_score', 0):.1f}")
                else:
                    completion = stats.get('completion_rate', 0) * 100
                    print(f"  {model}: {completion:.1f}% completion")
        
        errors = results.get('errors', [])
        if errors:
            print(f"\nErrors: {len(errors)}")
            for error in errors[:3]:
                print(f"  - {error.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"Error analyzing results: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LLM Testing Harness - Evaluate Language Models on Benchmark Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a default configuration
  llm-test create-config --type default --output config.yaml
  
  # Run a quick test with 3 questions
  llm-test run --config config.yaml --quick 3
  
  # Run full benchmark on specific models
  llm-test run --config config.yaml --models gpt-4o claude-3-5-sonnet-20240620
  
  # Validate configuration
  llm-test validate --config config.yaml
  
  # Analyze previous results
  llm-test analyze ./test_outputs/test_run_2024-01-01_12-00-00/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create configuration files')
    config_parser.add_argument(
        '--type', 
        choices=['default', 'minimal', 'samples'], 
        default='default',
        help='Type of configuration to create'
    )
    config_parser.add_argument(
        '--output', 
        default='config.yaml',
        help='Output configuration file path'
    )
    
    # Run benchmark command
    run_parser = subparsers.add_parser('run', help='Run benchmark tests')
    run_parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Configuration file path'
    )
    run_parser.add_argument(
        '--benchmark',
        help='Specific benchmark to run'
    )
    run_parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to test'
    )
    run_parser.add_argument(
        '--quick',
        type=int,
        help='Quick test with N questions'
    )
    run_parser.add_argument(
        '--output-dir',
        help='Override output directory'
    )
    run_parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    run_parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    # List models command
    models_parser = subparsers.add_parser('list-models', help='List available models')
    
    # Validate config command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file to validate'
    )
    
    # Analyze results command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze test results')
    analyze_parser.add_argument(
        'results_dir',
        help='Directory containing test results'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute commands
    if args.command == 'create-config':
        create_config_command(args)
    elif args.command == 'run':
        asyncio.run(run_benchmark_command(args))
    elif args.command == 'list-models':
        list_models_command(args)
    elif args.command == 'validate':
        validate_config_command(args)
    elif args.command == 'analyze':
        analyze_results_command(args)


if __name__ == "__main__":
    main()