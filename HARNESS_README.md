# LLM Testing Harness

A comprehensive testing framework for evaluating Large Language Models (LLMs) against benchmark problems. This harness provides automated testing, evaluation, and reporting capabilities for multiple LLM providers.

## Features

- **Multi-Provider Support**: Works with OpenAI, Anthropic, Google, Mistral, AWS Bedrock, and Azure AI
- **Configurable Testing**: YAML/JSON configuration files for easy setup
- **Automated Evaluation**: Both multiple-choice scoring and LLM-based evaluation
- **Statistical Analysis**: Bootstrap confidence intervals and performance metrics
- **Command Line Interface**: Easy-to-use CLI for running tests
- **Extensible Architecture**: Plugin-style support for new models and benchmarks
- **Comprehensive Testing**: Unit tests and integration tests included

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Setup API Keys

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_AI_STUDIO=your_google_key
# ... add other keys as needed
```

### 3. Create Configuration

```bash
python cli.py create-config --type minimal --output my_config.yaml
```

### 4. Run Tests

```bash
# Quick test with 3 questions
python cli.py run --config my_config.yaml --quick 3

# Full benchmark
python cli.py run --config my_config.yaml
```

## Command Line Interface

### Available Commands

- `create-config`: Create configuration files
- `run`: Run benchmark tests
- `validate`: Validate configuration files
- `list-models`: List available models
- `analyze`: Analyze previous test results

### Examples

```bash
# Create different types of configurations
python cli.py create-config --type default --output config.yaml
python cli.py create-config --type minimal --output minimal.yaml
python cli.py create-config --type samples  # Creates multiple example configs

# Validate configuration
python cli.py validate --config config.yaml

# Run specific models only
python cli.py run --config config.yaml --models gpt-4o claude-3-5-sonnet-20240620

# Run specific benchmark
python cli.py run --config config.yaml --benchmark linguistic_benchmark

# Quick test for development
python cli.py run --config config.yaml --quick 5 --quiet

# Analyze previous results
python cli.py analyze ./test_outputs/test_run_2024-01-01_12-00-00/
```

## Configuration

### Configuration File Structure

```yaml
models:
  - name: gpt-4o-mini
    service: litellm
    max_tokens: 500
    temperature: 0.0
    num_retries: 2

benchmarks:
  - name: linguistic_benchmark
    file_path: linguistic_benchmark.json
    question_type: open_ended
    description: Original benchmark questions
  - name: multiple_choice_benchmark
    file_path: linguistic_benchmark_multi_choice.json
    question_type: multiple_choice
    description: Multiple choice version

evaluation:
  evaluator_model: gpt-4o
  auto_eval: true
  multiple_choice_scoring: true
  bootstrap_samples: 1000

output:
  base_path: ./test_outputs
  timestamp_folders: true

batch_size: 2
parallel_execution: false
verbose: true
```

### Model Services

- **litellm**: Uses LiteLLM for unified API access (recommended)
- **custom**: Uses custom implementations for specific providers

### Question Types

- **open_ended**: Questions with free-form text answers
- **multiple_choice**: Questions with A/B/C/D answer choices

## Programming API

### Basic Usage

```python
from config import TestHarnessConfig
from test_harness import TestHarness
import asyncio

# Load configuration
config = TestHarnessConfig.from_file('config.yaml')

# Create and run test harness
harness = TestHarness(config)
result = await harness.run_benchmark()

print(f"Test completed in {result.duration:.2f} seconds")
print(f"Results saved to: {harness.output_dir}")
```

### Creating Custom Configurations

```python
from config import (
    TestHarnessConfig, ModelConfig, BenchmarkConfig, 
    EvaluationConfig, OutputConfig
)

config = TestHarnessConfig(
    models=[
        ModelConfig(
            name="gpt-4o-mini",
            service="litellm",
            max_tokens=200,
            temperature=0.0
        )
    ],
    benchmarks=[
        BenchmarkConfig(
            name="custom_benchmark",
            file_path="my_questions.json",
            question_type="open_ended"
        )
    ],
    evaluation=EvaluationConfig(
        evaluator_model="gpt-4o",
        auto_eval=True
    ),
    output=OutputConfig(
        base_path="./my_results"
    )
)

# Save configuration
config.save('my_custom_config.yaml')
```

### Quick Testing

```python
# Run a quick test with limited questions
result = await harness.quick_test(num_questions=3)

# Test specific models only
result = await harness.run_benchmark(
    model_names=['gpt-4o-mini', 'claude-3-5-sonnet-20240620']
)

# Test specific benchmark only
result = await harness.run_benchmark(
    benchmark_name='linguistic_benchmark'
)
```

## Output and Results

### Directory Structure

```
test_outputs/
├── test_run_2024-01-01_12-00-00/
│   ├── answers/
│   │   ├── linguistic_benchmark/
│   │   │   ├── final_answers-gpt-4o-mini.json
│   │   │   └── final_answers-claude-3-5-sonnet-20240620.json
│   │   └── multiple_choice_benchmark/
│   │       ├── final_answers-gpt-4o-mini.json
│   │       └── final_answers-claude-3-5-sonnet-20240620.json
│   ├── evaluations/
│   │   ├── auto_eval-gpt-4o-mini.json
│   │   └── auto_eval-claude-3-5-sonnet-20240620.json
│   ├── statistics/
│   │   ├── final_stats.csv
│   │   └── detailed_stats.json
│   ├── charts/
│   └── test_results.json
```

### Result Analysis

```python
import json
import pandas as pd

# Load test results
with open('test_outputs/test_run_*/test_results.json', 'r') as f:
    results = json.load(f)

# Load statistics
stats_df = pd.read_csv('test_outputs/test_run_*/statistics/final_stats.csv')
print(stats_df)

# Load detailed answers
answers_df = pd.read_json('test_outputs/test_run_*/answers/*/final_answers-*.json')
```

## Extending the Framework

### Adding New Models

1. Add model configuration to your config file
2. Ensure API keys are set in `.env`
3. For custom providers, extend the `custom_llm_service` class

### Adding New Benchmarks

1. Create a JSON file with your questions
2. Add benchmark configuration to your config file
3. Questions should follow this structure:

```json
[
    {
        "index": 1,
        "category": "Logic",
        "question": "Your question here",
        "human_answer": "Expected answer",
        "multiple_choice": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": "Option A"
    }
]
```

### Custom Evaluation

You can disable auto-evaluation and implement custom scoring:

```python
config.evaluation.auto_eval = False

# Run benchmark without auto-evaluation
result = await harness.run_benchmark()

# Implement custom evaluation logic
# Access raw answers via result.model_results
```

## Testing

Run the test suite to verify everything is working:

```bash
python tests/simple_test.py
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required API keys are set in `.env`
2. **Import Errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
3. **Configuration Errors**: Validate your config: `python cli.py validate --config your_config.yaml`
4. **Model Not Found**: Check that model names match exactly what the provider expects

### Debug Mode

Run with verbose output to see detailed information:

```bash
python cli.py run --config config.yaml --quick 1 --verbose
```

### Rate Limits

If you hit rate limits, try:

- Reducing `batch_size` in your configuration
- Adding delays between requests
- Using fewer models or questions for testing

## License

This project extends the original "Easy Problems That LLMs Get Wrong" research. See the main repository for license information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `python tests/simple_test.py`
5. Submit a pull request