"""
Configuration management for LLM testing harness
"""

import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a single LLM model"""
    name: str
    service: str  # "litellm" or "custom"
    max_tokens: int = 1000
    temperature: float = 0.0
    num_retries: int = 2


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark questions"""
    name: str
    file_path: str
    question_type: str  # "open_ended" or "multiple_choice"
    description: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings"""
    evaluator_model: str = "gpt-4o"
    auto_eval: bool = True
    multiple_choice_scoring: bool = False
    bootstrap_samples: int = 10000


@dataclass
class OutputConfig:
    """Configuration for output settings"""
    base_path: str = "./test_outputs"
    answers_path: str = "answers"
    evaluations_path: str = "evaluations"
    statistics_path: str = "statistics"
    charts_path: str = "charts"
    timestamp_folders: bool = True


@dataclass
class TestHarnessConfig:
    """Main configuration for the test harness"""
    models: List[ModelConfig]
    benchmarks: List[BenchmarkConfig]
    evaluation: EvaluationConfig
    output: OutputConfig
    batch_size: int = 1
    parallel_execution: bool = False
    verbose: bool = True

    @classmethod
    def from_file(cls, config_path: str) -> "TestHarnessConfig":
        """Load configuration from a YAML or JSON file"""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestHarnessConfig":
        """Create configuration from dictionary"""
        models = [ModelConfig(**model) for model in data.get('models', [])]
        benchmarks = [BenchmarkConfig(**benchmark) for benchmark in data.get('benchmarks', [])]
        evaluation = EvaluationConfig(**data.get('evaluation', {}))
        output = OutputConfig(**data.get('output', {}))
        
        return cls(
            models=models,
            benchmarks=benchmarks,
            evaluation=evaluation,
            output=output,
            batch_size=data.get('batch_size', 1),
            parallel_execution=data.get('parallel_execution', False),
            verbose=data.get('verbose', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save(self, config_path: str):
        """Save configuration to file"""
        path = Path(config_path)
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif path.suffix.lower() == '.json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")


def create_default_config() -> TestHarnessConfig:
    """Create a default configuration for testing"""
    return TestHarnessConfig(
        models=[
            ModelConfig(
                name="gpt-4o-mini",
                service="litellm",
                max_tokens=500,
                temperature=0.0,
                num_retries=2
            ),
            ModelConfig(
                name="claude-3-5-sonnet-20240620",
                service="litellm",
                max_tokens=500,
                temperature=0.0,
                num_retries=2
            )
        ],
        benchmarks=[
            BenchmarkConfig(
                name="linguistic_benchmark",
                file_path="linguistic_benchmark.json",
                question_type="open_ended",
                description="Original benchmark with open-ended questions"
            ),
            BenchmarkConfig(
                name="linguistic_benchmark_multiple_choice",
                file_path="linguistic_benchmark_multi_choice.json",
                question_type="multiple_choice",
                description="Multiple choice version of the benchmark"
            )
        ],
        evaluation=EvaluationConfig(
            evaluator_model="gpt-4o",
            auto_eval=True,
            multiple_choice_scoring=True,
            bootstrap_samples=1000  # Reduced for faster testing
        ),
        output=OutputConfig(
            base_path="./test_outputs",
            timestamp_folders=True
        ),
        batch_size=2,
        parallel_execution=False,
        verbose=True
    )


def create_sample_configs():
    """Create sample configuration files"""
    
    # Quick test configuration
    quick_config = TestHarnessConfig(
        models=[
            ModelConfig(
                name="gpt-4o-mini",
                service="litellm",
                max_tokens=200,
                temperature=0.0,
                num_retries=1
            )
        ],
        benchmarks=[
            BenchmarkConfig(
                name="quick_test",
                file_path="linguistic_benchmark.json",
                question_type="open_ended",
                description="Quick test with first 5 questions"
            )
        ],
        evaluation=EvaluationConfig(
            evaluator_model="gpt-4o-mini",
            auto_eval=True,
            bootstrap_samples=100
        ),
        output=OutputConfig(
            base_path="./quick_test_outputs"
        ),
        batch_size=1,
        verbose=True
    )
    
    # Full evaluation configuration
    full_config = create_default_config()
    full_config.models.extend([
        ModelConfig(name="gpt-4-turbo-preview", service="litellm"),
        ModelConfig(name="gemini-1.5-pro", service="litellm"),
        ModelConfig(name="mistral-large-latest", service="litellm")
    ])
    full_config.evaluation.bootstrap_samples = 10000
    full_config.batch_size = 5
    
    return {
        'quick_test': quick_config,
        'full_evaluation': full_config,
        'default': create_default_config()
    }


if __name__ == "__main__":
    # Create sample configurations
    configs = create_sample_configs()
    
    for name, config in configs.items():
        config.save(f"config_{name}.yaml")
        print(f"Created {name} configuration: config_{name}.yaml")