"""
LLM Testing Harness Framework
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import asdict

from config import TestHarnessConfig, ModelConfig, BenchmarkConfig
from llm_service import litellm_service, custom_llm_service, runner, message_parse
from utils import model_clean, save_answers_as_json, calculate_llm_stats
from auto_eval import (
    create_all_llm_eval_messages,
    get_llm_eval_responses,
    extract_all_scores,
    create_auto_eval_json,
    score_multiple_choice_answers,
    validation_func
)
from multiple_choice import construct_multiple_choice_question


class TestResult:
    """Container for test results"""
    
    def __init__(self, config: TestHarnessConfig):
        self.config = config
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = None
        self.model_results = {}
        self.statistics = {}
        self.errors = []
        self.metadata = {
            'config': asdict(config),
            'timestamp': self.start_time.isoformat(),
            'version': '1.0'
        }
    
    def add_model_result(self, model_name: str, answers_df: pd.DataFrame, scores: Optional[pd.DataFrame] = None):
        """Add results for a specific model"""
        self.model_results[model_name] = {
            'answers': answers_df,
            'scores': scores,
            'total_questions': len(answers_df),
            'completed_questions': len(answers_df.dropna(subset=['model_answer']))
        }
    
    def add_error(self, error: str, model: Optional[str] = None):
        """Add an error to the results"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'model': model
        })
    
    def finalize(self):
        """Finalize the test results"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.metadata['end_timestamp'] = self.end_time.isoformat()
        self.metadata['duration_seconds'] = self.duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization"""
        result_dict = {
            'metadata': self.metadata,
            'statistics': self.statistics,
            'errors': self.errors,
            'model_summary': {}
        }
        
        for model_name, results in self.model_results.items():
            result_dict['model_summary'][model_name] = {
                'total_questions': results['total_questions'],
                'completed_questions': results['completed_questions'],
                'completion_rate': results['completed_questions'] / results['total_questions'] if results['total_questions'] > 0 else 0
            }
            
            if results['scores'] is not None and 'score' in results['scores'].columns:
                scores = results['scores']['score'].dropna()
                if len(scores) > 0:
                    result_dict['model_summary'][model_name].update({
                        'mean_score': float(scores.mean()),
                        'std_score': float(scores.std()),
                        'min_score': float(scores.min()),
                        'max_score': float(scores.max()),
                        'median_score': float(scores.median())
                    })
        
        return result_dict


class TestHarness:
    """Main test harness for LLM benchmarking"""
    
    def __init__(self, config: TestHarnessConfig):
        self.config = config
        self.output_dir = self._setup_output_directory()
        self.benchmark_questions = {}
        
    def _setup_output_directory(self) -> Path:
        """Setup the output directory structure"""
        base_path = Path(self.config.output.base_path)
        
        if self.config.output.timestamp_folders:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = base_path / f"test_run_{timestamp}"
        else:
            output_dir = base_path
        
        # Create subdirectories
        for subdir in ['answers', 'evaluations', 'statistics', 'charts', 'logs']:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _load_benchmark_questions(self) -> Dict[str, List[Dict]]:
        """Load all benchmark question sets"""
        benchmark_questions = {}
        
        for benchmark in self.config.benchmarks:
            try:
                with open(benchmark.file_path, 'r') as f:
                    questions = json.load(f)
                
                # Process questions based on type
                if benchmark.question_type == "multiple_choice":
                    processed_questions = []
                    for q in questions:
                        prompt, correct_letter = construct_multiple_choice_question(q)
                        q_processed = q.copy()
                        q_processed['multi_choice_question'] = prompt
                        q_processed['correct_letter'] = correct_letter
                        processed_questions.append(q_processed)
                    questions = processed_questions
                
                benchmark_questions[benchmark.name] = questions
                
                if self.config.verbose:
                    print(f"Loaded {len(questions)} questions from {benchmark.file_path}")
                    
            except Exception as e:
                error_msg = f"Failed to load benchmark {benchmark.name}: {str(e)}"
                print(f"Error: {error_msg}")
                raise RuntimeError(error_msg)
        
        return benchmark_questions
    
    async def _get_model_answers(self, model_config: ModelConfig, questions: List[Dict], 
                                benchmark_name: str, question_type: str) -> pd.DataFrame:
        """Get answers from a specific model"""
        try:
            # Prepare messages
            question_key = "multi_choice_question" if question_type == "multiple_choice" else "question"
            messages = [
                [{"role": "user", "content": q[question_key]}]
                for q in questions
            ]
            
            # Get LLM service
            llm_service_func = (
                litellm_service() if model_config.service == "litellm" 
                else custom_llm_service()
            )
            
            # Prepare hyperparameters
            hyperparams = {
                'max_tokens': model_config.max_tokens,
                'temperature': model_config.temperature,
                'num_retries': model_config.num_retries,
                'batch_size': self.config.batch_size
            }
            
            if self.config.verbose:
                print(f"Getting answers from {model_config.name} for {benchmark_name}...")
            
            # Get answers
            answers = await runner(
                llm_service_func.completion,
                messages=messages,
                model=model_config.name,
                validation_func=validation_func if question_type == "multiple_choice" else lambda x: True,
                **hyperparams
            )
            
            # Save answers
            answers_path = self.output_dir / "answers" / benchmark_name
            answers_df = save_answers_as_json(
                answers, questions, model_config.name, str(answers_path)
            )
            
            return answers_df
            
        except Exception as e:
            error_msg = f"Error getting answers from {model_config.name}: {str(e)}"
            print(f"Error: {error_msg}")
            raise RuntimeError(error_msg)
    
    async def _evaluate_answers(self, model_answers: Dict[str, pd.DataFrame], 
                               benchmark_questions: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
        """Evaluate model answers"""
        all_evaluated_results = {}
        
        for benchmark_name, questions in benchmark_questions.items():
            benchmark_config = next(b for b in self.config.benchmarks if b.name == benchmark_name)
            
            if benchmark_config.question_type == "multiple_choice":
                # Use multiple choice scoring
                eval_path = str(self.output_dir / "evaluations" / benchmark_name)
                evaluated_results = score_multiple_choice_answers(
                    {model: df for model, df in model_answers.items() if benchmark_name in model},
                    eval_path
                )
                all_evaluated_results.update(evaluated_results)
                
            elif self.config.evaluation.auto_eval:
                # Use LLM-based evaluation
                eval_messages = create_all_llm_eval_messages(model_answers, questions)
                
                eval_hyperparams = {
                    'max_tokens': 1000,
                    'temperature': 0,
                    'num_retries': 2,
                    'batch_size': self.config.batch_size
                }
                
                eval_responses = await get_llm_eval_responses(
                    eval_messages,
                    (self.config.evaluation.evaluator_model, "litellm"),
                    eval_hyperparams,
                    validation_func
                )
                
                scores = extract_all_scores(eval_responses)
                
                eval_path = self.output_dir / "evaluations" / benchmark_name
                evaluated_results = create_auto_eval_json(
                    scores, eval_responses, model_answers, questions, str(eval_path)
                )
                all_evaluated_results.update(evaluated_results)
        
        return all_evaluated_results
    
    def _calculate_statistics(self, evaluated_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate statistics for all models"""
        stats_dict = calculate_llm_stats(evaluated_results, self.config.evaluation.bootstrap_samples)
        
        stats_df = pd.DataFrame(stats_dict).transpose().sort_values('mean_score', ascending=False)
        stats_df.index.name = 'model'
        
        # Save statistics
        stats_path = self.output_dir / "statistics"
        stats_df.to_csv(stats_path / "final_stats.csv")
        
        # Save detailed statistics as JSON
        with open(stats_path / "detailed_stats.json", 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        return stats_df
    
    async def run_benchmark(self, benchmark_name: Optional[str] = None, 
                           model_names: Optional[List[str]] = None,
                           limit_questions: Optional[int] = None) -> TestResult:
        """Run the complete benchmark suite"""
        result = TestResult(self.config)
        
        try:
            if self.config.verbose:
                print("Starting LLM Benchmark Test Harness")
                print(f"Output directory: {self.output_dir}")
            
            # Load benchmark questions
            all_benchmark_questions = self._load_benchmark_questions()
            
            # Filter benchmarks if specified
            if benchmark_name:
                if benchmark_name not in all_benchmark_questions:
                    raise ValueError(f"Benchmark '{benchmark_name}' not found")
                all_benchmark_questions = {benchmark_name: all_benchmark_questions[benchmark_name]}
            
            # Filter models if specified
            models_to_test = self.config.models
            if model_names:
                models_to_test = [m for m in self.config.models if m.name in model_names]
                if not models_to_test:
                    raise ValueError(f"No matching models found: {model_names}")
            
            # Limit questions if specified (for testing)
            if limit_questions:
                for bench_name in all_benchmark_questions:
                    all_benchmark_questions[bench_name] = all_benchmark_questions[bench_name][:limit_questions]
            
            # Get answers from all models for all benchmarks
            all_model_answers = {}
            for benchmark_name, questions in all_benchmark_questions.items():
                benchmark_config = next(b for b in self.config.benchmarks if b.name == benchmark_name)
                
                for model_config in models_to_test:
                    try:
                        answers_df = await self._get_model_answers(
                            model_config, questions, benchmark_name, benchmark_config.question_type
                        )
                        
                        model_key = f"{model_config.name}_{benchmark_name}"
                        all_model_answers[model_key] = answers_df
                        result.add_model_result(model_key, answers_df)
                        
                    except Exception as e:
                        result.add_error(str(e), model_config.name)
                        if self.config.verbose:
                            print(f"Skipping {model_config.name} due to error: {e}")
            
            # Evaluate answers if we have any results
            if all_model_answers and self.config.evaluation.auto_eval:
                if self.config.verbose:
                    print("Evaluating model answers...")
                
                try:
                    evaluated_results = await self._evaluate_answers(all_model_answers, all_benchmark_questions)
                    
                    # Update results with scores
                    for model_key, eval_df in evaluated_results.items():
                        if model_key in result.model_results:
                            result.model_results[model_key]['scores'] = eval_df
                    
                    # Calculate statistics
                    if self.config.verbose:
                        print("Calculating statistics...")
                    
                    stats_df = self._calculate_statistics(evaluated_results)
                    result.statistics = stats_df.to_dict('index')
                    
                    if self.config.verbose:
                        print("\nFinal Results:")
                        print(stats_df)
                
                except Exception as e:
                    result.add_error(f"Evaluation failed: {str(e)}")
                    if self.config.verbose:
                        print(f"Evaluation error: {e}")
            
        except Exception as e:
            result.add_error(f"Benchmark failed: {str(e)}")
            raise
        
        finally:
            result.finalize()
            
            # Save complete results
            results_path = self.output_dir / "test_results.json"
            with open(results_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            if self.config.verbose:
                print(f"\nTest completed in {result.duration:.2f} seconds")
                print(f"Results saved to: {self.output_dir}")
        
        return result
    
    async def quick_test(self, num_questions: int = 3, models: Optional[List[str]] = None) -> TestResult:
        """Run a quick test with limited questions"""
        if self.config.verbose:
            print(f"Running quick test with {num_questions} questions")
        
        return await self.run_benchmark(
            limit_questions=num_questions,
            model_names=models
        )


async def run_test_harness(config_path: str, **kwargs) -> TestResult:
    """Convenience function to run the test harness from a config file"""
    config = TestHarnessConfig.from_file(config_path)
    harness = TestHarness(config)
    return await harness.run_benchmark(**kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Testing Harness")
    parser.add_argument("--config", default="config_default.yaml", help="Configuration file path")
    parser.add_argument("--benchmark", help="Specific benchmark to run")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--quick", type=int, help="Quick test with N questions")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    async def main():
        # Load config
        try:
            config = TestHarnessConfig.from_file(args.config)
        except FileNotFoundError:
            print(f"Config file not found: {args.config}")
            print("Creating default configuration...")
            from config import create_default_config
            config = create_default_config()
            config.save(args.config)
            print(f"Default config saved to: {args.config}")
        
        if args.quiet:
            config.verbose = False
        
        # Run the test harness
        harness = TestHarness(config)
        
        if args.quick:
            result = await harness.quick_test(args.quick, args.models)
        else:
            result = await harness.run_benchmark(args.benchmark, args.models)
        
        print(f"\nTest completed. Results in: {harness.output_dir}")
        if result.errors:
            print(f"Errors encountered: {len(result.errors)}")
            for error in result.errors:
                print(f"  - {error['error']}")
    
    asyncio.run(main())