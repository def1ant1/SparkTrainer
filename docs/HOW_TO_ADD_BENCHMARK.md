# How to Add a New Benchmark

This guide explains how to add custom evaluation benchmarks to SparkTrainer for measuring model performance on specific tasks.

## Table of Contents

- [Overview](#overview)
- [Benchmark Architecture](#benchmark-architecture)
- [Step-by-Step Guide](#step-by-step-guide)
- [Example Implementations](#example-implementations)
- [Best Practices](#best-practices)
- [Integration with MLflow](#integration-with-mlflow)
- [Contributing Benchmarks](#contributing-benchmarks)

## Overview

SparkTrainer's benchmark system allows you to:
- Evaluate trained models on standardized tasks
- Compare model performance across different training runs
- Track performance metrics over time
- Reproduce evaluation results

### Existing Benchmarks

SparkTrainer includes several built-in benchmarks:
- **MMLU** (Massive Multitask Language Understanding) - 57 subjects
- **COCO** (Common Objects in Context) - Object detection, segmentation
- **ImageNet** - Image classification
- **Custom safety probes** - Toxicity, bias, jailbreak detection

## Benchmark Architecture

### Core Components

1. **Benchmark Interface**: Abstract base class defining the evaluation contract
2. **Dataset Loader**: Handles loading and preprocessing benchmark data
3. **Evaluator**: Runs the model on benchmark samples
4. **Metrics Calculator**: Computes performance metrics
5. **Results Reporter**: Formats and logs results to MLflow

### File Structure

```
src/spark_trainer/evaluation/
├── __init__.py
├── benchmark_interface.py     # Base class
├── mmlu_benchmark.py          # Example: MMLU implementation
├── coco_benchmark.py          # Example: COCO implementation
├── your_benchmark.py          # Your new benchmark
└── utils.py                   # Helper functions
```

## Step-by-Step Guide

### Step 1: Create Benchmark Class

Create a new file in `src/spark_trainer/evaluation/your_benchmark.py`:

```python
"""
Your Benchmark Name
Description of what this benchmark measures.
"""

from typing import Dict, List, Any, Optional
import torch
from datasets import load_dataset
from .benchmark_interface import BenchmarkInterface, BenchmarkResult


class YourBenchmark(BenchmarkInterface):
    """
    Your benchmark implementation.

    This benchmark evaluates models on [describe task].

    Attributes:
        name: Benchmark identifier
        description: Human-readable description
        dataset_name: HuggingFace dataset identifier
        num_samples: Number of samples to evaluate (None = all)
    """

    def __init__(
        self,
        name: str = "your_benchmark",
        dataset_name: str = "your/dataset",
        num_samples: Optional[int] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.dataset = None

    def load_dataset(self) -> None:
        """
        Load the benchmark dataset.

        This method should:
        1. Download/load the dataset
        2. Apply any necessary preprocessing
        3. Cache for future use
        """
        print(f"Loading {self.name} dataset: {self.dataset_name}")

        # Load from HuggingFace Hub or local path
        self.dataset = load_dataset(self.dataset_name, split='test')

        # Limit samples if specified
        if self.num_samples:
            self.dataset = self.dataset.select(range(self.num_samples))

        print(f"Loaded {len(self.dataset)} samples")

    def evaluate(self, model, tokenizer, device='cuda') -> BenchmarkResult:
        """
        Run evaluation on the model.

        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on

        Returns:
            BenchmarkResult with metrics and details
        """
        if self.dataset is None:
            self.load_dataset()

        model.eval()
        model.to(device)

        results = []
        correct = 0
        total = 0

        print(f"Evaluating {len(self.dataset)} samples...")

        with torch.no_grad():
            for i, sample in enumerate(self.dataset):
                # Extract question and answer from sample
                question = sample['question']
                correct_answer = sample['answer']

                # Format prompt
                prompt = self._format_prompt(question, sample.get('choices', []))

                # Tokenize
                inputs = tokenizer(prompt, return_tensors='pt').to(device)

                # Generate prediction
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=None,
                    top_p=None
                )

                # Decode prediction
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_answer = self._extract_answer(prediction)

                # Check correctness
                is_correct = self._check_answer(predicted_answer, correct_answer)

                if is_correct:
                    correct += 1
                total += 1

                # Store result
                results.append({
                    'question': question,
                    'predicted': predicted_answer,
                    'correct': correct_answer,
                    'is_correct': is_correct
                })

                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"Progress: {i + 1}/{len(self.dataset)} "
                          f"(Accuracy: {100 * correct / total:.2f}%)")

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'error_rate': 1.0 - accuracy
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            details=results
        )

    def _format_prompt(self, question: str, choices: List[str]) -> str:
        """
        Format the prompt for the model.

        Override this method to customize prompt formatting.
        """
        if choices:
            choices_text = '\n'.join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            return f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
        else:
            return f"Question: {question}\n\nAnswer:"

    def _extract_answer(self, prediction: str) -> str:
        """
        Extract the answer from model output.

        Override this method for custom answer extraction logic.
        """
        # Simple extraction: take first letter/word
        prediction = prediction.strip()
        if prediction:
            return prediction.split()[0].upper()
        return ""

    def _check_answer(self, predicted: str, correct: str) -> bool:
        """
        Check if predicted answer matches correct answer.

        Override this method for custom answer matching logic.
        """
        return predicted.strip().upper() == correct.strip().upper()
```

### Step 2: Define Benchmark Interface (if not exists)

The base interface in `benchmark_interface.py`:

```python
"""
Benchmark Interface for SparkTrainer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import mlflow


@dataclass
class BenchmarkResult:
    """Results from a benchmark evaluation"""
    benchmark_name: str
    metrics: Dict[str, float]
    details: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class BenchmarkInterface(ABC):
    """
    Abstract base class for all benchmarks.

    All benchmarks must implement:
    - load_dataset(): Load the benchmark dataset
    - evaluate(): Run evaluation and return results
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

    @abstractmethod
    def load_dataset(self) -> None:
        """Load and prepare the benchmark dataset"""
        pass

    @abstractmethod
    def evaluate(self, model, tokenizer, device='cuda') -> BenchmarkResult:
        """
        Evaluate model on this benchmark.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run on ('cuda' or 'cpu')

        Returns:
            BenchmarkResult with metrics and details
        """
        pass

    def log_to_mlflow(self, result: BenchmarkResult) -> None:
        """
        Log benchmark results to MLflow.

        Args:
            result: BenchmarkResult to log
        """
        # Log metrics
        for metric_name, metric_value in result.metrics.items():
            mlflow.log_metric(f"{self.name}_{metric_name}", metric_value)

        # Log detailed results as artifact
        import json
        results_file = f"{self.name}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'benchmark': result.benchmark_name,
                'metrics': result.metrics,
                'details': result.details
            }, f, indent=2)

        mlflow.log_artifact(results_file)
        print(f"Logged {self.name} results to MLflow")
```

### Step 3: Register Your Benchmark

Add your benchmark to `src/spark_trainer/evaluation/__init__.py`:

```python
from .benchmark_interface import BenchmarkInterface, BenchmarkResult
from .mmlu_benchmark import MMLUBenchmark
from .coco_benchmark import COCOBenchmark
from .your_benchmark import YourBenchmark  # Add this

# Benchmark registry
BENCHMARK_REGISTRY = {
    'mmlu': MMLUBenchmark,
    'coco': COCOBenchmark,
    'your_benchmark': YourBenchmark,  # Add this
}

def get_benchmark(name: str, **kwargs) -> BenchmarkInterface:
    """
    Get benchmark by name.

    Args:
        name: Benchmark name
        **kwargs: Benchmark-specific configuration

    Returns:
        Benchmark instance
    """
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. "
                        f"Available: {list(BENCHMARK_REGISTRY.keys())}")

    return BENCHMARK_REGISTRY[name](**kwargs)
```

### Step 4: Test Your Benchmark

Create a test file `tests/test_your_benchmark.py`:

```python
"""
Tests for Your Benchmark
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spark_trainer.evaluation import get_benchmark


def test_your_benchmark_loading():
    """Test benchmark dataset loading"""
    benchmark = get_benchmark('your_benchmark', num_samples=10)
    benchmark.load_dataset()
    assert len(benchmark.dataset) == 10


def test_your_benchmark_evaluation():
    """Test benchmark evaluation"""
    # Load a small model for testing
    model_name = 'gpt2'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create benchmark with small sample
    benchmark = get_benchmark('your_benchmark', num_samples=5)

    # Run evaluation
    result = benchmark.evaluate(model, tokenizer, device='cpu')

    # Check results
    assert result.benchmark_name == 'your_benchmark'
    assert 'accuracy' in result.metrics
    assert 0.0 <= result.metrics['accuracy'] <= 1.0
    assert len(result.details) == 5


def test_benchmark_prompt_formatting():
    """Test prompt formatting"""
    benchmark = get_benchmark('your_benchmark')

    question = "What is 2+2?"
    choices = ["3", "4", "5", "6"]

    prompt = benchmark._format_prompt(question, choices)

    assert "What is 2+2?" in prompt
    assert "A. 3" in prompt
    assert "B. 4" in prompt


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Step 5: Run Your Benchmark

Use your benchmark in training scripts:

```python
from spark_trainer.evaluation import get_benchmark
import mlflow

# Load your trained model
model = load_your_model()
tokenizer = load_your_tokenizer()

# Create benchmark
benchmark = get_benchmark('your_benchmark', num_samples=1000)

# Run evaluation
result = benchmark.evaluate(model, tokenizer)

# Print results
print(f"Benchmark: {result.benchmark_name}")
print(f"Metrics: {result.metrics}")

# Log to MLflow
with mlflow.start_run():
    benchmark.log_to_mlflow(result)
```

## Example Implementations

### Example 1: Multiple Choice Benchmark

```python
class MultipleChoiceBenchmark(BenchmarkInterface):
    """Generic multiple choice benchmark"""

    def evaluate(self, model, tokenizer, device='cuda') -> BenchmarkResult:
        results = []

        for sample in self.dataset:
            question = sample['question']
            choices = sample['choices']
            correct_idx = sample['answer']

            # Score each choice
            scores = []
            for choice in choices:
                prompt = f"{question}\nAnswer: {choice}"
                inputs = tokenizer(prompt, return_tensors='pt').to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use log probability as score
                    score = outputs.logits[:, -1, :].log_softmax(dim=-1).max().item()
                scores.append(score)

            # Predict choice with highest score
            predicted_idx = scores.index(max(scores))
            is_correct = (predicted_idx == correct_idx)

            results.append({
                'question': question,
                'predicted_idx': predicted_idx,
                'correct_idx': correct_idx,
                'is_correct': is_correct
            })

        accuracy = sum(r['is_correct'] for r in results) / len(results)

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics={'accuracy': accuracy},
            details=results
        )
```

### Example 2: Generation Benchmark

```python
class GenerationBenchmark(BenchmarkInterface):
    """Benchmark for open-ended generation tasks"""

    def evaluate(self, model, tokenizer, device='cuda') -> BenchmarkResult:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        results = []

        for sample in self.dataset:
            prompt = sample['prompt']
            reference = sample['reference']

            # Generate
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = model.generate(**inputs, max_new_tokens=100)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Score
            scores = scorer.score(reference, prediction)

            results.append({
                'prompt': prompt,
                'prediction': prediction,
                'reference': reference,
                'rouge_scores': {k: v.fmeasure for k, v in scores.items()}
            })

        # Average scores
        avg_rouge1 = sum(r['rouge_scores']['rouge1'] for r in results) / len(results)
        avg_rouge2 = sum(r['rouge_scores']['rouge2'] for r in results) / len(results)
        avg_rougeL = sum(r['rouge_scores']['rougeL'] for r in results) / len(results)

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics={
                'rouge1': avg_rouge1,
                'rouge2': avg_rouge2,
                'rougeL': avg_rougeL
            },
            details=results
        )
```

## Best Practices

### 1. Dataset Management

- **Cache datasets** to avoid repeated downloads
- **Version your datasets** using checksums
- **Provide small test sets** for quick iteration
- **Document dataset licenses** and usage restrictions

### 2. Reproducibility

- **Set random seeds** for deterministic evaluation
- **Log all hyperparameters** (temperature, top_p, etc.)
- **Save model predictions** for manual inspection
- **Version benchmark code** alongside results

### 3. Performance

- **Batch processing** when possible for speed
- **Use appropriate precision** (FP16/BF16 for speed, FP32 for accuracy)
- **Cache model outputs** to avoid redundant computation
- **Profile your code** to identify bottlenecks

### 4. Metrics

- **Choose appropriate metrics** for your task
- **Report multiple metrics** (accuracy, F1, ROUGE, etc.)
- **Include confidence intervals** when possible
- **Provide per-category breakdowns** for analysis

### 5. Documentation

- **Describe the task clearly** in docstrings
- **Provide usage examples** in comments
- **Link to papers/datasets** for reference
- **Explain metric calculations** explicitly

## Integration with MLflow

SparkTrainer automatically integrates benchmarks with MLflow:

```python
import mlflow
from spark_trainer.evaluation import get_benchmark

# Start MLflow run
with mlflow.start_run():
    # Log training params
    mlflow.log_params({
        'model': 'llama-2-7b',
        'recipe': 'lora',
        'epochs': 3
    })

    # Train model
    model = train_model()

    # Run benchmarks
    for benchmark_name in ['mmlu', 'your_benchmark']:
        benchmark = get_benchmark(benchmark_name)
        result = benchmark.evaluate(model, tokenizer)
        benchmark.log_to_mlflow(result)

    # Compare in MLflow UI
    print("View results at: http://localhost:5001")
```

## Contributing Benchmarks

Want to contribute your benchmark to SparkTrainer?

### Contribution Checklist

- [ ] Benchmark follows `BenchmarkInterface`
- [ ] Tests pass (`pytest tests/test_your_benchmark.py`)
- [ ] Documentation is complete
- [ ] Dataset is publicly accessible
- [ ] Code follows project style (Black, isort)
- [ ] Example usage provided
- [ ] MLflow integration works

### Submission Process

1. **Fork the repository**
2. **Create benchmark** in `src/spark_trainer/evaluation/`
3. **Add tests** in `tests/`
4. **Update documentation** in this file
5. **Submit pull request** with description

Use the "Benchmark Submission" issue template on GitHub.

## Resources

- [HuggingFace Datasets](https://huggingface.co/datasets)
- [Papers with Code Benchmarks](https://paperswithcode.com/datasets)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [PyTorch Evaluation](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

## FAQ

**Q: Can I use private datasets?**
A: Yes, but provide clear loading instructions. Consider using HuggingFace's private repos.

**Q: How do I handle large benchmarks?**
A: Use the `num_samples` parameter to limit evaluation size, or implement sampling strategies.

**Q: What if my benchmark requires external APIs?**
A: Document API requirements and provide mock implementations for testing.

**Q: How do I compare with published baselines?**
A: Include baseline scores in your benchmark documentation and MLflow logs.

## Support

For questions or help:
- [GitHub Discussions](https://github.com/def1ant1/SparkTrainer/discussions)
- [Issue Tracker](https://github.com/def1ant1/SparkTrainer/issues)
- [Documentation](https://github.com/def1ant1/SparkTrainer/tree/main/docs)

---

**Happy benchmarking!** 🎯
