# Evaluators Documentation

SparkTrainer provides comprehensive evaluation capabilities for assessing model performance across various domains and benchmarks.

## Overview

The evaluation system supports:
- **Language Models**: MMLU (57 subjects), TruthfulQA, HellaSwag
- **Vision Models**: COCO (detection, segmentation, captioning), ImageNet
- **Safety**: Toxicity, bias, and fairness probes
- **Custom Benchmarks**: Extensible framework for domain-specific evaluations

## Supported Benchmarks

### MMLU (Massive Multitask Language Understanding)

MMLU evaluates language models across 57 academic subjects spanning STEM, humanities, social sciences, and more.

#### Usage

```python
from spark_trainer.evaluation import MMLUEvaluator

evaluator = MMLUEvaluator(
    model_path="outputs/llama-7b-finetuned",
    subjects=["mathematics", "physics", "computer_science"],
    num_shots=5,
    output_dir="eval_results/mmlu"
)

results = evaluator.evaluate()
print(f"Overall accuracy: {results['overall_accuracy']:.3f}")
```

#### CLI Usage

```bash
python -m spark_trainer.evaluation.mmlu \
    --model_path outputs/llama-7b-finetuned \
    --subjects mathematics physics computer_science \
    --num_shots 5 \
    --output_dir eval_results/mmlu
```

#### Available Subjects

**STEM:**
- abstract_algebra
- anatomy
- astronomy
- college_biology
- college_chemistry
- college_computer_science
- college_mathematics
- college_physics
- computer_security
- conceptual_physics
- electrical_engineering
- elementary_mathematics
- high_school_biology
- high_school_chemistry
- high_school_computer_science
- high_school_mathematics
- high_school_physics
- high_school_statistics
- machine_learning

**Humanities:**
- formal_logic
- high_school_european_history
- high_school_us_history
- high_school_world_history
- international_law
- jurisprudence
- logical_fallacies
- moral_disputes
- moral_scenarios
- philosophy
- prehistory
- professional_law
- world_religions

**Social Sciences:**
- econometrics
- high_school_geography
- high_school_government_and_politics
- high_school_macroeconomics
- high_school_microeconomics
- high_school_psychology
- human_sexuality
- professional_psychology
- public_relations
- security_studies
- sociology
- us_foreign_policy

**Other:**
- business_ethics
- clinical_knowledge
- college_medicine
- global_facts
- human_aging
- management
- marketing
- medical_genetics
- miscellaneous
- nutrition
- professional_accounting
- professional_medicine
- virology

#### Output Format

```json
{
  "overall_accuracy": 0.652,
  "total_questions": 1500,
  "correct_answers": 978,
  "subjects": {
    "mathematics": {
      "accuracy": 0.678,
      "total": 100,
      "correct": 68
    },
    "physics": {
      "accuracy": 0.643,
      "total": 100,
      "correct": 64
    },
    "computer_science": {
      "accuracy": 0.735,
      "total": 100,
      "correct": 74
    }
  },
  "metadata": {
    "model_path": "outputs/llama-7b-finetuned",
    "num_shots": 5,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### COCO Evaluation

COCO (Common Objects in Context) evaluates vision models on object detection, instance segmentation, and image captioning.

#### Usage

```python
from spark_trainer.evaluation import COCOEvaluator

evaluator = COCOEvaluator(
    model_path="outputs/yolo-v8-finetuned",
    task="detection",  # or "segmentation", "captioning"
    dataset_path="datasets/coco_val",
    output_dir="eval_results/coco"
)

results = evaluator.evaluate()
print(f"mAP@0.5: {results['mAP_50']:.3f}")
```

#### Tasks

1. **Object Detection**
   - Metrics: mAP, mAP@0.5, mAP@0.75, mAP (small/medium/large)

2. **Instance Segmentation**
   - Metrics: Mask mAP, Box mAP

3. **Image Captioning**
   - Metrics: BLEU, METEOR, ROUGE-L, CIDEr, SPICE

#### Output Format

**Detection:**
```json
{
  "task": "detection",
  "mAP": 0.456,
  "mAP_50": 0.678,
  "mAP_75": 0.512,
  "mAP_small": 0.234,
  "mAP_medium": 0.467,
  "mAP_large": 0.589,
  "per_class": {
    "person": 0.612,
    "car": 0.578,
    "dog": 0.534
  }
}
```

**Captioning:**
```json
{
  "task": "captioning",
  "BLEU-1": 0.723,
  "BLEU-4": 0.312,
  "METEOR": 0.267,
  "ROUGE-L": 0.534,
  "CIDEr": 1.042,
  "SPICE": 0.198
}
```

### Safety Evaluation

Evaluates models for toxicity, bias, and fairness.

#### Usage

```python
from spark_trainer.evaluation import SafetyEvaluator

evaluator = SafetyEvaluator(
    model_path="outputs/llama-7b-finetuned",
    probes=["toxicity", "bias", "fairness"],
    output_dir="eval_results/safety"
)

results = evaluator.evaluate()
print(f"Toxicity score: {results['toxicity_score']:.3f}")
```

#### Probes

1. **Toxicity Detection**
   - Tests model's tendency to generate toxic content
   - Metrics: toxicity_score, profanity_rate, hate_speech_rate

2. **Bias Evaluation**
   - Tests for demographic biases (gender, race, religion, etc.)
   - Metrics: bias_score_gender, bias_score_race, bias_score_religion

3. **Fairness Testing**
   - Evaluates fairness across different demographic groups
   - Metrics: demographic_parity, equal_opportunity, equalized_odds

#### Output Format

```json
{
  "toxicity": {
    "toxicity_score": 0.042,
    "profanity_rate": 0.012,
    "hate_speech_rate": 0.008,
    "samples_tested": 1000
  },
  "bias": {
    "bias_score_gender": 0.156,
    "bias_score_race": 0.089,
    "bias_score_religion": 0.067,
    "samples_tested": 500
  },
  "fairness": {
    "demographic_parity": 0.892,
    "equal_opportunity": 0.876,
    "equalized_odds": 0.854
  }
}
```

## Custom Benchmarks

Create custom evaluation benchmarks for your domain.

### Example: Custom QA Benchmark

```python
from spark_trainer.evaluation import BaseEvaluator
import json

class CustomQAEvaluator(BaseEvaluator):
    def __init__(self, model_path, dataset_path, output_dir):
        super().__init__(model_path, output_dir)
        self.dataset = self.load_dataset(dataset_path)

    def load_dataset(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f]

    def evaluate_sample(self, sample):
        question = sample["question"]
        expected_answer = sample["answer"]

        # Generate model prediction
        prediction = self.model.generate(question)

        # Compute metric (e.g., exact match, F1, etc.)
        score = self.compute_metric(prediction, expected_answer)

        return {
            "question": question,
            "prediction": prediction,
            "expected": expected_answer,
            "score": score
        }

    def evaluate(self):
        results = []
        for sample in self.dataset:
            result = self.evaluate_sample(sample)
            results.append(result)

        # Aggregate results
        avg_score = sum(r["score"] for r in results) / len(results)

        return {
            "average_score": avg_score,
            "total_samples": len(results),
            "results": results
        }

# Usage
evaluator = CustomQAEvaluator(
    model_path="outputs/my-model",
    dataset_path="datasets/custom_qa.jsonl",
    output_dir="eval_results/custom_qa"
)

results = evaluator.evaluate()
```

## Evaluation Artifacts

All evaluations generate the following artifacts:

1. **results.json** - Main results file with all metrics
2. **predictions.jsonl** - Model predictions for each sample
3. **metrics.csv** - Metrics in CSV format for easy analysis
4. **report.html** - Visual HTML report with charts and tables

Example directory structure:
```
eval_results/
├── mmlu/
│   ├── results.json
│   ├── predictions.jsonl
│   ├── metrics.csv
│   └── report.html
├── coco/
│   ├── results.json
│   ├── detections.json
│   ├── metrics.csv
│   └── report.html
└── safety/
    ├── results.json
    ├── samples.jsonl
    ├── metrics.csv
    └── report.html
```

## Leaderboard Integration

Results are automatically published to the leaderboard:

```bash
# Run evaluation and publish to leaderboard
python -m spark_trainer.evaluation.mmlu \
    --model_path outputs/llama-7b-finetuned \
    --publish_to_leaderboard \
    --model_name "Llama-7B-Finetuned" \
    --tags "llama,instruction-tuning"
```

View the leaderboard:
- Web UI: http://localhost:5001/leaderboard
- API: http://localhost:5001/api/leaderboard?benchmark=mmlu&sort=accuracy

Export leaderboard data:
```bash
curl http://localhost:5001/api/leaderboard?format=csv > leaderboard.csv
```

## Best Practices

1. **Consistent Evaluation**
   - Use the same test set across model iterations
   - Document evaluation parameters (num_shots, temperature, etc.)

2. **Multiple Metrics**
   - Don't rely on a single metric
   - Evaluate across multiple dimensions (accuracy, safety, efficiency)

3. **Statistical Significance**
   - Run evaluations multiple times with different random seeds
   - Report confidence intervals

4. **Domain-Specific Tests**
   - Create custom benchmarks for your specific use case
   - Validate on real-world data

5. **Continuous Evaluation**
   - Integrate evaluations into your CI/CD pipeline
   - Monitor model performance over time

## API Reference

### BaseEvaluator

Base class for all evaluators.

```python
class BaseEvaluator:
    def __init__(self, model_path: str, output_dir: str):
        pass

    def load_model(self) -> Any:
        """Load the model for evaluation"""
        pass

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation and return results"""
        pass

    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        pass
```

### MMLUEvaluator

```python
class MMLUEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_path: str,
        subjects: List[str],
        num_shots: int = 5,
        output_dir: str = "eval_results/mmlu"
    ):
        pass
```

### COCOEvaluator

```python
class COCOEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_path: str,
        task: str,  # "detection", "segmentation", or "captioning"
        dataset_path: str,
        output_dir: str = "eval_results/coco"
    ):
        pass
```

### SafetyEvaluator

```python
class SafetyEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_path: str,
        probes: List[str],  # ["toxicity", "bias", "fairness"]
        output_dir: str = "eval_results/safety"
    ):
        pass
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Use smaller batch sizes
   - Enable gradient checkpointing
   - Use model quantization

2. **Slow Evaluation**
   - Use GPU acceleration
   - Parallelize across multiple GPUs
   - Cache model predictions

3. **Inconsistent Results**
   - Fix random seeds
   - Use deterministic operations
   - Verify data preprocessing

## Further Reading

- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [COCO Dataset](https://cocodataset.org/)
- [Safety Best Practices](https://www.anthropic.com/index/core-views-on-ai-safety)
