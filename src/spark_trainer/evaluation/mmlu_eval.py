"""
MMLU (Massive Multitask Language Understanding) Evaluation.

Evaluates language models on 57 subjects across STEM, humanities, social sciences, and more.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# MMLU subjects grouped by category
MMLU_SUBJECTS = {
    "stem": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning"
    ],
    "humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "social_sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy"
    ],
    "other": [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition", "professional_accounting",
        "professional_medicine", "virology"
    ]
}


@dataclass
class MMLUConfig:
    """Configuration for MMLU evaluation."""
    model_path: str
    output_dir: str
    subjects: Optional[List[str]] = None  # None means all subjects
    num_fewshot: int = 5
    batch_size: int = 1
    max_samples: Optional[int] = None  # For quick testing
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_cache: bool = True


@dataclass
class MMLUResult:
    """Result for a single MMLU subject."""
    subject: str
    category: str
    accuracy: float
    num_correct: int
    num_total: int
    per_question_results: List[Dict[str, Any]]


class MMLUEvaluator:
    """
    Evaluator for MMLU benchmark.

    Supports:
    - All 57 MMLU subjects
    - Few-shot evaluation (0-shot to n-shot)
    - Batch processing
    - Detailed per-question results
    """

    def __init__(self, config: MMLUConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer
        print(f"Loading model from {config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            device_map=config.device,
            trust_remote_code=True
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self) -> Dict[str, Any]:
        """
        Run full MMLU evaluation.

        Returns:
            Dictionary with overall results and per-subject results
        """
        subjects = self.config.subjects or self._get_all_subjects()

        print(f"Evaluating on {len(subjects)} subjects")

        results = []
        category_scores = defaultdict(list)

        for subject in tqdm(subjects, desc="MMLU Evaluation"):
            result = self._evaluate_subject(subject)
            results.append(result)
            category_scores[result.category].append(result.accuracy)

        # Calculate overall metrics
        overall_accuracy = np.mean([r.accuracy for r in results])
        category_accuracies = {
            cat: np.mean(scores) for cat, scores in category_scores.items()
        }

        final_results = {
            "overall_accuracy": overall_accuracy,
            "category_accuracies": category_accuracies,
            "subject_results": [asdict(r) for r in results],
            "num_subjects": len(subjects),
            "config": asdict(self.config)
        }

        # Save results
        self._save_results(final_results)

        print(f"\n{'='*50}")
        print(f"MMLU Evaluation Results")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print(f"\nCategory Accuracies:")
        for category, acc in category_accuracies.items():
            print(f"  {category:20s}: {acc:.2%}")

        return final_results

    def _evaluate_subject(self, subject: str) -> MMLUResult:
        """Evaluate model on a single MMLU subject."""
        # Load subject data
        dev_data = self._load_subject_data(subject, "dev")
        test_data = self._load_subject_data(subject, "test")

        if self.config.max_samples:
            test_data = test_data[:self.config.max_samples]

        # Get few-shot examples from dev set
        fewshot_examples = dev_data[:self.config.num_fewshot]

        # Evaluate on test set
        correct = 0
        per_question_results = []

        for question_data in tqdm(test_data, desc=subject, leave=False):
            prompt = self._format_prompt(question_data, fewshot_examples)
            prediction = self._get_model_prediction(prompt, question_data["choices"])
            is_correct = prediction == question_data["answer"]

            if is_correct:
                correct += 1

            per_question_results.append({
                "question": question_data["question"],
                "choices": question_data["choices"],
                "correct_answer": question_data["answer"],
                "predicted_answer": prediction,
                "is_correct": is_correct
            })

        accuracy = correct / len(test_data) if test_data else 0.0
        category = self._get_subject_category(subject)

        return MMLUResult(
            subject=subject,
            category=category,
            accuracy=accuracy,
            num_correct=correct,
            num_total=len(test_data),
            per_question_results=per_question_results
        )

    def _load_subject_data(self, subject: str, split: str) -> List[Dict[str, Any]]:
        """
        Load MMLU data for a subject.

        Expected format: CSV with columns: question,A,B,C,D,answer
        """
        # Try to load from HuggingFace datasets first
        try:
            from datasets import load_dataset
            dataset = load_dataset("cais/mmlu", subject, split=split)

            data = []
            for item in dataset:
                data.append({
                    "question": item["question"],
                    "choices": [item["choices"][i] for i in range(4)],
                    "answer": item["answer"]  # 0-3 for A-D
                })
            return data

        except Exception as e:
            print(f"Warning: Could not load {subject} from HuggingFace: {e}")

            # Fallback: try to load from local CSV files
            csv_path = Path(f"data/mmlu/{split}/{subject}_{split}.csv")
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found. Skipping {subject}.")
                return []

            import csv
            data = []
            with open(csv_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 6:
                        data.append({
                            "question": row[0],
                            "choices": row[1:5],
                            "answer": ord(row[5]) - ord('A')  # Convert A/B/C/D to 0/1/2/3
                        })
            return data

    def _format_prompt(self, question_data: Dict[str, Any],
                      fewshot_examples: List[Dict[str, Any]]) -> str:
        """Format question with few-shot examples."""
        prompt_parts = []

        # Add few-shot examples
        for example in fewshot_examples:
            example_text = self._format_question(example, include_answer=True)
            prompt_parts.append(example_text)

        # Add test question
        question_text = self._format_question(question_data, include_answer=False)
        prompt_parts.append(question_text)

        return "\n\n".join(prompt_parts)

    def _format_question(self, question_data: Dict[str, Any],
                        include_answer: bool = False) -> str:
        """Format a single question."""
        choices_text = "\n".join([
            f"{chr(65+i)}. {choice}"
            for i, choice in enumerate(question_data["choices"])
        ])

        text = f"Question: {question_data['question']}\n{choices_text}\nAnswer:"

        if include_answer:
            answer_letter = chr(65 + question_data["answer"])
            text += f" {answer_letter}"

        return text

    def _get_model_prediction(self, prompt: str, choices: List[str]) -> int:
        """
        Get model's predicted answer (0-3 for A-D).

        Uses log probabilities of answer tokens to determine most likely choice.
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Get logits for each possible answer
        answer_tokens = [self.tokenizer.encode(" " + chr(65+i), add_special_tokens=False)[0]
                        for i in range(4)]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Get logits for next token

        # Get probabilities for each answer token
        probs = torch.softmax(logits, dim=-1)
        answer_probs = [probs[token].item() for token in answer_tokens]

        # Return index of most likely answer
        return int(np.argmax(answer_probs))

    def _get_all_subjects(self) -> List[str]:
        """Get all MMLU subjects."""
        return [subject for subjects in MMLU_SUBJECTS.values() for subject in subjects]

    def _get_subject_category(self, subject: str) -> str:
        """Get category for a subject."""
        for category, subjects in MMLU_SUBJECTS.items():
            if subject in subjects:
                return category
        return "other"

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON."""
        output_file = self.output_dir / "mmlu_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Also save a summary
        summary_file = self.output_dir / "mmlu_summary.txt"
        with open(summary_file, "w") as f:
            f.write("MMLU Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {results['overall_accuracy']:.2%}\n\n")
            f.write("Category Accuracies:\n")
            for category, acc in results["category_accuracies"].items():
                f.write(f"  {category:20s}: {acc:.2%}\n")
            f.write("\nPer-Subject Accuracies:\n")
            for subject_result in results["subject_results"]:
                f.write(f"  {subject_result['subject']:40s}: "
                       f"{subject_result['accuracy']:.2%} "
                       f"({subject_result['num_correct']}/{subject_result['num_total']})\n")


def main():
    """CLI entry point for MMLU evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU benchmark")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model or HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                       help="Specific subjects to evaluate (default: all)")
    parser.add_argument("--num-fewshot", type=int, default=5,
                       help="Number of few-shot examples")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per subject (for testing)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    config = MMLUConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        subjects=args.subjects,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device
    )

    evaluator = MMLUEvaluator(config)
    results = evaluator.evaluate()

    return results


if __name__ == "__main__":
    main()
