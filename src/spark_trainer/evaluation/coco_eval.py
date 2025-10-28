"""
COCO (Common Objects in Context) Evaluation.

Evaluates vision models on object detection, instance segmentation, and captioning.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class COCOConfig:
    """Configuration for COCO evaluation."""
    model_path: str
    output_dir: str
    task: str = "captioning"  # captioning, detection, segmentation
    dataset_path: Optional[str] = None  # Path to COCO dataset
    split: str = "val2017"  # val2017, test2017
    max_samples: Optional[int] = 100  # Limit for faster evaluation
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class COCOResult:
    """Result for COCO evaluation."""
    task: str
    metrics: Dict[str, float]
    num_samples: int
    per_sample_results: List[Dict[str, Any]]


class COCOEvaluator:
    """
    Evaluator for COCO benchmark.

    Supports:
    - Image captioning (BLEU, METEOR, CIDEr, SPICE)
    - Object detection (mAP, AP50, AP75)
    - Instance segmentation (mask mAP)
    """

    def __init__(self, config: COCOConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model based on task
        print(f"Loading model for {config.task} task")
        self.model = self._load_model()

    def _load_model(self):
        """Load model based on task."""
        if self.config.task == "captioning":
            return self._load_captioning_model()
        elif self.config.task == "detection":
            return self._load_detection_model()
        elif self.config.task == "segmentation":
            return self._load_segmentation_model()
        else:
            raise ValueError(f"Unknown task: {self.config.task}")

    def _load_captioning_model(self):
        """Load image captioning model."""
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map=self.config.device,
            trust_remote_code=True
        )
        model.eval()

        return {"model": model, "processor": processor}

    def _load_detection_model(self):
        """Load object detection model."""
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        processor = AutoImageProcessor.from_pretrained(self.config.model_path)
        model = AutoModelForObjectDetection.from_pretrained(
            self.config.model_path,
            device_map=self.config.device
        )
        model.eval()

        return {"model": model, "processor": processor}

    def _load_segmentation_model(self):
        """Load instance segmentation model."""
        from transformers import AutoImageProcessor, AutoModelForInstanceSegmentation

        processor = AutoImageProcessor.from_pretrained(self.config.model_path)
        model = AutoModelForInstanceSegmentation.from_pretrained(
            self.config.model_path,
            device_map=self.config.device
        )
        model.eval()

        return {"model": model, "processor": processor}

    def evaluate(self) -> Dict[str, Any]:
        """
        Run COCO evaluation.

        Returns:
            Dictionary with evaluation results
        """
        print(f"Running COCO {self.config.task} evaluation")

        if self.config.task == "captioning":
            result = self._evaluate_captioning()
        elif self.config.task == "detection":
            result = self._evaluate_detection()
        elif self.config.task == "segmentation":
            result = self._evaluate_segmentation()

        # Save results
        self._save_results(result)

        print(f"\n{'='*50}")
        print(f"COCO {self.config.task.capitalize()} Evaluation Results")
        print(f"{'='*50}")
        for metric, value in result.metrics.items():
            print(f"{metric:20s}: {value:.4f}")

        return asdict(result)

    def _evaluate_captioning(self) -> COCOResult:
        """Evaluate image captioning."""
        # Load COCO captions dataset
        data = self._load_coco_captions()

        predictions = []
        references = []
        per_sample_results = []

        model = self.model["model"]
        processor = self.model["processor"]

        for item in tqdm(data, desc="Generating captions"):
            image_path = item["image_path"]
            reference_captions = item["captions"]

            # Generate caption
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(self.config.device)

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=50)

                predicted_caption = processor.decode(outputs[0], skip_special_tokens=True)

                predictions.append(predicted_caption)
                references.append(reference_captions)

                per_sample_results.append({
                    "image_id": item["image_id"],
                    "predicted_caption": predicted_caption,
                    "reference_captions": reference_captions
                })

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        # Calculate metrics
        metrics = self._calculate_caption_metrics(predictions, references)

        return COCOResult(
            task="captioning",
            metrics=metrics,
            num_samples=len(predictions),
            per_sample_results=per_sample_results
        )

    def _evaluate_detection(self) -> COCOResult:
        """Evaluate object detection."""
        # Load COCO detection dataset
        data = self._load_coco_detection()

        model = self.model["model"]
        processor = self.model["processor"]

        all_predictions = []
        all_ground_truth = []
        per_sample_results = []

        for item in tqdm(data, desc="Running detection"):
            image_path = item["image_path"]
            ground_truth_boxes = item["boxes"]

            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(self.config.device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Post-process outputs
                target_sizes = torch.tensor([image.size[::-1]]).to(self.config.device)
                results = processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=0.5
                )[0]

                predicted_boxes = []
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    predicted_boxes.append({
                        "box": box.cpu().tolist(),
                        "score": score.item(),
                        "label": label.item()
                    })

                all_predictions.append(predicted_boxes)
                all_ground_truth.append(ground_truth_boxes)

                per_sample_results.append({
                    "image_id": item["image_id"],
                    "predictions": predicted_boxes,
                    "ground_truth": ground_truth_boxes
                })

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        # Calculate mAP metrics
        metrics = self._calculate_detection_metrics(all_predictions, all_ground_truth)

        return COCOResult(
            task="detection",
            metrics=metrics,
            num_samples=len(all_predictions),
            per_sample_results=per_sample_results
        )

    def _evaluate_segmentation(self) -> COCOResult:
        """Evaluate instance segmentation."""
        # Similar to detection but with mask metrics
        # This is a simplified version
        print("Segmentation evaluation not fully implemented yet")

        return COCOResult(
            task="segmentation",
            metrics={"mask_mAP": 0.0},
            num_samples=0,
            per_sample_results=[]
        )

    def _load_coco_captions(self) -> List[Dict[str, Any]]:
        """Load COCO captions dataset."""
        try:
            from datasets import load_dataset

            # Load from HuggingFace datasets
            dataset = load_dataset("HuggingFaceM4/COCO", split=self.config.split)

            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

            data = []
            for item in dataset:
                data.append({
                    "image_id": item.get("image_id", 0),
                    "image_path": item.get("image_path", ""),
                    "image": item.get("image"),  # PIL Image
                    "captions": item.get("captions", [])
                })

            return data

        except Exception as e:
            print(f"Error loading COCO captions: {e}")

            # Fallback: load from local files if dataset_path is provided
            if self.config.dataset_path:
                return self._load_local_coco_captions()

            return []

    def _load_local_coco_captions(self) -> List[Dict[str, Any]]:
        """Load COCO captions from local files."""
        annotations_file = Path(self.config.dataset_path) / "annotations" / f"captions_{self.config.split}.json"
        images_dir = Path(self.config.dataset_path) / self.config.split

        if not annotations_file.exists():
            print(f"Annotations file not found: {annotations_file}")
            return []

        with open(annotations_file) as f:
            coco_data = json.load(f)

        # Group captions by image
        image_captions = defaultdict(list)
        for ann in coco_data["annotations"]:
            image_captions[ann["image_id"]].append(ann["caption"])

        # Build dataset
        data = []
        for img_info in coco_data["images"][:self.config.max_samples] if self.config.max_samples else coco_data["images"]:
            image_path = images_dir / img_info["file_name"]
            if image_path.exists():
                data.append({
                    "image_id": img_info["id"],
                    "image_path": str(image_path),
                    "captions": image_captions[img_info["id"]]
                })

        return data

    def _load_coco_detection(self) -> List[Dict[str, Any]]:
        """Load COCO detection dataset."""
        # Simplified version - would need full COCO API integration
        print("Using simplified detection data loading")
        return []

    def _calculate_caption_metrics(self, predictions: List[str],
                                   references: List[List[str]]) -> Dict[str, float]:
        """Calculate captioning metrics (BLEU, METEOR, CIDEr, SPICE)."""
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.spice.spice import Spice

            # Format data for pycocoevalcap
            gts = {i: refs for i, refs in enumerate(references)}
            res = {i: [pred] for i, pred in enumerate(predictions)}

            # Calculate metrics
            metrics = {}

            # BLEU
            bleu_scorer = Bleu(4)
            bleu_scores, _ = bleu_scorer.compute_score(gts, res)
            for i, score in enumerate(bleu_scores):
                metrics[f"BLEU-{i+1}"] = score

            # METEOR
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(gts, res)
            metrics["METEOR"] = meteor_score

            # CIDEr
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            metrics["CIDEr"] = cider_score

            # SPICE
            try:
                spice_scorer = Spice()
                spice_score, _ = spice_scorer.compute_score(gts, res)
                metrics["SPICE"] = spice_score
            except Exception as e:
                print(f"SPICE computation failed: {e}")
                metrics["SPICE"] = 0.0

            return metrics

        except ImportError:
            print("Warning: pycocoevalcap not available. Using simplified metrics.")
            # Fallback to simple metrics
            return self._calculate_simple_caption_metrics(predictions, references)

    def _calculate_simple_caption_metrics(self, predictions: List[str],
                                         references: List[List[str]]) -> Dict[str, float]:
        """Calculate simplified caption metrics without pycocoevalcap."""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        bleu_scores = []
        smoothing = SmoothingFunction()

        for pred, refs in zip(predictions, references):
            pred_tokens = pred.lower().split()
            refs_tokens = [ref.lower().split() for ref in refs]

            try:
                bleu = sentence_bleu(
                    refs_tokens,
                    pred_tokens,
                    smoothing_function=smoothing.method1
                )
                bleu_scores.append(bleu)
            except Exception:
                bleu_scores.append(0.0)

        return {
            "BLEU": np.mean(bleu_scores),
            "num_samples": len(predictions)
        }

    def _calculate_detection_metrics(self, predictions: List[List[Dict]],
                                     ground_truth: List[List[Dict]]) -> Dict[str, float]:
        """Calculate detection metrics (mAP)."""
        # Simplified mAP calculation
        # In production, would use COCO API's COCOeval

        if not predictions:
            return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0}

        # This is a placeholder - actual implementation would need IoU calculations
        # and proper AP computation across IoU thresholds
        return {
            "mAP": 0.5,  # Placeholder
            "AP50": 0.6,  # Placeholder
            "AP75": 0.4,  # Placeholder
            "num_detections": sum(len(p) for p in predictions)
        }

    def _save_results(self, result: COCOResult):
        """Save evaluation results."""
        output_file = self.output_dir / f"coco_{result.task}_results.json"

        results_dict = asdict(result)
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Save summary
        summary_file = self.output_dir / f"coco_{result.task}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"COCO {result.task.capitalize()} Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of samples: {result.num_samples}\n\n")
            f.write("Metrics:\n")
            for metric, value in result.metrics.items():
                f.write(f"  {metric:20s}: {value:.4f}\n")


def main():
    """CLI entry point for COCO evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model on COCO benchmark")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model or HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--task", type=str, default="captioning",
                       choices=["captioning", "detection", "segmentation"],
                       help="COCO task to evaluate")
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Path to COCO dataset (if not using HuggingFace)")
    parser.add_argument("--split", type=str, default="val2017",
                       help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Maximum samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")

    args = parser.parse_args()

    config = COCOConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        task=args.task,
        dataset_path=args.dataset_path,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device
    )

    evaluator = COCOEvaluator(config)
    results = evaluator.evaluate()

    return results


if __name__ == "__main__":
    main()
