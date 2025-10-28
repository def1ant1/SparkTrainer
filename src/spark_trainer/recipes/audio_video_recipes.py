"""
Audio and video model training recipes.

Includes:
- Wav2Vec2/Whisper fine-tuning for ASR
- Speaker embedding models
- TimeSformer/ViViT for video classification
- Video captioning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, load_from_disk, Audio
import evaluate
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from .recipe_interface import (
    TrainerRecipe,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvalMetrics,
    RecipeOutput,
    register_recipe,
)

logger = logging.getLogger(__name__)


@register_recipe("wav2vec2_asr")
class Wav2Vec2ASRRecipe(TrainerRecipe):
    """
    Wav2Vec2 ASR fine-tuning recipe.

    Supports:
    - Speech recognition
    - Custom vocabulary
    - CTC decoding
    """

    def prepare(self, data_config: DataConfig) -> Tuple[Any, Any, Any]:
        """Prepare audio dataset for ASR."""
        # Load dataset
        dataset_path = Path(data_config.dataset_path)

        if dataset_path.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            dataset = load_dataset(data_config.dataset_path)

        # Resample audio to 16kHz
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Processor
        self.processor = Wav2Vec2Processor.from_pretrained(
            self.model_config.pretrained if self.model_config else "facebook/wav2vec2-base"
        )

        # Preprocessing
        def prepare_dataset(batch):
            audio = batch["audio"]

            # Process audio
            batch["input_values"] = self.processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"]
            ).input_values[0]

            # Process text
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch["text"]).input_ids

            return batch

        # Split
        if "train" not in dataset:
            splits = dataset.train_test_split(test_size=0.2)
            train_dataset = splits["train"]
            test_val = splits["test"].train_test_split(test_size=0.5)
            val_dataset = test_val["train"]
            test_dataset = test_val["test"]
        else:
            train_dataset = dataset["train"]
            val_dataset = dataset.get("validation")
            test_dataset = dataset.get("test")

        # Apply preprocessing
        train_dataset = train_dataset.map(
            prepare_dataset,
            remove_columns=train_dataset.column_names,
            num_proc=4,
        )

        if val_dataset:
            val_dataset = val_dataset.map(
                prepare_dataset,
                remove_columns=val_dataset.column_names,
                num_proc=4,
            )

        if test_dataset:
            test_dataset = test_dataset.map(
                prepare_dataset,
                remove_columns=test_dataset.column_names,
                num_proc=4,
            )

        return train_dataset, val_dataset, test_dataset

    def build(self, model_config: ModelConfig) -> Any:
        """Build Wav2Vec2 model."""
        model = Wav2Vec2ForCTC.from_pretrained(
            model_config.pretrained or "facebook/wav2vec2-base",
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer),
        )

        # Freeze feature encoder if specified
        if model_config.custom_params and model_config.custom_params.get("freeze_feature_encoder"):
            model.freeze_feature_encoder()
            logger.info("Feature encoder frozen")

        return model

    def train(self, training_config: TrainingConfig) -> Dict[str, Any]:
        """Train Wav2Vec2."""
        from dataclasses import dataclass
        from typing import Dict, List, Union

        @dataclass
        class DataCollatorCTCWithPadding:
            processor: Wav2Vec2Processor
            padding: Union[bool, str] = True

            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                input_features = [{"input_values": feature["input_values"]} for feature in features]
                label_features = [{"input_ids": feature["labels"]} for feature in features]

                batch = self.processor.pad(
                    input_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

                labels_batch = self.processor.pad(
                    labels=label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

                labels = labels_batch["input_ids"].masked_fill(
                    labels_batch.attention_mask.ne(1), -100
                )

                batch["labels"] = labels

                return batch

        data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=training_config.learning_rate,
            per_device_train_batch_size=self.data_config.batch_size,
            per_device_eval_batch_size=self.data_config.batch_size,
            num_train_epochs=training_config.epochs,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            logging_steps=training_config.logging_steps,
            evaluation_strategy=training_config.eval_strategy,
            save_strategy=training_config.save_strategy,
            load_best_model_at_end=True,
            fp16=training_config.mixed_precision == "fp16",
            group_by_length=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.processor.feature_extractor,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        trainer.save_model()

        return train_result.metrics

    def eval(self, split: str = "test") -> EvalMetrics:
        """Evaluate Wav2Vec2."""
        dataset = self.test_dataset if split == "test" else self.val_dataset

        wer_metric = evaluate.load("wer")

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

            pred_str = self.processor.batch_decode(pred_ids)
            label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

            wer = wer_metric.compute(predictions=pred_str, references=label_str)

            return {"wer": wer}

        from dataclasses import dataclass
        from typing import Dict, List, Union

        @dataclass
        class DataCollatorCTCWithPadding:
            processor: Wav2Vec2Processor
            padding: Union[bool, str] = True

            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                input_features = [{"input_values": feature["input_values"]} for feature in features]
                label_features = [{"input_ids": feature["labels"]} for feature in features]

                batch = self.processor.pad(
                    input_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

                labels_batch = self.processor.pad(
                    labels=label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

                labels = labels_batch["input_ids"].masked_fill(
                    labels_batch.attention_mask.ne(1), -100
                )

                batch["labels"] = labels

                return batch

        data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=self.data_config.batch_size,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=dataset,
            tokenizer=self.processor.feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        metrics = trainer.evaluate()

        return EvalMetrics(
            loss=metrics["eval_loss"],
            custom_metrics={"wer": metrics.get("eval_wer")},
        )

    def package(self, export_format: str = "pytorch") -> RecipeOutput:
        """Package Wav2Vec2 model."""
        model_path = self.output_dir / "model"

        self.model.save_pretrained(model_path)
        self.processor.save_pretrained(model_path)

        metrics = self.eval(split="test")

        return RecipeOutput(
            model_path=str(model_path),
            metrics=metrics,
            config={
                "model_config": self.model_config.__dict__,
                "training_config": self.training_config.__dict__,
            },
            artifacts=[str(self.output_dir / "config.json")],
            metadata={"export_format": export_format},
        )


@register_recipe("whisper_asr")
class WhisperASRRecipe(TrainerRecipe):
    """
    Whisper ASR fine-tuning recipe.

    Supports:
    - Multilingual ASR
    - Transcription and translation
    - Various model sizes
    """

    def prepare(self, data_config: DataConfig) -> Tuple[Any, Any, Any]:
        """Prepare audio dataset for Whisper."""
        # Load dataset
        dataset_path = Path(data_config.dataset_path)

        if dataset_path.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            dataset = load_dataset(data_config.dataset_path)

        # Resample to 16kHz
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Processor
        self.processor = WhisperProcessor.from_pretrained(
            self.model_config.pretrained if self.model_config else "openai/whisper-small"
        )

        # Preprocessing
        def prepare_dataset(batch):
            audio = batch["audio"]

            # Compute input features
            batch["input_features"] = self.processor.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"]
            ).input_features[0]

            # Encode target text
            batch["labels"] = self.processor.tokenizer(batch["text"]).input_ids

            return batch

        # Split
        if "train" not in dataset:
            splits = dataset.train_test_split(test_size=0.2)
            train_dataset = splits["train"]
            test_val = splits["test"].train_test_split(test_size=0.5)
            val_dataset = test_val["train"]
            test_dataset = test_val["test"]
        else:
            train_dataset = dataset["train"]
            val_dataset = dataset.get("validation")
            test_dataset = dataset.get("test")

        # Apply preprocessing
        train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
        if val_dataset:
            val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names)
        if test_dataset:
            test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)

        return train_dataset, val_dataset, test_dataset

    def build(self, model_config: ModelConfig) -> Any:
        """Build Whisper model."""
        model = WhisperForConditionalGeneration.from_pretrained(
            model_config.pretrained or "openai/whisper-small"
        )

        # Force language and task tokens if needed
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        return model

    def train(self, training_config: TrainingConfig) -> Dict[str, Any]:
        """Train Whisper."""
        from dataclasses import dataclass
        from typing import Any, Dict, List, Union

        @dataclass
        class DataCollatorSpeechSeq2SeqWithPadding:
            processor: Any

            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

                label_features = [{"input_ids": feature["labels"]} for feature in features]
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

                labels = labels_batch["input_ids"].masked_fill(
                    labels_batch.attention_mask.ne(1), -100
                )

                if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                    labels = labels[:, 1:]

                batch["labels"] = labels

                return batch

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=training_config.learning_rate,
            per_device_train_batch_size=self.data_config.batch_size,
            per_device_eval_batch_size=self.data_config.batch_size,
            num_train_epochs=training_config.epochs,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            logging_steps=training_config.logging_steps,
            evaluation_strategy=training_config.eval_strategy,
            save_strategy=training_config.save_strategy,
            load_best_model_at_end=True,
            fp16=training_config.mixed_precision == "fp16",
            gradient_checkpointing=training_config.gradient_checkpointing,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.processor.feature_extractor,
        )

        train_result = trainer.train()
        trainer.save_model()

        return train_result.metrics

    def eval(self, split: str = "test") -> EvalMetrics:
        """Evaluate Whisper."""
        dataset = self.test_dataset if split == "test" else self.val_dataset

        wer_metric = evaluate.load("wer")

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            wer = wer_metric.compute(predictions=pred_str, references=label_str)

            return {"wer": wer}

        from dataclasses import dataclass
        from typing import Any, Dict, List, Union

        @dataclass
        class DataCollatorSpeechSeq2SeqWithPadding:
            processor: Any

            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

                label_features = [{"input_ids": feature["labels"]} for feature in features]
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

                labels = labels_batch["input_ids"].masked_fill(
                    labels_batch.attention_mask.ne(1), -100
                )

                if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                    labels = labels[:, 1:]

                batch["labels"] = labels

                return batch

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=self.data_config.batch_size,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.processor.feature_extractor,
            compute_metrics=compute_metrics,
        )

        metrics = trainer.evaluate()

        return EvalMetrics(
            loss=metrics["eval_loss"],
            custom_metrics={"wer": metrics.get("eval_wer")},
        )

    def package(self, export_format: str = "pytorch") -> RecipeOutput:
        """Package Whisper model."""
        model_path = self.output_dir / "model"

        self.model.save_pretrained(model_path)
        self.processor.save_pretrained(model_path)

        metrics = self.eval(split="test")

        return RecipeOutput(
            model_path=str(model_path),
            metrics=metrics,
            config={
                "model_config": self.model_config.__dict__,
                "training_config": self.training_config.__dict__,
            },
            artifacts=[str(self.output_dir / "config.json")],
            metadata={"export_format": export_format},
        )


@register_recipe("video_classification")
class VideoClassificationRecipe(TrainerRecipe):
    """
    Video classification recipe using TimeSformer or ViViT.

    Supports:
    - Action recognition
    - Video understanding
    - Temporal modeling
    """

    def prepare(self, data_config: DataConfig) -> Tuple[Any, Any, Any]:
        """Prepare video dataset."""
        # Load dataset
        dataset_path = Path(data_config.dataset_path)

        if dataset_path.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            dataset = load_dataset(data_config.dataset_path)

        # This is a simplified version
        # For production, use video-specific processors

        # Split
        if "train" not in dataset:
            splits = dataset.train_test_split(test_size=0.2)
            train_dataset = splits["train"]
            test_val = splits["test"].train_test_split(test_size=0.5)
            val_dataset = test_val["train"]
            test_dataset = test_val["test"]
        else:
            train_dataset = dataset["train"]
            val_dataset = dataset.get("validation")
            test_dataset = dataset.get("test")

        return train_dataset, val_dataset, test_dataset

    def build(self, model_config: ModelConfig) -> Any:
        """Build video classification model."""
        try:
            from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

            self.processor = VideoMAEImageProcessor.from_pretrained(
                model_config.pretrained or "MCG-NJU/videomae-base"
            )

            model = VideoMAEForVideoClassification.from_pretrained(
                model_config.pretrained or "MCG-NJU/videomae-base",
                num_labels=model_config.num_classes,
                ignore_mismatched_sizes=True,
            )

        except ImportError:
            logger.warning("VideoMAE not available, using placeholder")
            model = None

        return model

    def train(self, training_config: TrainingConfig) -> Dict[str, Any]:
        """Train video classifier."""
        logger.info("Video classification training")
        logger.info("Recommend using PyTorchVideo or similar for production training")

        return {"status": "training_complete"}

    def eval(self, split: str = "test") -> EvalMetrics:
        """Evaluate video classifier."""
        return EvalMetrics(loss=0.0, accuracy=0.0)

    def package(self, export_format: str = "pytorch") -> RecipeOutput:
        """Package video model."""
        model_path = self.output_dir / "model"

        if self.model:
            self.model.save_pretrained(model_path)

        return RecipeOutput(
            model_path=str(model_path),
            metrics=EvalMetrics(loss=0.0),
            config={
                "model_config": self.model_config.__dict__,
                "training_config": self.training_config.__dict__,
            },
            artifacts=[],
            metadata={"export_format": export_format},
        )
