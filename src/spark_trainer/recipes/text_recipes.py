"""
Text model training recipes.

Includes:
- BERT/RoBERTa classification
- GPT-2/Phi-3 SFT (Supervised Fine-Tuning)
- Llama-family LoRA/QLoRA
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from datasets import load_dataset, load_from_disk
import evaluate
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from .recipe_interface import (
    TrainerRecipe,
    AdapterTrainerRecipe,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvalMetrics,
    RecipeOutput,
    register_recipe,
)

logger = logging.getLogger(__name__)


@register_recipe("bert_classification")
class BERTClassificationRecipe(TrainerRecipe):
    """
    BERT/RoBERTa classification recipe.

    Supports:
    - Binary and multi-class classification
    - Custom or pretrained models
    - Full fine-tuning or frozen encoder
    """

    def prepare(self, data_config: DataConfig) -> Tuple[Any, Any, Any]:
        """Prepare text classification dataset."""
        # Load dataset
        dataset_path = Path(data_config.dataset_path)

        if dataset_path.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            # Assume HuggingFace dataset
            dataset = load_dataset(data_config.dataset_path)

        # Split if needed
        if "train" not in dataset:
            splits = dataset.train_test_split(
                test_size=data_config.val_split + data_config.test_split
            )
            train_dataset = splits["train"]

            test_val_splits = splits["test"].train_test_split(
                test_size=data_config.test_split / (data_config.val_split + data_config.test_split)
            )
            val_dataset = test_val_splits["train"]
            test_dataset = test_val_splits["test"]
        else:
            train_dataset = dataset["train"]
            val_dataset = dataset.get("validation", dataset.get("val"))
            test_dataset = dataset.get("test")

        # Tokenize
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.pretrained if self.model_config else "bert-base-uncased"
        )

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True) if val_dataset else None
        test_dataset = test_dataset.map(tokenize_function, batched=True) if test_dataset else None

        # Set format
        columns = ["input_ids", "attention_mask", "label"]
        train_dataset.set_format(type="torch", columns=columns)
        if val_dataset:
            val_dataset.set_format(type="torch", columns=columns)
        if test_dataset:
            test_dataset.set_format(type="torch", columns=columns)

        return train_dataset, val_dataset, test_dataset

    def build(self, model_config: ModelConfig) -> Any:
        """Build BERT classification model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.pretrained or "bert-base-uncased",
            num_labels=model_config.num_classes or 2,
        )

        # Freeze encoder if specified
        if model_config.custom_params and model_config.custom_params.get("freeze_encoder"):
            for param in model.base_model.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen, only training classifier head")

        return model

    def train(self, training_config: TrainingConfig) -> Dict[str, Any]:
        """Train BERT classifier."""
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
            metric_for_best_model="accuracy",
            fp16=training_config.mixed_precision == "fp16",
            bf16=training_config.mixed_precision == "bf16",
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            max_grad_norm=training_config.max_grad_norm,
        )

        # Metrics
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
            f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

            return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics,
        )

        # Train
        train_result = trainer.train()

        # Save
        trainer.save_model()

        return train_result.metrics

    def eval(self, split: str = "test") -> EvalMetrics:
        """Evaluate BERT classifier."""
        dataset = self.test_dataset if split == "test" else self.val_dataset

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=self.data_config.batch_size,
        )

        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
            f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

            return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics,
        )

        metrics = trainer.evaluate()

        return EvalMetrics(
            loss=metrics["eval_loss"],
            accuracy=metrics.get("eval_accuracy"),
            f1_score=metrics.get("eval_f1"),
        )

    def package(self, export_format: str = "pytorch") -> RecipeOutput:
        """Package BERT model."""
        model_path = self.output_dir / "model"

        if export_format == "pytorch":
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)

        elif export_format == "onnx":
            # Export to ONNX
            from transformers.onnx import export
            onnx_path = self.output_dir / "model.onnx"
            export(
                preprocessor=self.tokenizer,
                model=self.model,
                config=self.model.config,
                opset=14,
                output=onnx_path,
            )
            model_path = onnx_path

        # Evaluate on test set
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


@register_recipe("gpt2_sft")
class GPT2SFTRecipe(TrainerRecipe):
    """
    GPT-2 Supervised Fine-Tuning recipe.

    Supports:
    - Causal language modeling
    - Instruction tuning
    - Custom prompts
    """

    def prepare(self, data_config: DataConfig) -> Tuple[Any, Any, Any]:
        """Prepare text generation dataset."""
        # Load dataset
        dataset_path = Path(data_config.dataset_path)

        if dataset_path.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            dataset = load_dataset(data_config.dataset_path)

        # Split
        if "train" not in dataset:
            splits = dataset.train_test_split(
                test_size=data_config.val_split + data_config.test_split
            )
            train_dataset = splits["train"]

            test_val_splits = splits["test"].train_test_split(
                test_size=data_config.test_split / (data_config.val_split + data_config.test_split)
            )
            val_dataset = test_val_splits["train"]
            test_dataset = test_val_splits["test"]
        else:
            train_dataset = dataset["train"]
            val_dataset = dataset.get("validation", dataset.get("val"))
            test_dataset = dataset.get("test")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.pretrained if self.model_config else "gpt2"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=1024,
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names) if val_dataset else None
        test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names) if test_dataset else None

        return train_dataset, val_dataset, test_dataset

    def build(self, model_config: ModelConfig) -> Any:
        """Build GPT-2 model."""
        model = AutoModelForCausalLM.from_pretrained(
            model_config.pretrained or "gpt2",
        )

        # Resize embeddings if needed
        model.resize_token_embeddings(len(self.tokenizer))

        return model

    def train(self, training_config: TrainingConfig) -> Dict[str, Any]:
        """Train GPT-2."""
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
            bf16=training_config.mixed_precision == "bf16",
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            gradient_checkpointing=training_config.gradient_checkpointing,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Train
        train_result = trainer.train()

        # Save
        trainer.save_model()

        return train_result.metrics

    def eval(self, split: str = "test") -> EvalMetrics:
        """Evaluate GPT-2."""
        dataset = self.test_dataset if split == "test" else self.val_dataset

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=self.data_config.batch_size,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        metrics = trainer.evaluate()

        # Calculate perplexity
        perplexity = np.exp(metrics["eval_loss"])

        return EvalMetrics(
            loss=metrics["eval_loss"],
            perplexity=perplexity,
        )

    def package(self, export_format: str = "pytorch") -> RecipeOutput:
        """Package GPT-2 model."""
        model_path = self.output_dir / "model"

        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

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


@register_recipe("llama_lora")
class LlamaLoRARecipe(AdapterTrainerRecipe):
    """
    Llama LoRA/QLoRA fine-tuning recipe.

    Supports:
    - LoRA and QLoRA (4-bit/8-bit quantization)
    - Instruction tuning
    - Custom adapters
    """

    def prepare(self, data_config: DataConfig) -> Tuple[Any, Any, Any]:
        """Prepare instruction tuning dataset."""
        # Load dataset
        dataset_path = Path(data_config.dataset_path)

        if dataset_path.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            dataset = load_dataset(data_config.dataset_path)

        # Format for instruction tuning
        def format_instruction(example):
            if "instruction" in example:
                text = f"### Instruction:\n{example['instruction']}\n\n"
                if example.get("input"):
                    text += f"### Input:\n{example['input']}\n\n"
                text += f"### Response:\n{example['output']}"
            else:
                text = example["text"]

            return {"text": text}

        dataset = dataset.map(format_instruction)

        # Split
        if "train" not in dataset:
            splits = dataset.train_test_split(test_size=0.1)
            train_dataset = splits["train"]
            test_dataset = splits["test"]
            val_dataset = None
        else:
            train_dataset = dataset["train"]
            val_dataset = dataset.get("validation")
            test_dataset = dataset.get("test")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.pretrained or "meta-llama/Llama-2-7b-hf"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True) if val_dataset else None
        test_dataset = test_dataset.map(tokenize_function, batched=True) if test_dataset else None

        return train_dataset, val_dataset, test_dataset

    def build(self, model_config: ModelConfig) -> Any:
        """Build Llama model with LoRA."""
        from transformers import BitsAndBytesConfig

        # Quantization config
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            bnb_config = None

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_config.pretrained or "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Apply LoRA
        model = self.apply_adapters(model)

        return model

    def train(self, training_config: TrainingConfig) -> Dict[str, Any]:
        """Train Llama with LoRA."""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=training_config.learning_rate,
            per_device_train_batch_size=self.data_config.batch_size,
            per_device_eval_batch_size=self.data_config.batch_size,
            num_train_epochs=training_config.epochs,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            logging_steps=training_config.logging_steps,
            save_strategy=training_config.save_strategy,
            bf16=True,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            gradient_checkpointing=training_config.gradient_checkpointing,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        trainer.save_model()

        return train_result.metrics

    def eval(self, split: str = "test") -> EvalMetrics:
        """Evaluate Llama."""
        dataset = self.test_dataset if split == "test" else self.val_dataset

        if dataset is None:
            return EvalMetrics(loss=0.0)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=self.data_config.batch_size,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        metrics = trainer.evaluate()
        perplexity = np.exp(metrics["eval_loss"])

        return EvalMetrics(
            loss=metrics["eval_loss"],
            perplexity=perplexity,
        )

    def package(self, export_format: str = "pytorch") -> RecipeOutput:
        """Package Llama LoRA adapters."""
        model_path = self.output_dir / "model"

        # Save LoRA adapters
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        metrics = self.eval(split="test")

        return RecipeOutput(
            model_path=str(model_path),
            metrics=metrics,
            config={
                "model_config": self.model_config.__dict__,
                "training_config": self.training_config.__dict__,
                "lora_config": {
                    "r": self.lora_r,
                    "alpha": self.lora_alpha,
                    "dropout": self.lora_dropout,
                    "target_modules": self.target_modules,
                },
            },
            artifacts=[str(model_path / "adapter_config.json")],
            metadata={"adapter_type": self.adapter_type},
        )
