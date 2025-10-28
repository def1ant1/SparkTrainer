"""
LoRA and QLoRA training recipes.

Provides efficient fine-tuning with parameter-efficient methods.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset

from .base_recipe import BaseRecipe, RecipeConfig


@dataclass
class LoRAConfig(RecipeConfig):
    """Configuration for LoRA training."""
    # Base model
    base_model: str
    model_type: str = "causal_lm"  # causal_lm, seq2seq

    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # Auto-detect if None
    bias: str = "none"  # none, all, lora_only
    task_type: str = "CAUSAL_LM"  # CAUSAL_LM, SEQ_2_SEQ_LM, etc.

    # Quantization (for QLoRA)
    use_4bit: bool = False
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_seq_length: int = 512
    weight_decay: float = 0.0

    # Dataset
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    text_column: str = "text"
    train_split: str = "train"
    eval_split: Optional[str] = "validation"

    # Output
    output_dir: str = "./lora_output"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10


class LoRARecipe(BaseRecipe):
    """
    LoRA (Low-Rank Adaptation) training recipe.

    Efficient fine-tuning by training low-rank decomposition matrices
    instead of full model weights.

    Features:
    - Automatic target module detection
    - Support for 4-bit and 8-bit quantization (QLoRA)
    - Memory-efficient training
    - Compatible with all transformer models
    """

    def __init__(self, config: LoRAConfig):
        super().__init__(config)
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def prepare(self) -> Dict[str, Any]:
        """Prepare dataset."""
        print("Loading dataset...")

        if self.config.dataset_name:
            dataset = load_dataset(self.config.dataset_name)
        elif self.config.dataset_path:
            dataset = load_dataset("json", data_files=self.config.dataset_path)
        else:
            raise ValueError("Either dataset_name or dataset_path must be provided")

        # Get splits
        train_dataset = dataset[self.config.train_split]
        eval_dataset = dataset[self.config.eval_split] if self.config.eval_split else None

        print(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Eval samples: {len(eval_dataset)}")

        return {
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset
        }

    def build(self) -> Dict[str, Any]:
        """Build model with LoRA adapters."""
        print(f"Loading base model: {self.config.base_model}")

        # Setup quantization config for QLoRA
        quantization_config = None
        if self.config.use_4bit or self.config.use_8bit:
            print("Enabling quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.use_4bit,
                load_in_8bit=self.config.use_8bit,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        if self.config.model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if not quantization_config else None
            )
        elif self.config.model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.base_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

        # Prepare model for k-bit training if quantized
        if quantization_config:
            model = prepare_model_for_kbit_training(model)

        # Setup LoRA config
        target_modules = self.config.target_modules
        if target_modules is None:
            # Auto-detect target modules based on model architecture
            target_modules = self._detect_target_modules(model)
            print(f"Auto-detected target modules: {target_modules}")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=getattr(TaskType, self.config.task_type),
        )

        # Apply LoRA
        print("Applying LoRA adapters...")
        self.model = get_peft_model(model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "lora_config": lora_config
        }

    def train(self, data: Dict[str, Any], model_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute training."""
        print("Starting LoRA training...")

        train_dataset = data["train_dataset"]
        eval_dataset = data.get("eval_dataset")

        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.config.text_column],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length"
            )

        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        if eval_dataset:
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=True if torch.cuda.is_available() else False,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit" if (self.config.use_4bit or self.config.use_8bit) else "adamw_torch",
            report_to=["tensorboard"],
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        train_result = self.trainer.train()

        # Save LoRA adapters
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "output_dir": self.config.output_dir
        }

    def eval(self, data: Dict[str, Any], model_dict: Dict[str, Any],
            train_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained model."""
        if not data.get("eval_dataset"):
            print("No evaluation dataset provided")
            return {}

        print("Evaluating model...")
        eval_result = self.trainer.evaluate()

        return {
            "eval_loss": eval_result["eval_loss"],
            "eval_runtime": eval_result["eval_runtime"],
            "eval_samples_per_second": eval_result["eval_samples_per_second"]
        }

    def package(self, outputs: Dict[str, Any]) -> str:
        """Package trained model."""
        print(f"Model and LoRA adapters saved to: {self.config.output_dir}")

        # Save training config
        config_path = f"{self.config.output_dir}/training_config.json"
        import json
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        return self.config.output_dir

    def _detect_target_modules(self, model) -> List[str]:
        """Auto-detect target modules for LoRA based on model architecture."""
        # Common patterns for different model families
        module_patterns = {
            "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "phi": ["q_proj", "v_proj", "k_proj", "dense"],
            "gpt": ["c_attn", "c_proj"],
            "t5": ["q", "v", "k", "o"],
            "bert": ["query", "value", "key"],
        }

        # Get model type from config
        model_type = model.config.model_type.lower()

        # Check for known patterns
        for pattern_key, modules in module_patterns.items():
            if pattern_key in model_type:
                return modules

        # Default: target all linear layers in attention
        print("Using default target modules (all attention projections)")
        return ["q_proj", "v_proj"]


class QLoRARecipe(LoRARecipe):
    """
    QLoRA (Quantized LoRA) recipe.

    Combines 4-bit quantization with LoRA for maximum memory efficiency.
    Enables training of 70B+ models on consumer GPUs.

    Features:
    - 4-bit NormalFloat (NF4) quantization
    - Double quantization for extra memory savings
    - Paged optimizers for handling large batches
    """

    def __init__(self, config: LoRAConfig):
        # Force 4-bit quantization for QLoRA
        config.use_4bit = True
        config.bnb_4bit_compute_dtype = "float16"
        config.bnb_4bit_quant_type = "nf4"
        config.use_nested_quant = True

        super().__init__(config)

    def build(self) -> Dict[str, Any]:
        """Build model with QLoRA (4-bit + LoRA)."""
        print("Building QLoRA model (4-bit quantization + LoRA adapters)")
        return super().build()


def create_lora_recipe(config_dict: Dict[str, Any], use_qlora: bool = False) -> LoRARecipe:
    """
    Factory function to create LoRA or QLoRA recipe.

    Args:
        config_dict: Configuration dictionary
        use_qlora: If True, create QLoRA recipe with 4-bit quantization

    Returns:
        LoRARecipe or QLoRARecipe instance
    """
    config = LoRAConfig(**config_dict)

    if use_qlora:
        return QLoRARecipe(config)
    else:
        return LoRARecipe(config)


# Register recipes
try:
    from .recipe_registry import register_recipe

    @register_recipe("lora")
    class RegisteredLoRARecipe(LoRARecipe):
        """Registered LoRA recipe."""
        pass

    @register_recipe("qlora")
    class RegisteredQLoRARecipe(QLoRARecipe):
        """Registered QLoRA recipe."""
        pass

except ImportError:
    # Recipe registry not available
    pass
