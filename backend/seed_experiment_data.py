"""
Seed script for populating base models, recipes, and sample adapters.
Run this after running migrations to populate initial data.
"""
import uuid
from database import get_db_session
from models import BaseModel, Recipe, Adapter
from datetime import datetime


def seed_base_models(session):
    """Seed sample base models."""
    print("Seeding base models...")

    models = [
        {
            'id': str(uuid.uuid4()),
            'name': 'Llama-2-7B',
            'family': 'llama',
            'description': 'Meta Llama 2 7B parameter model',
            'params_b': 7.0,
            'dtype': 'bf16',
            'context_length': 4096,
            'hidden_size': 4096,
            'num_layers': 32,
            'architecture': 'transformer',
            'modality': 'text',
            'trainable': True,
            'servable': True,
            'quantized': False,
            'is_gguf': False,
            'stage': 'production',
            'status': 'active',
            'storage_path': '/models/llama-2-7b',
            'hf_repo_id': 'meta-llama/Llama-2-7b-hf',
            'tokenizer_path': '/models/llama-2-7b/tokenizer',
            'vocab_size': 32000,
            'tags': ['llm', 'chat', 'instruction'],
            'metadata': {}
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Mistral-7B-v0.1',
            'family': 'mistral',
            'description': 'Mistral 7B base model',
            'params_b': 7.3,
            'dtype': 'bf16',
            'context_length': 8192,
            'hidden_size': 4096,
            'num_layers': 32,
            'architecture': 'transformer',
            'modality': 'text',
            'trainable': True,
            'servable': True,
            'quantized': False,
            'is_gguf': False,
            'stage': 'production',
            'status': 'active',
            'storage_path': '/models/mistral-7b-v0.1',
            'hf_repo_id': 'mistralai/Mistral-7B-v0.1',
            'tokenizer_path': '/models/mistral-7b-v0.1/tokenizer',
            'vocab_size': 32000,
            'tags': ['llm', 'efficient', 'sliding-window'],
            'metadata': {}
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Llama-2-13B',
            'family': 'llama',
            'description': 'Meta Llama 2 13B parameter model',
            'params_b': 13.0,
            'dtype': 'bf16',
            'context_length': 4096,
            'hidden_size': 5120,
            'num_layers': 40,
            'architecture': 'transformer',
            'modality': 'text',
            'trainable': True,
            'servable': True,
            'quantized': False,
            'is_gguf': False,
            'stage': 'production',
            'status': 'active',
            'storage_path': '/models/llama-2-13b',
            'hf_repo_id': 'meta-llama/Llama-2-13b-hf',
            'tokenizer_path': '/models/llama-2-13b/tokenizer',
            'vocab_size': 32000,
            'tags': ['llm', 'chat', 'instruction'],
            'metadata': {}
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Llama-2-70B',
            'family': 'llama',
            'description': 'Meta Llama 2 70B parameter model',
            'params_b': 70.0,
            'dtype': 'bf16',
            'context_length': 4096,
            'hidden_size': 8192,
            'num_layers': 80,
            'architecture': 'transformer',
            'modality': 'text',
            'trainable': True,
            'servable': True,
            'quantized': False,
            'is_gguf': False,
            'stage': 'production',
            'status': 'active',
            'storage_path': '/models/llama-2-70b',
            'hf_repo_id': 'meta-llama/Llama-2-70b-hf',
            'tokenizer_path': '/models/llama-2-70b/tokenizer',
            'vocab_size': 32000,
            'tags': ['llm', 'chat', 'large'],
            'metadata': {}
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Qwen-7B-Chat',
            'family': 'qwen',
            'description': 'Qwen 7B chat model',
            'params_b': 7.7,
            'dtype': 'bf16',
            'context_length': 8192,
            'hidden_size': 4096,
            'num_layers': 32,
            'architecture': 'transformer',
            'modality': 'text',
            'trainable': True,
            'servable': True,
            'quantized': False,
            'is_gguf': False,
            'stage': 'staging',
            'status': 'active',
            'storage_path': '/models/qwen-7b-chat',
            'hf_repo_id': 'Qwen/Qwen-7B-Chat',
            'tokenizer_path': '/models/qwen-7b-chat/tokenizer',
            'vocab_size': 151936,
            'tags': ['llm', 'chat', 'multilingual'],
            'metadata': {}
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'ViT-B-16',
            'family': 'vit',
            'description': 'Vision Transformer Base 16x16 patch',
            'params_b': 0.086,
            'dtype': 'fp32',
            'context_length': None,
            'hidden_size': 768,
            'num_layers': 12,
            'architecture': 'vit',
            'modality': 'image',
            'trainable': True,
            'servable': True,
            'quantized': False,
            'is_gguf': False,
            'stage': 'production',
            'status': 'active',
            'storage_path': '/models/vit-b-16',
            'hf_repo_id': 'google/vit-base-patch16-224',
            'tokenizer_path': None,
            'vocab_size': None,
            'tags': ['vision', 'classification'],
            'metadata': {'image_size': 224, 'patch_size': 16}
        },
    ]

    for model_data in models:
        model = BaseModel(**model_data)
        session.add(model)

    session.commit()
    print(f"✓ Seeded {len(models)} base models")


def seed_recipes(session):
    """Seed training recipes."""
    print("Seeding recipes...")

    recipes = [
        {
            'id': str(uuid.uuid4()),
            'name': 'llm_qlora',
            'display_name': 'QLoRA (Recommended for LLMs)',
            'description': 'Memory-efficient fine-tuning using 4-bit quantization and LoRA',
            'recipe_type': 'qlora',
            'modality': 'text',
            'train_styles': ['qlora', 'lora'],
            'default_config': {
                'lora_r': 64,
                'lora_alpha': 16,
                'lora_dropout': 0.05,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                'quant_type': 'nf4',
            },
            'required_fields': ['learning_rate', 'max_steps', 'batch_size'],
            'optional_fields': ['lora_r', 'lora_alpha', 'target_modules'],
            'supported_architectures': ['transformer'],
            'min_gpu_memory_gb': 16.0,
            'supports_distributed': True,
            'tags': ['efficient', 'recommended'],
            'metadata': {},
            'is_active': True
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'llm_lora',
            'display_name': 'LoRA',
            'description': 'Low-Rank Adaptation for efficient fine-tuning',
            'recipe_type': 'lora',
            'modality': 'text',
            'train_styles': ['lora'],
            'default_config': {
                'lora_r': 64,
                'lora_alpha': 16,
                'lora_dropout': 0.05,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            },
            'required_fields': ['learning_rate', 'max_steps', 'batch_size'],
            'optional_fields': ['lora_r', 'lora_alpha', 'target_modules'],
            'supported_architectures': ['transformer'],
            'min_gpu_memory_gb': 24.0,
            'supports_distributed': True,
            'tags': ['efficient', 'fast'],
            'metadata': {},
            'is_active': True
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'llm_full_ft',
            'display_name': 'Full Fine-Tune',
            'description': 'Full parameter fine-tuning for maximum quality',
            'recipe_type': 'full_ft',
            'modality': 'text',
            'train_styles': ['full'],
            'default_config': {},
            'required_fields': ['learning_rate', 'max_steps', 'batch_size'],
            'optional_fields': [],
            'supported_architectures': ['transformer'],
            'min_gpu_memory_gb': 80.0,
            'supports_distributed': True,
            'tags': ['high-quality', 'expensive'],
            'metadata': {},
            'is_active': True
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'vision_lora',
            'display_name': 'Vision LoRA',
            'description': 'LoRA fine-tuning for vision models',
            'recipe_type': 'vision_lora',
            'modality': 'image',
            'train_styles': ['lora'],
            'default_config': {
                'lora_r': 32,
                'lora_alpha': 32,
                'target_modules': ['qkv', 'proj'],
            },
            'required_fields': ['learning_rate', 'max_steps', 'batch_size'],
            'optional_fields': ['lora_r', 'lora_alpha'],
            'supported_architectures': ['vit', 'cnn'],
            'min_gpu_memory_gb': 16.0,
            'supports_distributed': True,
            'tags': ['vision', 'efficient'],
            'metadata': {},
            'is_active': True
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'prompt_tuning',
            'display_name': 'Prompt Tuning',
            'description': 'Minimal parameter tuning using soft prompts',
            'recipe_type': 'prompt_tuning',
            'modality': 'text',
            'train_styles': ['prompt_tuning'],
            'default_config': {
                'num_virtual_tokens': 20,
                'prompt_tuning_init': 'random',
            },
            'required_fields': ['learning_rate', 'max_steps'],
            'optional_fields': ['num_virtual_tokens'],
            'supported_architectures': ['transformer'],
            'min_gpu_memory_gb': 12.0,
            'supports_distributed': True,
            'tags': ['ultra-efficient', 'experimental'],
            'metadata': {},
            'is_active': True
        },
    ]

    for recipe_data in recipes:
        recipe = Recipe(**recipe_data)
        session.add(recipe)

    session.commit()
    print(f"✓ Seeded {len(recipes)} recipes")


def seed_adapters(session):
    """Seed sample adapters (optional)."""
    print("Seeding sample adapters...")

    # Get a base model to attach adapters to
    llama_model = session.query(BaseModel).filter(BaseModel.name == 'Llama-2-7B').first()

    if not llama_model:
        print("⚠ No Llama-2-7B model found, skipping adapter seeding")
        return

    adapters_data = [
        {
            'id': str(uuid.uuid4()),
            'name': 'llama-2-7b-code-adapter',
            'base_model_id': llama_model.id,
            'description': 'LoRA adapter trained on code generation tasks',
            'adapter_type': 'lora',
            'rank': 64,
            'alpha': 16,
            'dropout': 0.05,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            'status': 'ready',
            'storage_path': f'/models/adapters/{llama_model.id}/code-adapter',
            'size_bytes': 134217728,  # 128 MB
            'metrics': {'final_loss': 0.342, 'eval_accuracy': 0.87},
            'tags': ['code', 'programming'],
            'metadata': {'trained_on': 'code-instruct-50k', 'epochs': 3}
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'llama-2-7b-math-adapter',
            'base_model_id': llama_model.id,
            'description': 'LoRA adapter trained on math reasoning',
            'adapter_type': 'lora',
            'rank': 64,
            'alpha': 16,
            'dropout': 0.05,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            'status': 'ready',
            'storage_path': f'/models/adapters/{llama_model.id}/math-adapter',
            'size_bytes': 134217728,  # 128 MB
            'metrics': {'final_loss': 0.298, 'eval_accuracy': 0.82},
            'tags': ['math', 'reasoning'],
            'metadata': {'trained_on': 'math-qa-30k', 'epochs': 5}
        },
    ]

    for adapter_data in adapters_data:
        adapter = Adapter(**adapter_data)
        session.add(adapter)

    session.commit()
    print(f"✓ Seeded {len(adapters_data)} adapters")


def main():
    """Run all seed functions."""
    print("\n" + "="*60)
    print("Seeding Experiment Data")
    print("="*60 + "\n")

    session = get_db_session()

    try:
        # Check if data already exists
        existing_models = session.query(BaseModel).count()
        existing_recipes = session.query(Recipe).count()

        if existing_models > 0 or existing_recipes > 0:
            print(f"⚠ Found existing data ({existing_models} models, {existing_recipes} recipes)")
            response = input("Do you want to add more data anyway? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return

        seed_base_models(session)
        seed_recipes(session)
        seed_adapters(session)

        print("\n" + "="*60)
        print("✓ Seeding completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        session.rollback()
        print(f"\n✗ Error during seeding: {e}")
        raise
    finally:
        session.close()


if __name__ == '__main__':
    main()
