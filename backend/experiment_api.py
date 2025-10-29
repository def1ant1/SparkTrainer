"""
API routes for enhanced experiment creation with base models, recipes, and adapters.
"""
from flask import Blueprint, request, jsonify
from typing import Dict, List, Optional
import uuid
from datetime import datetime

from database import get_db_session
from models import BaseModel, Recipe, Adapter, Experiment, Dataset, Project
from compatibility_engine import CompatibilityEngine
from smart_defaults import SmartDefaults

# Create blueprint
experiment_bp = Blueprint('experiment_api', __name__)


# ============================================================================
# Base Models API
# ============================================================================

@experiment_bp.route('/api/base-models', methods=['GET', 'POST'])
def base_models():
    """List all base models or create a new one."""
    session = get_db_session()

    try:
        if request.method == 'GET':
            # Query parameters for filtering
            family = request.args.get('family')
            modality = request.args.get('modality')
            stage = request.args.get('stage')
            trainable = request.args.get('trainable')
            search = request.args.get('search')

            query = session.query(BaseModel)

            # Apply filters
            if family:
                query = query.filter(BaseModel.family == family)
            if modality:
                query = query.filter(BaseModel.modality == modality)
            if stage:
                query = query.filter(BaseModel.stage == stage)
            if trainable is not None:
                trainable_bool = trainable.lower() == 'true'
                query = query.filter(BaseModel.trainable == trainable_bool)
            if search:
                search_pattern = f"%{search}%"
                query = query.filter(
                    (BaseModel.name.ilike(search_pattern)) |
                    (BaseModel.description.ilike(search_pattern))
                )

            # Order by params desc (largest first)
            query = query.order_by(BaseModel.params_b.desc())

            models = query.all()

            return jsonify({
                'models': [_serialize_base_model(m) for m in models],
                'total': len(models)
            }), 200

        elif request.method == 'POST':
            data = request.json

            # Create new base model
            model = BaseModel(
                id=str(uuid.uuid4()),
                name=data['name'],
                family=data['family'],
                description=data.get('description'),
                params_b=data.get('params_b'),
                dtype=data['dtype'],
                context_length=data.get('context_length'),
                hidden_size=data.get('hidden_size'),
                num_layers=data.get('num_layers'),
                architecture=data.get('architecture'),
                modality=data['modality'],
                trainable=data.get('trainable', True),
                servable=data.get('servable', True),
                quantized=data.get('quantized', False),
                is_gguf=data.get('is_gguf', False),
                stage=data.get('stage', 'staging'),
                status=data.get('status', 'active'),
                storage_path=data['storage_path'],
                size_bytes=data.get('size_bytes', 0),
                checksum=data.get('checksum'),
                hf_repo_id=data.get('hf_repo_id'),
                hf_revision=data.get('hf_revision'),
                tokenizer_path=data.get('tokenizer_path'),
                vocab_size=data.get('vocab_size'),
                tags=data.get('tags', []),
                metadata=data.get('metadata', {}),
                model_card=data.get('model_card'),
            )

            session.add(model)
            session.commit()

            return jsonify(_serialize_base_model(model)), 201

    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@experiment_bp.route('/api/base-models/<model_id>', methods=['GET', 'PUT', 'DELETE'])
def base_model_detail(model_id):
    """Get, update, or delete a specific base model."""
    session = get_db_session()

    try:
        model = session.query(BaseModel).filter(BaseModel.id == model_id).first()
        if not model:
            return jsonify({'error': 'Base model not found'}), 404

        if request.method == 'GET':
            return jsonify(_serialize_base_model(model)), 200

        elif request.method == 'PUT':
            data = request.json

            # Update fields
            for key, value in data.items():
                if hasattr(model, key) and key != 'id':
                    setattr(model, key, value)

            session.commit()
            return jsonify(_serialize_base_model(model)), 200

        elif request.method == 'DELETE':
            session.delete(model)
            session.commit()
            return jsonify({'message': 'Base model deleted'}), 200

    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


# ============================================================================
# Recipes API
# ============================================================================

@experiment_bp.route('/api/recipes', methods=['GET', 'POST'])
def recipes():
    """List all recipes or create a new one."""
    session = get_db_session()

    try:
        if request.method == 'GET':
            # Query parameters
            modality = request.args.get('modality')
            recipe_type = request.args.get('type')
            active_only = request.args.get('active_only', 'true').lower() == 'true'

            query = session.query(Recipe)

            # Apply filters
            if modality:
                query = query.filter(Recipe.modality == modality)
            if recipe_type:
                query = query.filter(Recipe.recipe_type == recipe_type)
            if active_only:
                query = query.filter(Recipe.is_active == True)

            recipes = query.all()

            return jsonify({
                'recipes': [_serialize_recipe(r) for r in recipes],
                'total': len(recipes)
            }), 200

        elif request.method == 'POST':
            data = request.json

            recipe = Recipe(
                id=str(uuid.uuid4()),
                name=data['name'],
                display_name=data['display_name'],
                description=data.get('description'),
                recipe_type=data['recipe_type'],
                modality=data['modality'],
                train_styles=data.get('train_styles', []),
                default_config=data.get('default_config', {}),
                required_fields=data.get('required_fields', []),
                optional_fields=data.get('optional_fields', []),
                supported_architectures=data.get('supported_architectures', []),
                min_gpu_memory_gb=data.get('min_gpu_memory_gb'),
                supports_distributed=data.get('supports_distributed', True),
                template_path=data.get('template_path'),
                script_template=data.get('script_template'),
                tags=data.get('tags', []),
                metadata=data.get('metadata', {}),
                is_active=data.get('is_active', True),
            )

            session.add(recipe)
            session.commit()

            return jsonify(_serialize_recipe(recipe)), 201

    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@experiment_bp.route('/api/recipes/<recipe_id>', methods=['GET', 'PUT', 'DELETE'])
def recipe_detail(recipe_id):
    """Get, update, or delete a specific recipe."""
    session = get_db_session()

    try:
        recipe = session.query(Recipe).filter(Recipe.id == recipe_id).first()
        if not recipe:
            return jsonify({'error': 'Recipe not found'}), 404

        if request.method == 'GET':
            return jsonify(_serialize_recipe(recipe)), 200

        elif request.method == 'PUT':
            data = request.json
            for key, value in data.items():
                if hasattr(recipe, key) and key != 'id':
                    setattr(recipe, key, value)

            session.commit()
            return jsonify(_serialize_recipe(recipe)), 200

        elif request.method == 'DELETE':
            session.delete(recipe)
            session.commit()
            return jsonify({'message': 'Recipe deleted'}), 200

    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


# ============================================================================
# Adapters API
# ============================================================================

@experiment_bp.route('/api/adapters', methods=['GET', 'POST'])
def adapters():
    """List all adapters or create a new one."""
    session = get_db_session()

    try:
        if request.method == 'GET':
            base_model_id = request.args.get('base_model_id')
            adapter_type = request.args.get('type')
            status = request.args.get('status')

            query = session.query(Adapter)

            if base_model_id:
                query = query.filter(Adapter.base_model_id == base_model_id)
            if adapter_type:
                query = query.filter(Adapter.adapter_type == adapter_type)
            if status:
                query = query.filter(Adapter.status == status)

            adapters = query.all()

            return jsonify({
                'adapters': [_serialize_adapter(a) for a in adapters],
                'total': len(adapters)
            }), 200

        elif request.method == 'POST':
            data = request.json

            adapter = Adapter(
                id=str(uuid.uuid4()),
                name=data['name'],
                base_model_id=data['base_model_id'],
                description=data.get('description'),
                adapter_type=data['adapter_type'],
                rank=data.get('rank'),
                alpha=data.get('alpha'),
                dropout=data.get('dropout', 0.0),
                target_modules=data.get('target_modules', []),
                status=data.get('status', 'training'),
                training_experiment_id=data.get('training_experiment_id'),
                storage_path=data['storage_path'],
                size_bytes=data.get('size_bytes', 0),
                checksum=data.get('checksum'),
                metrics=data.get('metrics', {}),
                tags=data.get('tags', []),
                metadata=data.get('metadata', {}),
            )

            session.add(adapter)
            session.commit()

            return jsonify(_serialize_adapter(adapter)), 201

    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@experiment_bp.route('/api/adapters/<adapter_id>', methods=['GET', 'PUT', 'DELETE'])
def adapter_detail(adapter_id):
    """Get, update, or delete a specific adapter."""
    session = get_db_session()

    try:
        adapter = session.query(Adapter).filter(Adapter.id == adapter_id).first()
        if not adapter:
            return jsonify({'error': 'Adapter not found'}), 404

        if request.method == 'GET':
            return jsonify(_serialize_adapter(adapter)), 200

        elif request.method == 'PUT':
            data = request.json
            for key, value in data.items():
                if hasattr(adapter, key) and key != 'id':
                    setattr(adapter, key, value)

            session.commit()
            return jsonify(_serialize_adapter(adapter)), 200

        elif request.method == 'DELETE':
            session.delete(adapter)
            session.commit()
            return jsonify({'message': 'Adapter deleted'}), 200

    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


# ============================================================================
# Experiment Preflight & Smart Defaults
# ============================================================================

@experiment_bp.route('/api/experiments/preflight', methods=['POST'])
def experiment_preflight():
    """
    Run preflight checks for an experiment configuration.
    Returns compatibility warnings/errors and resource estimates.
    """
    session = get_db_session()

    try:
        data = request.json

        # Fetch components
        base_model = None
        dataset = None
        recipe = None
        adapters_list = []

        if data.get('base_model_id'):
            base_model = session.query(BaseModel).filter(
                BaseModel.id == data['base_model_id']
            ).first()

        if data.get('dataset_id'):
            dataset = session.query(Dataset).filter(
                Dataset.id == data['dataset_id']
            ).first()

        if data.get('recipe_id'):
            recipe = session.query(Recipe).filter(
                Recipe.id == data['recipe_id']
            ).first()

        if data.get('adapters'):
            adapter_ids = [a.get('adapter_id') for a in data['adapters']]
            adapters_list = session.query(Adapter).filter(
                Adapter.id.in_(adapter_ids)
            ).all()

        # Run compatibility checks
        compat_ok, warnings, errors = CompatibilityEngine.check_compatibility(
            _serialize_base_model(base_model) if base_model else None,
            _serialize_dataset(dataset) if dataset else None,
            _serialize_recipe(recipe) if recipe else None,
            [_serialize_adapter(a) for a in adapters_list] if adapters_list else None
        )

        # Estimate VRAM and throughput
        vram_estimate = {}
        throughput_estimate = {}

        if base_model and recipe:
            train_config = data.get('train', {})
            resources = data.get('resources', {})

            vram_estimate = SmartDefaults.estimate_vram(
                params_b=base_model.params_b or 1.0,
                dtype=base_model.dtype,
                recipe_type=recipe.recipe_type,
                batch_size=train_config.get('global_batch_size', 1),
                grad_accum=train_config.get('grad_accum', 1),
            )

            if resources.get('gpu_type'):
                throughput_estimate = SmartDefaults.estimate_throughput(
                    params_b=base_model.params_b or 1.0,
                    gpu_type=resources['gpu_type'],
                    batch_size=train_config.get('global_batch_size', 1),
                )

        return jsonify({
            'ok': compat_ok and len(errors) == 0,
            'warnings': warnings,
            'errors': errors,
            'estimated_vram_mb': int(vram_estimate.get('total_with_buffer_gb', 0) * 1024),
            'vram_breakdown': vram_estimate,
            'time_per_step_ms': throughput_estimate.get('time_per_step_ms', 0),
            'throughput': throughput_estimate,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@experiment_bp.route('/api/experiments/smart-defaults', methods=['POST'])
def experiment_smart_defaults():
    """
    Calculate smart defaults for an experiment configuration.
    """
    session = get_db_session()

    try:
        data = request.json

        # Fetch components
        base_model = None
        dataset = None
        recipe = None

        if data.get('base_model_id'):
            base_model = session.query(BaseModel).filter(
                BaseModel.id == data['base_model_id']
            ).first()

        if data.get('dataset_id'):
            dataset = session.query(Dataset).filter(
                Dataset.id == data['dataset_id']
            ).first()

        if data.get('recipe_id'):
            recipe = session.query(Recipe).filter(
                Recipe.id == data['recipe_id']
            ).first()

        if not base_model or not dataset or not recipe:
            return jsonify({'error': 'Missing required components'}), 400

        # Calculate defaults
        defaults = SmartDefaults.calculate_defaults(
            _serialize_base_model(base_model),
            _serialize_dataset(dataset),
            _serialize_recipe(recipe),
            num_gpus=data.get('num_gpus', 1),
            gpu_type=data.get('gpu_type')
        )

        return jsonify(defaults), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


# ============================================================================
# Helper Functions
# ============================================================================

def _serialize_base_model(model: BaseModel) -> Dict:
    """Serialize a BaseModel to a dict."""
    if not model:
        return None

    return {
        'id': model.id,
        'name': model.name,
        'family': model.family,
        'description': model.description,
        'params_b': model.params_b,
        'dtype': model.dtype,
        'context_length': model.context_length,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'architecture': model.architecture,
        'modality': model.modality,
        'trainable': model.trainable,
        'servable': model.servable,
        'quantized': model.quantized,
        'is_gguf': model.is_gguf,
        'stage': model.stage,
        'status': model.status,
        'storage_path': model.storage_path,
        'size_bytes': model.size_bytes,
        'checksum': model.checksum,
        'hf_repo_id': model.hf_repo_id,
        'hf_revision': model.hf_revision,
        'tokenizer_path': model.tokenizer_path,
        'vocab_size': model.vocab_size,
        'tags': model.tags,
        'metadata': model.metadata,
        'model_card': model.model_card,
        'created_at': model.created_at.isoformat() if model.created_at else None,
        'updated_at': model.updated_at.isoformat() if model.updated_at else None,
    }


def _serialize_recipe(recipe: Recipe) -> Dict:
    """Serialize a Recipe to a dict."""
    if not recipe:
        return None

    return {
        'id': recipe.id,
        'name': recipe.name,
        'display_name': recipe.display_name,
        'description': recipe.description,
        'recipe_type': recipe.recipe_type,
        'modality': recipe.modality,
        'train_styles': recipe.train_styles,
        'default_config': recipe.default_config,
        'required_fields': recipe.required_fields,
        'optional_fields': recipe.optional_fields,
        'supported_architectures': recipe.supported_architectures,
        'min_gpu_memory_gb': recipe.min_gpu_memory_gb,
        'supports_distributed': recipe.supports_distributed,
        'template_path': recipe.template_path,
        'tags': recipe.tags,
        'metadata': recipe.metadata,
        'is_active': recipe.is_active,
        'created_at': recipe.created_at.isoformat() if recipe.created_at else None,
        'updated_at': recipe.updated_at.isoformat() if recipe.updated_at else None,
    }


def _serialize_adapter(adapter: Adapter) -> Dict:
    """Serialize an Adapter to a dict."""
    if not adapter:
        return None

    return {
        'id': adapter.id,
        'name': adapter.name,
        'base_model_id': adapter.base_model_id,
        'description': adapter.description,
        'adapter_type': adapter.adapter_type,
        'rank': adapter.rank,
        'alpha': adapter.alpha,
        'dropout': adapter.dropout,
        'target_modules': adapter.target_modules,
        'status': adapter.status,
        'training_experiment_id': adapter.training_experiment_id,
        'storage_path': adapter.storage_path,
        'size_bytes': adapter.size_bytes,
        'checksum': adapter.checksum,
        'metrics': adapter.metrics,
        'tags': adapter.tags,
        'metadata': adapter.metadata,
        'created_at': adapter.created_at.isoformat() if adapter.created_at else None,
        'updated_at': adapter.updated_at.isoformat() if adapter.updated_at else None,
    }


def _serialize_dataset(dataset: Dataset) -> Dict:
    """Serialize a Dataset to a dict."""
    if not dataset:
        return None

    return {
        'id': dataset.id,
        'project_id': dataset.project_id,
        'name': dataset.name,
        'version': dataset.version,
        'description': dataset.description,
        'modality': dataset.modality,
        'size_bytes': dataset.size_bytes,
        'num_samples': dataset.num_samples,
        'manifest_path': dataset.manifest_path,
        'storage_path': dataset.storage_path,
        'checksum': dataset.checksum,
        'integrity_checked': dataset.integrity_checked,
        'integrity_passed': dataset.integrity_passed,
        'integrity_report': dataset.integrity_report,
        'statistics': dataset.statistics,
        'tags': dataset.tags,
        'metadata': dataset.metadata,
        'created_at': dataset.created_at.isoformat() if dataset.created_at else None,
        'updated_at': dataset.updated_at.isoformat() if dataset.updated_at else None,
    }
