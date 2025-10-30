"""
API routes for enhanced experiment creation with base models, recipes, and adapters.
"""
from flask import Blueprint, request, jsonify
from typing import Dict, List, Optional
import uuid
from datetime import datetime

from database import get_db_session
from models import BaseModel, Recipe, Adapter, Experiment, Dataset, Project, Activity
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
            servable = request.args.get('servable')
            search = request.args.get('search')
            q = request.args.get('q')  # Alias for search
            project_id = request.args.get('project')
            limit = request.args.get('limit', type=int)
            offset = request.args.get('offset', type=int, default=0)

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
            if servable is not None:
                servable_bool = servable.lower() == 'true'
                query = query.filter(BaseModel.servable == servable_bool)

            # Search filter
            search_term = search or q
            if search_term:
                search_pattern = f"%{search_term}%"
                query = query.filter(
                    (BaseModel.name.ilike(search_pattern)) |
                    (BaseModel.description.ilike(search_pattern)) |
                    (BaseModel.family.ilike(search_pattern))
                )

            # Order by params desc (largest first), then by name
            query = query.order_by(BaseModel.params_b.desc(), BaseModel.name)

            # Count total before pagination
            total = query.count()

            # Apply pagination
            if limit:
                query = query.limit(limit).offset(offset)

            models = query.all()

            return jsonify({
                'models': [_serialize_base_model(m) for m in models],
                'total': total,
                'limit': limit,
                'offset': offset
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
                meta=data.get('metadata', {}),
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
            data = request.json or {}
            if 'metadata' in data and 'meta' not in data:
                data['meta'] = data.pop('metadata')

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
                meta=data.get('metadata', {}),
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
            data = request.json or {}
            if 'metadata' in data and 'meta' not in data:
                data['meta'] = data.pop('metadata')
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
                meta=data.get('metadata', {}),
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
            data = request.json or {}
            if 'metadata' in data and 'meta' not in data:
                data['meta'] = data.pop('metadata')
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
# Datasets API with Compatibility
# ============================================================================

@experiment_bp.route('/api/datasets', methods=['GET'])
def list_datasets():
    """
    List datasets with optional filtering and compatibility checking.

    Query parameters:
        - project: Filter by project ID
        - modality: Filter by modality (text, image, audio, video, multimodal)
        - compatible_with_model: Filter to only show datasets compatible with given model ID
        - search or q: Search query
        - limit: Pagination limit
        - offset: Pagination offset
    """
    session = get_db_session()

    try:
        project_id = request.args.get('project')
        modality = request.args.get('modality')
        compatible_with_model = request.args.get('compatible_with_model')
        search = request.args.get('search') or request.args.get('q')
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', type=int, default=0)

        query = session.query(Dataset)

        # Apply basic filters
        if project_id:
            query = query.filter(Dataset.project_id == project_id)
        if modality:
            query = query.filter(Dataset.modality == modality)
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                (Dataset.name.ilike(search_pattern)) |
                (Dataset.description.ilike(search_pattern))
            )

        # Apply model compatibility filter
        if compatible_with_model:
            base_model = session.query(BaseModel).filter(
                BaseModel.id == compatible_with_model
            ).first()

            if base_model:
                model_modality = base_model.modality
                # Get compatible modalities
                from compatibility_engine import CompatibilityEngine
                compat_modalities = CompatibilityEngine.MODALITY_COMPAT.get(model_modality, [])

                # Filter datasets to compatible modalities
                query = query.filter(Dataset.modality.in_(compat_modalities))

        # Order by creation date (newest first)
        query = query.order_by(Dataset.created_at.desc())

        # Count total before pagination
        total = query.count()

        # Apply pagination
        if limit:
            query = query.limit(limit).offset(offset)

        datasets = query.all()

        # Build response with compatibility info if model specified
        dataset_list = []
        for ds in datasets:
            ds_dict = _serialize_dataset(ds)

            # Add compatibility status if model specified
            if compatible_with_model and base_model:
                _, warnings, errors = CompatibilityEngine.check_compatibility(
                    _serialize_base_model(base_model),
                    ds_dict,
                    None,
                    None
                )
                ds_dict['compatibility'] = {
                    'compatible': len(errors) == 0,
                    'warnings': warnings,
                    'errors': errors
                }

            dataset_list.append(ds_dict)

        return jsonify({
            'datasets': dataset_list,
            'total': total,
            'limit': limit,
            'offset': offset
        }), 200

    except Exception as e:
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
# Activity Feed API
# ============================================================================

@experiment_bp.route('/api/activity', methods=['GET', 'POST'])
def activity_feed():
    """
    Get activity feed or create new activity event.

    GET query parameters:
        - limit: Number of items to return (default 100)
        - offset: Pagination offset
        - event_type: Filter by event type
        - entity_type: Filter by entity type
        - user_id: Filter by user
        - project_id: Filter by project
        - unread_only: Show only unread items (true/false)
    """
    session = get_db_session()

    try:
        if request.method == 'GET':
            limit = request.args.get('limit', type=int, default=100)
            offset = request.args.get('offset', type=int, default=0)
            event_type = request.args.get('event_type')
            entity_type = request.args.get('entity_type')
            user_id = request.args.get('user_id')
            project_id = request.args.get('project_id')
            unread_only = request.args.get('unread_only', 'false').lower() == 'true'

            query = session.query(Activity)

            # Apply filters
            if event_type:
                query = query.filter(Activity.event_type == event_type)
            if entity_type:
                query = query.filter(Activity.entity_type == entity_type)
            if user_id:
                query = query.filter(Activity.user_id == user_id)
            if project_id:
                query = query.filter(Activity.project_id == project_id)
            if unread_only:
                query = query.filter(Activity.read == False)

            # Order by creation date (newest first)
            query = query.order_by(Activity.created_at.desc())

            # Count total
            total = query.count()

            # Apply pagination
            query = query.limit(limit).offset(offset)

            activities = query.all()

            return jsonify({
                'activities': [_serialize_activity(a) for a in activities],
                'total': total,
                'limit': limit,
                'offset': offset
            }), 200

        elif request.method == 'POST':
            data = request.json

            activity = Activity(
                id=str(uuid.uuid4()),
                event_type=data['event_type'],
                entity_type=data['entity_type'],
                entity_id=data.get('entity_id'),
                title=data['title'],
                message=data.get('message'),
                status=data.get('status'),
                user_id=data.get('user_id'),
                project_id=data.get('project_id'),
                metadata=data.get('metadata', {}),
                read=data.get('read', False),
            )

            session.add(activity)
            session.commit()

            return jsonify(_serialize_activity(activity)), 201

    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@experiment_bp.route('/api/activity/<activity_id>/mark-read', methods=['POST'])
def mark_activity_read(activity_id):
    """Mark an activity as read."""
    session = get_db_session()

    try:
        activity = session.query(Activity).filter(Activity.id == activity_id).first()
        if not activity:
            return jsonify({'error': 'Activity not found'}), 404

        activity.read = True
        session.commit()

        return jsonify(_serialize_activity(activity)), 200

    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@experiment_bp.route('/api/activity/mark-all-read', methods=['POST'])
def mark_all_activities_read():
    """Mark all activities as read for a user/project."""
    session = get_db_session()

    try:
        data = request.json or {}
        user_id = data.get('user_id')
        project_id = data.get('project_id')

        query = session.query(Activity).filter(Activity.read == False)

        if user_id:
            query = query.filter(Activity.user_id == user_id)
        if project_id:
            query = query.filter(Activity.project_id == project_id)

        updated_count = query.update({'read': True})
        session.commit()

        return jsonify({'marked_read': updated_count}), 200

    except Exception as e:
        session.rollback()
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
        'metadata': model.meta,
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
        'metadata': recipe.meta,
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
        'metadata': adapter.meta,
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
        'metadata': dataset.meta,
        'created_at': dataset.created_at.isoformat() if dataset.created_at else None,
        'updated_at': dataset.updated_at.isoformat() if dataset.updated_at else None,
    }


def _serialize_activity(activity: Activity) -> Dict:
    """Serialize an Activity to a dict."""
    if not activity:
        return None

    return {
        'id': activity.id,
        'event_type': activity.event_type,
        'entity_type': activity.entity_type,
        'entity_id': activity.entity_id,
        'title': activity.title,
        'message': activity.message,
        'status': activity.status,
        'user_id': activity.user_id,
        'project_id': activity.project_id,
        'metadata': activity.metadata,
        'read': activity.read,
        'created_at': activity.created_at.isoformat() if activity.created_at else None,
    }
