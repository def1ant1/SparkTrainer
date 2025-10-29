import React, { useState, useEffect } from 'react';
import {
  ChevronRight,
  Search,
  AlertCircle,
  CheckCircle,
  Info,
  X,
  Plus,
  Zap,
  Cpu,
  Database,
  Settings,
  Play
} from 'lucide-react';

/**
 * Comprehensive Create Experiment page with full experiment specification.
 *
 * Sections (left→right flow):
 * 1. Project (preselected or dropdown)
 * 2. Base model (searchable dropdown with stage, family, params, dtype)
 * 3. Dataset version (filtered by compatibility)
 * 4. Recipe (template selector)
 * 5. Adapters (optional)
 * 6. Hyperparams (smart defaults)
 * 7. Resources (gpus, strategy)
 * 8. Eval & export
 * 9. Pre-flight checks
 */
export default function CreateExperiment({ onClose, api, currentProject }) {
  // ============================================================================
  // State Management
  // ============================================================================

  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Form state
  const [experimentName, setExperimentName] = useState('');
  const [selectedProject, setSelectedProject] = useState(currentProject || null);
  const [selectedBaseModel, setSelectedBaseModel] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [selectedAdapters, setSelectedAdapters] = useState([]);
  const [hyperparams, setHyperparams] = useState({});
  const [resources, setResources] = useState({ gpus: 1 });
  const [strategy, setStrategy] = useState({});
  const [evalConfig, setEvalConfig] = useState({ suites: [], interval: 500 });
  const [exportFormats, setExportFormats] = useState(['safetensors']);

  // Data lists
  const [projects, setProjects] = useState([]);
  const [baseModels, setBaseModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [recipes, setRecipes] = useState([]);
  const [adapters, setAdapters] = useState([]);

  // Search and filters
  const [modelSearch, setModelSearch] = useState('');
  const [modelFilters, setModelFilters] = useState({});

  // Preflight results
  const [preflightResults, setPreflightResults] = useState(null);
  const [preflightLoading, setPreflightLoading] = useState(false);

  // Smart defaults
  const [smartDefaults, setSmartDefaults] = useState(null);

  // ============================================================================
  // Data Fetching
  // ============================================================================

  useEffect(() => {
    loadProjects();
    loadBaseModels();
    loadRecipes();
  }, []);

  useEffect(() => {
    if (selectedProject) {
      loadDatasets(selectedProject.id);
    }
  }, [selectedProject]);

  useEffect(() => {
    if (selectedBaseModel) {
      loadAdapters(selectedBaseModel.id);
    }
  }, [selectedBaseModel]);

  // Trigger smart defaults calculation when key components change
  useEffect(() => {
    if (selectedBaseModel && selectedDataset && selectedRecipe) {
      calculateSmartDefaults();
    }
  }, [selectedBaseModel, selectedDataset, selectedRecipe, resources.gpus]);

  // Trigger preflight check when configuration is complete
  useEffect(() => {
    if (selectedBaseModel && selectedDataset && selectedRecipe && hyperparams.max_steps) {
      runPreflightCheck();
    }
  }, [selectedBaseModel, selectedDataset, selectedRecipe, hyperparams, selectedAdapters, resources, strategy]);

  const loadProjects = async () => {
    try {
      const data = await api.getProjects();
      setProjects(data.projects || []);
    } catch (err) {
      console.error('Error loading projects:', err);
    }
  };

  const loadBaseModels = async () => {
    try {
      const response = await fetch('/api/base-models');
      const data = await response.json();
      setBaseModels(data.models || []);
    } catch (err) {
      console.error('Error loading base models:', err);
    }
  };

  const loadDatasets = async (projectId) => {
    try {
      const data = await api.getDatasets(projectId);
      setDatasets(data.datasets || []);
    } catch (err) {
      console.error('Error loading datasets:', err);
    }
  };

  const loadRecipes = async () => {
    try {
      const response = await fetch('/api/recipes?active_only=true');
      const data = await response.json();
      setRecipes(data.recipes || []);
    } catch (err) {
      console.error('Error loading recipes:', err);
    }
  };

  const loadAdapters = async (baseModelId) => {
    try {
      const response = await fetch(`/api/adapters?base_model_id=${baseModelId}&status=ready`);
      const data = await response.json();
      setAdapters(data.adapters || []);
    } catch (err) {
      console.error('Error loading adapters:', err);
    }
  };

  // ============================================================================
  // Smart Defaults & Preflight
  // ============================================================================

  const calculateSmartDefaults = async () => {
    try {
      const response = await fetch('/api/experiments/smart-defaults', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_model_id: selectedBaseModel.id,
          dataset_id: selectedDataset.id,
          recipe_id: selectedRecipe.id,
          num_gpus: resources.gpus || 1,
          gpu_type: resources.gpu_type
        })
      });

      const data = await response.json();
      setSmartDefaults(data);

      // Apply defaults if not already set
      if (!hyperparams.max_steps) {
        setHyperparams(data.train || {});
        setStrategy(data.strategy || {});
        setResources(prev => ({ ...prev, ...data.resources }));
        setEvalConfig(data.eval || {});
        setExportFormats(data.export || ['safetensors']);
      }
    } catch (err) {
      console.error('Error calculating smart defaults:', err);
    }
  };

  const runPreflightCheck = async () => {
    setPreflightLoading(true);
    try {
      const response = await fetch('/api/experiments/preflight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_model_id: selectedBaseModel.id,
          dataset_id: selectedDataset.id,
          recipe_id: selectedRecipe.id,
          adapters: selectedAdapters.map(a => ({ adapter_id: a.id, mode: 'attach' })),
          train: hyperparams,
          strategy,
          resources
        })
      });

      const data = await response.json();
      setPreflightResults(data);
    } catch (err) {
      console.error('Error running preflight check:', err);
    } finally {
      setPreflightLoading(false);
    }
  };

  // ============================================================================
  // Experiment Creation
  // ============================================================================

  const handleCreateExperiment = async () => {
    if (!experimentName || !selectedProject || !selectedBaseModel || !selectedDataset || !selectedRecipe) {
      setError('Please fill in all required fields');
      return;
    }

    if (preflightResults && !preflightResults.ok) {
      setError('Preflight checks failed. Please resolve errors before creating experiment.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const experimentData = {
        name: experimentName,
        project_id: selectedProject.id,
        base_model_id: selectedBaseModel.id,
        dataset_id: selectedDataset.id,
        recipe_id: selectedRecipe.id,
        adapters: selectedAdapters.map(a => ({ adapter_id: a.id, mode: 'attach' })),
        train: hyperparams,
        strategy,
        resources,
        eval: evalConfig,
        export: exportFormats,
        preflight_summary: preflightResults
      };

      const response = await fetch('/api/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(experimentData)
      });

      if (!response.ok) {
        throw new Error('Failed to create experiment');
      }

      const result = await response.json();

      // Success! Close modal and refresh
      if (onClose) {
        onClose(result);
      }
    } catch (err) {
      setError(err.message || 'Failed to create experiment');
    } finally {
      setLoading(false);
    }
  };

  // ============================================================================
  // Render Helpers
  // ============================================================================

  const getFilteredModels = () => {
    let filtered = baseModels;

    if (modelSearch) {
      const search = modelSearch.toLowerCase();
      filtered = filtered.filter(m =>
        m.name.toLowerCase().includes(search) ||
        m.family.toLowerCase().includes(search)
      );
    }

    if (modelFilters.family) {
      filtered = filtered.filter(m => m.family === modelFilters.family);
    }

    if (modelFilters.stage) {
      filtered = filtered.filter(m => m.stage === modelFilters.stage);
    }

    if (modelFilters.trainable !== undefined) {
      filtered = filtered.filter(m => m.trainable === modelFilters.trainable);
    }

    return filtered;
  };

  const getCompatibleDatasets = () => {
    if (!selectedBaseModel) return datasets;

    // Filter datasets by modality compatibility
    return datasets.filter(d => {
      const modelModality = selectedBaseModel.modality;
      const datasetModality = d.modality;

      if (modelModality === 'multimodal') return true;
      if (datasetModality === 'multimodal') return true;
      return modelModality === datasetModality;
    });
  };

  const getCompatibleRecipes = () => {
    if (!selectedBaseModel) return recipes;

    return recipes.filter(r => {
      const modelModality = selectedBaseModel.modality;
      const recipeModality = r.modality;

      return modelModality === recipeModality ||
        modelModality === 'multimodal' ||
        recipeModality === 'multimodal';
    });
  };

  const canProceed = () => {
    // Must have all required fields
    if (!experimentName || !selectedProject || !selectedBaseModel || !selectedDataset || !selectedRecipe) {
      return false;
    }

    // Must pass preflight checks
    if (preflightResults && !preflightResults.ok) {
      return false;
    }

    return true;
  };

  // ============================================================================
  // Section Renderers
  // ============================================================================

  const renderProjectSection = () => (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <Database className="w-5 h-5" />
        Project
      </h3>

      {projects.length > 0 ? (
        <select
          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          value={selectedProject?.id || ''}
          onChange={(e) => {
            const proj = projects.find(p => p.id === e.target.value);
            setSelectedProject(proj);
          }}
        >
          <option value="">Select a project...</option>
          {projects.map(proj => (
            <option key={proj.id} value={proj.id}>
              {proj.name}
            </option>
          ))}
        </select>
      ) : (
        <div className="text-gray-500 text-sm">No projects available</div>
      )}

      {selectedProject && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm">
          <div className="font-medium">{selectedProject.name}</div>
          {selectedProject.description && (
            <div className="text-gray-600 mt-1">{selectedProject.description}</div>
          )}
        </div>
      )}
    </div>
  );

  const renderBaseModelSection = () => (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <Cpu className="w-5 h-5" />
        Base Model
      </h3>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          placeholder="Search models..."
          className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          value={modelSearch}
          onChange={(e) => setModelSearch(e.target.value)}
        />
      </div>

      {/* Filters */}
      <div className="flex gap-2">
        <select
          className="px-3 py-1 border border-gray-300 rounded text-sm"
          value={modelFilters.stage || ''}
          onChange={(e) => setModelFilters({ ...modelFilters, stage: e.target.value || undefined })}
        >
          <option value="">All Stages</option>
          <option value="production">Production</option>
          <option value="staging">Staging</option>
        </select>

        <select
          className="px-3 py-1 border border-gray-300 rounded text-sm"
          value={modelFilters.trainable === undefined ? '' : modelFilters.trainable.toString()}
          onChange={(e) => setModelFilters({
            ...modelFilters,
            trainable: e.target.value === '' ? undefined : e.target.value === 'true'
          })}
        >
          <option value="">All</option>
          <option value="true">Trainable</option>
          <option value="false">Not Trainable</option>
        </select>
      </div>

      {/* Model List */}
      <div className="border border-gray-300 rounded-lg max-h-96 overflow-y-auto">
        {getFilteredModels().length > 0 ? (
          getFilteredModels().map(model => (
            <div
              key={model.id}
              className={`p-3 border-b border-gray-200 cursor-pointer hover:bg-gray-50 ${
                selectedBaseModel?.id === model.id ? 'bg-blue-50 border-blue-500' : ''
              }`}
              onClick={() => setSelectedBaseModel(model)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="font-medium">{model.name}</div>
                  <div className="text-sm text-gray-600 mt-1">
                    {model.family} • {model.params_b}B params • {model.dtype}
                  </div>
                  <div className="flex gap-2 mt-2">
                    <span className={`px-2 py-0.5 text-xs rounded ${
                      model.stage === 'production' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {model.stage}
                    </span>
                    {model.trainable && (
                      <span className="px-2 py-0.5 text-xs rounded bg-blue-100 text-blue-800">
                        Trainable
                      </span>
                    )}
                    {model.servable && (
                      <span className="px-2 py-0.5 text-xs rounded bg-purple-100 text-purple-800">
                        Servable
                      </span>
                    )}
                    {model.quantized && (
                      <span className="px-2 py-0.5 text-xs rounded bg-orange-100 text-orange-800">
                        Quantized
                      </span>
                    )}
                    {model.is_gguf && (
                      <span className="px-2 py-0.5 text-xs rounded bg-red-100 text-red-800">
                        GGUF
                      </span>
                    )}
                  </div>
                </div>
                {selectedBaseModel?.id === model.id && (
                  <CheckCircle className="w-5 h-5 text-blue-600 flex-shrink-0" />
                )}
              </div>
            </div>
          ))
        ) : (
          <div className="p-4 text-center text-gray-500 text-sm">
            No models found
          </div>
        )}
      </div>
    </div>
  );

  // Continue in next section...
  const renderDatasetSection = () => (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <Database className="w-5 h-5" />
        Dataset
      </h3>

      {!selectedProject ? (
        <div className="text-gray-500 text-sm">Please select a project first</div>
      ) : getCompatibleDatasets().length === 0 ? (
        <div className="text-gray-500 text-sm">No compatible datasets found</div>
      ) : (
        <div className="border border-gray-300 rounded-lg max-h-80 overflow-y-auto">
          {getCompatibleDatasets().map(dataset => (
            <div
              key={dataset.id}
              className={`p-3 border-b border-gray-200 cursor-pointer hover:bg-gray-50 ${
                selectedDataset?.id === dataset.id ? 'bg-blue-50 border-blue-500' : ''
              }`}
              onClick={() => setSelectedDataset(dataset)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="font-medium">{dataset.name}</div>
                  <div className="text-sm text-gray-600 mt-1">
                    {dataset.modality} • {dataset.num_samples?.toLocaleString()} samples • v{dataset.version}
                  </div>
                  {dataset.integrity_checked && (
                    <div className="mt-2">
                      {dataset.integrity_passed ? (
                        <span className="px-2 py-0.5 text-xs rounded bg-green-100 text-green-800">
                          ✓ Verified
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 text-xs rounded bg-red-100 text-red-800">
                          ✗ Failed validation
                        </span>
                      )}
                    </div>
                  )}
                </div>
                {selectedDataset?.id === dataset.id && (
                  <CheckCircle className="w-5 h-5 text-blue-600 flex-shrink-0" />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderRecipeSection = () => (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <Settings className="w-5 h-5" />
        Recipe
      </h3>

      {!selectedBaseModel ? (
        <div className="text-gray-500 text-sm">Please select a base model first</div>
      ) : getCompatibleRecipes().length === 0 ? (
        <div className="text-gray-500 text-sm">No compatible recipes found</div>
      ) : (
        <div className="space-y-2">
          {getCompatibleRecipes().map(recipe => (
            <div
              key={recipe.id}
              className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                selectedRecipe?.id === recipe.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300'
              }`}
              onClick={() => setSelectedRecipe(recipe)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="font-medium flex items-center gap-2">
                    {recipe.display_name}
                    {recipe.name.includes('recommended') && (
                      <span className="px-2 py-0.5 text-xs rounded bg-green-100 text-green-800">
                        Recommended
                      </span>
                    )}
                  </div>
                  {recipe.description && (
                    <div className="text-sm text-gray-600 mt-1">{recipe.description}</div>
                  )}
                  <div className="flex gap-2 mt-2 text-xs text-gray-500">
                    <span>Type: {recipe.recipe_type}</span>
                    {recipe.min_gpu_memory_gb && (
                      <span>• Min GPU: {recipe.min_gpu_memory_gb}GB</span>
                    )}
                  </div>
                </div>
                {selectedRecipe?.id === recipe.id && (
                  <CheckCircle className="w-5 h-5 text-blue-600 flex-shrink-0" />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderAdaptersSection = () => (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <Plus className="w-5 h-5" />
        Adapters <span className="text-sm font-normal text-gray-500">(Optional)</span>
      </h3>

      {!selectedBaseModel ? (
        <div className="text-gray-500 text-sm">Please select a base model first</div>
      ) : adapters.length === 0 ? (
        <div className="text-gray-500 text-sm">No adapters available for this model</div>
      ) : (
        <div className="space-y-2">
          {adapters.map(adapter => (
            <label
              key={adapter.id}
              className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50"
            >
              <input
                type="checkbox"
                checked={selectedAdapters.some(a => a.id === adapter.id)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedAdapters([...selectedAdapters, adapter]);
                  } else {
                    setSelectedAdapters(selectedAdapters.filter(a => a.id !== adapter.id));
                  }
                }}
                className="w-4 h-4"
              />
              <div className="flex-1">
                <div className="font-medium">{adapter.name}</div>
                <div className="text-sm text-gray-600">
                  {adapter.adapter_type} • r={adapter.rank} • α={adapter.alpha}
                </div>
              </div>
            </label>
          ))}
        </div>
      )}
    </div>
  );

  const renderHyperparamsSection = () => (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Hyperparameters
        </h3>
        {smartDefaults && (
          <button
            onClick={() => setHyperparams(smartDefaults.train || {})}
            className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
          >
            <Zap className="w-4 h-4" />
            Use Smart Defaults
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-sm font-medium mb-1">Max Steps</label>
          <input
            type="number"
            className="w-full p-2 border border-gray-300 rounded"
            value={hyperparams.max_steps || ''}
            onChange={(e) => setHyperparams({ ...hyperparams, max_steps: parseInt(e.target.value) || 0 })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Batch Size</label>
          <input
            type="number"
            className="w-full p-2 border border-gray-300 rounded"
            value={hyperparams.global_batch_size || ''}
            onChange={(e) => setHyperparams({ ...hyperparams, global_batch_size: parseInt(e.target.value) || 1 })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Gradient Accumulation</label>
          <input
            type="number"
            className="w-full p-2 border border-gray-300 rounded"
            value={hyperparams.grad_accum || ''}
            onChange={(e) => setHyperparams({ ...hyperparams, grad_accum: parseInt(e.target.value) || 1 })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Learning Rate</label>
          <input
            type="number"
            step="0.00001"
            className="w-full p-2 border border-gray-300 rounded"
            value={hyperparams.learning_rate || ''}
            onChange={(e) => setHyperparams({ ...hyperparams, learning_rate: parseFloat(e.target.value) || 0 })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Warmup Steps</label>
          <input
            type="number"
            className="w-full p-2 border border-gray-300 rounded"
            value={hyperparams.warmup_steps || ''}
            onChange={(e) => setHyperparams({ ...hyperparams, warmup_steps: parseInt(e.target.value) || 0 })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Checkpoint Interval</label>
          <input
            type="number"
            className="w-full p-2 border border-gray-300 rounded"
            value={hyperparams.checkpoint_interval || ''}
            onChange={(e) => setHyperparams({ ...hyperparams, checkpoint_interval: parseInt(e.target.value) || 100 })}
          />
        </div>
      </div>

      <details className="border border-gray-200 rounded-lg">
        <summary className="px-3 py-2 cursor-pointer font-medium text-sm">
          Advanced Settings
        </summary>
        <div className="p-3 space-y-3 border-t">
          <div>
            <label className="block text-sm font-medium mb-1">LR Scheduler</label>
            <select
              className="w-full p-2 border border-gray-300 rounded"
              value={hyperparams.lr_scheduler || 'cosine'}
              onChange={(e) => setHyperparams({ ...hyperparams, lr_scheduler: e.target.value })}
            >
              <option value="cosine">Cosine</option>
              <option value="linear">Linear</option>
              <option value="constant">Constant</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Weight Decay</label>
            <input
              type="number"
              step="0.001"
              className="w-full p-2 border border-gray-300 rounded"
              value={hyperparams.weight_decay || ''}
              onChange={(e) => setHyperparams({ ...hyperparams, weight_decay: parseFloat(e.target.value) || 0 })}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Max Gradient Norm</label>
            <input
              type="number"
              step="0.1"
              className="w-full p-2 border border-gray-300 rounded"
              value={hyperparams.max_grad_norm || ''}
              onChange={(e) => setHyperparams({ ...hyperparams, max_grad_norm: parseFloat(e.target.value) || 1.0 })}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Random Seed</label>
            <input
              type="number"
              className="w-full p-2 border border-gray-300 rounded"
              value={hyperparams.seed || ''}
              onChange={(e) => setHyperparams({ ...hyperparams, seed: parseInt(e.target.value) || 42 })}
            />
          </div>
        </div>
      </details>
    </div>
  );

  const renderResourcesSection = () => (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <Cpu className="w-5 h-5" />
        Resources & Strategy
      </h3>

      <div>
        <label className="block text-sm font-medium mb-1">Number of GPUs</label>
        <input
          type="number"
          min="1"
          className="w-full p-2 border border-gray-300 rounded"
          value={resources.gpus || 1}
          onChange={(e) => setResources({ ...resources, gpus: parseInt(e.target.value) || 1 })}
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">GPU Type (Optional)</label>
        <select
          className="w-full p-2 border border-gray-300 rounded"
          value={resources.gpu_type || ''}
          onChange={(e) => setResources({ ...resources, gpu_type: e.target.value || undefined })}
        >
          <option value="">Auto-detect</option>
          <option value="H100">H100 (80GB)</option>
          <option value="A100">A100 (80GB)</option>
          <option value="A100-40GB">A100 (40GB)</option>
          <option value="V100">V100 (32GB)</option>
          <option value="A10">A10 (24GB)</option>
          <option value="L4">L4 (24GB)</option>
          <option value="L40">L40 (48GB)</option>
        </select>
      </div>

      {(resources.gpus || 1) > 1 && (
        <>
          <div>
            <label className="block text-sm font-medium mb-1">Distribution Strategy</label>
            <select
              className="w-full p-2 border border-gray-300 rounded"
              value={strategy.type || ''}
              onChange={(e) => setStrategy({ ...strategy, type: e.target.value || undefined })}
            >
              <option value="">Auto-select</option>
              <option value="ddp">DDP (Data Parallel)</option>
              <option value="fsdp">FSDP (Fully Sharded)</option>
              <option value="deepspeed">DeepSpeed</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Mixed Precision</label>
            <select
              className="w-full p-2 border border-gray-300 rounded"
              value={strategy.mixed_precision || ''}
              onChange={(e) => setStrategy({ ...strategy, mixed_precision: e.target.value || undefined })}
            >
              <option value="">None</option>
              <option value="fp16">FP16</option>
              <option value="bf16">BF16</option>
            </select>
          </div>
        </>
      )}
    </div>
  );

  const renderEvalExportSection = () => (
    <div className="space-y-4">
      {/* Eval Config */}
      <div className="space-y-3">
        <h3 className="text-lg font-semibold">Evaluation</h3>

        <div>
          <label className="block text-sm font-medium mb-1">Eval Interval (steps)</label>
          <input
            type="number"
            className="w-full p-2 border border-gray-300 rounded"
            value={evalConfig.interval || ''}
            onChange={(e) => setEvalConfig({ ...evalConfig, interval: parseInt(e.target.value) || 500 })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Eval Suites</label>
          <div className="space-y-2">
            {['perplexity', 'mmlu', 'hellaswag', 'accuracy', 'f1'].map(suite => (
              <label key={suite} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={evalConfig.suites?.includes(suite)}
                  onChange={(e) => {
                    const suites = evalConfig.suites || [];
                    if (e.target.checked) {
                      setEvalConfig({ ...evalConfig, suites: [...suites, suite] });
                    } else {
                      setEvalConfig({ ...evalConfig, suites: suites.filter(s => s !== suite) });
                    }
                  }}
                  className="w-4 h-4"
                />
                <span className="text-sm">{suite}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Export Formats */}
      <div className="space-y-3">
        <h3 className="text-lg font-semibold">Export Formats</h3>
        <div className="space-y-2">
          {['safetensors', 'onnx', 'gguf'].map(format => (
            <label key={format} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={exportFormats.includes(format)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setExportFormats([...exportFormats, format]);
                  } else {
                    setExportFormats(exportFormats.filter(f => f !== format));
                  }
                }}
                className="w-4 h-4"
              />
              <span className="text-sm uppercase">{format}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );

  const renderPreflightSection = () => (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <AlertCircle className="w-5 h-5" />
        Pre-flight Checks
      </h3>

      {preflightLoading ? (
        <div className="text-center py-4 text-gray-500">Running checks...</div>
      ) : !preflightResults ? (
        <div className="text-center py-4 text-gray-500">Complete configuration to run checks</div>
      ) : (
        <div className="space-y-3">
          {/* Status */}
          <div className={`p-3 rounded-lg ${
            preflightResults.ok ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
          }`}>
            <div className="flex items-center gap-2 font-medium">
              {preflightResults.ok ? (
                <><CheckCircle className="w-5 h-5 text-green-600" /> Ready to Launch</>
              ) : (
                <><AlertCircle className="w-5 h-5 text-red-600" /> Issues Detected</>
              )}
            </div>
          </div>

          {/* VRAM Estimate */}
          {preflightResults.vram_breakdown && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="font-medium mb-2">Estimated VRAM Usage</div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Model:</span>
                  <span>{preflightResults.vram_breakdown.model_gb} GB</span>
                </div>
                <div className="flex justify-between">
                  <span>Optimizer:</span>
                  <span>{preflightResults.vram_breakdown.optimizer_gb} GB</span>
                </div>
                <div className="flex justify-between">
                  <span>Gradients:</span>
                  <span>{preflightResults.vram_breakdown.gradients_gb} GB</span>
                </div>
                <div className="flex justify-between">
                  <span>Activations:</span>
                  <span>{preflightResults.vram_breakdown.activations_gb} GB</span>
                </div>
                <div className="flex justify-between font-medium border-t border-blue-300 pt-1 mt-1">
                  <span>Total (with buffer):</span>
                  <span>{preflightResults.vram_breakdown.total_with_buffer_gb} GB</span>
                </div>
              </div>
            </div>
          )}

          {/* Throughput */}
          {preflightResults.throughput && (
            <div className="p-3 bg-purple-50 border border-purple-200 rounded-lg">
              <div className="font-medium mb-2">Estimated Throughput</div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Tokens/sec:</span>
                  <span>{preflightResults.throughput.tokens_per_sec}</span>
                </div>
                <div className="flex justify-between">
                  <span>Time per step:</span>
                  <span>{preflightResults.throughput.time_per_step_ms} ms</span>
                </div>
              </div>
            </div>
          )}

          {/* Warnings */}
          {preflightResults.warnings && preflightResults.warnings.length > 0 && (
            <div className="space-y-2">
              <div className="font-medium text-yellow-800 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                Warnings
              </div>
              {preflightResults.warnings.map((warn, i) => (
                <div key={i} className="p-2 bg-yellow-50 border border-yellow-200 rounded text-sm">
                  {warn}
                </div>
              ))}
            </div>
          )}

          {/* Errors */}
          {preflightResults.errors && preflightResults.errors.length > 0 && (
            <div className="space-y-2">
              <div className="font-medium text-red-800 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                Errors
              </div>
              {preflightResults.errors.map((err, i) => (
                <div key={i} className="p-2 bg-red-50 border border-red-200 rounded text-sm">
                  {err}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );

  // ============================================================================
  // Main Render
  // ============================================================================

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-7xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-2xl font-bold">Create Experiment</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-6">
            {/* Experiment Name */}
            <div>
              <label className="block text-sm font-medium mb-2">Experiment Name *</label>
              <input
                type="text"
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="Enter experiment name..."
                value={experimentName}
                onChange={(e) => setExperimentName(e.target.value)}
              />
            </div>

            {/* Sections Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {renderProjectSection()}
              {renderBaseModelSection()}
              {renderDatasetSection()}
              {renderRecipeSection()}
              {renderAdaptersSection()}
              {renderHyperparamsSection()}
              {renderResourcesSection()}
              {renderEvalExportSection()}
            </div>

            {/* Preflight - Full Width */}
            <div className="border-t border-gray-200 pt-6">
              {renderPreflightSection()}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
          <div>
            {error && (
              <div className="text-red-600 text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
            )}
          </div>

          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              disabled={loading}
            >
              Cancel
            </button>

            <button
              onClick={handleCreateExperiment}
              disabled={!canProceed() || loading}
              className={`px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                canProceed() && !loading
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              {loading ? (
                'Creating...'
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Create Experiment
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
