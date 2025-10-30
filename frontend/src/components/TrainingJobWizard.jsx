import React, { useState, useEffect } from 'react';
import { X, ChevronLeft, ChevronRight, Check, AlertCircle, Loader2, Zap, Database, Settings, Cpu } from 'lucide-react';
import BaseModelSelector from './BaseModelSelector';

/**
 * TrainingJobWizard - Multi-step wizard for creating training jobs from templates
 *
 * Steps:
 * 1. Base Model Selection
 * 2. Dataset Selection (filtered by model compatibility)
 * 3. Training Style (QLoRA/LoRA/Full)
 * 4. Hyperparameters (smart defaults)
 * 5. Resources (GPUs, strategy, partition)
 * 6. Pre-flight Check (VRAM/time estimates)
 * 7. Review & Create
 */
const TrainingJobWizard = ({ templateId, onClose, onSuccess }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [template, setTemplate] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Form state
  const [formData, setFormData] = useState({
    jobName: '',
    baseModelId: null,
    datasetId: null,
    recipeId: null,
    trainStyle: 'qlora', // qlora, lora, full
    hyperparams: {
      max_steps: 1000,
      global_batch_size: 8,
      grad_accum: 4,
      learning_rate: 2e-5,
      warmup_steps: 100,
      seed: 42,
      checkpoint_interval: 100,
    },
    resources: {
      gpus: 1,
      gpu_type: 'a100',
      strategy: 'ddp',
      mixed_precision: 'bf16',
    },
    preflight: null,
  });

  const [datasets, setDatasets] = useState([]);
  const [recipes, setRecipes] = useState([]);
  const [preflightResult, setPreflightResult] = useState(null);

  // Load template
  useEffect(() => {
    if (templateId) {
      loadTemplate();
    }
  }, [templateId]);

  const loadTemplate = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/recipes/${templateId}`);
      if (!response.ok) throw new Error('Failed to load template');
      const data = await response.json();
      setTemplate(data);

      // Set initial recipe
      setFormData(prev => ({
        ...prev,
        recipeId: data.id,
        jobName: `${data.display_name} Training - ${new Date().toLocaleDateString()}`,
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Load datasets when model is selected
  useEffect(() => {
    if (formData.baseModelId) {
      loadCompatibleDatasets();
    }
  }, [formData.baseModelId]);

  const loadCompatibleDatasets = async () => {
    try {
      const response = await fetch(
        `/api/datasets?compatible_with_model=${formData.baseModelId}&limit=50`
      );
      if (!response.ok) throw new Error('Failed to load datasets');
      const data = await response.json();
      setDatasets(data.datasets || []);
    } catch (err) {
      console.error('Failed to load datasets:', err);
    }
  };

  // Run preflight checks
  const runPreflight = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/experiments/preflight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_model_id: formData.baseModelId,
          dataset_id: formData.datasetId,
          recipe_id: formData.recipeId,
          train: formData.hyperparams,
          resources: formData.resources,
        }),
      });

      if (!response.ok) throw new Error('Preflight checks failed');
      const result = await response.json();
      setPreflightResult(result);

      if (!result.ok) {
        setError('Configuration has errors. Please review and fix.');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Create and start job
  const createJob = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formData.jobName,
          base_model_id: formData.baseModelId,
          dataset_id: formData.datasetId,
          recipe_id: formData.recipeId,
          train: formData.hyperparams,
          resources: formData.resources,
          metadata: {
            origin: 'template',
            template_id: templateId,
          },
        }),
      });

      if (!response.ok) throw new Error('Failed to create job');
      const job = await response.json();

      // Show success notification
      if (onSuccess) {
        onSuccess(job);
      }

      // Redirect to job detail page
      window.location.href = `/jobs/${job.id}`;
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Wizard steps configuration
  const steps = [
    {
      id: 'base-model',
      title: 'Base Model',
      icon: Cpu,
      validate: () => formData.baseModelId !== null,
    },
    {
      id: 'dataset',
      title: 'Dataset',
      icon: Database,
      validate: () => formData.datasetId !== null,
    },
    {
      id: 'training-style',
      title: 'Training Style',
      icon: Zap,
      validate: () => formData.trainStyle !== null,
    },
    {
      id: 'hyperparams',
      title: 'Hyperparameters',
      icon: Settings,
      validate: () => true,
    },
    {
      id: 'resources',
      title: 'Resources',
      icon: Cpu,
      validate: () => formData.resources.gpus > 0,
    },
    {
      id: 'preflight',
      title: 'Pre-flight',
      icon: Check,
      validate: () => preflightResult && preflightResult.ok,
    },
  ];

  const currentStepConfig = steps[currentStep];
  const canProceed = currentStepConfig.validate();

  const nextStep = async () => {
    if (currentStep === steps.length - 2) {
      // Run preflight before final step
      await runPreflight();
    }
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              Create Training Job
            </h2>
            {template && (
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Template: {template.display_name}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Progress indicator */}
        <div className="px-6 pt-6">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center ${
                      index < currentStep
                        ? 'bg-green-600 text-white'
                        : index === currentStep
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-400'
                    }`}
                  >
                    {index < currentStep ? (
                      <Check className="w-5 h-5" />
                    ) : (
                      <step.icon className="w-5 h-5" />
                    )}
                  </div>
                  <span className="text-xs mt-2 text-center">{step.title}</span>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={`h-1 flex-1 ${
                      index < currentStep
                        ? 'bg-green-600'
                        : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-16rem)]">
          {error && (
            <div className="mb-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-800 dark:text-red-200">
                  {error}
                </p>
              </div>
            </div>
          )}

          {/* Step 0: Base Model */}
          {currentStep === 0 && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                Select Base Model
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                Choose the base model you want to fine-tune. Only trainable models are shown.
              </p>
              <BaseModelSelector
                selectedModelId={formData.baseModelId}
                onSelectModel={(model) =>
                  setFormData(prev => ({ ...prev, baseModelId: model.id }))
                }
                filterTrainable={true}
              />
            </div>
          )}

          {/* Step 1: Dataset */}
          {currentStep === 1 && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                Select Dataset
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                Choose a dataset compatible with your selected model.
              </p>
              {datasets.length === 0 ? (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No compatible datasets found</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {datasets.map((dataset) => (
                    <button
                      key={dataset.id}
                      onClick={() =>
                        setFormData(prev => ({ ...prev, datasetId: dataset.id }))
                      }
                      className={`w-full p-4 text-left border rounded-lg transition-colors ${
                        formData.datasetId === dataset.id
                          ? 'border-blue-600 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
                      }`}
                    >
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {dataset.name}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {dataset.num_samples?.toLocaleString()} samples • {dataset.modality}
                      </div>
                      {dataset.compatibility && !dataset.compatibility.compatible && (
                        <div className="mt-2 text-xs text-yellow-600 dark:text-yellow-400">
                          ⚠ {dataset.compatibility.warnings?.[0]}
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Step 2: Training Style */}
          {currentStep === 2 && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                Training Style
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                Choose how to fine-tune the model.
              </p>
              <div className="space-y-3">
                {['qlora', 'lora', 'full'].map((style) => (
                  <button
                    key={style}
                    onClick={() =>
                      setFormData(prev => ({ ...prev, trainStyle: style }))
                    }
                    className={`w-full p-4 text-left border rounded-lg transition-colors ${
                      formData.trainStyle === style
                        ? 'border-blue-600 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
                    }`}
                  >
                    <div className="font-medium text-gray-900 dark:text-gray-100 capitalize">
                      {style === 'qlora' ? 'QLoRA' : style === 'lora' ? 'LoRA' : 'Full Fine-Tuning'}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {style === 'qlora' && 'Memory-efficient quantized training (Recommended)'}
                      {style === 'lora' && 'Low-rank adaptation with good quality/speed'}
                      {style === 'full' && 'Full parameter fine-tuning (Best quality, most VRAM)'}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 3: Hyperparameters */}
          {currentStep === 3 && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                Hyperparameters
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                Configure training parameters (smart defaults are pre-filled).
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Steps
                  </label>
                  <input
                    type="number"
                    value={formData.hyperparams.max_steps}
                    onChange={(e) =>
                      setFormData(prev => ({
                        ...prev,
                        hyperparams: { ...prev.hyperparams, max_steps: parseInt(e.target.value) },
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={formData.hyperparams.global_batch_size}
                    onChange={(e) =>
                      setFormData(prev => ({
                        ...prev,
                        hyperparams: { ...prev.hyperparams, global_batch_size: parseInt(e.target.value) },
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    step="0.00001"
                    value={formData.hyperparams.learning_rate}
                    onChange={(e) =>
                      setFormData(prev => ({
                        ...prev,
                        hyperparams: { ...prev.hyperparams, learning_rate: parseFloat(e.target.value) },
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Gradient Accumulation
                  </label>
                  <input
                    type="number"
                    value={formData.hyperparams.grad_accum}
                    onChange={(e) =>
                      setFormData(prev => ({
                        ...prev,
                        hyperparams: { ...prev.hyperparams, grad_accum: parseInt(e.target.value) },
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Resources */}
          {currentStep === 4 && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                Resource Configuration
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                Configure GPU and training strategy.
              </p>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Number of GPUs
                  </label>
                  <input
                    type="number"
                    min="1"
                    value={formData.resources.gpus}
                    onChange={(e) =>
                      setFormData(prev => ({
                        ...prev,
                        resources: { ...prev.resources, gpus: parseInt(e.target.value) },
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Mixed Precision
                  </label>
                  <select
                    value={formData.resources.mixed_precision}
                    onChange={(e) =>
                      setFormData(prev => ({
                        ...prev,
                        resources: { ...prev.resources, mixed_precision: e.target.value },
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  >
                    <option value="bf16">BF16 (Recommended)</option>
                    <option value="fp16">FP16</option>
                    <option value="fp32">FP32 (Full Precision)</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* Step 5: Preflight */}
          {currentStep === 5 && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                Pre-flight Checks
              </h3>
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                </div>
              ) : preflightResult ? (
                <div className="space-y-4">
                  {/* Errors */}
                  {preflightResult.errors && preflightResult.errors.length > 0 && (
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                      <h4 className="font-medium text-red-800 dark:text-red-200 mb-2">
                        Errors
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-sm text-red-700 dark:text-red-300">
                        {preflightResult.errors.map((error, i) => (
                          <li key={i}>{error}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Warnings */}
                  {preflightResult.warnings && preflightResult.warnings.length > 0 && (
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                      <h4 className="font-medium text-yellow-800 dark:text-yellow-200 mb-2">
                        Warnings
                      </h4>
                      <ul className="list-disc list-inside space-y-1 text-sm text-yellow-700 dark:text-yellow-300">
                        {preflightResult.warnings.map((warning, i) => (
                          <li key={i}>{warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Resource Estimates */}
                  {preflightResult.ok && (
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                      <h4 className="font-medium text-green-800 dark:text-green-200 mb-3">
                        ✓ Configuration Valid
                      </h4>
                      <div className="space-y-2 text-sm text-green-700 dark:text-green-300">
                        <p>
                          <strong>Estimated VRAM:</strong>{' '}
                          {(preflightResult.estimated_vram_mb / 1024).toFixed(2)} GB
                        </p>
                        {preflightResult.time_per_step_ms && (
                          <p>
                            <strong>Time per Step:</strong>{' '}
                            {preflightResult.time_per_step_ms.toFixed(0)} ms
                          </p>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-gray-500 dark:text-gray-400">
                  Click "Next" to run preflight checks...
                </p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={prevStep}
            disabled={currentStep === 0}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-4 h-4 inline mr-1" />
            Back
          </button>

          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
            >
              Cancel
            </button>
            {currentStep < steps.length - 1 ? (
              <button
                onClick={nextStep}
                disabled={!canProceed || loading}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRight className="w-4 h-4 inline ml-1" />
              </button>
            ) : (
              <button
                onClick={createJob}
                disabled={!canProceed || loading}
                className="px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 inline mr-1 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Check className="w-4 h-4 inline mr-1" />
                    Create & Start
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingJobWizard;
