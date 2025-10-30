import React, { useState, useEffect, useCallback } from 'react';
import { Search, Loader2, AlertCircle, CheckCircle2, Cpu, Database, Layers, Tag, Zap } from 'lucide-react';

/**
 * BaseModelSelector - Searchable dropdown for selecting base models
 *
 * Features:
 * - Search and filter models
 * - Display model metadata as chips (params, dtype, family, stage, trainable/servable)
 * - Empty state with helpful message
 * - Pagination for large model lists
 * - Compatibility highlighting when used with dataset selection
 */
const BaseModelSelector = ({
  selectedModelId,
  onSelectModel,
  projectId = null,
  filterTrainable = true,
  compatibleWithDatasetId = null,
  className = ""
}) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [total, setTotal] = useState(0);

  // Fetch models from API
  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (searchQuery) params.append('q', searchQuery);
      if (filterTrainable) params.append('trainable', 'true');
      if (projectId) params.append('project', projectId);
      params.append('limit', '50');

      const response = await fetch(`/api/base-models?${params}`);
      if (!response.ok) throw new Error('Failed to fetch models');

      const data = await response.json();
      setModels(data.models || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message);
      setModels([]);
    } finally {
      setLoading(false);
    }
  }, [searchQuery, filterTrainable, projectId]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Get selected model details
  const selectedModel = models.find(m => m.id === selectedModelId);

  // Handle model selection
  const handleSelectModel = (model) => {
    onSelectModel(model);
    setIsOpen(false);
  };

  // Format parameter count
  const formatParams = (params_b) => {
    if (!params_b) return null;
    if (params_b >= 1) return `${params_b.toFixed(1)}B`;
    return `${(params_b * 1000).toFixed(0)}M`;
  };

  // Render model chip badges
  const ModelChips = ({ model }) => (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {model.params_b && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-md bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
          <Cpu size={12} />
          {formatParams(model.params_b)}
        </span>
      )}
      {model.dtype && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-md bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
          <Database size={12} />
          {model.dtype}
        </span>
      )}
      {model.family && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-md bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200">
          <Layers size={12} />
          {model.family}
        </span>
      )}
      {model.stage && (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-md ${
          model.stage === 'production' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
          model.stage === 'staging' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
          'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
        }`}>
          <Tag size={12} />
          {model.stage}
        </span>
      )}
      {model.trainable && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-md bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
          <Zap size={12} />
          Trainable
        </span>
      )}
      {model.servable && !model.trainable && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-md bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
          <CheckCircle2 size={12} />
          Servable
        </span>
      )}
    </div>
  );

  // Empty state
  const EmptyState = () => (
    <div className="text-center py-12 px-4">
      <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 dark:bg-gray-800 mb-4">
        <Database className="w-8 h-8 text-gray-400" />
      </div>
      <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
        No models yet
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        Create one from Templates or import from HuggingFace
      </p>
      <div className="flex gap-2 justify-center">
        <a
          href="/models?tab=templates"
          className="inline-flex items-center px-3 py-2 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700"
        >
          Browse Templates
        </a>
        <a
          href="/models?tab=import"
          className="inline-flex items-center px-3 py-2 text-sm font-medium rounded-md border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-700"
        >
          Import from HF
        </a>
      </div>
    </div>
  );

  return (
    <div className={`relative ${className}`}>
      {/* Selected model display / trigger */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 text-left border border-gray-300 rounded-lg bg-white dark:bg-gray-800 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
      >
        <div className="flex-1 min-w-0">
          {selectedModel ? (
            <div>
              <div className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                {selectedModel.name}
              </div>
              <ModelChips model={selectedModel} />
            </div>
          ) : (
            <span className="text-sm text-gray-500 dark:text-gray-400">
              Select a base model...
            </span>
          )}
        </div>
        <Search className="ml-2 w-5 h-5 text-gray-400 flex-shrink-0" />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-50 mt-2 w-full bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 max-h-96 overflow-hidden">
          {/* Search input */}
          <div className="p-3 border-b border-gray-200 dark:border-gray-700">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search models..."
                className="w-full pl-10 pr-4 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                autoFocus
              />
            </div>
          </div>

          {/* Results */}
          <div className="overflow-y-auto max-h-80">
            {loading && (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
              </div>
            )}

            {error && (
              <div className="p-4 text-center">
                <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-2" />
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}

            {!loading && !error && models.length === 0 && <EmptyState />}

            {!loading && !error && models.length > 0 && (
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {models.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => handleSelectModel(model)}
                    className={`w-full px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${
                      selectedModelId === model.id ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                          {model.name}
                        </div>
                        {model.description && (
                          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 line-clamp-2">
                            {model.description}
                          </div>
                        )}
                        <ModelChips model={model} />
                      </div>
                      {selectedModelId === model.id && (
                        <CheckCircle2 className="ml-2 w-5 h-5 text-blue-600 flex-shrink-0" />
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Footer with count */}
          {!loading && !error && total > 0 && (
            <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Showing {models.length} of {total} models
                {total > 50 && ' (refine search to see more)'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Click outside to close */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
};

export default BaseModelSelector;
