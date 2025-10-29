import React, { useEffect, useState } from 'react';

/**
 * GatingMetricsViewer - Comprehensive visualization for gating mechanism metrics
 *
 * Displays expert utilization heatmaps, capacity overflow, z-loss, and other
 * gating-specific metrics for MoE, MoE-LoRA, Routerless MoE, Mixture-of-Depths,
 * FiLM gating, and Span Routing.
 */
export default function GatingMetricsViewer({ jobId, api }) {
  const [metrics, setMetrics] = useState(null);
  const [expertUtilization, setExpertUtilization] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('overview'); // overview | heatmap | timeline

  useEffect(() => {
    if (!jobId || !api) return;

    const fetchMetrics = async () => {
      try {
        setLoading(true);

        // Fetch gating metrics
        const metricsRes = await fetch(`${api}/api/jobs/${jobId}/gating/metrics`);
        const metricsData = await metricsRes.json();

        if (!metricsData.enabled) {
          setMetrics({ enabled: false });
          setLoading(false);
          return;
        }

        setMetrics(metricsData);

        // Fetch expert utilization heatmap data
        const utilRes = await fetch(`${api}/api/jobs/${jobId}/gating/expert-utilization`);
        if (utilRes.ok) {
          const utilData = await utilRes.json();
          setExpertUtilization(utilData);
        }

        setError(null);
      } catch (err) {
        console.error('Failed to fetch gating metrics:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();

    // Auto-refresh every 5 seconds during training
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, [jobId, api]);

  if (loading && !metrics) {
    return (
      <div className="p-6 text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        <p className="mt-2 text-sm text-text/60">Loading gating metrics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
        <p className="text-sm text-red-600 dark:text-red-400">Error loading gating metrics: {error}</p>
      </div>
    );
  }

  if (!metrics || !metrics.enabled) {
    return (
      <div className="p-6 bg-gray-50 dark:bg-gray-900/20 border border-border rounded-lg">
        <p className="text-sm text-text/60">Gating is not enabled for this job.</p>
      </div>
    );
  }

  const { type, metrics: metricsData } = metrics;
  const summary = metricsData?.summary || {};

  // Render overview panel
  const renderOverview = () => {
    const getGatingTypeInfo = () => {
      const types = {
        moe: { name: 'Mixture of Experts (MoE)', desc: 'Token-level Top-K routing with capacity factors' },
        moe_lora: { name: 'MoE-LoRA', desc: 'Per-expert LoRA adapters for reduced VRAM' },
        routerless: { name: 'Routerless MoE', desc: 'DeepSeek-style MoE without explicit routing' },
        mixture_of_depths: { name: 'Mixture-of-Depths', desc: 'Dynamic layer selection with early exit' },
        film_gates: { name: 'FiLM Gating', desc: 'Multi-modal feature-wise modulation' },
        span_routing: { name: 'Span Routing', desc: 'Contiguous token spans for efficiency' },
      };
      return types[type] || { name: type, desc: 'Custom gating mechanism' };
    };

    const typeInfo = getGatingTypeInfo();

    return (
      <div className="space-y-4">
        {/* Header */}
        <div className="bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/20 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-1">{typeInfo.name}</h3>
          <p className="text-sm text-text/70">{typeInfo.desc}</p>
        </div>

        {/* Key metrics grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Capacity Overflow */}
          {summary.capacity_overflow !== undefined && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text/60 mb-1">Capacity Overflow</div>
              <div className="text-2xl font-bold">
                {summary.capacity_overflow.toFixed(2)}%
              </div>
              <div className="text-xs text-text/60 mt-1">
                {summary.capacity_overflow < 5 ? '✓ Good load balance' : '⚠️ High overflow'}
              </div>
            </div>
          )}

          {/* Z-loss */}
          {summary.z_loss !== undefined && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text/60 mb-1">Z-Loss (Auxiliary)</div>
              <div className="text-2xl font-bold">
                {summary.z_loss.toFixed(4)}
              </div>
              <div className="text-xs text-text/60 mt-1">
                {summary.z_loss < 0.05 ? '✓ Stable routing' : '⚠️ Check routing stability'}
              </div>
            </div>
          )}

          {/* Gate Entropy */}
          {summary.gate_entropy !== undefined && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text/60 mb-1">Gate Entropy</div>
              <div className="text-2xl font-bold">
                {summary.gate_entropy.toFixed(2)}
              </div>
              <div className="text-xs text-text/60 mt-1">
                Routing diversity
              </div>
            </div>
          )}

          {/* Expert Load Variance */}
          {summary.expert_load_variance !== undefined && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text/60 mb-1">Load Variance</div>
              <div className="text-2xl font-bold">
                {summary.expert_load_variance.toFixed(4)}
              </div>
              <div className="text-xs text-text/60 mt-1">
                {summary.expert_load_variance < 0.05 ? '✓ Balanced' : '⚠️ Imbalanced'}
              </div>
            </div>
          )}

          {/* Average Depth (Mixture-of-Depths) */}
          {summary.avg_depth !== undefined && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text/60 mb-1">Average Depth</div>
              <div className="text-2xl font-bold">
                {summary.avg_depth.toFixed(2)}
              </div>
              <div className="text-xs text-text/60 mt-1">
                layers per token
              </div>
            </div>
          )}

          {/* Exit Rate (Mixture-of-Depths) */}
          {summary.exit_rate !== undefined && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text/60 mb-1">Early Exit Rate</div>
              <div className="text-2xl font-bold">
                {(summary.exit_rate * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-text/60 mt-1">
                tokens exiting early
              </div>
            </div>
          )}

          {/* Routing Confidence */}
          {summary.routing_confidence !== undefined && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text/60 mb-1">Routing Confidence</div>
              <div className="text-2xl font-bold">
                {(summary.routing_confidence * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-text/60 mt-1">
                average confidence
              </div>
            </div>
          )}

          {/* Number of Spans */}
          {summary.num_spans !== undefined && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text/60 mb-1">Number of Spans</div>
              <div className="text-2xl font-bold">
                {summary.num_spans}
              </div>
              <div className="text-xs text-text/60 mt-1">
                routed spans
              </div>
            </div>
          )}
        </div>

        {/* Modality Distribution (FiLM gating) */}
        {summary.modality_distribution && (
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="text-sm font-semibold mb-3">Modality Distribution</div>
            <div className="space-y-2">
              {summary.modality_distribution.map((val, idx) => {
                const modalities = ['Text', 'Image', 'Audio', 'Video'];
                return (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="text-xs w-16">{modalities[idx] || `Mod ${idx}`}</div>
                    <div className="flex-1 bg-border rounded-full h-4 overflow-hidden">
                      <div
                        className="bg-primary h-full transition-all duration-300"
                        style={{ width: `${(val * 100).toFixed(1)}%` }}
                      ></div>
                    </div>
                    <div className="text-xs w-12 text-right">{(val * 100).toFixed(1)}%</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Training Progress */}
        {(summary.step || summary.epoch) && (
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="text-sm font-semibold mb-2">Training Progress</div>
            <div className="flex items-center gap-4 text-sm">
              {summary.epoch && <div>Epoch: <span className="font-mono">{summary.epoch}</span></div>}
              {summary.step && <div>Step: <span className="font-mono">{summary.step}</span></div>}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render expert utilization heatmap
  const renderHeatmap = () => {
    if (!expertUtilization || !expertUtilization.expert_utilization) {
      return (
        <div className="p-6 text-center text-sm text-text/60">
          No expert utilization data available yet
        </div>
      );
    }

    const { steps, expert_utilization, num_experts, expert_stats, summary: utilSummary } = expertUtilization;

    // Normalize utilization for visualization
    const allValues = expert_utilization.flat();
    const maxUtil = Math.max(...allValues);
    const minUtil = Math.min(...allValues);

    return (
      <div className="space-y-4">
        {/* Summary stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="text-xs text-text/60 mb-1">Number of Experts</div>
            <div className="text-2xl font-bold">{num_experts}</div>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="text-xs text-text/60 mb-1">Total Steps</div>
            <div className="text-2xl font-bold">{utilSummary.total_steps}</div>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="text-xs text-text/60 mb-1">Load Balance Score</div>
            <div className="text-2xl font-bold">{(utilSummary.load_balance_score * 100).toFixed(1)}%</div>
            <div className="text-xs text-text/60 mt-1">
              {utilSummary.load_balance_score > 0.8 ? '✓ Well balanced' : '⚠️ Consider rebalancing'}
            </div>
          </div>
        </div>

        {/* Heatmap */}
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-sm font-semibold mb-3">Expert Utilization Over Time</div>
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              {/* Y-axis labels (experts) */}
              <div className="flex">
                <div className="w-20 flex-shrink-0">
                  <div className="h-6"></div>
                  {Array.from({ length: num_experts }, (_, i) => (
                    <div key={i} className="h-6 flex items-center justify-end pr-2 text-xs text-text/60">
                      Expert {i}
                    </div>
                  ))}
                </div>

                {/* Heatmap cells */}
                <div className="flex-1 overflow-x-auto">
                  <div className="flex flex-col">
                    {/* X-axis labels (steps) - show every Nth step */}
                    <div className="flex h-6 mb-1">
                      {steps.filter((_, idx) => idx % Math.max(1, Math.floor(steps.length / 20)) === 0).map((step, idx) => (
                        <div key={idx} className="text-xs text-text/60 px-1" style={{ width: '40px' }}>
                          {step}
                        </div>
                      ))}
                    </div>

                    {/* Heatmap rows */}
                    {Array.from({ length: num_experts }, (_, expertIdx) => (
                      <div key={expertIdx} className="flex h-6">
                        {expert_utilization.map((stepUtil, stepIdx) => {
                          const value = stepUtil[expertIdx] || 0;
                          const normalized = (value - minUtil) / (maxUtil - minUtil || 1);
                          const intensity = Math.floor(normalized * 255);
                          const color = `rgb(${255 - intensity}, ${100 + intensity * 0.6}, ${255 - intensity})`;

                          return (
                            <div
                              key={stepIdx}
                              className="border border-border/20"
                              style={{
                                width: '4px',
                                minWidth: '4px',
                                backgroundColor: color,
                              }}
                              title={`Step ${steps[stepIdx]}, Expert ${expertIdx}: ${value.toFixed(1)} tokens`}
                            ></div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Color scale legend */}
          <div className="mt-4 flex items-center gap-2 text-xs">
            <span className="text-text/60">Utilization:</span>
            <div className="flex-1 h-4 rounded-full overflow-hidden flex">
              {Array.from({ length: 20 }, (_, i) => {
                const normalized = i / 19;
                const intensity = Math.floor(normalized * 255);
                const color = `rgb(${255 - intensity}, ${100 + intensity * 0.6}, ${255 - intensity})`;
                return (
                  <div key={i} style={{ backgroundColor: color, flex: 1 }}></div>
                );
              })}
            </div>
            <span className="text-text/60">{minUtil.toFixed(0)}</span>
            <span className="text-text/60 ml-2">{maxUtil.toFixed(0)}</span>
          </div>
        </div>

        {/* Per-expert statistics */}
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="text-sm font-semibold mb-3">Expert Statistics</div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {expert_stats.map((stat) => (
              <div key={stat.expert_id} className="border border-border rounded p-3">
                <div className="text-xs font-semibold mb-2">Expert {stat.expert_id}</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-text/60">Mean:</span>
                    <span className="font-mono">{stat.mean_utilization.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text/60">Std:</span>
                    <span className="font-mono">{stat.std_utilization.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text/60">Range:</span>
                    <span className="font-mono">{stat.min_utilization.toFixed(0)}-{stat.max_utilization.toFixed(0)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Header with view mode selector */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Gating Metrics</h2>
          <p className="text-sm text-text/60">Monitor expert utilization and routing efficiency</p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('overview')}
            className={`px-3 py-1.5 text-sm rounded transition-colors ${
              viewMode === 'overview'
                ? 'bg-primary text-white'
                : 'bg-card border border-border hover:bg-border/50'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setViewMode('heatmap')}
            className={`px-3 py-1.5 text-sm rounded transition-colors ${
              viewMode === 'heatmap'
                ? 'bg-primary text-white'
                : 'bg-card border border-border hover:bg-border/50'
            }`}
            disabled={!expertUtilization}
          >
            Heatmap
          </button>
        </div>
      </div>

      {/* Content */}
      <div>
        {viewMode === 'overview' && renderOverview()}
        {viewMode === 'heatmap' && renderHeatmap()}
      </div>

      {/* Last updated */}
      {metrics.last_updated && (
        <div className="text-xs text-text/60 text-right">
          Last updated: {new Date(metrics.last_updated).toLocaleString()}
        </div>
      )}
    </div>
  );
}
