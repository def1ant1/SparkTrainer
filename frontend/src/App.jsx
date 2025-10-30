import React, { useState, useEffect } from 'react';
import JobWizard from './components/JobWizard';
import ArchitectureBuilder from './components/ArchitectureBuilder';
import HPOViewer from './components/HPOViewer';
import CommandPalette from './components/CommandPalette';
import QuickStart from './components/QuickStart';
import { Activity, Cpu, Database, Plus, List, Settings, Sun, Moon, Monitor, LayoutDashboard, BarChart3, ChevronLeft, ChevronRight, User, Search } from 'lucide-react';
import { ToastProvider, Button, Input, Modal, useToast, PageTransition } from './components/ui';
import { ModelsPage, ModelDetail, ModelCompare } from './components/Models';
import { DatasetsPage } from './components/Datasets';
import ExperimentsPage from './components/Experiments';
import PipelinesPage from './components/Pipelines';
import Documentation from './components/Documentation';
const ProfilePage = React.lazy(() => import('./components/Profile'));
import Labeling from './components/Labeling';

// API Service
// Use Vite env var if provided (e.g., VITE_API_URL=http://backend:5000), otherwise rely on relative '/api'.
const API_BASE = (import.meta.env && import.meta.env.VITE_API_URL)
  ? `${String(import.meta.env.VITE_API_URL).replace(/\/$/, '')}/api`
  : '/api';

// Helper: robust JSON parsing that tolerates empty bodies
const jsonOrEmpty = async (res) => {
  const text = await res.text();
  if (!res.ok) {
    // Surface server-provided error text if available
    throw new Error(text || res.statusText);
  }
  if (!text) return {};
  try { return JSON.parse(text); } catch (e) { throw new Error('Invalid JSON response'); }
};

const api = {
  getSystemInfo: () => fetch(`${API_BASE}/system/info`).then(jsonOrEmpty),
  getPartitions: () => fetch(`${API_BASE}/gpu/partitions`).then(jsonOrEmpty),
  getPartitionConfig: () => fetch(`${API_BASE}/gpu/partition/config`).then(jsonOrEmpty),
  applyPartitionConfig: (payload) => fetch(`${API_BASE}/gpu/partition/apply`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then(jsonOrEmpty),
  getMetricsHistory: () => fetch(`${API_BASE}/system/metrics/history`).then(jsonOrEmpty),
  getFrameworks: () => fetch(`${API_BASE}/frameworks`).then(jsonOrEmpty),
  getJobs: () => fetch(`${API_BASE}/jobs`).then(jsonOrEmpty),
  getJob: (id) => fetch(`${API_BASE}/jobs/${id}`).then(jsonOrEmpty),
  createJob: (data) => fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  }).then(jsonOrEmpty),
  cancelJob: (id) => fetch(`${API_BASE}/jobs/${id}/cancel`, {
    method: 'POST'
  }).then(jsonOrEmpty),
  getModels: () => fetch(`${API_BASE}/models`).then(jsonOrEmpty),
  getModelsRaw: (query) => fetch(`${API_BASE}/models${query?`?${query}`:''}`).then(jsonOrEmpty),
  getModel: (id) => fetch(`${API_BASE}/models/${id}`).then(jsonOrEmpty),
  updateModelMetadata: (id, payload) => fetch(`${API_BASE}/models/${id}/metadata`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(jsonOrEmpty),
  updateModelCard: (id, payload) => fetch(`${API_BASE}/models/${id}/card`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(jsonOrEmpty),
  bulkDeleteModels: (ids) => fetch(`${API_BASE}/models/bulk_delete`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ids})}).then(jsonOrEmpty),
  exportModelsUrl: (ids) => `${API_BASE}/models/export?ids=${ids.join(',')}`,
  saveModel: (payload) => fetch(`${API_BASE}/models/save`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(jsonOrEmpty)
};
// Extended API for wizard
api.validateJob = (payload) => fetch(`${API_BASE}/jobs/validate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then(async r => r.json());
// Datasets API
api.getDatasets = () => fetch(`${API_BASE}/datasets`).then(r => r.json());
api.getDataset = (name) => fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}`).then(r => r.json());
api.uploadDataset = async (name, file, version) => {
  const fd = new FormData();
  fd.append('name', name);
  if (version) fd.append('version', version);
  fd.append('file', file);
  const res = await fetch(`${API_BASE}/datasets/upload`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};
api.syncDatasets = (payload) => fetch(`${API_BASE}/datasets/sync`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r => r.json());
api.updateDatasetMetadata = (name, payload) => fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/metadata`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) }).then(r => r.json());
api.getDatasetSamples = (name, params) => {
  const qs = new URLSearchParams(Object.entries(params||{})).toString();
  return fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/samples${qs?`?${qs}`:''}`).then(r=>r.json());
};
// Large-file streaming ingestion
api.ingestStreamStart = (payload) => fetch(`${API_BASE}/datasets/ingest/stream_start`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
api.ingestStreamChunk = async (formData) => {
  const res = await fetch(`${API_BASE}/datasets/ingest/stream_chunk`, { method:'POST', body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};
api.ingestStreamFinalize = (payload) => fetch(`${API_BASE}/datasets/ingest/stream_finalize`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
// Annotations export/queue/prelabel
api.exportYolo = (name, params) => {
  const qs = new URLSearchParams(Object.entries(params||{})).toString();
  return fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/annotations/export/yolo${qs?`?${qs}`:''}`);
};
api.exportCoco = (name, params) => {
  const qs = new URLSearchParams(Object.entries(params||{})).toString();
  return fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/annotations/export/coco${qs?`?${qs}`:''}`).then(r=>r.json());
};
api.queueAnnotations = (name, params) => {
  const qs = new URLSearchParams(Object.entries(params||{})).toString();
  return fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/annotations/queue${qs?`?${qs}`:''}`).then(r=>r.json());
};
api.prelabel = (name, payload) => fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/annotations/prelabel`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
api.applyQuality = (name, payload) => fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/quality/apply`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
// HPO studies persistence
api.listHpoStudies = () => fetch(`${API_BASE}/hpo/studies`).then(r=>r.json());
api.saveHpoStudy = (study) => fetch(`${API_BASE}/hpo/studies/save`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(study)}).then(r=>r.json());
api.createDatasetTemplate = (payload) => fetch(`${API_BASE}/datasets/template`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
api.createDatasetVersion = (name, payload) => fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/version/create`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
api.diffDatasetVersions = (name, payload) => fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/version/diff`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
api.rollbackDatasetVersion = (name, payload) => fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/version/rollback`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
api.ingestPreview = (payload) => fetch(`${API_BASE}/datasets/ingest/preview`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
api.ingestApply = (payload) => fetch(`${API_BASE}/datasets/ingest/apply`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
api.getAnnotations = (name, params) => {
  const qs = new URLSearchParams(Object.entries(params||{})).toString();
  return fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/annotations${qs?`?${qs}`:''}`).then(r=>r.json());
};
api.saveAnnotations = (name, payload) => fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/annotations/save`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).then(r=>r.json());
// Adapters API
api.listModelAdapters = (id) => fetch(`${API_BASE}/models/${id}/adapters`).then(r => r.json());
api.mergeModelAdapter = (id, name) => fetch(`${API_BASE}/models/${id}/adapters/merge`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name })}).then(async r => { const t = await r.text(); try { return JSON.parse(t) } catch { return { error: t } } });

// Dashboard Component
const Dashboard = ({ onNavigate, systemInfo, partitions, metrics }) => {
  const [expanded, setExpanded] = useState({});
  const toggle = (idx) => setExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }));
  const colorFor = (pct) => pct < 70 ? 'bg-green-500' : pct < 85 ? 'bg-yellow-500' : 'bg-red-500';
  const fmtGiB = (mib) => (mib != null ? (mib / 1024).toFixed(1) : '0.0');
  const fmtPct = (n) => (n != null ? Math.round(n) : 0);
  const fmtRate = (bps) => {
    if (bps == null) return '0 B/s';
    const K = 1024, M = K*1024, G = M*1024;
    if (bps >= G) return (bps/G).toFixed(2) + ' GB/s';
    if (bps >= M) return (bps/M).toFixed(2) + ' MB/s';
    if (bps >= K) return (bps/K).toFixed(2) + ' KB/s';
    return bps.toFixed(0) + ' B/s';
  };
  const [cpuExpanded, setCpuExpanded] = useState(false);
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">DGX AI Trainer Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text/70">GPUs Available</p>
              <p className="text-2xl font-bold text-blue-600">{systemInfo.gpus?.length || 0}</p>
            </div>
            <Cpu className="text-blue-500" size={32} />
          </div>
        </div>
        

        
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text/70">Running Jobs</p>
              <p className="text-2xl font-bold text-green-600">{systemInfo.jobs_running || 0}</p>
            </div>
            <Activity className="text-green-500" size={32} />
          </div>
        </div>
        
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text/70">Queued Jobs</p>
              <p className="text-2xl font-bold text-yellow-600">{systemInfo.jobs_queued || 0}</p>
            </div>
            <List className="text-yellow-500" size={32} />
          </div>
        </div>
        
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text/70">Saved Models</p>
              <p className="text-2xl font-bold text-purple-600">{systemInfo.models_available || 0}</p>
            </div>
            <Database className="text-purple-500" size={32} />
          </div>
        </div>

        {/* GPU Memory Summary */}
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
          <div className="flex items-center justify-between">
            <div className="flex-1 mr-4">
              <p className="text-sm text-text/70">GPU Memory</p>
              <div className="mt-2">
                {(() => {
                  const total = systemInfo.memory_total_mib ?? 0;
                  const used = systemInfo.memory_used_mib ?? 0;
                  const pct = systemInfo.memory_used_pct ?? (total ? (used/total)*100 : 0);
                  return (
                    <>
                      <div className="w-full bg-muted rounded h-3">
                        <div className={`${colorFor(pct)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
                      </div>
                      <div className="text-xs text-text/70 mt-1">{fmtGiB(used)} / {fmtGiB(total)} GiB</div>
                    </>
                  );
                })()}
              </div>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold">{Math.round(systemInfo.memory_used_pct ?? 0)}%</p>
            </div>
          </div>
        </div>

        {/* System RAM Summary */}
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
          <div className="flex items-center justify-between">
            <div className="flex-1 mr-4">
              <p className="text-sm text-text/70">System RAM</p>
              <div className="mt-2">
                {(() => {
                  const total = systemInfo.memory?.total_mib ?? 0;
                  const used = systemInfo.memory?.used_mib ?? 0;
                  const pct = systemInfo.memory?.used_pct ?? (total ? (used/total)*100 : 0);
                  return (
                    <>
                      <div className="w-full bg-muted rounded h-3">
                        <div className={`${colorFor(pct)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
                      </div>
                      <div className="text-xs text-text/70 mt-1">{fmtGiB(used)} / {fmtGiB(total)} GiB</div>
                    </>
                  );
                })()}
              </div>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold">{Math.round(systemInfo.memory?.used_pct ?? 0)}%</p>
            </div>
          </div>
        </div>

        {/* GPU Allocations Summary */}
        <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text/70">GPU Allocations</p>
              {(() => {
                const gpus = partitions?.gpus || [];
                const gpuAlloc = gpus.filter(g => (g.allocated_by_jobs||[]).length>0).length;
                const migTotal = gpus.flatMap(g => (g.instances||[])).length;
                const migAlloc = gpus.flatMap(g => (g.instances||[])).filter(i => (i.allocated_by_jobs||[]).length>0).length;
                return (
                  <p className="text-2xl font-bold">{gpuAlloc} GPU, {migAlloc}/{migTotal} MIG</p>
                );
              })()}
            </div>
          </div>
        </div>
      </div>
      {/* Jobs filters belong to Jobs page; removed duplicate unintended filter row here */}

      <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
        <h2 className="text-xl font-semibold mb-4">GPU Status</h2>
        <div className="space-y-3">
          {systemInfo.gpus?.length > 0 ? (
            systemInfo.gpus.map((gpu, idx) => {
              const usedMiB = gpu.memory_used_mib ?? 0;
              const totalMiB = gpu.memory_total_mib ?? 0;
              const freeMiB = gpu.memory_free_mib ?? Math.max(totalMiB - usedMiB, 0);
              const pct = gpu.memory_used_pct ?? (totalMiB ? Math.round((usedMiB / totalMiB) * 100) : 0);
              const partGpu = (partitions?.gpus || []).find(x => x.index === (gpu.index ?? idx));
              const allocCount = (partGpu?.allocated_by_jobs || []).length + (partGpu?.instances || []).reduce((acc, inst) => acc + ((inst.allocated_by_jobs||[]).length), 0);
              return (
                <div key={idx} className="p-4 bg-muted rounded-lg border border-border">
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="font-semibold">{gpu.name}</p>
                        <span className="px-2 py-0.5 text-xs rounded-full bg-primary/10 text-primary border border-primary/30">{allocCount} alloc</span>
                      </div>
                      <p className="text-sm text-text/70">GPU {gpu.index ?? idx}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-text/70">{fmtGiB(usedMiB)} / {fmtGiB(totalMiB)} GiB</p>
                      <p className="font-semibold">{Math.round(pct)}%</p>
                    </div>
                  </div>
                  <div className="mt-3">
                    <div className="w-full bg-muted rounded h-3">
                      <div
                        className={`${colorFor(pct)} h-3 rounded`}
                        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                      />
                    </div>
                  </div>
                    <div className="mt-3 text-right">
                    <button onClick={() => toggle(idx)} className="text-blue-600 hover:text-blue-800 text-sm">
                      {expanded[idx] ? 'Hide details' : 'Show details'}
                    </button>
                  </div>
                  {expanded[idx] && (
                    <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
                      <div className="p-3 bg-surface rounded border border-border"><p className="text-text/70">Used</p><p className="font-semibold">{fmtGiB(usedMiB)} GiB</p></div>
                      <div className="p-3 bg-surface rounded border border-border"><p className="text-text/70">Free</p><p className="font-semibold">{fmtGiB(freeMiB)} GiB</p></div>
                      <div className="p-3 bg-surface rounded border border-border"><p className="text-text/70">Total</p><p className="font-semibold">{fmtGiB(totalMiB)} GiB</p></div>
                      {gpu.utilization_gpu_pct != null && (
                        <div className="p-3 bg-surface rounded border border-border"><p className="text-text/70">Utilization</p><p className="font-semibold">{gpu.utilization_gpu_pct}%</p></div>
                      )}
                      {gpu.temperature_gpu_c != null && (
                        <div className="p-3 bg-surface rounded border border-border"><p className="text-text/70">Temperature</p><p className="font-semibold">{gpu.temperature_gpu_c} °C</p></div>
                      )}
                    </div>
                  )}
                  {expanded[idx] && (
                    <div className="mt-4">
                      <p className="text-sm text-text/70 mb-1">Active Allocations</p>
                      <div className="text-xs text-gray-700">
                        {(() => {
                          const g = (partitions?.gpus || []).find(x => x.index === (gpu.index ?? idx));
                          const jobs = g?.allocated_by_jobs || [];
                          return jobs.length ? (
                            <div className="flex flex-wrap gap-2">
                              {jobs.map((j,i) => (<span key={i} className="px-2 py-1 bg-blue-50 border border-blue-200 rounded">{j.job_name}</span>))}
                            </div>
                          ) : <span className="text-gray-500">None</span>;
                        })()}
                      </div>
                      {(() => {
                        const g = (partitions?.gpus || []).find(x => x.index === (gpu.index ?? idx));
                        if (!g || !g.instances?.length) return null;
                        return (
                          <div className="mt-3">
                            <p className="text-sm text-text/70 mb-1">MIG Instances</p>
                            <div className="space-y-2">
                              {g.instances.map((inst, k) => (
                                <div key={k} className="p-2 bg-surface rounded border border-border text-xs flex items-center justify-between">
                                  <div>
                                    <div className="font-semibold">{inst.profile} — Device {inst.device_id}</div>
                                    <div className="text-gray-500">{inst.uuid?.slice(0,22)}...</div>
                                  </div>
                                  <div>
                                    {(inst.allocated_by_jobs||[]).length ? (
                                      <div className="flex gap-2 flex-wrap">
                                        {inst.allocated_by_jobs.map((j,i) => (<span key={i} className="px-2 py-1 bg-green-50 border border-green-200 rounded">{j.job_name}</span>))}
                                      </div>
                                    ) : <span className="text-gray-500">Free</span>}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <p className="text-gray-500">No GPU information available</p>
          )}
        </div>
      </div>

      {/* CPU Overview */}
      <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">CPU</h2>
          <button onClick={() => setCpuExpanded(v => !v)} className="text-blue-600 hover:text-blue-800 text-sm">
            {cpuExpanded ? 'Hide details' : 'Show details'}
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="p-4 bg-muted rounded border border-border">
            <p className="text-sm text-text/70">Total Usage</p>
            <div className="flex items-center justify-between mt-1">
              <div className="flex-1 mr-3">
                <div className="w-full bg-muted rounded h-3">
                  <div className={`${colorFor(systemInfo.cpu?.total_pct ?? 0)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, systemInfo.cpu?.total_pct ?? 0))}%` }} />
                </div>
              </div>
              <div className="text-sm font-semibold">{fmtPct(systemInfo.cpu?.total_pct)}%</div>
            </div>
          </div>
          <div className="p-4 bg-muted rounded border border-border">
            <p className="text-sm text-text/70">Cores</p>
            <p className="font-semibold mt-1">{systemInfo.cpu?.count ?? '-'}</p>
          </div>
          <div className="p-4 bg-muted rounded border border-border">
            <p className="text-sm text-text/70">Load Avg (1,5,15)</p>
            <p className="font-semibold mt-1">{systemInfo.cpu?.load_avg ? systemInfo.cpu.load_avg.map(x=>x.toFixed(2)).join(' / ') : '-'}</p>
          </div>
        </div>
        {cpuExpanded && Array.isArray(systemInfo.cpu?.per_core_pct) && systemInfo.cpu.per_core_pct.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2">
            {systemInfo.cpu.per_core_pct.map((v, i) => (
              <div key={i} className="p-2 bg-surface rounded border border-border">
                <div className="text-xs text-text/70 mb-1">CPU {i}</div>
                <div className="w-full bg-muted rounded h-2">
                  <div className={`${colorFor(v)} h-2 rounded`} style={{ width: `${Math.min(100, Math.max(0, v))}%` }} />
                </div>
                <div className="text-xs text-right mt-1">{fmtPct(v)}%</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Memory & Swap */}
      <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
        <h2 className="text-xl font-semibold mb-4">Memory</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-text/70">RAM</p>
            <div className="flex items-center justify-between mt-1">
              <div className="flex-1 mr-3">
                <div className="w-full bg-muted rounded h-3">
                  <div className={`${colorFor(systemInfo.memory?.used_pct ?? 0)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, systemInfo.memory?.used_pct ?? 0))}%` }} />
                </div>
              </div>
              <div className="text-sm font-semibold">{fmtPct(systemInfo.memory?.used_pct)}%</div>
            </div>
            <div className="text-xs text-text/70 mt-1">{fmtGiB(systemInfo.memory?.used_mib)} / {fmtGiB(systemInfo.memory?.total_mib)} GiB</div>
          </div>
          <div>
            <p className="text-sm text-text/70">Swap</p>
            <div className="flex items-center justify-between mt-1">
              <div className="flex-1 mr-3">
                <div className="w-full bg-muted rounded h-3">
                  <div className={`${colorFor(systemInfo.swap?.used_pct ?? 0)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, systemInfo.swap?.used_pct ?? 0))}%` }} />
                </div>
              </div>
              <div className="text-sm font-semibold">{fmtPct(systemInfo.swap?.used_pct)}%</div>
            </div>
            <div className="text-xs text-text/70 mt-1">{fmtGiB(systemInfo.swap?.used_mib)} / {fmtGiB(systemInfo.swap?.total_mib)} GiB</div>
          </div>
        </div>
      </div>

      {/* Network & Disks */}
      <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
        <h2 className="text-xl font-semibold mb-4">I/O</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 bg-muted rounded border border-border">
            <p className="text-sm text-text/70">Network</p>
            <div className="mt-2 text-sm">
              <div><span className="text-text/70">Receive:</span> <span className="font-semibold">{fmtRate(systemInfo.net?.rx_rate_bps)}</span></div>
              <div><span className="text-text/70">Transmit:</span> <span className="font-semibold">{fmtRate(systemInfo.net?.tx_rate_bps)}</span></div>
            </div>
          </div>
          <div className="p-4 bg-muted rounded border border-border">
            <p className="text-sm text-text/70 mb-2">Disks</p>
            <div className="space-y-2">
              {(systemInfo.disks || []).map((d, i) => (
                <div key={i}>
                  <div className="flex justify-between text-xs text-text/70"><span>{d.path}</span><span>{d.used_gib} / {d.total_gib} GiB</span></div>
                  <div className="w-full bg-muted rounded h-2">
                    <div className={`${colorFor(d.used_pct ?? 0)} h-2 rounded`} style={{ width: `${Math.min(100, Math.max(0, d.used_pct ?? 0))}%` }} />
                  </div>
                </div>
              ))}
              {(!systemInfo.disks || systemInfo.disks.length === 0) && (
                <div className="text-xs text-text/60">No disk info</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Timelines */}
      <div className="bg-surface p-6 rounded-lg shadow-md border border-border">
        <h2 className="text-xl font-semibold mb-4">Resource Utilization Timeline (Last {Math.round((metrics?.window_seconds||3600)/60)} min)</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-text/70 mb-2">GPU Utilization (%)</p>
            <LineChart data={metrics?.samples||[]} seriesFrom={(s)=>{
              const map = {};
              (s[0]?.gpus||[]).forEach(g => { map[`gpu-${g.index}`] = {label:`GPU ${g.index}`, color: colorFor((g.index||0)*10).replace('bg-','text-')}; });
              return map;
            }} yKeyFor={(s, key)=>{
              const idx = parseInt(String(key).split('-')[1]||'0');
              const g = (s.gpus||[]).find(x => x.index===idx);
              return g?.util_pct ?? 0;
            }} />
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm text-text/70">Memory Usage (%)</p>
              <a href={`${API_BASE}/system/metrics/history.csv`} className="text-primary hover:brightness-110 text-sm">Export CSV</a>
            </div>
            <LineChart data={metrics?.samples||[]} seriesFrom={()=>({
              'gpu-mem': {label:'GPU Mem', color:'blue'},
              'sys-mem': {label:'System RAM', color:'green'}
            })} yKeyFor={(s,key)=> key==='gpu-mem' ? (s.gpu_mem_used_pct ?? 0) : (s.sys_mem_used_pct ?? 0)} />
          </div>
        </div>
        <div className="mt-6">
          <p className="text-sm text-text/70 mb-2">Active Jobs</p>
          <JobsTimeline metrics={metrics} partitions={partitions} />
        </div>
      </div>

      <div className="flex gap-4">
        <Button variant="primary" className="flex-1 py-4" onClick={()=>onNavigate('create')} leftIcon={<Plus size={20} />}>Create New Training Job</Button>
        <Button variant="ghost" className="flex-1 py-4" onClick={()=>onNavigate('jobs')} leftIcon={<List size={20} />}>View All Jobs</Button>
      </div>
    </div>
  );
};

// Create Job Component
const CreateJob = ({ onNavigate, frameworks, partitions }) => {
  const [jobType, setJobType] = useState('train');
  const [framework, setFramework] = useState('pytorch');
  const [jobName, setJobName] = useState('');
  const [gpuMode, setGpuMode] = useState('auto'); // 'auto' | 'select'
  const [gpuSelection, setGpuSelection] = useState('');
  const [gpuAutoPrefer, setGpuAutoPrefer] = useState('mig_first'); // 'mig_first' | 'gpu_first'
  const [datasets, setDatasets] = useState([]);
  const [datasetMode, setDatasetMode] = useState('none'); // none | registry
  const [datasetId, setDatasetId] = useState('');
  const [datasetVersion, setDatasetVersion] = useState('');

  // Default to first free MIG, else first GPU when switching to select mode or when partitions update
  useEffect(() => {
    if (gpuMode !== 'select') return;
    if (gpuSelection) return;
    const gpus = partitions?.gpus || [];
    let pick = null;
    for (const g of gpus) {
      for (const inst of (g.instances||[])) {
        if (!(inst.allocated_by_jobs||[]).length) {
          pick = JSON.stringify({type:'mig', gpu_index:g.index, gpu_uuid:g.uuid, mig_uuid:inst.uuid});
          break;
        }
      }
      if (pick) break;
    }
    if (!pick && gpus.length) {
      pick = JSON.stringify({type:'gpu', gpu_index:gpus[0].index, gpu_uuid:gpus[0].uuid});
    }
    if (pick) setGpuSelection(pick);
  }, [gpuMode, partitions]);
  useEffect(() => { (async()=>{ try{ setDatasets(await api.getDatasets()); } catch{} })(); }, []);
  const [config, setConfig] = useState({
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    architecture: 'custom',
    input_size: 784,
    output_size: 10,
    hidden_layers: [512, 256, 128],
    num_classes: 10
  });
  const [cfgErrors, setCfgErrors] = useState({});
  
  const handleSubmit = async () => {
    const errs = {};
    if (!Number.isFinite(config.epochs) || config.epochs <= 0) errs.epochs = 'Epochs must be greater than 0';
    if (!Number.isFinite(config.batch_size) || config.batch_size <= 0) errs.batch_size = 'Batch size must be greater than 0';
    if (!Number.isFinite(config.learning_rate) || config.learning_rate <= 0) errs.learning_rate = 'Learning rate must be greater than 0';
    if (!Number.isFinite(config.num_classes) || config.num_classes <= 0) errs.num_classes = 'Number of classes must be greater than 0';
    setCfgErrors(errs);
    if (Object.keys(errs).length) { alert('Please fix validation errors'); return; }
    const jobData = {
      name: jobName || `Training Job ${Date.now()}`,
      type: jobType,
      framework: framework,
      config: config
    };
    if (datasetMode === 'registry' && datasetId) {
      jobData.config = { ...jobData.config, dataset_id: datasetId, dataset_version: datasetVersion || undefined, data: { ...(jobData.config.data||{}), dataset_id: datasetId, dataset_version: datasetVersion || undefined } };
    }
    if (gpuMode === 'auto') {
      jobData.gpu_prefer = gpuAutoPrefer;
    }
    if (gpuMode === 'select' && gpuSelection) {
      try {
        jobData.gpu = JSON.parse(gpuSelection);
      } catch {}
      // Validate MIG availability (prevent selecting already allocated instance)
      if (jobData.gpu?.type === 'mig' && jobData.gpu?.mig_uuid) {
        const found = (partitions?.gpus || []).flatMap(g => (g.instances||[])).find(inst => inst.uuid === jobData.gpu.mig_uuid);
        if (found && (found.allocated_by_jobs||[]).length > 0) {
          alert('Selected MIG instance is already allocated to another job. Please choose a free instance.');
          return;
        }
      }
    }
    
    try {
      const result = await api.createJob(jobData);
      alert(`Job created successfully! ID: ${result.id}`);
      onNavigate('jobs');
    } catch (error) {
      alert('Failed to create job: ' + error.message);
    }
  };
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Create Training Job</h1>
        <Button variant="ghost" onClick={()=>onNavigate('dashboard')}>← Back to Dashboard</Button>
      </div>
      
      <div className="bg-surface p-6 rounded-lg shadow-md border border-border space-y-6">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Compute Resource</label>
          <div className="flex gap-3 mb-3">
            <label className="inline-flex items-center gap-2">
              <input type="radio" name="gpuMode" value="auto" checked={gpuMode==='auto'} onChange={()=>setGpuMode('auto')} />
              <span>Auto</span>
            </label>
            <label className="inline-flex items-center gap-2">
              <input type="radio" name="gpuMode" value="select" checked={gpuMode==='select'} onChange={()=>setGpuMode('select')} />
              <span>Select GPU/MIG</span>
            </label>
          </div>
          {gpuMode==='auto' && (
            <div className="flex gap-3 mb-2">
              <label className="inline-flex items-center gap-2">
                <input type="radio" name="gpuPref" checked={gpuAutoPrefer==='mig_first'} onChange={()=>setGpuAutoPrefer('mig_first')} />
                <span>Prefer MIG first</span>
              </label>
              <label className="inline-flex items-center gap-2">
                <input type="radio" name="gpuPref" checked={gpuAutoPrefer==='gpu_first'} onChange={()=>setGpuAutoPrefer('gpu_first')} />
                <span>Prefer whole GPU first</span>
              </label>
              <span className="text-xs text-gray-500">(You can choose MIG/GPU order. Applied on submit.)</span>
            </div>
          )}
          {gpuMode==='select' && (
            <select value={gpuSelection} onChange={e=>setGpuSelection(e.target.value)} className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
              <option value="">Choose...</option>
              {(partitions?.gpus || []).map((g) => (
                <option key={`g-${g.index}`} value={JSON.stringify({type:'gpu', gpu_index:g.index, gpu_uuid:g.uuid})}>{`GPU ${g.index} — ${g.name}`}</option>
              ))}
              {(partitions?.gpus || []).flatMap(g => (g.instances||[]).map((inst, k) => (
                <option key={`m-${g.index}-${k}`} disabled={(inst.allocated_by_jobs||[]).length>0} value={JSON.stringify({type:'mig', gpu_index:g.index, gpu_uuid:g.uuid, mig_uuid:inst.uuid})}>{`GPU ${g.index} — ${inst.profile} (MIG ${inst.device_id})${(inst.allocated_by_jobs||[]).length>0 ? ' — Allocated' : ''}`}</option>
              )))}
            </select>
          )}
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Dataset</label>
          <div className="flex gap-3 mb-2">
            <label className="inline-flex items-center gap-2"><input type="radio" name="dsMode" value="none" checked={datasetMode==='none'} onChange={()=>setDatasetMode('none')} /> None / Manual</label>
            <label className="inline-flex items-center gap-2"><input type="radio" name="dsMode" value="registry" checked={datasetMode==='registry'} onChange={()=>setDatasetMode('registry')} /> From Registry</label>
          </div>
          {datasetMode==='registry' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <select className="w-full border rounded px-3 py-2" value={datasetId} onChange={e=>{setDatasetId(e.target.value); setDatasetVersion('');}}>
                <option value="">Select dataset</option>
                {datasets.map(d => <option key={d.name} value={d.name}>{d.name} {d.latest?`(latest ${d.latest})`:''}</option>)}
              </select>
              <input className="w-full border rounded px-3 py-2" placeholder="Version (optional)" value={datasetVersion} onChange={e=>setDatasetVersion(e.target.value)} />
          <div className="text-xs text-text/60 flex items-center">If empty, latest version is used.</div>
            </div>
          )}
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Job Name</label>
          <input
            type="text"
            value={jobName}
            onChange={(e) => setJobName(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="My Training Job"
          />
        </div>
        
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Job Type</label>
          <div className="flex gap-4">
            <button
              onClick={() => setJobType('train')}
              className={`flex-1 py-3 px-4 rounded-lg border-2 transition ${
                jobType === 'train'
                  ? 'border-blue-600 bg-blue-50 text-blue-700'
                  : 'border-border bg-surface hover:bg-muted'
              }`}
            >
              <div className="font-semibold">Train from Scratch</div>
              <div className="text-xs mt-1">Create a new model</div>
            </button>
            <button
              onClick={() => setJobType('finetune')}
              className={`flex-1 py-3 px-4 rounded-lg border-2 transition ${
                jobType === 'finetune'
                  ? 'border-blue-600 bg-blue-50 text-blue-700'
                  : 'border-border bg-surface hover:bg-muted'
              }`}
            >
              <div className="font-semibold">Fine-tune</div>
              <div className="text-xs mt-1">Fine-tune existing model</div>
            </button>
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Framework</label>
          <select
            value={framework}
            onChange={(e) => setFramework(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            {Object.entries(frameworks).map(([key, fw]) => (
              <option key={key} value={key}>{fw.name}</option>
            ))}
          </select>
        </div>
        
        {jobType === 'finetune' && (
          <Input label="Base Model" placeholder="resnet18, bert-base-uncased, etc." value={config.model_name || ''} onChange={(e)=>setConfig({...config, model_name: e.target.value})} />
        )}
        
        <div className="grid grid-cols-2 gap-4">
          <Input label="Epochs" type="number" value={config.epochs} onChange={(e)=>setConfig({...config, epochs: parseInt(e.target.value)})} error={!!cfgErrors.epochs} helperText={cfgErrors.epochs} />
          
          <Input label="Batch Size" type="number" value={config.batch_size} onChange={(e)=>setConfig({...config, batch_size: parseInt(e.target.value)})} error={!!cfgErrors.batch_size} helperText={cfgErrors.batch_size} />
          
          <Input label="Learning Rate" type="number" step="0.0001" value={config.learning_rate} onChange={(e)=>setConfig({...config, learning_rate: parseFloat(e.target.value)})} error={!!cfgErrors.learning_rate} helperText={cfgErrors.learning_rate} />
          
          <Input label="Number of Classes" type="number" value={config.num_classes} onChange={(e)=>setConfig({...config, num_classes: parseInt(e.target.value)})} error={!!cfgErrors.num_classes} helperText={cfgErrors.num_classes} />
          <Input label="Project (optional)" value={config.project || ''} onChange={(e)=>setConfig({...config, project: e.target.value})} placeholder="experiment/project name" />
          <Input label="User (optional)" value={config.user || ''} onChange={(e)=>setConfig({...config, user: e.target.value})} placeholder="your-name" />
        </div>
        
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Experiment Tracking</label>
          <div className="flex gap-4 text-sm">
            <label className="inline-flex items-center gap-2"><input type="checkbox" checked={!!(config.tracking && config.tracking.wandb)} onChange={e=> setConfig({...config, tracking: { ...(config.tracking||{}), wandb: e.target.checked }}) }/> Weights & Biases</label>
            <label className="inline-flex items-center gap-2"><input type="checkbox" checked={!!(config.tracking && config.tracking.tensorboard)} onChange={e=> setConfig({...config, tracking: { ...(config.tracking||{}), tensorboard: e.target.checked }}) }/> TensorBoard</label>
          </div>
          <div className="text-xs text-gray-500 mt-1">Optional. W&B requires wandb installed + WANDB_API_KEY on server. TensorBoard requires tensorboard package.</div>
        </div>

        <Button className="w-full py-3" variant="primary" onClick={handleSubmit}>Create Training Job</Button>
      </div>
    </div>
  );
};

// Jobs List Component
const JobsList = ({ onNavigate, partitions, onOpenHpo }) => {
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const toast = useToast();
  const [checkpoints, setCheckpoints] = useState([]);
  const [cpLoading, setCpLoading] = useState(false);
  const [q, setQ] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [frameworkFilter, setFrameworkFilter] = useState('');
  const [userFilter, setUserFilter] = useState('');
  const [groupByProject, setGroupByProject] = useState(false);
  const [experimentFilter, setExperimentFilter] = useState('');
  const [groupByExperiment, setGroupByExperiment] = useState(false);
  const [selectedJobs, setSelectedJobs] = useState({});
  
  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 5000);
    return () => clearInterval(interval);
  }, []);
  
  const loadJobs = async () => {
    try {
      const data = await api.getJobs();
      setJobs(data);
    } catch (error) {
      console.error('Failed to load jobs:', error);
    }
  };
  
  const handleCancelJob = async (jobId) => {
    if (confirm('Are you sure you want to cancel this job?')) {
      try {
        await api.cancelJob(jobId);
        loadJobs();
        toast.push({ type:'success', title:'Job cancelled' });
      } catch (error) {
        toast.push({ type:'error', title:'Cancel failed', message: String(error.message||error) });
      }
    }
  };

  const handleDeleteJob = async (jobId) => {
    if (confirm('Delete this job permanently? This will remove it from the list but keep logs/checkpoints.')) {
      try {
        const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || 'Delete failed');
        }
        if (selectedJob && selectedJob.id === jobId) {
          setSelectedJob(null);
        }
        loadJobs();
        toast.push({ type: 'success', title: 'Job deleted' });
      } catch (error) {
        toast.push({ type: 'error', title: 'Delete failed', message: String(error.message || error) });
      }
    }
  };

  const handleRestartJob = async (jobId) => {
    if (confirm('Restart this job? It will be re-queued and run from scratch.')) {
      try {
        const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/restart`, { method: 'POST' });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || 'Restart failed');
        }
        loadJobs();
        toast.push({ type: 'success', title: 'Job restarted', message: 'Job has been re-queued' });
      } catch (error) {
        toast.push({ type: 'error', title: 'Restart failed', message: String(error.message || error) });
      }
    }
  };

  const selectedJobIds = Object.keys(selectedJobs).filter(k => selectedJobs[k]);

  const toggleAllJobs = (filtered) => {
    if (selectedJobIds.length === filtered.length && filtered.length > 0) {
      setSelectedJobs({});
    } else {
      setSelectedJobs(Object.fromEntries(filtered.map(j => [j.id, true])));
    }
  };

  const bulkCancelJobs = async () => {
    const jobsToCanel = selectedJobIds.filter(id => {
      const job = jobs.find(j => j.id === id);
      return job && job.status === 'running';
    });

    if (jobsToCanel.length === 0) {
      toast.push({ type: 'warning', title: 'No running jobs selected' });
      return;
    }

    if (!confirm(`Cancel ${jobsToCanel.length} running job${jobsToCanel.length > 1 ? 's' : ''}?`)) return;

    let success = 0, failed = 0;
    for (const jobId of jobsToCanel) {
      try {
        await api.cancelJob(jobId);
        success++;
      } catch {
        failed++;
      }
    }
    setSelectedJobs({});
    loadJobs();
    toast.push({
      type: success > 0 ? 'success' : 'error',
      title: `Cancelled ${success} job${success !== 1 ? 's' : ''}`,
      message: failed > 0 ? `${failed} failed` : undefined
    });
  };

  const bulkDeleteJobs = async () => {
    const jobsToDelete = selectedJobIds.filter(id => {
      const job = jobs.find(j => j.id === id);
      return job && ['completed', 'failed', 'cancelled'].includes(job.status);
    });

    if (jobsToDelete.length === 0) {
      toast.push({ type: 'warning', title: 'No deletable jobs selected', message: 'Only completed, failed, or cancelled jobs can be deleted' });
      return;
    }

    if (!confirm(`Delete ${jobsToDelete.length} job${jobsToDelete.length > 1 ? 's' : ''}? This cannot be undone.`)) return;

    let success = 0, failed = 0;
    for (const jobId of jobsToDelete) {
      try {
        const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
        if (res.ok) success++;
        else failed++;
      } catch {
        failed++;
      }
    }
    setSelectedJobs({});
    loadJobs();
    toast.push({
      type: success > 0 ? 'success' : 'error',
      title: `Deleted ${success} job${success !== 1 ? 's' : ''}`,
      message: failed > 0 ? `${failed} failed` : undefined
    });
  };

  useEffect(() => {
    if (!selectedJob) { setCheckpoints([]); return; }
    (async () => {
      try {
        setCpLoading(true);
        const res = await fetch(`${API_BASE}/jobs/${selectedJob.id}/checkpoints`);
        const data = await res.json();
        setCheckpoints(data.checkpoints || []);
      } catch (e) {
        setCheckpoints([]);
      } finally {
        setCpLoading(false);
      }
    })();
  }, [selectedJob]);
  
  const getStatusColor = (status) => {
    const colors = {
      'queued': 'bg-warning/10 text-warning border border-warning/30',
      'running': 'bg-primary/10 text-primary border border-primary/30',
      'completed': 'bg-success/10 text-success border border-success/30',
      'failed': 'bg-danger/10 text-danger border border-danger/30',
      'cancelled': 'bg-text/10 text-text/70 border border-text/20',
      'paused': 'bg-secondary/10 text-secondary border border-secondary/30',
      'blocked': 'bg-text/10 text-text/60 border border-text/20'
    };
    return colors[status] || 'bg-text/10 text-text/70 border border-text/20';
  };
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Training Jobs</h1>
        <div className="flex gap-2">
          {selectedJobIds.length > 0 && (
            <>
              <span className="px-3 py-2 bg-muted rounded border border-border text-sm">
                {selectedJobIds.length} selected
              </span>
              <button
                onClick={bulkCancelJobs}
                className="px-3 py-2 bg-warning/10 text-warning border border-warning/30 rounded hover:bg-warning/20"
              >
                Cancel Selected
              </button>
              <button
                onClick={bulkDeleteJobs}
                className="px-3 py-2 bg-danger/10 text-danger border border-danger/30 rounded hover:bg-danger/20"
              >
                Delete Selected
              </button>
            </>
          )}
          <button
            onClick={() => onNavigate('create')}
            className="bg-primary text-on-primary py-2 px-4 rounded-lg hover:brightness-110 transition flex items-center gap-2"
          >
            <Plus size={16} />
            New Job
          </button>
          <button
            onClick={() => onNavigate('dashboard')}
            className="text-primary hover:brightness-110"
          >
            ← Dashboard
          </button>
        </div>
      </div>
      <div className="bg-surface rounded-lg border border-border p-4 grid grid-cols-1 md:grid-cols-7 gap-3">
        <input className="border border-border rounded px-3 py-2 bg-surface text-text" placeholder="Search jobs" value={q} onChange={e=>setQ(e.target.value)} />
        <select className="border border-border rounded px-3 py-2 bg-surface text-text" value={statusFilter} onChange={e=>setStatusFilter(e.target.value)}>
          <option value="">All Statuses</option>
          {['queued','blocked','running','paused','completed','failed','cancelled'].map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <select className="border border-border rounded px-3 py-2 bg-surface text-text" value={frameworkFilter} onChange={e=>setFrameworkFilter(e.target.value)}>
          <option value="">All Frameworks</option>
          {['pytorch','huggingface','tensorflow'].map(f => <option key={f} value={f}>{f}</option>)}
        </select>
        <input className="border border-border rounded px-3 py-2 bg-surface text-text" placeholder="Filter by user" value={userFilter} onChange={e=>setUserFilter(e.target.value)} />
        <input className="border border-border rounded px-3 py-2 bg-surface text-text" placeholder="Filter by experiment" value={experimentFilter} onChange={e=>setExperimentFilter(e.target.value)} />
        <div className="flex items-center gap-4 text-sm text-text col-span-2">
          <label className="flex items-center gap-2"><input type="checkbox" checked={groupByProject} onChange={e=>setGroupByProject(e.target.checked)} /> Group by project</label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={groupByExperiment} onChange={e=>setGroupByExperiment(e.target.checked)} /> Group by experiment</label>
        </div>
      </div>

      <div className="bg-surface rounded-lg shadow-md border border-border overflow-hidden">
        <table className="w-full">
          <thead className="bg-muted border-b border-border">
            <tr>
              <th className="px-3 py-3 text-left">
                <input
                  type="checkbox"
                  checked={selectedJobIds.length > 0 && jobs.filter(j =>
                    (!q || j.name.toLowerCase().includes(q.toLowerCase())) &&
                    (!statusFilter || j.status===statusFilter) &&
                    (!frameworkFilter || j.framework===frameworkFilter) &&
                    (!userFilter || ((j.config && j.config.user && String(j.config.user).toLowerCase().includes(userFilter.toLowerCase())) || (j.user && String(j.user).toLowerCase().includes(userFilter.toLowerCase())))) &&
                    (!experimentFilter || ((j.experiment && j.experiment.name && String(j.experiment.name).toLowerCase().includes(experimentFilter.toLowerCase()))))
                  ).every(j => selectedJobs[j.id])}
                  onChange={(e) => toggleAllJobs(jobs.filter(j =>
                    (!q || j.name.toLowerCase().includes(q.toLowerCase())) &&
                    (!statusFilter || j.status===statusFilter) &&
                    (!frameworkFilter || j.framework===frameworkFilter) &&
                    (!userFilter || ((j.config && j.config.user && String(j.config.user).toLowerCase().includes(userFilter.toLowerCase())) || (j.user && String(j.user).toLowerCase().includes(userFilter.toLowerCase())))) &&
                    (!experimentFilter || ((j.experiment && j.experiment.name && String(j.experiment.name).toLowerCase().includes(experimentFilter.toLowerCase()))))
                  ))}
                  className="w-4 h-4"
                />
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase">Name</th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase">Type</th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase">Framework</th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase">GPU</th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase">Status</th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase">Experiment</th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase">Created</th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {(function(){
              const filtered = jobs.filter(j =>
                (!q || j.name.toLowerCase().includes(q.toLowerCase())) &&
                (!statusFilter || j.status===statusFilter) &&
                (!frameworkFilter || j.framework===frameworkFilter) &&
                (!userFilter || ((j.config && j.config.user && String(j.config.user).toLowerCase().includes(userFilter.toLowerCase())) || (j.user && String(j.user).toLowerCase().includes(userFilter.toLowerCase())))) &&
                (!experimentFilter || ((j.experiment && j.experiment.name && String(j.experiment.name).toLowerCase().includes(experimentFilter.toLowerCase()))))
              );
              if (!groupByProject && !groupByExperiment) {
                return filtered.map((job) => (
                   <tr key={job.id} className="hover:bg-muted">
                  <td className="px-3 py-4">
                    <input
                      type="checkbox"
                      checked={!!selectedJobs[job.id]}
                      onChange={(e) => setSelectedJobs(prev => ({ ...prev, [job.id]: e.target.checked }))}
                      className="w-4 h-4"
                    />
                  </td>
                  <td className="px-6 py-4 text-sm font-medium">{job.name}</td>
                  <td className="px-6 py-4 text-sm text-text/80 capitalize">{job.type}</td>
                  <td className="px-6 py-4 text-sm text-text/80 capitalize">{job.framework}</td>
                  <td className="px-6 py-4 text-sm text-text/80">
                    {(() => {
                      const g = job.gpu;
                      if (!g) return <span className="text-text/60">auto</span>;
                      if (g.type === 'gpu') {
                        if (g.gpu_index != null) return `GPU ${g.gpu_index}`;
                        if (g.gpu_uuid) return `GPU ${String(g.gpu_uuid).slice(0,8)}…`;
                        return 'GPU';
                      }
                      if (g.type === 'mig') {
                        // Try to map to profile+device id
                        const inst = (partitions?.gpus || []).flatMap(gg => (gg.instances||[])).find(i => i.uuid === g.mig_uuid);
                        if (inst) {
                          const parent = (partitions?.gpus || []).find(gg => (gg.instances||[]).some(i => i.uuid === g.mig_uuid));
                          return `MIG GPU ${parent?.index ?? ''} ${inst.profile} (dev ${inst.device_id})`;
                        }
                        return `MIG ${String(g.mig_uuid).slice(0,10)}…`;
                      }
                      return '-';
                    })()}
                  </td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(job.status)}`}>
                      {job.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-text/80">{job.experiment?.name ? (<span className="px-2 py-0.5 border rounded text-xs">{job.experiment.name}</span>) : '-'}</td>
                  <td className="px-6 py-4 text-sm text-text/80">
                    <div>{new Date(job.created).toLocaleString()}</div>
                    {job.progress!=null && <div className="text-xs text-text/70">{Math.round(job.progress)}%{job.eta_seconds!=null && ` • ETA ${Math.ceil(job.eta_seconds/60)}m`}</div>}
                  </td>
                      <td className="px-6 py-4 text-sm">
                        <button
                          onClick={() => setSelectedJob(job)}
                          className="text-primary hover:brightness-110 mr-3"
                        >
                          View
                        </button>
                        {job.framework==='huggingface' && job.config && job.config.hpo && job.config.hpo.enabled && (
                          <button onClick={()=> onOpenHpo && onOpenHpo(job.id)} className="text-secondary hover:brightness-110 mr-3">HPO</button>
                        )}
                    {job.status === 'running' && (
                      <>
                        <button onClick={async()=>{await fetch(`${API_BASE}/jobs/${job.id}/pause`,{method:'POST'}); loadJobs(); toast.push({type:'info', title:'Job paused'});}} className="text-warning hover:brightness-110 mr-3">Pause</button>
                        <button onClick={async()=>{await fetch(`${API_BASE}/jobs/${job.id}/resume`,{method:'POST'}); loadJobs(); toast.push({type:'success', title:'Job resumed'});}} className="text-success hover:brightness-110 mr-3">Resume</button>
                        <button onClick={() => handleCancelJob(job.id)} className="text-danger hover:brightness-110 mr-3">Cancel</button>
                      </>
                    )}
                    {(job.status === 'failed' || job.status === 'cancelled') && (
                      <button onClick={() => handleRestartJob(job.id)} className="text-success hover:brightness-110 mr-3">Restart</button>
                    )}
                    {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') && (
                      <button onClick={() => handleDeleteJob(job.id)} className="text-danger hover:brightness-110 mr-3">Delete</button>
                    )}
                    <button onClick={async()=>{await fetch(`${API_BASE}/jobs/${job.id}/clone`,{method:'POST'}); loadJobs(); toast.push({type:'success', title:'Job cloned'});}} className="text-text/70 hover:text-text mr-3">Clone</button>
                  </td>
                </tr>
              ));
              }
              // Group by project (config.project)
              const groups = {};
              filtered.forEach(j => {
                const key = groupByExperiment ? (j.experiment && j.experiment.name ? `Experiment: ${j.experiment.name}` : 'No experiment') : ((j.config && j.config.project) ? String(j.config.project) : 'No project');
                (groups[key] ||= []).push(j);
              });
              const keys = Object.keys(groups).sort();
              return keys.flatMap((k) => {
                const rows = [];
                rows.push(
                  <tr key={`g-${k}`} className="bg-muted">
                    <td colSpan={8} className="px-6 py-2 text-xs font-semibold text-text">{k} <span className="text-text/60 font-normal">({groups[k].length})</span></td>
                  </tr>
                );
                groups[k].forEach(job => {
                  rows.push(
                    <tr key={job.id} className="hover:bg-muted">
                      <td className="px-6 py-4 text-sm font-medium">{job.name}</td>
                      <td className="px-6 py-4 text-sm text-text/80 capitalize">{job.type}</td>
                      <td className="px-6 py-4 text-sm text-text/80 capitalize">{job.framework}</td>
                  <td className="px-6 py-4 text-sm text-text/80">
                        {(() => {
                          const g = job.gpu;
                          if (!g) return <span className="text-text/60">auto</span>;
                          if (g.type === 'gpu') {
                            if (g.gpu_index != null) return `GPU ${g.gpu_index}`;
                            if (g.gpu_uuid) return `GPU ${String(g.gpu_uuid).slice(0,8)}…`;
                            return 'GPU';
                          }
                          if (g.type === 'mig') {
                            const inst = (partitions?.gpus || []).flatMap(gg => (gg.instances||[])).find(i => i.uuid === g.mig_uuid);
                            if (inst) {
                              const parent = (partitions?.gpus || []).find(gg => (gg.instances||[]).some(i => i.uuid === g.mig_uuid));
                              return `MIG GPU ${parent?.index ?? ''} ${inst.profile} (dev ${inst.device_id})`;
                            }
                            return `MIG ${String(g.mig_uuid).slice(0,10)}…`;
                          }
                          return '-';
                        })()}
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-text/80">{job.experiment?.name ? (<span className="px-2 py-0.5 border rounded text-xs">{job.experiment.name}</span>) : '-'}</td>
                      <td className="px-6 py-4 text-sm text-text/80">
                        <div>{new Date(job.created).toLocaleString()}</div>
                    {job.progress!=null && <div className="text-xs text-text/70">{Math.round(job.progress)}%{job.eta_seconds!=null && ` • ETA ${Math.ceil(job.eta_seconds/60)}m`}</div>}
                      </td>
                  <td className="px-6 py-4 text-sm">
                        <button
                          onClick={() => setSelectedJob(job)}
                          className="text-primary hover:brightness-110 mr-3"
                        >
                          View
                        </button>
                        {job.framework==='huggingface' && job.config && job.config.hpo && job.config.hpo.enabled && (
                          <button onClick={()=> onOpenHpo && onOpenHpo(job.id)} className="text-secondary hover:brightness-110 mr-3">HPO</button>
                        )}
                        {job.status === 'running' && (
                          <>
                            <button onClick={async()=>{await fetch(`${API_BASE}/jobs/${job.id}/pause`,{method:'POST'}); loadJobs(); toast.push({type:'info', title:'Job paused'});}} className="text-warning hover:brightness-110 mr-3">Pause</button>
                            <button onClick={async()=>{await fetch(`${API_BASE}/jobs/${job.id}/resume`,{method:'POST'}); loadJobs(); toast.push({type:'success', title:'Job resumed'});}} className="text-success hover:brightness-110 mr-3">Resume</button>
                            <button onClick={() => handleCancelJob(job.id)} className="text-danger hover:brightness-110 mr-3">Cancel</button>
                          </>
                        )}
                    {(job.status === 'failed' || job.status === 'cancelled') && (
                      <button onClick={() => handleRestartJob(job.id)} className="text-success hover:brightness-110 mr-3">Restart</button>
                    )}
                    {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') && (
                      <button onClick={() => handleDeleteJob(job.id)} className="text-danger hover:brightness-110 mr-3">Delete</button>
                    )}
                    <button onClick={async()=>{await fetch(`${API_BASE}/jobs/${job.id}/clone`,{method:'POST'}); loadJobs(); toast.push({type:'success', title:'Job cloned'});}} className="text-text/70 hover:text-text mr-3">Clone</button>
                      </td>
                    </tr>
                  );
                });
                return rows;
              });
            })()}
          </tbody>
        </table>
      </div>
      
      {selectedJob && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-surface border border-border rounded-lg max-w-3xl w-full max-h-[80vh] overflow-auto p-6">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-2xl font-bold">{selectedJob.name}</h2>
              <div className="flex gap-2 items-center">
                {selectedJob.status === 'running' && (
                  <button onClick={() => handleCancelJob(selectedJob.id)} className="px-3 py-1 text-sm bg-danger/10 text-danger border border-danger/30 rounded hover:bg-danger/20">
                    Cancel
                  </button>
                )}
                {(selectedJob.status === 'failed' || selectedJob.status === 'cancelled') && (
                  <button onClick={() => handleRestartJob(selectedJob.id)} className="px-3 py-1 text-sm bg-success/10 text-success border border-success/30 rounded hover:bg-success/20">
                    Restart
                  </button>
                )}
                {(selectedJob.status === 'completed' || selectedJob.status === 'failed' || selectedJob.status === 'cancelled') && (
                  <button onClick={() => handleDeleteJob(selectedJob.id)} className="px-3 py-1 text-sm bg-danger/10 text-danger border border-danger/30 rounded hover:bg-danger/20">
                    Delete
                  </button>
                )}
                <button
                  onClick={() => setSelectedJob(null)}
                  className="text-text/60 hover:text-text text-xl"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-text/70">Status</p>
                  <p className="font-semibold capitalize">{selectedJob.status}</p>
                </div>
                <div>
                  <p className="text-sm text-text/70">Framework</p>
                  <p className="font-semibold capitalize">{selectedJob.framework}</p>
                </div>
                <div>
                  <p className="text-sm text-text/70">Type</p>
                  <p className="font-semibold capitalize">{selectedJob.type}</p>
                </div>
                <div>
                  <p className="text-sm text-text/70">Created</p>
                  <p className="font-semibold">{new Date(selectedJob.created).toLocaleString()}</p>
                </div>
                {selectedJob.progress!=null && (
                  <div className="col-span-2">
                    <div className="text-sm text-text/70">Progress</div>
                    <div className="w-full bg-muted rounded h-3">
                      <div className="bg-primary h-3 rounded" style={{width:`${Math.min(100, Math.max(0, selectedJob.progress))}%`}} />
                    </div>
                    <div className="text-xs text-text/70 mt-1">{Math.round(selectedJob.progress)}% {selectedJob.eta_seconds!=null && `(ETA ${Math.ceil(selectedJob.eta_seconds/60)}m)`}</div>
                  </div>
                )}
              </div>
              
              <div>
                <p className="text-sm text-text/70 mb-2">Configuration</p>
                <pre className="bg-muted p-4 rounded-lg text-xs overflow-auto">
                  {JSON.stringify(selectedJob.config, null, 2)}
                </pre>
              </div>

              <div>
                <p className="text-sm text-text/70 mb-2">GPU Allocation</p>
                {selectedJob.gpu ? (
                  <div className="p-4 bg-muted rounded-lg text-xs">
                    <div><span className="text-text/70">Type:</span> <span className="font-semibold">{selectedJob.gpu.type?.toUpperCase()}</span></div>
                    {selectedJob.gpu.gpu_index != null && (<div><span className="text-text/70">GPU Index:</span> <span className="font-semibold">{selectedJob.gpu.gpu_index}</span></div>)}
                    {selectedJob.gpu.gpu_uuid && (<div><span className="text-text/70">GPU UUID:</span> <span className="font-semibold">{selectedJob.gpu.gpu_uuid}</span></div>)}
                    {selectedJob.gpu.mig_uuid && (<div><span className="text-text/70">MIG UUID:</span> <span className="font-semibold">{selectedJob.gpu.mig_uuid}</span></div>)}
                  </div>
                ) : (
                  <div className="text-xs text-text/60">No explicit GPU/MIG selected (auto)</div>
                )}
              </div>

              {selectedJob.resource && (
                <div>
                  <p className="text-sm text-text/70 mb-2">Resource Usage</p>
                  <div className="p-4 bg-muted rounded-lg text-xs grid grid-cols-3 gap-3">
                    {'rss_mib' in selectedJob.resource && (<div><span className="text-text/70">RSS:</span> <span className="font-semibold">{selectedJob.resource.rss_mib} MiB</span></div>)}
                    {'cpu_time_sec' in selectedJob.resource && (<div><span className="text-text/70">CPU Time:</span> <span className="font-semibold">{selectedJob.resource.cpu_time_sec}s</span></div>)}
                    {'gpu_mem_mib' in selectedJob.resource && (<div><span className="text-text/70">GPU Mem:</span> <span className="font-semibold">{selectedJob.resource.gpu_mem_mib} MiB</span></div>)}
                  </div>
                </div>
              )}

              <div>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm text-text/70">Checkpoints</p>
                  <a className="text-primary text-sm" href={`${API_BASE}/models/export?ids=${selectedJob.id}`} target="_blank" rel="noreferrer">Export Model</a>
                </div>
                {cpLoading ? (
                  <div className="text-xs text-text/60">Loading...</div>
                ) : checkpoints.length ? (
                  <ul className="text-xs text-text bg-muted rounded border border-border p-2 space-y-1">
                    {checkpoints.map((c,i)=> (
                      <li key={i} className="flex justify-between"><span>{c.file}</span><span className="text-text/60">{c.size != null ? `${c.size} B` : ''}</span></li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-xs text-text/60">No checkpoints found</div>
                )}
              </div>

              <JobLiveLogs jobId={selectedJob.id} />
              <JobLiveMetrics jobId={selectedJob.id} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Main App Component
export default function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [systemInfo, setSystemInfo] = useState({});
  const [frameworks, setFrameworks] = useState({});
  const [partitions, setPartitions] = useState({ gpus: [] });
  const [metrics, setMetrics] = useState({ window_seconds: 3600, interval_seconds: 5, samples: [], jobs: [] });
  const [modelView, setModelView] = useState({ id: null, compareIds: [] });
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [quickStartOpen, setQuickStartOpen] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [hpoJobId, setHpoJobId] = useState(null);
  const [gpuSeries, setGpuSeries] = useState({}); // { [gpuIndex]: number[] }
  const [powerSeries, setPowerSeries] = useState({}); // { [gpuIndex]: number[] } for power draw
  const [netSeries, setNetSeries] = useState({ rx: [], tx: [] });
  const userId = 'anonymous';
  const widgetKey = `dashboard.widgets:${userId}`;
  const [widgetPrefs, setWidgetPrefs] = useState(() => {
    try { return JSON.parse(localStorage.getItem(widgetKey) || '{}'); } catch { return {}; }
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try { return localStorage.getItem('sidebar.collapsed') === '1'; } catch { return false; }
  });
  useEffect(()=>{ try { localStorage.setItem('sidebar.collapsed', sidebarCollapsed ? '1' : '0'); } catch {} }, [sidebarCollapsed]);
  useEffect(() => { try { localStorage.setItem(widgetKey, JSON.stringify(widgetPrefs)); } catch {} }, [widgetPrefs]);
  
  useEffect(() => {
    loadFrameworks();
    // Prefer SSE for live updates; fall back to polling
    let es;
    try {
      es = new EventSource(`${API_BASE}/system/stream`);
      es.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          setSystemInfo(data);
          // update live series
          setGpuSeries(prev => {
            const gmap = { ...prev };
            (data.gpus || []).forEach(g => {
              const k = String(g.index ?? 0);
              const arr = (gmap[k] || []).slice(-119);
              arr.push(g.utilization_gpu_pct ?? 0);
              gmap[k] = arr;
            });
            return gmap;
          });
          setPowerSeries(prev => {
            const pmap = { ...prev };
            (data.gpus || []).forEach(g => {
              const k = String(g.index ?? 0);
              const arr = (pmap[k] || []).slice(-119);
              arr.push(g.power_draw_w ?? 0);
              pmap[k] = arr;
            });
            return pmap;
          });
          setNetSeries(prev => {
            const rx = (prev.rx || []).slice(-119); rx.push(data.net?.rx_rate_bps || 0);
            const tx = (prev.tx || []).slice(-119); tx.push(data.net?.tx_rate_bps || 0);
            return { rx, tx };
          });
        } catch {}
      };
    } catch (e) {
      // fallback
      loadSystemInfo();
      const interval = setInterval(loadSystemInfo, 10000);
      return () => clearInterval(interval);
    }
    return () => { if (es) es.close(); };
  }, []);

  useEffect(() => {
    const load = async () => {
      try { setPartitions(await api.getPartitions()); } catch {}
    };
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const load = async () => {
      try { setMetrics(await api.getMetricsHistory()); } catch {}
    };
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const load = async () => {
      try { setJobs(await api.getJobs()); } catch {}
    };
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  // Global search shortcut (Cmd+K / Ctrl+K)
  useEffect(() => {
    const onKey = (e) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        setPaletteOpen(true);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  
  const loadSystemInfo = async () => {
    try {
      const data = await api.getSystemInfo();
      setSystemInfo(data);
    } catch (error) {
      console.error('Failed to load system info:', error);
    }
  };
  
  const loadFrameworks = async () => {
    try {
      const data = await api.getFrameworks();
      setFrameworks(data);
    } catch (error) {
      console.error('Failed to load frameworks:', error);
    }
  };
  
  return (
    <ToastProvider>
    <div className="min-h-screen bg-bg flex">
      <Sidebar collapsed={sidebarCollapsed} setCollapsed={setSidebarCollapsed} current={currentPage} onNavigate={setCurrentPage} />
      <div className="flex-1 flex flex-col min-w-0">
        <HeaderBar breadcrumbs={<Breadcrumbs currentPage={currentPage} modelView={modelView} onNavigate={setCurrentPage} />} onSearch={()=>setPaletteOpen(true)}>
          <ThemeToggle />
          <UserMenu />
        </HeaderBar>
        <main className="max-w-7xl mx-auto px-4 py-6 w-full">
          <PageTransition>
        {currentPage === 'dashboard' && (
          <Dashboard3Col
            onNavigate={setCurrentPage}
            systemInfo={systemInfo}
            partitions={partitions}
            metrics={metrics}
            gpuSeries={gpuSeries}
            powerSeries={powerSeries}
            netSeries={netSeries}
            widgetPrefs={widgetPrefs}
            setWidgetPrefs={setWidgetPrefs}
            jobs={jobs}
          />
        )}
        {currentPage === 'create' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <CreateJob onNavigate={setCurrentPage} frameworks={frameworks} partitions={partitions} />
          </PageWithSidebars>
        )}
        {currentPage === 'wizard' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <JobWizard onNavigate={setCurrentPage} frameworks={frameworks} partitions={partitions} api={api} />
          </PageWithSidebars>
        )}
        {currentPage === 'jobs' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <JobsList onNavigate={setCurrentPage} partitions={partitions} onOpenHpo={(id)=>{ setHpoJobId(id); setCurrentPage('hpo'); }} />
          </PageWithSidebars>
        )}
        {currentPage === 'hpo' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <HPOViewer jobId={hpoJobId} apiBase={API_BASE} />
          </PageWithSidebars>
        )}
        {currentPage === 'labeling' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <Labeling api={api} />
          </PageWithSidebars>
        )}
        {currentPage === 'builder' && (
          <ArchitectureBuilder api={api} />
        )}
        {currentPage === 'models' && !modelView.id && modelView.compareIds.length===0 && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <ModelsPage api={api} onOpen={(id)=> setModelView({ id, compareIds: [] })} onCompare={(ids)=> setModelView({ id: null, compareIds: ids })} onNavigate={setCurrentPage} />
          </PageWithSidebars>
        )}
        {currentPage === 'models' && modelView.id && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <ModelDetail id={modelView.id} api={api} onBack={()=> setModelView({ id: null, compareIds: [] })} />
          </PageWithSidebars>
        )}
        {currentPage === 'models' && modelView.compareIds.length>0 && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <ModelCompare ids={modelView.compareIds} api={api} onBack={()=> setModelView({ id: null, compareIds: [] })} />
          </PageWithSidebars>
        )}
        {currentPage === 'admin' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <AdminPartitions />
          </PageWithSidebars>
        )}
        {currentPage === 'hardware_gpus' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <AdminPartitions />
          </PageWithSidebars>
        )}
        {currentPage === 'hardware_storage' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <HardwareStorage systemInfo={systemInfo} />
          </PageWithSidebars>
        )}
        {currentPage === 'hardware_network' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <HardwareNetwork systemInfo={systemInfo} />
          </PageWithSidebars>
        )}
        {currentPage === 'settings_profile' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <SettingsProfile />
          </PageWithSidebars>
        )}
        {currentPage === 'settings_teams' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <SettingsTeams />
          </PageWithSidebars>
        )}
        {currentPage === 'settings_billing' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <SettingsBilling />
          </PageWithSidebars>
        )}
        {currentPage === 'datasets' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <DatasetsPage api={api} />
          </PageWithSidebars>
        )}
        {currentPage === 'experiments' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <ExperimentsPage />
          </PageWithSidebars>
        )}
        {currentPage === 'pipelines' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <PipelinesPage />
          </PageWithSidebars>
        )}
        {currentPage === 'profile' && (
          <PageWithSidebars onNavigate={setCurrentPage} jobs={jobs} systemInfo={systemInfo}>
            <React.Suspense fallback={<div className="p-6">Loading profile...</div>}>
              <ProfilePage />
            </React.Suspense>
          </PageWithSidebars>
        )}
        {currentPage === 'docs' && (
          <Documentation onNavigate={setCurrentPage} />
        )}
          </PageTransition>
        </main>
      </div>
      <CommandPalette open={paletteOpen} onClose={()=>setPaletteOpen(false)} api={api} onNavigate={setCurrentPage} />
      <QuickStart open={quickStartOpen} onClose={()=>setQuickStartOpen(false)} api={api} onDone={()=>setCurrentPage('jobs')} />
    </div>
    </ToastProvider>
  );
}

function ThemeToggle(){
  const [theme, setTheme] = React.useState(() => localStorage.getItem('theme') || 'system');
  React.useEffect(() => {
    const apply = (t) => {
      const doc = document.documentElement;
      const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (t === 'dark' || (t === 'system' && prefersDark)) {
        doc.classList.add('dark');
      } else {
        doc.classList.remove('dark');
      }
      doc.setAttribute('data-theme', t);
      localStorage.setItem('theme', t);
    };
    apply(theme);
    const mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => { if ((localStorage.getItem('theme') || 'system') === 'system') apply('system'); };
    try { mq && mq.addEventListener ? mq.addEventListener('change', handler) : mq && mq.addListener && mq.addListener(handler); } catch {}
    return () => { try { mq && mq.removeEventListener ? mq.removeEventListener('change', handler) : mq && mq.removeListener && mq.removeListener(handler);} catch {} };
  }, [theme]);
  const cycle = () => setTheme(prev => prev === 'light' ? 'dark' : prev === 'dark' ? 'system' : 'light');
  const Icon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor;
  const label = theme === 'light' ? 'Light' : theme === 'dark' ? 'Dark' : 'System';
  return (
    <button onClick={cycle} title={`Theme: ${label}`} className="px-2 py-2 rounded border border-border hover:bg-muted transition">
      <Icon size={18} />
    </button>
  );
}

function JobLiveLogs({ jobId }){
  const [lines, setLines] = useState([]);
  const [q, setQ] = useState('');
  const [level, setLevel] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const preRef = React.useRef(null);
  useEffect(()=>{
    const es = new EventSource(`${API_BASE}/jobs/${jobId}/logs/stream`);
    es.onmessage = (e)=>{
      try{ const p = JSON.parse(e.data); setLines(p.lines||[]); }catch{}
    };
    return ()=> es.close();
  }, [jobId]);
  useEffect(()=>{
    if (autoScroll && preRef.current){ preRef.current.scrollTop = preRef.current.scrollHeight; }
  }, [lines, autoScroll]);
  const filtered = lines.filter(l => (!q || l.toLowerCase().includes(q.toLowerCase())) && (!level || (level==='error'? l.toLowerCase().includes('error') : level==='warn' ? l.toLowerCase().includes('warn') : true)) );
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm text-text/70">Logs</p>
        <div className="flex gap-2 text-xs">
          <input className="border rounded px-2 py-1" placeholder="Search logs" value={q} onChange={e=>setQ(e.target.value)} />
          <select className="border rounded px-2 py-1" value={level} onChange={e=>setLevel(e.target.value)}>
            <option value="">All</option>
            <option value="info">Info</option>
            <option value="warn">Warning</option>
            <option value="error">Error</option>
          </select>
          <a className="px-2 py-1 border rounded" href={`${API_BASE}/jobs/${jobId}/logs?export=txt`} target="_blank">Export TXT</a>
          <a className="px-2 py-1 border rounded" href={`${API_BASE}/jobs/${jobId}/logs?export=json`} target="_blank">Export JSON</a>
          <label className="inline-flex items-center gap-1"><input type="checkbox" checked={autoScroll} onChange={e=>setAutoScroll(e.target.checked)} /> Auto-scroll</label>
        </div>
      </div>
      <div ref={preRef} className="bg-gray-900 p-4 rounded-lg text-xs overflow-auto max-h-64 font-mono">
        {filtered.map((ln, i) => {
          const low = ln.toLowerCase();
          const isErr = low.includes('error') || low.includes('exception') || low.includes('traceback');
          const isWarn = !isErr && (low.includes('warn'));
          const cls = isErr ? 'text-red-300' : isWarn ? 'text-yellow-200' : 'text-green-300';
          return <div key={i} className={cls}>{ln}</div>;
        })}
      </div>
    </div>
  );
}

function JobLiveMetrics({ jobId }){
  const [data, setData] = useState([]);
  const [tracking, setTracking] = useState({});
  useEffect(()=>{
    const es = new EventSource(`${API_BASE}/jobs/${jobId}/metrics/stream`);
    es.onmessage = (e)=>{ try{ const p = JSON.parse(e.data); const arr = p.metrics||[]; setData(arr); const t = arr.filter(m=>m.kind==='tracking'); if (t.length){ const last=t[t.length-1]; setTracking(prev=>({...prev, ...last})); } }catch{}};
    return ()=> es.close();
  }, [jobId]);
  const losses = data.filter(m => m.loss!=null || m.avg_loss!=null || m.avg_train_loss!=null);
  const accs = data.filter(m => (m.accuracy!=null) || (m.accuracy_pct!=null) || (m.val_acc_pct!=null));
  const lrs = data.filter(m => (m.lr!=null) || (m.learning_rate!=null) || (m.logs && (m.logs.learning_rate!=null)));
  return (
    <div>
      <p className="text-sm text-text/70 mb-2">Metrics</p>
      {/* Tracking links */}
      {(() => {
        const wb = tracking && tracking.wandb;
        const mf = tracking && tracking.mlflow;
        if (!wb && !mf) return null;
        return (
          <div className="mb-3 p-2 bg-muted rounded border border-border text-xs flex items-center gap-3">
            {wb && wb.url && (<a className="text-primary underline" href={wb.url} target="_blank" rel="noreferrer">Open W&B Run</a>)}
            {mf && mf.run_id && (<span>MLflow run: <code className="px-1 py-0.5 bg-surface border border-border rounded">{mf.run_id.slice(0,8)}…</code></span>)}
          </div>
        );
      })()}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MiniLine series={losses} y={(m)=> m.loss ?? m.avg_loss ?? m.avg_train_loss} label="Loss" />
        <MiniLine series={accs} y={(m)=> (m.accuracy ?? m.accuracy_pct ?? m.val_acc_pct)} label="Accuracy" />
        <MiniLine series={lrs} y={(m)=> (m.lr ?? m.learning_rate ?? (m.logs?m.logs.learning_rate:undefined))} label="LR" />
      </div>
      {/* Custom numeric metrics from logs */}
      {(() => {
        const keys = new Set();
        data.forEach(m => { if (m.logs) Object.entries(m.logs).forEach(([k,v])=>{ if (typeof v === 'number' && !['learning_rate'].includes(k)) keys.add(k); }); });
        const keyList = Array.from(keys).slice(0,4); // cap to avoid clutter
        if (!keyList.length) return null;
        return (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            {keyList.map(k => (
              <MiniLine key={k} series={data.filter(m=>m.logs && typeof m.logs[k] === 'number')} y={(m)=> m.logs[k]} label={k} />
            ))}
          </div>
        );
      })()}
    </div>
  );
}

function MiniLine({ series, y, label }){
  const w=600, h=140, pad=20; if (!series||!series.length) return <div className="text-xs text-gray-500">No {label} yet</div>;
  const ys = series.map(y).filter(v=>v!=null);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pts = series.map((s,i)=>{
    const xv = i/(Math.max(1, series.length-1));
    const yv = (y(s)-yMin)/(Math.max(1e-6, (yMax-yMin||1)));
    const x = pad + xv*(w-2*pad);
    const yy = pad + (1-yv)*(h-2*pad);
    return `${x},${yy}`;
  }).join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[140px] bg-muted rounded border border-border">
      <text x={pad} y={12} fontSize="10" fill="#374151">{label}</text>
      <polyline fill="none" stroke="#2563eb" strokeWidth="2" points={pts} />
    </svg>
  );
}
function LineChart({ data, seriesFrom, yKeyFor, height=180 }) {
  const w = 600; // virtual width
  const h = height;
  const padding = 24;
  const series = seriesFrom(data);
  const entries = Object.entries(series);
  if (!data || data.length === 0 || entries.length === 0) return <div className="text-gray-500 text-sm">No data</div>;
  const maxX = Math.max(1, data.length - 1);
  const toPoint = (i, val)=>{
    const x = padding + (i/maxX) * (w - 2*padding);
    const y = padding + (1 - Math.min(100, Math.max(0, val))/100) * (h - 2*padding);
    return `${x},${y}`;
  };
  const colors = ['#2563eb','#16a34a','#dc2626','#7c3aed','#ca8a04','#0891b2','#b91c1c','#0e7490'];
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[180px] bg-muted rounded border border-border">
      {/* axes */}
      <line x1={padding} y1={padding} x2={padding} y2={h-padding} stroke="#e5e7eb" />
      <line x1={padding} y1={h-padding} x2={w-padding} y2={h-padding} stroke="#e5e7eb" />
      {/* 0/50/100 grid */}
      {[0,50,100].map((v,i)=>{
        const y = padding + (1 - v/100) * (h - 2*padding);
        return <g key={i}>
          <line x1={padding} x2={w-padding} y1={y} y2={y} stroke="#f3f4f6" />
          <text x={w-padding+4} y={y+4} fontSize="10" fill="#6b7280">{v}%</text>
        </g>;
      })}
      {entries.map(([key, meta], si) => {
        const pts = data.map((s, i) => toPoint(i, yKeyFor(s, key))).join(' ');
        const color = colors[si % colors.length];
        return <g key={key}>
          <polyline fill="none" stroke={color} strokeWidth="2" points={pts} />
          <text x={padding+6+si*70} y={padding+12} fontSize="10" fill={color}>{meta.label}</text>
        </g>;
      })}
    </svg>
  );
}

function JobsTimeline({ metrics, partitions }) {
  const samples = metrics?.samples || [];
  const jobs = metrics?.jobs || [];
  if (!samples.length) return <div className="text-gray-500 text-sm">No data</div>;
  const winSec = metrics.window_seconds || 3600;
  const parseTs = (s)=> new Date(s).getTime()/1000;
  const now = Date.now()/1000;
  const winStart = now - winSec;
  const pct = (t)=> Math.min(100, Math.max(0, (t - winStart) / winSec * 100));
  const labelFor = (g)=>{
    if (!g) return 'auto';
    if (g.type === 'gpu') return g.gpu_index != null ? `GPU ${g.gpu_index}` : 'GPU';
    if (g.type === 'mig') {
      const inst = (partitions?.gpus||[]).flatMap(gg => (gg.instances||[])).find(i => i.uuid===g.mig_uuid);
      if (inst) {
        const parent = (partitions?.gpus||[]).find(gg => (gg.instances||[]).some(i => i.uuid===g.mig_uuid));
        return `MIG GPU ${parent?.index ?? ''} ${inst.profile} (dev ${inst.device_id})`;
      }
      return 'MIG';
    }
    return '-';
  };
  return (
    <div className="space-y-2">
      {jobs.length ? jobs.map(j => {
        const s = j.started ? parseTs(j.started) : now;
        const e = j.completed ? parseTs(j.completed) : now;
        const left = pct(s);
        const right = pct(e);
        const width = Math.max(1, right - left);
        return (
          <div key={j.id} className="p-2 bg-muted rounded border border-border">
            <div className="flex items-center justify-between text-xs text-gray-700">
              <div className="font-semibold">{j.name}</div>
              <div className="text-gray-500">{labelFor(j.gpu)}</div>
            </div>
            <div className="mt-1 w-full bg-muted h-2 rounded relative">
              <div className="absolute top-0 left-0 h-2 bg-blue-600 rounded" style={{ left: `${left}%`, width: `${width}%` }} />
            </div>
          </div>
        );
      }) : <div className="text-gray-500 text-sm">No jobs active in the last hour</div>}
    </div>
  );
}

// New 3-column dashboard with widgets
function Dashboard3Col({ onNavigate, systemInfo, partitions, metrics, gpuSeries, powerSeries, netSeries, widgetPrefs, setWidgetPrefs, jobs }) {
  // Helpers reused from original dashboard
  const colorFor = (pct) => pct < 70 ? 'bg-green-500' : pct < 85 ? 'bg-yellow-500' : 'bg-red-500';
  const fmtGiB = (mib) => (mib != null ? (mib / 1024).toFixed(1) : '0.0');
  const fmtPct = (n) => (n != null ? Math.round(n) : 0);
  const fmtRate = (bps) => {
    if (bps == null) return '0 B/s';
    const K = 1024, M = K*1024, G = M*1024;
    if (bps >= G) return (bps/G).toFixed(2) + ' GB/s';
    if (bps >= M) return (bps/M).toFixed(2) + ' MB/s';
    if (bps >= K) return (bps/K).toFixed(2) + ' KB/s';
    return bps.toFixed(0) + ' B/s';
  };

  const defaultWidgets = [
    { id: 'gpu_spark', title: 'GPU Utilization', size: 'md', visible: true },
    { id: 'temp_heat', title: 'Temperature Heatmap', size: 'sm', visible: true },
    { id: 'power', title: 'Power Meters', size: 'sm', visible: true },
    { id: 'net', title: 'Network Throughput', size: 'md', visible: true },
    { id: 'gpu_status', title: 'GPU Status', size: 'lg', visible: true },
    { id: 'mem', title: 'Memory', size: 'md', visible: true },
    { id: 'io', title: 'I/O', size: 'md', visible: true },
    { id: 'timeline', title: 'Timelines', size: 'lg', visible: true },
  ];
  const widgets = (widgetPrefs?.order?.length ? widgetPrefs.order : defaultWidgets.map(w => w.id)).map(id => {
    const base = defaultWidgets.find(w => w.id === id) || { id, size: 'md', visible: true };
    const ow = (widgetPrefs?.items || {})[id] || {};
    return { ...base, ...ow };
  });
  const sizeCls = (sz) => sz === 'lg' ? 'col-span-12' : sz === 'md' ? 'col-span-12 md:col-span-6' : 'col-span-12 md:col-span-4';
  const updateItem = (id, patch) => {
    const items = { ...(widgetPrefs.items||{}) };
    items[id] = { ...(items[id]||{}), ...patch };
    setWidgetPrefs({ items, order: widgetPrefs.order && widgetPrefs.order.length ? widgetPrefs.order : defaultWidgets.map(w=>w.id) });
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">DGX AI Trainer Dashboard</h1>
      <div className="grid grid-cols-12 gap-4">
        {/* Left: customize widgets only */}
        <div className="col-span-12 lg:col-span-3 space-y-4">
          <div className="bg-surface p-4 rounded-lg border border-border">
            <div className="text-sm text-text/70 mb-2">Customize Widgets</div>
            {defaultWidgets.map(w => (
              <label key={w.id} className="flex items-center justify-between text-sm py-1">
                <span>{w.title}</span>
                <input type="checkbox" checked={(widgetPrefs.items?.[w.id]?.visible ?? w.visible)} onChange={e=>updateItem(w.id, { visible: e.target.checked })} />
              </label>
            ))}
          </div>
        </div>

        {/* Center: main widgets */}
        <div className="col-span-12 lg:col-span-6 space-y-4">
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-12 md:col-span-6 bg-surface p-6 rounded-lg border border-border">
              <div className="flex items-center justify-between"><div><div className="text-sm text-text/70">GPUs</div><div className="text-2xl font-bold text-blue-600">{systemInfo.gpus?.length||0}</div></div><Cpu className="text-blue-500" size={32} /></div>
            </div>
            <div className="col-span-12 md:col-span-6 bg-surface p-6 rounded-lg border border-border">
              <div className="flex items-center justify-between"><div><div className="text-sm text-text/70">Jobs Running</div><div className="text-2xl font-bold text-green-600">{systemInfo.jobs_running ?? 0}</div></div><Activity className="text-green-500" size={32} /></div>
            </div>
          </div>
          <div className="grid grid-cols-12 gap-4">
            {widgets.filter(w => w.visible !== false).map(w => (
              <div key={w.id} className={`${sizeCls(w.size)} bg-surface p-4 rounded-lg border border-border`} draggable onDragStart={(e)=>{ e.dataTransfer.setData('text/plain', w.id); }} onDragOver={(e)=>e.preventDefault()} onDrop={(e)=>{
                const from = e.dataTransfer.getData('text/plain');
                if (!from || from === w.id) return;
                const ord = (widgetPrefs.order && widgetPrefs.order.length ? widgetPrefs.order : defaultWidgets.map(x=>x.id)).slice();
                const a = ord.indexOf(from); const b = ord.indexOf(w.id);
                if (a===-1||b===-1) return;
                ord.splice(a,1); ord.splice(b,0,from);
                setWidgetPrefs({ items: widgetPrefs.items||{}, order: ord });
              }}>
                <div className="flex items-center justify-between mb-2">
                  <div className="text-sm font-semibold">{defaultWidgets.find(d=>d.id===w.id)?.title || w.id}</div>
                  <select className="border border-border rounded px-2 py-1 bg-surface text-xs" value={w.size} onChange={e=>updateItem(w.id, { size: e.target.value })}>
                    <option value="sm">Small</option>
                    <option value="md">Medium</option>
                    <option value="lg">Large</option>
                  </select>
                </div>
                {w.id === 'gpu_spark' && <GPUSparklines systemInfo={systemInfo} gpuSeries={gpuSeries} />}
                {w.id === 'temp_heat' && <GPUTemperatureHeatmap systemInfo={systemInfo} />}
                {w.id === 'power' && <GPUPowerMeters systemInfo={systemInfo} powerSeries={powerSeries} />}
                {w.id === 'net' && <NetworkSpark netSeries={netSeries} />}
                {w.id === 'gpu_status' && (
                  <div className="space-y-3">
                    {systemInfo.gpus?.length ? systemInfo.gpus.map((gpu, i) => {
                      const usedMiB = gpu.memory_used_mib ?? 0;
                      const totalMiB = gpu.memory_total_mib ?? 0;
                      const pct = gpu.memory_used_pct ?? (totalMiB ? Math.round((usedMiB / totalMiB) * 100) : 0);
                      return (
                        <div key={i} className="p-3 bg-muted rounded-lg border border-border">
                          <div className="flex justify-between text-sm"><div className="font-semibold">{gpu.name} — GPU {gpu.index ?? i}</div><div>{fmtGiB(usedMiB)} / {fmtGiB(totalMiB)} GiB</div></div>
                          <div className="mt-2 w-full bg-muted rounded h-3"><div className={`${colorFor(pct)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} /></div>
                        </div>
                      );
                    }) : <div className="text-sm text-text/60">No GPUs</div>}
                  </div>
                )}
                {w.id === 'mem' && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <p className="text-sm text-text/70">RAM</p>
                      <div className="flex items-center justify-between mt-1">
                        <div className="flex-1 mr-3">
                          <div className="w-full bg-muted rounded h-3">
                            <div className={`h-3 rounded ${colorFor(systemInfo.memory?.used_pct ?? 0)}`} style={{ width: `${Math.min(100, Math.max(0, systemInfo.memory?.used_pct ?? 0))}%` }} />
                          </div>
                        </div>
                        <div className="text-sm font-semibold">{fmtPct(systemInfo.memory?.used_pct)}%</div>
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-text/70">Swap</p>
                      <div className="flex items-center justify-between mt-1">
                        <div className="flex-1 mr-3">
                          <div className="w-full bg-muted rounded h-3">
                            <div className={`h-3 rounded ${colorFor(systemInfo.swap?.used_pct ?? 0)}`} style={{ width: `${Math.min(100, Math.max(0, systemInfo.swap?.used_pct ?? 0))}%` }} />
                          </div>
                        </div>
                        <div className="text-sm font-semibold">{fmtPct(systemInfo.swap?.used_pct)}%</div>
                      </div>
                    </div>
                  </div>
                )}
                {w.id === 'io' && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="p-4 bg-muted rounded border border-border">
                      <p className="text-sm text-text/70">Network</p>
                      <div className="mt-2 text-sm">
                        <div><span className="text-text/70">Receive:</span> <span className="font-semibold">{fmtRate(systemInfo.net?.rx_rate_bps)}</span></div>
                        <div><span className="text-text/70">Transmit:</span> <span className="font-semibold">{fmtRate(systemInfo.net?.tx_rate_bps)}</span></div>
                      </div>
                    </div>
                    <div className="p-4 bg-muted rounded border border-border">
                      <p className="text-sm text-text/70 mb-2">Disks</p>
                      <div className="space-y-2">
                        {(systemInfo.disks || []).map((d, i) => (
                          <div key={i}>
                            <div className="flex justify-between text-xs text-text/70"><span>{d.path}</span><span>{d.used_gib} / {d.total_gib} GiB</span></div>
                            <div className="w-full bg-muted rounded h-2"><div className={`${colorFor(d.used_pct ?? 0)} h-2 rounded`} style={{ width: `${Math.min(100, Math.max(0, d.used_pct ?? 0))}%` }} /></div>
                          </div>
                        ))}
                        {(!systemInfo.disks || systemInfo.disks.length === 0) && (<div className="text-xs text-text/60">No disk info</div>)}
                      </div>
                    </div>
                  </div>
                )}
                {w.id === 'timeline' && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <div>
                        <p className="text-sm text-text/70 mb-2">GPU Utilization (%)</p>
                        <LineChart data={metrics?.samples||[]} seriesFrom={(s)=>{
                          const map = {};
                          (s[0]?.gpus||[]).forEach(g => { map[`gpu-${g.index}`] = {label:`GPU ${g.index}`, color: 'blue'}; });
                          return map;
                        }} yKeyFor={(s, key)=>{
                          const idx = parseInt(String(key).split('-')[1]||'0');
                          const g = (s.gpus||[]).find(x => x.index===idx);
                          return g?.util_pct ?? 0;
                        }} />
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-sm text-text/70">Memory Usage (%)</p>
                          <a href={`${API_BASE}/system/metrics/history.csv`} className="text-primary hover:brightness-110 text-sm">Export CSV</a>
                        </div>
                        <LineChart data={metrics?.samples||[]} seriesFrom={()=>({ 'gpu-mem': {label:'GPU Mem', color:'blue'}, 'sys-mem': {label:'System RAM', color:'green'} })} yKeyFor={(s,key)=> key==='gpu-mem' ? (s.gpu_mem_used_pct ?? 0) : (s.sys_mem_used_pct ?? 0)} />
                      </div>
                    </div>
                    <div className="mt-2">
                      <p className="text-sm text-text/70 mb-2">Active Jobs</p>
                      <JobsTimeline metrics={metrics} partitions={partitions} />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="flex gap-4">
            <Button variant="primary" className="flex-1 py-4" onClick={()=>onNavigate('create')} leftIcon={<Plus size={20} />}>Create New Training Job</Button>
            <Button variant="ghost" className="flex-1 py-4" onClick={()=>onNavigate('jobs')} leftIcon={<List size={20} />}>View All Jobs</Button>
          </div>
        </div>

        {/* Right: activity & notifications */}
        <div className="col-span-12 lg:col-span-3 space-y-4">
          <div className="bg-surface p-4 rounded-lg border border-border">
            <div className="text-sm font-semibold mb-2">Activity Feed</div>
            <div className="space-y-2 max-h-[420px] overflow-auto">
              {(jobs||[]).slice().reverse().slice(0,20).map(j => (
                <div key={j.id} className="p-2 bg-muted rounded border border-border text-sm">
                  <div className="flex justify-between"><div className="font-semibold truncate mr-2">{j.name}</div><div className="text-xs text-text/60">{j.status}</div></div>
                  <div className="text-xs text-text/70">{j.framework}</div>
                </div>
              ))}
              {(!jobs || jobs.length===0) && <div className="text-sm text-text/60">No recent activity</div>}
            </div>
          </div>
          <div className="bg-surface p-4 rounded-lg border border-border">
            <div className="text-sm font-semibold mb-2">Notifications</div>
            <div className="space-y-2 text-sm">
              {(() => {
                const alerts = [];
                (systemInfo.gpus||[]).forEach((g,i)=>{
                  if ((g.temperature_gpu_c||0) >= 80) alerts.push(`GPU ${g.index ?? i} high temperature: ${g.temperature_gpu_c} °C`);
                  if ((g.utilization_gpu_pct||0) >= 95) alerts.push(`GPU ${g.index ?? i} high utilization: ${g.utilization_gpu_pct}%`);
                });
                if ((systemInfo.memory?.used_pct||0) >= 90) alerts.push(`System RAM high: ${Math.round(systemInfo.memory.used_pct)}%`);
                return alerts.length ? alerts.map((a, i)=>(<div key={i} className="p-2 bg-yellow-50 border border-yellow-200 rounded">{a}</div>)) : <div className="text-text/60">No alerts</div>;
              })()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function GPUSparklines({ systemInfo, gpuSeries }){
  const keys = (systemInfo.gpus||[]).map(g => String(g.index ?? 0));
  return (
    <div className="space-y-2">
      {keys.map(k => {
        const arr = gpuSeries[k] || [];
        return (
          <div key={k} className="flex items-center gap-2">
            <div className="w-16 text-xs text-text/70">GPU {k}</div>
            <Sparkline data={arr} color="#2563eb" />
            <div className="text-xs text-text/70 w-10 text-right">{arr.length? Math.round(arr[arr.length-1]) : 0}%</div>
          </div>
        );
      })}
      {keys.length===0 && <div className="text-sm text-text/60">No GPUs</div>}
    </div>
  );
}

function GPUTemperatureHeatmap({ systemInfo }){
  const temps = (systemInfo.gpus||[]).map(g => g.temperature_gpu_c||0);
  const color = (t)=> t>=85? '#ef4444' : t>=75? '#f59e0b' : '#22c55e';
  return (
    <div className="flex flex-wrap gap-2">
      {temps.map((t,i)=>(
        <div key={i} title={`GPU ${i}: ${t} °C`} className="w-8 h-8 rounded" style={{ background: color(t) }} />
      ))}
      {temps.length===0 && <div className="text-sm text-text/60">No data</div>}
    </div>
  );
}

function GPUPowerMeters({ systemInfo, powerSeries = {} }){
  const gpus = systemInfo.gpus||[];
  return (
    <div className="space-y-3">
      {gpus.map((g,i)=>{
        const p = g.power_draw_w ?? 0; const lim = g.power_limit_w || Math.max(p, 1);
        const pct = Math.min(100, Math.round((p/lim)*100));
        const k = String(g.index ?? i);
        const powerHistory = powerSeries[k] || [];
        const isDgxSpark = g.is_dgx_spark || false;
        return (
          <div key={i} className="text-xs space-y-1">
            <div className="flex justify-between mb-1">
              <span className="flex items-center gap-1">
                GPU {g.index ?? i}
                {isDgxSpark && <span className="text-[10px] bg-primary/20 text-primary px-1 rounded">DGX</span>}
              </span>
              <span>{p? p.toFixed(0):'-'}W / {lim? lim.toFixed(0):'-'}W</span>
            </div>
            <div className="w-full bg-muted rounded h-2">
              <div className="h-2 rounded" style={{ width: pct+'%', background: pct>85? '#ef4444' : pct>70? '#f59e0b' : '#22c55e' }} />
            </div>
            {powerHistory.length > 0 && (
              <div className="flex items-center gap-2 pt-1">
                <span className="text-[10px] text-text/50 w-12">History</span>
                <Sparkline
                  data={powerHistory}
                  color={pct>85? '#ef4444' : pct>70? '#f59e0b' : '#22c55e'}
                  width={180}
                  height={24}
                />
              </div>
            )}
          </div>
        );
      })}
      {gpus.length===0 && <div className="text-sm text-text/60">No GPUs</div>}
    </div>
  );
}

function NetworkSpark({ netSeries }){
  return (
    <div className="space-y-3">
      <div>
        <div className="text-xs text-text/70 mb-1">RX</div>
        <Sparkline data={netSeries.rx||[]} color="#16a34a" formatter={(v)=>formatRate(v)} />
      </div>
      <div>
        <div className="text-xs text-text/70 mb-1">TX</div>
        <Sparkline data={netSeries.tx||[]} color="#f59e0b" formatter={(v)=>formatRate(v)} />
      </div>
    </div>
  );
}

function formatRate(bps){
  const K=1024,M=K*1024,G=M*1024;
  if (bps>=G) return (bps/G).toFixed(1)+' GB/s';
  if (bps>=M) return (bps/M).toFixed(1)+' MB/s';
  if (bps>=K) return (bps/K).toFixed(1)+' KB/s';
  return (bps||0).toFixed(0)+' B/s';
}

function Sparkline({ data=[], color="#2563eb", width=240, height=32, formatter }){
  const w = width, h = height, pad=2;
  const max = Math.max(1, ...data);
  const min = Math.min(0, ...data);
  const pts = data.map((v,i)=>{
    const x = pad + (i/(Math.max(1,data.length-1))) * (w-2*pad);
    const y = pad + (1 - (v-min)/(max-min || 1)) * (h-2*pad);
    return `${x},${y}`;
  }).join(' ');
  const last = data[data.length-1]||0;
  return (
    <div className="flex items-center gap-2">
      <svg width={w} height={h} className="block">
        <polyline fill="none" stroke={color} strokeWidth="2" points={pts} />
      </svg>
      {formatter && <div className="text-xs text-text/70 w-16">{formatter(last)}</div>}
    </div>
  );
}

// Secondary pages
function HardwareStorage({ systemInfo }){
  const [usage, setUsage] = React.useState(null);
  const [io, setIo] = React.useState(null);
  React.useEffect(()=>{ (async()=>{ try{ const res = await fetch(`${API_BASE}/storage/usage`); setUsage(await res.json()); }catch{} })(); }, []);
  React.useEffect(()=>{
    let t = null;
    const poll = async () => { try { const r = await fetch(`${API_BASE}/system/io`); setIo(await r.json()); } catch {} t = setTimeout(poll, 3000); };
    poll();
    return ()=>{ if (t) clearTimeout(t); };
  }, []);
  const fmt = (b) => `${(b/1e9).toFixed(1)} GB`;
  return (
    <div className="space-y-4">
      <h1 className="text-3xl font-bold">Storage</h1>
      <div className="bg-surface p-6 rounded-lg border border-border">
        <div className="text-sm text-text/70 mb-2">Disks</div>
        <div className="space-y-2">
          {(systemInfo.disks || []).map((d, i) => (
            <div key={i}>
              <div className="flex justify-between text-xs text-text/70"><span>{d.path}</span><span>{d.used_gib} / {d.total_gib} GiB</span></div>
              <div className="w-full bg-muted rounded h-2"><div className={`h-2 rounded ${ (d.used_pct||0) > 85 ? 'bg-red-500' : (d.used_pct||0) > 70 ? 'bg-yellow-500' : 'bg-green-500' }`} style={{ width: `${Math.min(100, Math.max(0, d.used_pct ?? 0))}%` }} /></div>
            </div>
          ))}
          {(!systemInfo.disks || systemInfo.disks.length === 0) && (<div className="text-xs text-text/60">No disk info</div>)}
        </div>
      </div>
      <div className="bg-surface p-6 rounded-lg border border-border">
        <div className="text-sm font-semibold mb-2">I/O Profiling</div>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div><div className="text-text/70">Reads/s</div><div className="font-semibold">{io ? Math.round(io.reads_per_sec) : '-'}</div></div>
          <div><div className="text-text/70">Writes/s</div><div className="font-semibold">{io ? Math.round(io.writes_per_sec) : '-'}</div></div>
        </div>
      </div>
      <div className="bg-surface p-6 rounded-lg border border-border">
        <div className="text-sm font-semibold mb-2">Directories</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          {usage && Object.entries(usage.usage||{}).map(([k,v]) => (
            <div key={k} className="p-3 border border-border rounded">
              <div className="text-text/70">{k}</div>
              <div className="font-semibold">{fmt(v.total_bytes)}</div>
              <div className="text-[11px] text-text/60 truncate" title={v.path}>{v.path}</div>
            </div>
          ))}
        </div>
        {usage && (
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-sm font-semibold mb-1">Largest Files</div>
              <div className="text-xs max-h-48 overflow-auto">
                {Object.entries(usage.usage||{}).flatMap(([k,v]) => (v.largest_files||[]).slice(0,3).map(it => ({...it, scope:k}))).sort((a,b)=>b.size_bytes-a.size_bytes).slice(0,10).map((it,i)=>(
                  <div key={i} className="flex justify-between gap-2"><span className="truncate">[{it.scope}] {it.file}</span><span className="text-text/70">{fmt(it.size_bytes)}</span></div>
                ))}
              </div>
            </div>
            <div>
              <div className="text-sm font-semibold mb-1">Cleanup Recommendations</div>
              <div className="text-xs">
                {(usage.suggestions||[]).map((s,i)=>(<div key={i} className="flex justify-between gap-2"><span className="truncate">{s.action} — {s.target}</span><span className="text-text/70">{s.reason||''}</span></div>))}
                {(usage.suggestions||[]).length===0 && <div className="text-text/60">No suggestions</div>}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function HardwareNetwork({ systemInfo }){
  return (
    <div className="space-y-4">
      <h1 className="text-3xl font-bold">Network</h1>
      <div className="bg-surface p-6 rounded-lg border border-border">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-text/70">Receive</div>
            <div className="font-semibold">{formatRate(systemInfo.net?.rx_rate_bps || 0)}</div>
          </div>
          <div>
            <div className="text-text/70">Transmit</div>
            <div className="font-semibold">{formatRate(systemInfo.net?.tx_rate_bps || 0)}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SettingsProfile(){
  return (
    <div className="space-y-4">
      <h1 className="text-3xl font-bold">Profile</h1>
      <div className="bg-surface p-6 rounded-lg border border-border text-sm">Coming soon</div>
    </div>
  );
}
function SettingsTeams(){
  return (
    <div className="space-y-4">
      <h1 className="text-3xl font-bold">Teams</h1>
      <div className="bg-surface p-6 rounded-lg border border-border text-sm">Coming soon</div>
    </div>
  );
}
function SettingsBilling(){
  return (
    <div className="space-y-4">
      <h1 className="text-3xl font-bold">Billing</h1>
      <div className="bg-surface p-6 rounded-lg border border-border text-sm">Coming soon</div>
    </div>
  );
}

function AdminPartitions() {
  const [cfg, setCfg] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [form, setForm] = React.useState({});
  const [sys, setSys] = React.useState(null);
  const [presetSel, setPresetSel] = React.useState('');
  const [modelForRec, setModelForRec] = React.useState('');
  const [batchForRec, setBatchForRec] = React.useState(8);

  React.useEffect(() => {
    (async () => {
      try {
        const data = await api.getPartitionConfig();
        setCfg(data);
        try { setSys(await api.getSystemInfo()); } catch {}
        setLoading(false);
      } catch (e) {
        setError(String(e));
        setLoading(false);
      }
    })();
  }, []);

  const handleSubmit = async (gpuIndex) => {
    setError('');
    try {
      const payload = { gpu_index: gpuIndex, enable_mig: true, config: form[gpuIndex] || {} };
      const res = await api.applyPartitionConfig(payload);
      alert('Request accepted: ' + (res.note || ''));
    } catch (e) {
      setError(String(e));
    }
  };

  const totalInstances = (gpuIndex) => {
    const c = form[gpuIndex] || {};
    return Object.values(c).reduce((a, v) => a + (parseInt(v)||0), 0);
  };

  const profileMem = (p) => ({
    '1g.5gb': 5, '2g.10gb': 10, '3g.20gb': 20, '4g.20gb': 20, '7g.40gb': 40,
    '1g.10gb': 10, '2g.20gb': 20, '3g.40gb': 40, '7g.80gb': 80,
  }[p] || 0);

  const applyPreset = (gpuIndex, key) => {
    if (!key) return;
    const map = {
      small4: { '1g.5gb': 4 },
      med2: { '3g.20gb': 2 },
      large1: { '7g.40gb': 1 },
      mixed: { '3g.20gb': 1, '1g.5gb': 2 },
    };
    setForm(prev => ({ ...prev, [gpuIndex]: map[key] }));
  };

  const recommend = async (gpuIndex) => {
    try {
      const res = await fetch(`${API_BASE}/gpu/partition/recommend`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model_id: modelForRec || undefined, batch_size: parseInt(batchForRec)||8, training: true })});
      const out = await res.json();
      if (out.config) setForm(prev => ({ ...prev, [gpuIndex]: out.config }));
    } catch (e) { alert('Recommend failed'); }
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div className="text-red-600">{error}</div>;
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">GPU Partitioning</h1>
      {!cfg?.admin_enabled && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded">Admin operations disabled on server. Viewing topology only.</div>
      )}
      {(cfg?.partitions?.gpus || []).map((g) => (
        <div key={g.index} className="bg-surface p-6 rounded-lg shadow-md border border-border">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-semibold">GPU {g.index} — {g.name}</div>
              <div className="text-sm text-text/70">UUID: {g.uuid}</div>
              <div className="text-sm text-text/70">MIG Mode: {g.mig_mode}</div>
            </div>
          </div>
          <div className="mt-3">
            <p className="text-sm text-text/70 mb-2">Instances</p>
            <div className="space-y-2">
              {(g.instances || []).map((inst, i) => (
                <div key={i} className="p-2 bg-muted rounded border border-border text-sm flex justify-between">
                  <div>{inst.profile} — Device {inst.device_id}</div>
                  <div className="text-gray-500">{inst.uuid?.slice(0,22)}...</div>
                </div>
              ))}
              {(!g.instances || g.instances.length === 0) && (
                <div className="text-xs text-gray-500">No MIG instances detected</div>
              )}
            </div>
          </div>
          <div className="mt-4">
            <p className="text-sm font-semibold text-gray-700 mb-2">Configure (dry‑run)</p>
            <div className="mb-2 flex items-center gap-2 text-xs">
              <span>Total instances: {totalInstances(g.index)}/4</span>
              {totalInstances(g.index) > 4 && <span className="text-red-600">Limit exceeded (max 4)</span>}
            </div>
            {/* Visual partition designer: memory bar */}
            <div className="mb-3">
              <div className="text-xs text-text/70 mb-1">GPU Memory</div>
              <div className="w-full h-3 bg-muted rounded overflow-hidden border border-border">
                {Object.entries(form[g.index]||{}).flatMap(([p,c]) => Array.from({length: parseInt(c)||0}).map((_,i)=>({p, i}))).map((it,idx)=>{
                  const totalGiB = Math.max(1, Math.round(((sys?.gpus||[])[g.index]?.memory_total_mib||40000)/1024));
                  const w = Math.min(100, (profileMem(it.p)/totalGiB)*100);
                  const colors = ['#22c55e','#3b82f6','#f59e0b','#ef4444'];
                  return <div key={idx} style={{ width: `${w}%`, height:'100%', background: colors[idx%colors.length], display:'inline-block' }} title={`${it.p}`} />;
                })}
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {(cfg.supported_profiles || []).map((p) => (
                <div key={p} className="flex items-center gap-2">
                  <label className="text-sm text-gray-700 w-24">{p}</label>
                  <input type="number" min="0" className="flex-1 px-2 py-1 border border-border rounded bg-surface" value={form[g.index]?.[p] || ''} onChange={(e)=>setForm((prev)=>({ ...prev, [g.index]: { ...(prev[g.index]||{}), [p]: e.target.value ? parseInt(e.target.value) : undefined } }))} />
                </div>
              ))}
            </div>
            <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
              <div>
                <label className="text-sm">Presets</label>
                <select className="w-full border border-border rounded px-3 py-2 bg-surface" value={presetSel} onChange={e=>{ setPresetSel(e.target.value); applyPreset(g.index, e.target.value); }}>
                  <option value="">Choose…</option>
                  <option value="small4">4x Small Jobs (4x 1g.5gb)</option>
                  <option value="med2">2x Medium + Inference (2x 3g.20gb)</option>
                  <option value="large1">1x Large Training (1x 7g.40gb)</option>
                  <option value="mixed">Mixed Workload (1x 3g.20gb + 2x 1g.5gb)</option>
                </select>
              </div>
              <div>
                <label className="text-sm">Auto-size by Model</label>
                <input className="w-full border border-border rounded px-3 py-2 bg-surface" placeholder="model id (local)" value={modelForRec} onChange={e=>setModelForRec(e.target.value)} />
                <div className="text-[11px] text-text/60 mt-1">Optional: leave empty and we’ll assume params from job.</div>
              </div>
              <div className="flex items-end gap-2">
                <label className="text-sm">Batch<input type="number" className="w-24 ml-2 border border-border rounded px-2 py-1 bg-surface" value={batchForRec} onChange={e=>setBatchForRec(parseInt(e.target.value)||8)} /></label>
                <button onClick={()=>recommend(g.index)} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Recommend</button>
              </div>
            </div>
            <div className="mt-3">
              <button onClick={()=>handleSubmit(g.index)} className={`px-4 py-2 rounded ${cfg.admin_enabled ? 'bg-primary text-on-primary hover:brightness-110' : 'bg-muted text-text/60'}`} disabled={!cfg.admin_enabled || totalInstances(g.index) > 4}>
                Apply Configuration
              </button>
              {totalInstances(g.index) > 4 && <div className="text-xs text-red-600 mt-1">Cannot apply: total instances exceed maximum of 4 per GPU.</div>}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// Shared quick actions and right-side panels
function ActivityNotifications({ jobs, systemInfo }){
  return (
    <div className="space-y-4">
      <div className="bg-surface p-4 rounded-lg border border-border">
        <div className="text-sm font-semibold mb-2">Activity Feed</div>
        <div className="space-y-2 max-h-[420px] overflow-auto">
          {(jobs||[]).slice().reverse().slice(0,20).map(j => (
            <div key={j.id} className="p-2 bg-muted rounded border border-border text-sm">
              <div className="flex justify-between"><div className="font-semibold truncate mr-2">{j.name}</div><div className="text-xs text-text/60">{j.status}</div></div>
              <div className="text-xs text-text/70">{j.framework}</div>
            </div>
          ))}
          {(!jobs || jobs.length===0) && <div className="text-sm text-text/60">No recent activity</div>}
        </div>
      </div>
      <div className="bg-surface p-4 rounded-lg border border-border">
        <div className="text-sm font-semibold mb-2">Notifications</div>
        <div className="space-y-2 text-sm">
          {(() => {
            const alerts = [];
            (systemInfo.gpus||[]).forEach((g,i)=>{
              if ((g.temperature_gpu_c||0) >= 80) alerts.push(`GPU ${g.index ?? i} high temperature: ${g.temperature_gpu_c} °C`);
              if ((g.utilization_gpu_pct||0) >= 95) alerts.push(`GPU ${g.index ?? i} high utilization: ${g.utilization_gpu_pct}%`);
            });
            if ((systemInfo.memory?.used_pct||0) >= 90) alerts.push(`System RAM high: ${Math.round(systemInfo.memory.used_pct)}%`);
            return alerts.length ? alerts.map((a, i)=>(<div key={i} className="p-2 bg-yellow-50 border border-yellow-200 rounded">{a}</div>)) : <div className="text-text/60">No alerts</div>;
          })()}
        </div>
      </div>
    </div>
  );
}

function PageWithSidebars({ children, onNavigate, jobs, systemInfo }){
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-9 space-y-4">
        {children}
      </div>
      <div className="col-span-12 lg:col-span-3 space-y-4">
        <ActivityNotifications jobs={jobs} systemInfo={systemInfo} />
      </div>
    </div>
  );
}

function SecondaryNav({ currentPage, onNavigate }){
  const groups = [
    { name: 'Training', items: [
      { key: 'jobs', label: 'Jobs' },
      { key: 'experiments', label: 'Experiments' },
      { key: 'pipelines', label: 'Pipelines' },
      { key: 'models', label: 'Models' },
      { key: 'datasets', label: 'Datasets' },
      { key: 'labeling', label: 'Labeling' },
      { key: 'builder', label: 'Builder' },
    ]},
    { name: 'Hardware', items: [
      { key: 'hardware_gpus', label: 'GPUs' },
      { key: 'hardware_storage', label: 'Storage' },
      { key: 'hardware_network', label: 'Network' },
    ]},
    { name: 'Settings', items: [
      { key: 'settings_profile', label: 'Profile' },
      { key: 'settings_teams', label: 'Teams' },
      { key: 'settings_billing', label: 'Billing' },
    ]},
  ];
  const isActive = (key) => currentPage === key;
  return (
    <div className="bg-surface border-b border-border">
      <div className="max-w-7xl mx-auto px-4 py-2 flex flex-wrap gap-6">
        {groups.map((g) => (
          <div key={g.name} className="flex items-center gap-2 flex-wrap">
            <div className="text-xs text-text/60 uppercase tracking-wide mr-1">{g.name}</div>
            {g.items.map(it => (
              <button key={it.key} onClick={()=>onNavigate(it.key)} className={`px-3 py-1 rounded text-sm border ${isActive(it.key)?'bg-primary text-on-primary border-primary':'border-border hover:bg-muted'}`}>{it.label}</button>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// Persistent collapsible sidebar
function Sidebar({ collapsed, setCollapsed, current, onNavigate }){
  const items = [
    { key: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={18} /> },
    { key: 'jobs', label: 'Jobs', icon: <Activity size={18} /> },
    { key: 'experiments', label: 'Experiments', icon: <List size={18} /> },
    { key: 'pipelines', label: 'Pipelines', icon: <List size={18} /> },
    { key: 'models', label: 'Models', icon: <BarChart3 size={18} /> },
    { key: 'datasets', label: 'Datasets', icon: <Database size={18} /> },
    { key: 'labeling', label: 'Labeling', icon: <List size={18} /> },
    { key: 'profile', label: 'Profile', icon: <User size={18} /> },
    { key: 'docs', label: 'Docs', icon: <List size={18} /> },
    { key: 'builder', label: 'Builder', icon: <Monitor size={18} /> },
    { key: 'wizard', label: 'Quick Start', icon: <List size={18} /> },
    { key: 'admin', label: 'Admin', icon: <Settings size={18} /> },
  ];
  return (
    <aside className={`${collapsed? 'w-16':'w-60'} bg-surface border-r border-border transition-all duration-200 flex flex-col`}>
      <div className="h-14 flex items-center justify-between px-3 border-b border-border">
        <div className={`font-bold ${collapsed?'text-center w-full':'truncate'}`}>{collapsed? 'DT' : 'DGX Trainer'}</div>
        {!collapsed && (
          <button className="p-1 rounded hover:bg-muted" onClick={()=>setCollapsed(true)} title="Collapse">
            <ChevronLeft size={16} />
          </button>
        )}
      </div>
      <nav className="flex-1 py-3">
        {items.map(it => (
          <button key={it.key} onClick={()=>onNavigate(it.key)} className={`w-full flex items-center gap-3 px-3 py-2 text-sm hover:bg-muted ${current===it.key?'bg-muted font-semibold':'text-text'}`} title={collapsed ? it.label : undefined}>
            <span className="shrink-0">{it.icon}</span>
            {!collapsed && <span className="truncate">{it.label}</span>}
          </button>
        ))}
      </nav>
      <div className="p-3 border-t border-border">
        {collapsed ? (
          <button className="w-full p-2 rounded hover:bg-muted" onClick={()=>setCollapsed(false)} title="Expand">
            <ChevronRight size={16} />
          </button>
        ) : (
          <button className="w-full px-3 py-2 rounded border border-border hover:bg-muted text-sm" onClick={()=>setCollapsed(true)}>Collapse</button>
        )}
      </div>
    </aside>
  );
}

// Header with breadcrumbs and profile menu
function HeaderBar({ breadcrumbs, onSearch, children }){
  return (
    <header className="h-14 bg-surface border-b border-border flex items-center justify-between px-4">
      <div className="flex items-center gap-3 min-w-0">
        <nav className="text-sm text-text/70 truncate">{breadcrumbs}</nav>
      </div>
      <div className="flex items-center gap-2">
        <button className="px-2 py-1 rounded border border-border hover:bg-muted" onClick={onSearch} title="Search (Cmd/Ctrl+K)"><Search size={16} /></button>
        {children}
      </div>
    </header>
  );
}

function Breadcrumbs({ currentPage, modelView, onNavigate }){
  const items = [{ key:'dashboard', label:'Dashboard' }];
  const add = (key,label)=>items.push({ key,label });
  if (currentPage !== 'dashboard') {
    if (currentPage === 'models') {
      add('models','Models');
      if (modelView?.id) add('model-detail','Detail');
      if (modelView?.compareIds?.length) add('model-compare','Compare');
    } else if (currentPage === 'jobs') {
      add('jobs','Jobs');
    } else if (currentPage === 'experiments') {
      add('experiments','Experiments');
    } else if (currentPage === 'pipelines') {
      add('pipelines','Pipelines');
    } else if (currentPage === 'create') {
      add('jobs','Jobs'); add('create','New');
    } else if (currentPage === 'wizard') {
      add('wizard','Quick Start');
    } else if (currentPage === 'datasets') {
      add('datasets','Datasets');
    } else if (currentPage === 'hpo') {
      add('jobs','Jobs'); add('hpo','HPO');
    } else if (currentPage === 'labeling') {
      add('datasets','Datasets'); add('labeling','Labeling');
    } else if (currentPage === 'builder') {
      add('builder','Builder');
    } else if (currentPage === 'admin') {
      add('admin','Admin');
    } else if (currentPage === 'hardware_gpus') {
      add('hardware','Hardware'); add('hardware_gpus','GPUs');
    } else if (currentPage === 'hardware_storage') {
      add('hardware','Hardware'); add('hardware_storage','Storage');
    } else if (currentPage === 'hardware_network') {
      add('hardware','Hardware'); add('hardware_network','Network');
    } else if (currentPage === 'settings_profile') {
      add('settings','Settings'); add('settings_profile','Profile');
    } else if (currentPage === 'settings_teams') {
      add('settings','Settings'); add('settings_teams','Teams');
    } else if (currentPage === 'settings_billing') {
      add('settings','Settings'); add('settings_billing','Billing');
    } else if (currentPage === 'docs') {
      add('docs','Documentation');
    }
  }
  const pageKeys = new Set(['dashboard','jobs','experiments','pipelines','models','datasets','admin','wizard','create','builder','hpo','labeling']);
  return (
    <ol className="flex items-center gap-2">
      {items.map((it,i)=> (
        <li key={i} className="flex items-center gap-2">
          <button
            onClick={()=> { if (i===items.length-1) return; const k = pageKeys.has(it.key) ? it.key : 'dashboard'; onNavigate(k); }}
            className={`text-sm ${i===items.length-1? 'text-text font-semibold' : 'text-text/70 hover:text-text'} truncate`}
          >
            {it.label}
          </button>
          {i<items.length-1 && <span className="text-text/40">/</span>}
        </li>
      ))}
    </ol>
  );
}

function UserMenu(){
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);
  React.useEffect(()=>{
    const onDoc = (e)=>{ if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return ()=> document.removeEventListener('mousedown', onDoc);
  },[]);
  return (
    <div className="relative" ref={ref}>
      <button onClick={()=>setOpen(o=>!o)} className="px-2 py-1 rounded border border-border hover:bg-muted flex items-center gap-2">
        <User size={16} />
        <span className="hidden sm:inline text-sm">User</span>
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-40 bg-surface border border-border rounded shadow-md text-sm z-10">
          <button className="w-full text-left px-3 py-2 hover:bg-muted">Profile</button>
          <button className="w-full text-left px-3 py-2 hover:bg-muted">Settings</button>
          <button className="w-full text-left px-3 py-2 hover:bg-muted text-danger">Sign out</button>
        </div>
      )}
    </div>
  );
}
