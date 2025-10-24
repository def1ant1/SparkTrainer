import React, { useState, useEffect } from 'react';
import { Camera, Activity, Cpu, Database, Plus, List, Settings } from 'lucide-react';

// API Service
// Use Vite env var if provided (e.g., VITE_API_URL=http://backend:5000), otherwise rely on relative '/api'.
const API_BASE = (import.meta.env && import.meta.env.VITE_API_URL)
  ? `${String(import.meta.env.VITE_API_URL).replace(/\/$/, '')}/api`
  : '/api';

const api = {
  getSystemInfo: () => fetch(`${API_BASE}/system/info`).then(r => r.json()),
  getPartitions: () => fetch(`${API_BASE}/gpu/partitions`).then(r => r.json()),
  getPartitionConfig: () => fetch(`${API_BASE}/gpu/partition/config`).then(r => r.json()),
  applyPartitionConfig: (payload) => fetch(`${API_BASE}/gpu/partition/apply`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then(async r => { if(!r.ok) { const t = await r.text(); throw new Error(t || r.statusText); } return r.json(); }),
  getFrameworks: () => fetch(`${API_BASE}/frameworks`).then(r => r.json()),
  getJobs: () => fetch(`${API_BASE}/jobs`).then(r => r.json()),
  getJob: (id) => fetch(`${API_BASE}/jobs/${id}`).then(r => r.json()),
  createJob: (data) => fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  }).then(r => r.json()),
  cancelJob: (id) => fetch(`${API_BASE}/jobs/${id}/cancel`, {
    method: 'POST'
  }).then(r => r.json()),
  getModels: () => fetch(`${API_BASE}/models`).then(r => r.json())
};

// Dashboard Component
const Dashboard = ({ onNavigate, systemInfo, partitions }) => {
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
      <h1 className="text-3xl font-bold text-gray-900">DGX AI Trainer Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">GPUs Available</p>
              <p className="text-2xl font-bold text-blue-600">{systemInfo.gpus?.length || 0}</p>
            </div>
            <Cpu className="text-blue-500" size={32} />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Running Jobs</p>
              <p className="text-2xl font-bold text-green-600">{systemInfo.jobs_running || 0}</p>
            </div>
            <Activity className="text-green-500" size={32} />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Queued Jobs</p>
              <p className="text-2xl font-bold text-yellow-600">{systemInfo.jobs_queued || 0}</p>
            </div>
            <List className="text-yellow-500" size={32} />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Saved Models</p>
              <p className="text-2xl font-bold text-purple-600">{systemInfo.models_available || 0}</p>
            </div>
            <Database className="text-purple-500" size={32} />
          </div>
        </div>

        {/* GPU Memory Summary */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex-1 mr-4">
              <p className="text-sm text-gray-600">GPU Memory</p>
              <div className="mt-2">
                {(() => {
                  const total = systemInfo.memory_total_mib ?? 0;
                  const used = systemInfo.memory_used_mib ?? 0;
                  const pct = systemInfo.memory_used_pct ?? (total ? (used/total)*100 : 0);
                  return (
                    <>
                      <div className="w-full bg-gray-200 rounded h-3">
                        <div className={`${colorFor(pct)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
                      </div>
                      <div className="text-xs text-gray-600 mt-1">{fmtGiB(used)} / {fmtGiB(total)} GiB</div>
                    </>
                  );
                })()}
              </div>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-gray-900">{Math.round(systemInfo.memory_used_pct ?? 0)}%</p>
            </div>
          </div>
        </div>

        {/* System RAM Summary */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex-1 mr-4">
              <p className="text-sm text-gray-600">System RAM</p>
              <div className="mt-2">
                {(() => {
                  const total = systemInfo.memory?.total_mib ?? 0;
                  const used = systemInfo.memory?.used_mib ?? 0;
                  const pct = systemInfo.memory?.used_pct ?? (total ? (used/total)*100 : 0);
                  return (
                    <>
                      <div className="w-full bg-gray-200 rounded h-3">
                        <div className={`${colorFor(pct)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
                      </div>
                      <div className="text-xs text-gray-600 mt-1">{fmtGiB(used)} / {fmtGiB(total)} GiB</div>
                    </>
                  );
                })()}
              </div>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-gray-900">{Math.round(systemInfo.memory?.used_pct ?? 0)}%</p>
            </div>
          </div>
        </div>

        {/* GPU Allocations Summary */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">GPU Allocations</p>
              {(() => {
                const gpus = partitions?.gpus || [];
                const gpuAlloc = gpus.filter(g => (g.allocated_by_jobs||[]).length>0).length;
                const migTotal = gpus.flatMap(g => (g.instances||[])).length;
                const migAlloc = gpus.flatMap(g => (g.instances||[])).filter(i => (i.allocated_by_jobs||[]).length>0).length;
                return (
                  <p className="text-2xl font-bold text-gray-900">{gpuAlloc} GPU, {migAlloc}/{migTotal} MIG</p>
                );
              })()}
            </div>
          </div>
        </div>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
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
                <div key={idx} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="font-semibold">{gpu.name}</p>
                        <span className="px-2 py-0.5 text-xs rounded-full bg-blue-100 text-blue-800 border border-blue-200">{allocCount} alloc</span>
                      </div>
                      <p className="text-sm text-gray-600">GPU {gpu.index ?? idx}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600">{fmtGiB(usedMiB)} / {fmtGiB(totalMiB)} GiB</p>
                      <p className="font-semibold">{Math.round(pct)}%</p>
                    </div>
                  </div>
                  <div className="mt-3">
                    <div className="w-full bg-gray-200 rounded h-3">
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
                      <div className="p-3 bg-white rounded border"><p className="text-gray-600">Used</p><p className="font-semibold">{fmtGiB(usedMiB)} GiB</p></div>
                      <div className="p-3 bg-white rounded border"><p className="text-gray-600">Free</p><p className="font-semibold">{fmtGiB(freeMiB)} GiB</p></div>
                      <div className="p-3 bg-white rounded border"><p className="text-gray-600">Total</p><p className="font-semibold">{fmtGiB(totalMiB)} GiB</p></div>
                      {gpu.utilization_gpu_pct != null && (
                        <div className="p-3 bg-white rounded border"><p className="text-gray-600">Utilization</p><p className="font-semibold">{gpu.utilization_gpu_pct}%</p></div>
                      )}
                      {gpu.temperature_gpu_c != null && (
                        <div className="p-3 bg-white rounded border"><p className="text-gray-600">Temperature</p><p className="font-semibold">{gpu.temperature_gpu_c} °C</p></div>
                      )}
                    </div>
                  )}
                  {expanded[idx] && (
                    <div className="mt-4">
                      <p className="text-sm text-gray-600 mb-1">Active Allocations</p>
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
                            <p className="text-sm text-gray-600 mb-1">MIG Instances</p>
                            <div className="space-y-2">
                              {g.instances.map((inst, k) => (
                                <div key={k} className="p-2 bg-white rounded border text-xs flex items-center justify-between">
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
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">CPU</h2>
          <button onClick={() => setCpuExpanded(v => !v)} className="text-blue-600 hover:text-blue-800 text-sm">
            {cpuExpanded ? 'Hide details' : 'Show details'}
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="p-4 bg-gray-50 rounded border">
            <p className="text-sm text-gray-600">Total Usage</p>
            <div className="flex items-center justify-between mt-1">
              <div className="flex-1 mr-3">
                <div className="w-full bg-gray-200 rounded h-3">
                  <div className={`${colorFor(systemInfo.cpu?.total_pct ?? 0)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, systemInfo.cpu?.total_pct ?? 0))}%` }} />
                </div>
              </div>
              <div className="text-sm font-semibold">{fmtPct(systemInfo.cpu?.total_pct)}%</div>
            </div>
          </div>
          <div className="p-4 bg-gray-50 rounded border">
            <p className="text-sm text-gray-600">Cores</p>
            <p className="font-semibold mt-1">{systemInfo.cpu?.count ?? '-'}</p>
          </div>
          <div className="p-4 bg-gray-50 rounded border">
            <p className="text-sm text-gray-600">Load Avg (1,5,15)</p>
            <p className="font-semibold mt-1">{systemInfo.cpu?.load_avg ? systemInfo.cpu.load_avg.map(x=>x.toFixed(2)).join(' / ') : '-'}</p>
          </div>
        </div>
        {cpuExpanded && Array.isArray(systemInfo.cpu?.per_core_pct) && systemInfo.cpu.per_core_pct.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2">
            {systemInfo.cpu.per_core_pct.map((v, i) => (
              <div key={i} className="p-2 bg-white rounded border">
                <div className="text-xs text-gray-600 mb-1">CPU {i}</div>
                <div className="w-full bg-gray-200 rounded h-2">
                  <div className={`${colorFor(v)} h-2 rounded`} style={{ width: `${Math.min(100, Math.max(0, v))}%` }} />
                </div>
                <div className="text-xs text-right mt-1">{fmtPct(v)}%</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Memory & Swap */}
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
        <h2 className="text-xl font-semibold mb-4">Memory</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-gray-600">RAM</p>
            <div className="flex items-center justify-between mt-1">
              <div className="flex-1 mr-3">
                <div className="w-full bg-gray-200 rounded h-3">
                  <div className={`${colorFor(systemInfo.memory?.used_pct ?? 0)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, systemInfo.memory?.used_pct ?? 0))}%` }} />
                </div>
              </div>
              <div className="text-sm font-semibold">{fmtPct(systemInfo.memory?.used_pct)}%</div>
            </div>
            <div className="text-xs text-gray-600 mt-1">{fmtGiB(systemInfo.memory?.used_mib)} / {fmtGiB(systemInfo.memory?.total_mib)} GiB</div>
          </div>
          <div>
            <p className="text-sm text-gray-600">Swap</p>
            <div className="flex items-center justify-between mt-1">
              <div className="flex-1 mr-3">
                <div className="w-full bg-gray-200 rounded h-3">
                  <div className={`${colorFor(systemInfo.swap?.used_pct ?? 0)} h-3 rounded`} style={{ width: `${Math.min(100, Math.max(0, systemInfo.swap?.used_pct ?? 0))}%` }} />
                </div>
              </div>
              <div className="text-sm font-semibold">{fmtPct(systemInfo.swap?.used_pct)}%</div>
            </div>
            <div className="text-xs text-gray-600 mt-1">{fmtGiB(systemInfo.swap?.used_mib)} / {fmtGiB(systemInfo.swap?.total_mib)} GiB</div>
          </div>
        </div>
      </div>

      {/* Network & Disks */}
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
        <h2 className="text-xl font-semibold mb-4">I/O</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 bg-gray-50 rounded border">
            <p className="text-sm text-gray-600">Network</p>
            <div className="mt-2 text-sm">
              <div><span className="text-gray-600">Receive:</span> <span className="font-semibold">{fmtRate(systemInfo.net?.rx_rate_bps)}</span></div>
              <div><span className="text-gray-600">Transmit:</span> <span className="font-semibold">{fmtRate(systemInfo.net?.tx_rate_bps)}</span></div>
            </div>
          </div>
          <div className="p-4 bg-gray-50 rounded border">
            <p className="text-sm text-gray-600 mb-2">Disks</p>
            <div className="space-y-2">
              {(systemInfo.disks || []).map((d, i) => (
                <div key={i}>
                  <div className="flex justify-between text-xs text-gray-600"><span>{d.path}</span><span>{d.used_gib} / {d.total_gib} GiB</span></div>
                  <div className="w-full bg-gray-200 rounded h-2">
                    <div className={`${colorFor(d.used_pct ?? 0)} h-2 rounded`} style={{ width: `${Math.min(100, Math.max(0, d.used_pct ?? 0))}%` }} />
                  </div>
                </div>
              ))}
              {(!systemInfo.disks || systemInfo.disks.length === 0) && (
                <div className="text-xs text-gray-500">No disk info</div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <div className="flex gap-4">
        <button
          onClick={() => onNavigate('create')}
          className="flex-1 bg-blue-600 text-white py-4 px-6 rounded-lg hover:bg-blue-700 transition flex items-center justify-center gap-2 font-semibold"
        >
          <Plus size={20} />
          Create New Training Job
        </button>
        
        <button
          onClick={() => onNavigate('jobs')}
          className="flex-1 bg-gray-600 text-white py-4 px-6 rounded-lg hover:bg-gray-700 transition flex items-center justify-center gap-2 font-semibold"
        >
          <List size={20} />
          View All Jobs
        </button>
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
  
  const handleSubmit = async () => {
    const jobData = {
      name: jobName || `Training Job ${Date.now()}`,
      type: jobType,
      framework: framework,
      config: config
    };
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
        <h1 className="text-3xl font-bold text-gray-900">Create Training Job</h1>
        <button
          onClick={() => onNavigate('dashboard')}
          className="text-blue-600 hover:text-blue-800"
        >
          ← Back to Dashboard
        </button>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 space-y-6">
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
                  : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
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
                  : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
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
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Base Model</label>
            <input
              type="text"
              value={config.model_name || ''}
              onChange={(e) => setConfig({...config, model_name: e.target.value})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              placeholder="resnet18, bert-base-uncased, etc."
            />
          </div>
        )}
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Epochs</label>
            <input
              type="number"
              value={config.epochs}
              onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Batch Size</label>
            <input
              type="number"
              value={config.batch_size}
              onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Learning Rate</label>
            <input
              type="number"
              step="0.0001"
              value={config.learning_rate}
              onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Number of Classes</label>
            <input
              type="number"
              value={config.num_classes}
              onChange={(e) => setConfig({...config, num_classes: parseInt(e.target.value)})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
        
        <button
          onClick={handleSubmit}
          className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition font-semibold"
        >
          Create Training Job
        </button>
      </div>
    </div>
  );
};

// Jobs List Component
const JobsList = ({ onNavigate, partitions }) => {
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  
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
      } catch (error) {
        alert('Failed to cancel job: ' + error.message);
      }
    }
  };
  
  const getStatusColor = (status) => {
    const colors = {
      'queued': 'bg-yellow-100 text-yellow-800',
      'running': 'bg-blue-100 text-blue-800',
      'completed': 'bg-green-100 text-green-800',
      'failed': 'bg-red-100 text-red-800',
      'cancelled': 'bg-gray-100 text-gray-800'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">Training Jobs</h1>
        <div className="flex gap-2">
          <button
            onClick={() => onNavigate('create')}
            className="bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition flex items-center gap-2"
          >
            <Plus size={16} />
            New Job
          </button>
          <button
            onClick={() => onNavigate('dashboard')}
            className="text-blue-600 hover:text-blue-800"
          >
            ← Dashboard
          </button>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Name</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Type</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Framework</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">GPU</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Status</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Created</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {jobs.length > 0 ? (
              jobs.map((job) => (
                <tr key={job.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">{job.name}</td>
                  <td className="px-6 py-4 text-sm text-gray-600 capitalize">{job.type}</td>
                  <td className="px-6 py-4 text-sm text-gray-600 capitalize">{job.framework}</td>
                  <td className="px-6 py-4 text-sm text-gray-600">
                    {(() => {
                      const g = job.gpu;
                      if (!g) return <span className="text-gray-400">auto</span>;
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
                  <td className="px-6 py-4 text-sm text-gray-600">
                    {new Date(job.created).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-sm">
                    <button
                      onClick={() => setSelectedJob(job)}
                      className="text-blue-600 hover:text-blue-800 mr-3"
                    >
                      View
                    </button>
                    {job.status === 'running' && (
                      <button
                        onClick={() => handleCancelJob(job.id)}
                        className="text-red-600 hover:text-red-800"
                      >
                        Cancel
                      </button>
                    )}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="6" className="px-6 py-8 text-center text-gray-500">
                  No jobs found. Create your first training job!
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      {selectedJob && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-3xl w-full max-h-[80vh] overflow-auto p-6">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-2xl font-bold">{selectedJob.name}</h2>
              <button
                onClick={() => setSelectedJob(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Status</p>
                  <p className="font-semibold capitalize">{selectedJob.status}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Framework</p>
                  <p className="font-semibold capitalize">{selectedJob.framework}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Type</p>
                  <p className="font-semibold capitalize">{selectedJob.type}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Created</p>
                  <p className="font-semibold">{new Date(selectedJob.created).toLocaleString()}</p>
                </div>
              </div>
              
              <div>
                <p className="text-sm text-gray-600 mb-2">Configuration</p>
                <pre className="bg-gray-50 p-4 rounded-lg text-xs overflow-auto">
                  {JSON.stringify(selectedJob.config, null, 2)}
                </pre>
              </div>

              <div>
                <p className="text-sm text-gray-600 mb-2">GPU Allocation</p>
                {selectedJob.gpu ? (
                  <div className="p-4 bg-gray-50 rounded-lg text-xs">
                    <div><span className="text-gray-600">Type:</span> <span className="font-semibold">{selectedJob.gpu.type?.toUpperCase()}</span></div>
                    {selectedJob.gpu.gpu_index != null && (<div><span className="text-gray-600">GPU Index:</span> <span className="font-semibold">{selectedJob.gpu.gpu_index}</span></div>)}
                    {selectedJob.gpu.gpu_uuid && (<div><span className="text-gray-600">GPU UUID:</span> <span className="font-semibold">{selectedJob.gpu.gpu_uuid}</span></div>)}
                    {selectedJob.gpu.mig_uuid && (<div><span className="text-gray-600">MIG UUID:</span> <span className="font-semibold">{selectedJob.gpu.mig_uuid}</span></div>)}
                  </div>
                ) : (
                  <div className="text-xs text-gray-500">No explicit GPU/MIG selected (auto)</div>
                )}
              </div>
              
              {selectedJob.logs && (
                <div>
                  <p className="text-sm text-gray-600 mb-2">Logs</p>
                  <pre className="bg-gray-900 text-green-400 p-4 rounded-lg text-xs overflow-auto max-h-64">
                    {selectedJob.logs}
                  </pre>
                </div>
              )}
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
  
  useEffect(() => {
    loadFrameworks();
    // Prefer SSE for live updates; fall back to polling
    let es;
    try {
      es = new EventSource(`${API_BASE}/system/stream`);
      es.onmessage = (e) => {
        try { setSystemInfo(JSON.parse(e.data)); } catch {}
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
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Database className="text-blue-600" size={32} />
              <h1 className="text-2xl font-bold text-gray-900">DGX AI Trainer</h1>
            </div>
            <div className="flex gap-4">
              <button
                onClick={() => setCurrentPage('dashboard')}
                className={`px-4 py-2 rounded-lg transition ${
                  currentPage === 'dashboard'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setCurrentPage('jobs')}
                className={`px-4 py-2 rounded-lg transition ${
                  currentPage === 'jobs'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Jobs
              </button>
              <button
                onClick={() => setCurrentPage('admin')}
                className={`px-4 py-2 rounded-lg transition ${
                  currentPage === 'admin'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Admin
              </button>
            </div>
          </div>
        </div>
      </nav>
      
      <main className="max-w-7xl mx-auto px-4 py-8">
        {currentPage === 'dashboard' && (
          <Dashboard onNavigate={setCurrentPage} systemInfo={systemInfo} partitions={partitions} />
        )}
        {currentPage === 'create' && (
          <CreateJob onNavigate={setCurrentPage} frameworks={frameworks} partitions={partitions} />
        )}
        {currentPage === 'jobs' && (
          <JobsList onNavigate={setCurrentPage} partitions={partitions} />
        )}
        {currentPage === 'admin' && (
          <AdminPartitions />
        )}
      </main>
    </div>
  );
}

function AdminPartitions() {
  const [cfg, setCfg] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [form, setForm] = React.useState({});

  React.useEffect(() => {
    (async () => {
      try {
        const data = await api.getPartitionConfig();
        setCfg(data);
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

  if (loading) return <div>Loading...</div>;
  if (error) return <div className="text-red-600">{error}</div>;
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">GPU Partitioning</h1>
      {!cfg?.admin_enabled && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded">Admin operations disabled on server. Viewing topology only.</div>
      )}
      {(cfg?.partitions?.gpus || []).map((g) => (
        <div key={g.index} className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-semibold">GPU {g.index} — {g.name}</div>
              <div className="text-sm text-gray-600">UUID: {g.uuid}</div>
              <div className="text-sm text-gray-600">MIG Mode: {g.mig_mode}</div>
            </div>
          </div>
          <div className="mt-3">
            <p className="text-sm text-gray-600 mb-2">Instances</p>
            <div className="space-y-2">
              {(g.instances || []).map((inst, i) => (
                <div key={i} className="p-2 bg-gray-50 rounded border text-sm flex justify-between">
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
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {(cfg.supported_profiles || []).map((p) => (
                <div key={p} className="flex items-center gap-2">
                  <label className="text-sm text-gray-700 w-24">{p}</label>
                  <input type="number" min="0" className="flex-1 px-2 py-1 border rounded" value={form[g.index]?.[p] || ''} onChange={(e)=>setForm((prev)=>({ ...prev, [g.index]: { ...(prev[g.index]||{}), [p]: e.target.value ? parseInt(e.target.value) : undefined } }))} />
                </div>
              ))}
            </div>
            <div className="mt-3">
              <button onClick={()=>handleSubmit(g.index)} className={`px-4 py-2 rounded ${cfg.admin_enabled ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-300 text-gray-600'}`} disabled={!cfg.admin_enabled}>
                Apply Configuration
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
