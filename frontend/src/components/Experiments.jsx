import React, { useEffect, useMemo, useState } from 'react';
import GatingMetricsViewer from './GatingMetricsViewer';

export default function ExperimentsPage(){
  const [items, setItems] = useState([]);
  const [q, setQ] = useState('');
  const [favOnly, setFavOnly] = useState(false);
  const [sortBy, setSortBy] = useState('updated');
  const [creating, setCreating] = useState('');
  const [detail, setDetail] = useState(null);
  const [edit, setEdit] = useState({ name:'', description:'', tags:'' });
  const [metrics, setMetrics] = useState(null);
  const [comparing, setComparing] = useState([]);
  const [compareData, setCompareData] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [gatingEnabled, setGatingEnabled] = useState(false);
  const [selectedJobForGating, setSelectedJobForGating] = useState(null);

  const load = async () => {
    try{
      const params = new URLSearchParams();
      if (q) params.set('q', q);
      const res = await fetch(`/api/experiments${params.toString()?`?${params.toString()}`:''}`);
      const j = await res.json();
      setItems(j.items || []);
    }catch{}
  };
  useEffect(()=>{ load(); }, [q]);

  const filtered = useMemo(()=>{
    let arr = items.slice();
    if (favOnly) arr = arr.filter(e => !!e.favorite);
    const key = (e) => sortBy==='runs' ? (e.run_count||0) : sortBy==='name' ? (e.name||'').toLowerCase() : (e.updated||e.created||'');
    return arr.sort((a,b)=> key(a) > key(b) ? -1 : 1);
  }, [items, favOnly, sortBy]);

  const create = async () => {
    const name = creating.trim(); if (!name) return;
    try{
      const res = await fetch('/api/experiments', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
      if (res.ok){ setCreating(''); load(); }
      else alert(await res.text());
    }catch(e){ alert(String(e)); }
  };

  const openDetail = async (id) => {
    try{
      const res = await fetch(`/api/experiments/${encodeURIComponent(id)}`);
      const j = await res.json();
      setDetail(j);
      setEdit({ name: j.name||'', description: j.description||'', tags:(j.tags||[]).join(',') });
      loadMetrics(id);
      checkGatingEnabled(id);
    }catch{}
  };

  const checkGatingEnabled = async (expId) => {
    try{
      const res = await fetch(`/api/experiments/${encodeURIComponent(expId)}/gating/summary`);
      const j = await res.json();
      setGatingEnabled(j.enabled || false);

      // If gating is enabled, find the first job with gating for display
      if (j.enabled) {
        const jobsRes = await fetch(`/api/experiments/${encodeURIComponent(expId)}/jobs`);
        const jobsData = await jobsRes.json();
        const jobWithGating = jobsData.jobs?.find(job => {
          return job.config?.gating?.enabled || job.metadata?.gating_metrics;
        });
        if (jobWithGating) {
          setSelectedJobForGating(jobWithGating.id);
        }
      }
    }catch{
      setGatingEnabled(false);
    }
  };

  const loadMetrics = async (id) => {
    try{
      const res = await fetch(`/api/experiments/${encodeURIComponent(id)}/metrics`);
      const j = await res.json();
      setMetrics(j);
    }catch{
      setMetrics(null);
    }
  };

  const saveDetail = async () => {
    if (!detail) return;
    try{
      const payload = { name: edit.name, description: edit.description, tags: edit.tags.split(',').map(s=>s.trim()).filter(Boolean) };
      const res = await fetch(`/api/experiments/${encodeURIComponent(detail.id)}`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if (res.ok){ openDetail(detail.id); load(); }
      else alert(await res.text());
    }catch(e){ alert(String(e)); }
  };

  const deleteExperiment = async (id) => {
    if (!confirm('Delete this experiment? This will NOT delete the associated jobs/models.')) return;
    try{
      const res = await fetch(`/api/experiments/${encodeURIComponent(id)}`, { method:'DELETE' });
      if (res.ok){
        setDetail(null);
        load();
      } else {
        alert('Delete failed');
      }
    }catch(e){ alert(String(e)); }
  };

  const toggleStar = async (it) => {
    try{
      const res = await fetch(`/api/experiments/${encodeURIComponent(it.id)}/star`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ favorite: !it.favorite }) });
      if (res.ok) load();
    }catch{}
  };

  const compareExperiments = async () => {
    if (comparing.length < 2) {
      alert('Select at least 2 experiments to compare');
      return;
    }
    try{
      const res = await fetch('/api/experiments/compare', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ experiment_ids: comparing })
      });
      const j = await res.json();
      setCompareData(j);
    }catch(e){ alert(String(e)); }
  };

  const toggleCompare = (id) => {
    setComparing(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Experiments</h1>
        <div className="flex gap-2 text-sm">
          {comparing.length > 0 && (
            <>
              <span className="px-3 py-2 bg-accent/10 text-accent rounded border border-accent/30">
                {comparing.length} selected
              </span>
              <button
                className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted"
                onClick={compareExperiments}
              >
                Compare
              </button>
              <button
                className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted"
                onClick={()=>setComparing([])}
              >
                Clear
              </button>
            </>
          )}
          <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Create new experiment" value={creating} onChange={e=>setCreating(e.target.value)} onKeyPress={e=>e.key==='Enter'&&create()} />
          <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={create}>Create</button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
        <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Search experiments..." value={q} onChange={e=>setQ(e.target.value)} />
        <label className="inline-flex items-center gap-2 text-sm px-3 py-2">
          <input type="checkbox" checked={favOnly} onChange={e=>setFavOnly(e.target.checked)} />
          Favorites only
        </label>
        <select className="border border-border rounded px-3 py-2 bg-surface" value={sortBy} onChange={e=>setSortBy(e.target.value)}>
          <option value="updated">Recently Updated</option>
          <option value="runs">Most Runs</option>
          <option value="name">Name (A-Z)</option>
        </select>
        <div className="text-sm text-text/60 px-3 py-2">
          {filtered.length} experiment{filtered.length !== 1 ? 's' : ''}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {filtered.map(e => (
          <div key={e.id} className="bg-surface p-4 rounded border border-border hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <div className="font-semibold text-lg">{e.name}</div>
                <div className="text-xs text-text/60 mt-1">
                  {e.run_count||0} run{(e.run_count||0) !== 1 ? 's' : ''} •
                  Updated {new Date(e.updated || e.created).toLocaleDateString()}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={comparing.includes(e.id)}
                  onChange={() => toggleCompare(e.id)}
                  className="mr-1"
                  title="Select for comparison"
                />
                <button
                  className={`text-lg px-2 py-1 rounded transition-colors ${e.favorite?'bg-yellow-500/20 text-yellow-600 hover:bg-yellow-500/30':'text-text/40 hover:text-yellow-600 hover:bg-yellow-500/10'}`}
                  onClick={()=>toggleStar(e)}
                  title={e.favorite ? 'Remove from favorites' : 'Add to favorites'}
                >
                  {e.favorite?'★':'☆'}
                </button>
              </div>
            </div>

            {e.description && (
              <div className="text-sm text-text/70 mt-2 line-clamp-2 min-h-[2.5rem]">
                {e.description}
              </div>
            )}

            {Array.isArray(e.tags) && e.tags.length>0 && (
              <div className="mt-3 flex flex-wrap gap-1">
                {e.tags.slice(0,4).map((t,i)=>(
                  <span key={i} className="px-2 py-0.5 text-xs bg-muted rounded border border-border text-text/80">
                    {t}
                  </span>
                ))}
                {e.tags.length>4 && (
                  <span className="px-2 py-0.5 text-xs text-text/60">
                    +{e.tags.length-4}
                  </span>
                )}
              </div>
            )}

            <div className="mt-3 pt-3 border-t border-border flex gap-2">
              <button
                className="flex-1 text-sm px-3 py-2 border border-border rounded bg-surface hover:bg-muted transition-colors"
                onClick={()=>openDetail(e.id)}
              >
                View Details
              </button>
              <button
                className="text-sm px-3 py-2 border border-danger/50 rounded text-danger hover:bg-danger/10 transition-colors"
                onClick={()=>deleteExperiment(e.id)}
                title="Delete experiment"
              >
                Delete
              </button>
            </div>
          </div>
        ))}

        {filtered.length === 0 && (
          <div className="col-span-3 text-center py-12 text-text/60">
            <div className="text-4xl mb-2">🔬</div>
            <div className="text-lg">No experiments found</div>
            <div className="text-sm mt-1">Create one to get started</div>
          </div>
        )}
      </div>

      {/* Experiment Detail Modal */}
      {detail && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-surface border border-border rounded max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div className="flex items-center gap-3">
                <div className="text-2xl font-bold">{detail.name}</div>
                {detail.favorite && <span className="text-yellow-600 text-xl">★</span>}
              </div>
              <div className="flex gap-2">
                <button
                  className="px-3 py-2 border border-danger/50 rounded text-danger hover:bg-danger/10"
                  onClick={()=>deleteExperiment(detail.id)}
                >
                  Delete
                </button>
                <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={()=>setDetail(null)}>
                  Close
                </button>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 px-4 pt-2 border-b border-border">
              {['overview', 'metrics', 'runs', ...(gatingEnabled ? ['gating'] : [])].map(tab => (
                <button
                  key={tab}
                  className={`px-4 py-2 rounded-t transition-colors ${
                    activeTab === tab
                      ? 'bg-accent/10 text-accent border-b-2 border-accent'
                      : 'text-text/60 hover:text-text hover:bg-muted'
                  }`}
                  onClick={() => setActiveTab(tab)}
                >
                  {tab === 'gating' ? '🚀 Gating' : tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-auto p-4">
              {activeTab === 'overview' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <label className="md:col-span-1 flex flex-col gap-1">
                      <span className="text-sm font-medium">Name</span>
                      <input
                        className="w-full border border-border rounded px-3 py-2 bg-surface"
                        value={edit.name}
                        onChange={e=>setEdit({...edit, name:e.target.value})}
                      />
                    </label>
                    <label className="md:col-span-2 flex flex-col gap-1">
                      <span className="text-sm font-medium">Description</span>
                      <textarea
                        className="w-full border border-border rounded px-3 py-2 bg-surface h-24 resize-none"
                        value={edit.description}
                        onChange={e=>setEdit({...edit, description:e.target.value})}
                        placeholder="Describe this experiment..."
                      />
                    </label>
                    <label className="md:col-span-3 flex flex-col gap-1">
                      <span className="text-sm font-medium">Tags (comma-separated)</span>
                      <input
                        className="w-full border border-border rounded px-3 py-2 bg-surface"
                        value={edit.tags}
                        onChange={e=>setEdit({...edit, tags:e.target.value})}
                        placeholder="nlp, transformer, bert"
                      />
                    </label>
                  </div>
                  <div className="flex justify-end">
                    <button
                      className="px-4 py-2 bg-accent text-white rounded hover:bg-accent/90"
                      onClick={saveDetail}
                    >
                      Save Changes
                    </button>
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-3 gap-3 pt-4">
                    <div className="bg-muted rounded p-3 border border-border">
                      <div className="text-sm text-text/60">Total Runs</div>
                      <div className="text-2xl font-bold">{detail.runs?.length || 0}</div>
                    </div>
                    <div className="bg-muted rounded p-3 border border-border">
                      <div className="text-sm text-text/60">Created</div>
                      <div className="text-sm font-medium">{new Date(detail.created).toLocaleDateString()}</div>
                    </div>
                    <div className="bg-muted rounded p-3 border border-border">
                      <div className="text-sm text-text/60">Last Updated</div>
                      <div className="text-sm font-medium">{new Date(detail.updated || detail.created).toLocaleDateString()}</div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'metrics' && (
                <div className="space-y-4">
                  {metrics ? (
                    <>
                      <div className="grid grid-cols-4 gap-3">
                        <div className="bg-muted rounded p-3 border border-border">
                          <div className="text-sm text-text/60">Total Jobs</div>
                          <div className="text-2xl font-bold">{metrics.total_jobs}</div>
                        </div>
                        <div className="bg-green-500/10 rounded p-3 border border-green-500/30">
                          <div className="text-sm text-green-600">Completed</div>
                          <div className="text-2xl font-bold text-green-600">{metrics.completed}</div>
                        </div>
                        <div className="bg-blue-500/10 rounded p-3 border border-blue-500/30">
                          <div className="text-sm text-blue-600">Running</div>
                          <div className="text-2xl font-bold text-blue-600">{metrics.running}</div>
                        </div>
                        <div className="bg-red-500/10 rounded p-3 border border-red-500/30">
                          <div className="text-sm text-red-600">Failed</div>
                          <div className="text-2xl font-bold text-red-600">{metrics.failed}</div>
                        </div>
                      </div>

                      {Object.keys(metrics.best_metrics || {}).length > 0 && (
                        <div>
                          <h3 className="text-lg font-semibold mb-3">Best Metrics</h3>
                          <div className="grid grid-cols-2 gap-3">
                            {Object.entries(metrics.best_metrics).map(([metric, data]) => (
                              <div key={metric} className="bg-muted rounded p-3 border border-border">
                                <div className="text-sm text-text/60 capitalize">{metric.replace(/_/g, ' ')}</div>
                                <div className="text-xl font-bold">{typeof data.value === 'number' ? data.value.toFixed(4) : data.value}</div>
                                <div className="text-xs text-text/60 mt-1">Job: {data.job_name || data.job_id?.slice(0, 8)}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {metrics.job_metrics && metrics.job_metrics.length > 0 && (
                        <div>
                          <h3 className="text-lg font-semibold mb-3">Job Metrics</h3>
                          <div className="space-y-2">
                            {metrics.job_metrics.map(jm => (
                              <div key={jm.job_id} className="bg-surface border border-border rounded p-3">
                                <div className="flex justify-between items-start mb-2">
                                  <div>
                                    <div className="font-medium">{jm.name}</div>
                                    <div className="text-xs text-text/60">
                                      {jm.job_id.slice(0, 8)} • {jm.status}
                                    </div>
                                  </div>
                                </div>
                                <div className="grid grid-cols-4 gap-2 text-sm">
                                  {Object.entries(jm.metrics || {}).map(([k, v]) => (
                                    <div key={k}>
                                      <span className="text-text/60">{k}:</span>
                                      <span className="font-medium ml-1">
                                        {typeof v === 'number' ? v.toFixed(4) : v}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-center py-12 text-text/60">
                      <div className="text-2xl mb-2">📊</div>
                      <div>Loading metrics...</div>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'runs' && (
                <div>
                  <div className="text-sm text-text/60 mb-3">
                    {detail.runs?.length || 0} run{(detail.runs?.length || 0) !== 1 ? 's' : ''} in this experiment
                  </div>
                  <div className="border border-border rounded overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-muted border-b border-border">
                        <tr>
                          <th className="text-left px-3 py-2">Job ID</th>
                          <th className="text-left px-3 py-2">Name</th>
                          <th className="text-left px-3 py-2">Status</th>
                          <th className="text-left px-3 py-2">Framework</th>
                          <th className="text-left px-3 py-2">Created</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {(detail.runs||[]).map(r => (
                          <tr key={r.id} className="hover:bg-muted/50">
                            <td className="px-3 py-2 font-mono text-xs">{r.id.slice(0,12)}</td>
                            <td className="px-3 py-2">{r.name}</td>
                            <td className="px-3 py-2">
                              <span className={`px-2 py-0.5 rounded text-xs ${
                                r.status === 'completed' ? 'bg-green-500/20 text-green-600' :
                                r.status === 'running' ? 'bg-blue-500/20 text-blue-600' :
                                r.status === 'failed' ? 'bg-red-500/20 text-red-600' :
                                'bg-muted text-text/60'
                              }`}>
                                {r.status}
                              </span>
                            </td>
                            <td className="px-3 py-2">{r.framework}</td>
                            <td className="px-3 py-2 text-xs">{new Date(r.created).toLocaleString()}</td>
                          </tr>
                        ))}
                        {(!detail.runs || detail.runs.length===0) && (
                          <tr>
                            <td className="px-3 py-8 text-center text-text/60" colSpan={5}>
                              <div className="text-2xl mb-2">🔬</div>
                              <div>No runs yet</div>
                              <div className="text-xs mt-1">Create a job and associate it with this experiment</div>
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {activeTab === 'gating' && (
                <div>
                  {selectedJobForGating ? (
                    <GatingMetricsViewer jobId={selectedJobForGating} api="" />
                  ) : (
                    <div className="text-center py-12 text-text/60">
                      <div className="text-2xl mb-2">🚀</div>
                      <div>No job with gating found in this experiment</div>
                      <div className="text-xs mt-1">Create a job with gating enabled to see metrics here</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Comparison Modal */}
      {compareData && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-surface border border-border rounded max-w-6xl w-full max-h-[90vh] overflow-auto p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold">Experiment Comparison</h2>
              <button
                className="px-3 py-2 border border-border rounded hover:bg-muted"
                onClick={()=>setCompareData(null)}
              >
                Close
              </button>
            </div>

            <div className="space-y-6">
              {/* Experiments Overview */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {compareData.experiments?.map(exp => (
                  <div key={exp.id} className="bg-muted rounded p-4 border border-border">
                    <div className="font-semibold text-lg mb-2">{exp.name}</div>
                    <div className="text-sm space-y-1">
                      <div>Total Jobs: <span className="font-medium">{exp.total_jobs}</span></div>
                      <div>Completed: <span className="font-medium text-green-600">{exp.completed_jobs}</span></div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Metrics Comparison */}
              {Object.keys(compareData.metrics_comparison || {}).length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Metrics Comparison</h3>
                  <div className="space-y-4">
                    {Object.entries(compareData.metrics_comparison).map(([metric, data]) => (
                      <div key={metric} className="bg-surface border border-border rounded p-4">
                        <div className="font-medium mb-3 capitalize">{metric.replace(/_/g, ' ')}</div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          {Object.entries(data.experiments || {}).map(([exp_id, exp_data]) => (
                            <div key={exp_id} className="bg-muted rounded p-3">
                              <div className="text-sm font-medium mb-2">{exp_data.name}</div>
                              <div className="text-xs space-y-1">
                                <div>Mean: <span className="font-mono">{exp_data.mean.toFixed(4)}</span></div>
                                <div>Min: <span className="font-mono">{exp_data.min.toFixed(4)}</span></div>
                                <div>Max: <span className="font-mono">{exp_data.max.toFixed(4)}</span></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
