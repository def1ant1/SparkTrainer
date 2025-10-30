import React, { useEffect, useMemo, useState } from 'react';

// Toast hook for notifications
const useToast = () => {
  return {
    push: (msg) => {
      const { type = 'info', title, message } = msg;
      const colors = { success: '#10b981', error: '#ef4444', info: '#3b82f6', warning: '#f59e0b' };
      const color = colors[type] || colors.info;
      const toast = document.createElement('div');
      toast.style.cssText = `position:fixed;top:20px;right:20px;background:${color};color:white;padding:12px 20px;border-radius:8px;z-index:9999;box-shadow:0 4px 6px rgba(0,0,0,0.2)`;
      toast.textContent = title + (message ? ': ' + message : '');
      document.body.appendChild(toast);
      setTimeout(() => toast.remove(), 3000);
    }
  };
};

export function ModelsPage({ api, onOpen, onCompare, onNavigate }) {
  const [mode, setMode] = useState('models'); // 'models' or 'templates'
  const [items, setItems] = useState([]);
  const [view, setView] = useState('grid');
  const [q, setQ] = useState('');
  const [framework, setFramework] = useState('');
  const [architecture, setArchitecture] = useState('');
  const [sort, setSort] = useState('date');
  const [order, setOrder] = useState('desc');
  const [sel, setSel] = useState({});
  const [size, setSize] = useState('');
  const [license, setLicense] = useState('');
  const [tag, setTag] = useState('');
  const toast = useToast();

  const load = async () => {
    const params = new URLSearchParams();
    if (q) params.set('q', q);
    if (framework) params.set('framework', framework);
    if (architecture) params.set('architecture', architecture);
    if (size) params.set('size', size);
    if (license) params.set('license', license);
    if (tag) params.set('tag', tag);
    params.set('sort', sort); params.set('order', order);
    const data = await api.getModelsRaw(params.toString());
    setItems(data);
  };
  useEffect(() => { load(); }, [q, framework, architecture, sort, order, size, license, tag]);

  const allSelected = useMemo(()=>items.length>0 && items.every(m => sel[m.id]), [items, sel]);
  const toggleAll = () => {
    if (allSelected) setSel({}); else setSel(Object.fromEntries(items.map(m => [m.id, true])));
  };
  const selectedIds = Object.keys(sel).filter(k => sel[k]);

  const bulkDelete = async () => {
    if (!selectedIds.length) {
      toast.push({ type: 'warning', title: 'No models selected' });
      return;
    }
    if (!confirm(`Delete ${selectedIds.length} model${selectedIds.length > 1 ? 's' : ''}? This cannot be undone.`)) return;
    try {
      await api.bulkDeleteModels(selectedIds);
      setSel({});
      load();
      toast.push({ type: 'success', title: `Deleted ${selectedIds.length} model${selectedIds.length > 1 ? 's' : ''}` });
    } catch (e) {
      toast.push({ type: 'error', title: 'Delete failed', message: e.message });
    }
  };

  const deleteModel = async (modelId) => {
    if (!confirm('Delete this model? This cannot be undone.')) return;
    try {
      await api.deleteModel(modelId);
      load();
      toast.push({ type: 'success', title: 'Model deleted' });
    } catch (e) {
      toast.push({ type: 'error', title: 'Delete failed', message: e.message });
    }
  };

  const bulkExport = async () => {
    if (!selectedIds.length) {
      toast.push({ type: 'warning', title: 'No models selected' });
      return;
    }
    const url = api.exportModelsUrl(selectedIds);
    window.open(url, '_blank');
    toast.push({ type: 'info', title: `Exporting ${selectedIds.length} model${selectedIds.length > 1 ? 's' : ''}` });
  };

  const bulkCompare = () => {
    if (selectedIds.length < 2) {
      toast.push({ type: 'warning', title: 'Select at least 2 models', message: 'Comparison requires multiple models' });
      return;
    }
    onCompare(selectedIds);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Models</h1>
        <div className="flex gap-2">
          <div className="flex border border-border rounded overflow-hidden">
            <button
              onClick={() => setMode('models')}
              className={`px-4 py-2 ${mode === 'models' ? 'bg-primary text-on-primary' : 'bg-surface hover:bg-muted'}`}
            >
              My Models
            </button>
            <button
              onClick={() => setMode('templates')}
              className={`px-4 py-2 ${mode === 'templates' ? 'bg-primary text-on-primary' : 'bg-surface hover:bg-muted'}`}
            >
              Templates
            </button>
          </div>
          {mode === 'models' && (
            <>
              <button onClick={()=>setView(view==='grid'?'list':'grid')} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">{view==='grid'?'List View':'Grid View'}</button>
              <button onClick={bulkCompare} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Compare</button>
              <button onClick={bulkExport} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Export</button>
              <button onClick={bulkDelete} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted text-danger">Delete</button>
            </>
          )}
        </div>
      </div>

      {mode === 'templates' ? (
        <ModelTemplatesView api={api} toast={toast} onNavigate={onNavigate} />
      ) : (
        <>

      <div className="grid grid-cols-1 md:grid-cols-6 gap-3">
        <input placeholder="Search" value={q} onChange={e=>setQ(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface"/>
        <select value={framework} onChange={e=>setFramework(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface">
          <option value="">All Frameworks</option>
          <option value="pytorch">PyTorch</option>
          <option value="huggingface">Hugging Face</option>
          <option value="tensorflow">TensorFlow</option>
        </select>
        <input placeholder="Architecture (e.g., resnet)" value={architecture} onChange={e=>setArchitecture(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface"/>
        <select value={size} onChange={e=>setSize(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface"><option value="">Any Size</option><option value="small">Small</option><option value="base">Base</option><option value="large">Large</option><option value="xl">XL</option></select>
        <input placeholder="Tag/Domain" value={tag} onChange={e=>setTag(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface"/>
        <input placeholder="License" value={license} onChange={e=>setLicense(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface"/>
        <div className="flex gap-2">
          <select value={sort} onChange={e=>setSort(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface">
            <option value="date">Date</option>
            <option value="size">Size</option>
            <option value="accuracy">Accuracy</option>
            <option value="popular">Popular</option>
            <option value="name">Name</option>
          </select>
          <select value={order} onChange={e=>setOrder(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface">
            <option value="desc">Desc</option>
            <option value="asc">Asc</option>
          </select>
        </div>
      </div>

      {view==='grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {items.map(m => (
            <ModelCard key={m.id} m={m} selected={!!sel[m.id]} onSelect={v=>setSel(s=>({...s, [m.id]: v}))} onOpen={()=>onOpen(m.id)} onDelete={()=>deleteModel(m.id)} />
          ))}
        </div>
      ) : (
        <table className="w-full">
          <thead className="bg-muted border-b border-border"><tr>
            <th className="px-3 py-2"><input type="checkbox" checked={allSelected} onChange={toggleAll}/></th>
            <th className="px-3 py-2 text-left">Name</th>
            <th className="px-3 py-2 text-left">Framework</th>
            <th className="px-3 py-2 text-left">Arch</th>
            <th className="px-3 py-2 text-left">Metrics</th>
            <th className="px-3 py-2 text-left">Created</th>
            <th className="px-3 py-2"></th>
          </tr></thead>
          <tbody className="divide-y divide-border">
            {items.map(m => (
              <tr key={m.id} className="hover:bg-muted">
                <td className="px-3 py-2"><input type="checkbox" checked={!!sel[m.id]} onChange={e=>setSel(s=>({...s, [m.id]: e.target.checked}))}/></td>
                <td className="px-3 py-2 font-semibold">{m.name}</td>
                <td className="px-3 py-2">{m.framework}</td>
                <td className="px-3 py-2">{m.architecture || '-'}</td>
                <td className="px-3 py-2 text-sm">{m.metrics?.eval_accuracy!=null ? `Acc ${m.metrics.eval_accuracy.toFixed(3)}` : '-'}</td>
                <td className="px-3 py-2 text-sm">{m.created ? new Date(m.created).toLocaleString() : '-'}</td>
                <td className="px-3 py-2 text-right">
                  <button onClick={()=>onOpen(m.id)} className="text-primary hover:brightness-110 mr-3">Open</button>
                  <button onClick={()=>deleteModel(m.id)} className="text-danger hover:brightness-110">Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
        </>
      )}
    </div>
  );
}

export function ModelCard({ m, selected, onSelect, onOpen, onDelete }) {
  const size = (m.size_bytes||0);
  const fmtSize = size >= (1<<30) ? (size/(1<<30)).toFixed(1)+' GB' : size >= (1<<20) ? (size/(1<<20)).toFixed(1)+' MB' : size+' B';
  return (
    <div className="border border-border rounded-lg bg-surface shadow-sm p-4 flex flex-col gap-2 group relative hover:shadow-lg transition-shadow">
      <div className="flex justify-between items-start">
        <div className="flex-1 min-w-0 pr-2">
          <div className="font-semibold text-lg truncate">{m.name}</div>
          <div className="text-xs text-text/70">{m.framework} {m.architecture ? `• ${m.architecture}` : ''} {m.size_category ? `• ${m.size_category}` : ''}</div>
        </div>
        <input type="checkbox" checked={selected} onChange={e=>onSelect(e.target.checked)} className="shrink-0" />
      </div>
      {m.card_excerpt && <div className="text-xs text-text/80 line-clamp-3">{m.card_excerpt}</div>}
      <div className="text-sm">
        {m.metrics?.eval_accuracy!=null && <div>Accuracy: <span className="font-semibold">{m.metrics.eval_accuracy.toFixed(3)}</span></div>}
        {m.parameters!=null && <div>Params: <span className="font-semibold">{m.parameters.toLocaleString()}</span></div>}
        <div>Size: <span className="font-semibold">{fmtSize}</span></div>
      </div>
      {(m.tags||[]).length>0 && (
        <div className="flex flex-wrap gap-2 text-xs">
          {(m.tags||[]).slice(0,5).map((t,i)=> <span key={i} className="px-2 py-1 bg-muted rounded border border-border">{t}</span>)}
          {(m.tags||[]).length>5 && <span className="text-text/60">+{(m.tags||[]).length-5}</span>}
        </div>
      )}
      {m.popularity && <div className="text-[10px] text-text/60">Views {m.popularity.views||0} • Exports {m.popularity.exports||0}</div>}
      <div className="flex justify-between gap-2">
        <button onClick={onOpen} className="flex-1 px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Open</button>
        <button onClick={(e)=>{e.stopPropagation(); onDelete();}} className="px-3 py-2 text-xs bg-danger/10 text-danger border border-danger/30 rounded hover:bg-danger/20">
          Delete
        </button>
      </div>
    </div>
  );
}

export function ModelDetail({ id, api, onBack }) {
  const [info, setInfo] = useState(null);
  const [tab, setTab] = useState('overview');
  const [tags, setTags] = useState('');
  const [kv, setKv] = useState('');
  const [card, setCard] = useState('');
  const [adapters, setAdapters] = useState([]);
  const [mergeName, setMergeName] = useState('');
  const [merging, setMerging] = useState(false);
  const toast = useToast();

  const load = async () => {
    const d = await api.getModel(id);
    setInfo(d);
    setTags((d.metadata?.tags||[]).join(','));
    setKv(JSON.stringify(d.metadata?.custom||{}, null, 2));
    setCard(d.card_md||'');
  };
  useEffect(()=>{ load(); }, [id]);

  const saveMeta = async () => {
    try {
      const payload = {
        tags: tags.split(',').map(s=>s.trim()).filter(Boolean),
        custom: (()=>{ try { return JSON.parse(kv||'{}'); } catch { return {}; } })(),
      };
      await api.updateModelMetadata(id, payload);
      load();
      toast.push({ type: 'success', title: 'Metadata saved' });
    } catch (e) {
      toast.push({ type: 'error', title: 'Save failed', message: e.message });
    }
  };

  const saveCard = async () => {
    try {
      await api.updateModelCard(id, { content: card });
      load();
      toast.push({ type: 'success', title: 'Card saved' });
    } catch (e) {
      toast.push({ type: 'error', title: 'Save failed', message: e.message });
    }
  };
  if (!info) return <div>Loading...</div>;
  const cfg = info.config || {};
  const metrics = info.metrics || {};
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">{cfg.name || id}</h1>
          <div className="text-sm text-text/70">{cfg.framework} {cfg.architecture ? `• ${cfg.architecture}` : ''} {cfg.created ? `• ${new Date(cfg.created).toLocaleString()}` : ''}</div>
        </div>
        <div className="flex gap-2">
          <button onClick={()=>window.open(`/api/models/${id}/card.html`,'_blank')} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Export Card (HTML)</button>
          <button onClick={onBack} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Back</button>
        </div>
      </div>
      <div className="flex gap-2 text-sm">
        {['overview','architecture','metrics','evals','files','adapters','gallery','similar','card'].map(t => (
          <button key={t} onClick={()=>setTab(t)} className={`px-3 py-2 rounded ${tab===t?'bg-primary text-on-primary':'bg-muted'}`}>{t[0].toUpperCase()+t.slice(1)}</button>
        ))}
      </div>

      {tab==='overview' && (
        <div className="bg-surface p-4 rounded border border-border space-y-3">
          <div><span className="text-text/70">Parameters:</span> <span className="font-semibold">{(cfg.parameters||0).toLocaleString()}</span></div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-semibold">Tags</label>
              <input className="w-full border border-border rounded px-3 py-2 bg-surface" value={tags} onChange={e=>setTags(e.target.value)} placeholder="comma,separated"/>
            </div>
            <div>
              <label className="text-sm font-semibold">Custom Metadata (JSON)</label>
              <textarea className="w-full border border-border rounded px-3 py-2 h-24 bg-surface" value={kv} onChange={e=>setKv(e.target.value)} />
            </div>
          </div>
          <div className="flex justify-end"><button onClick={saveMeta} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Save Metadata</button></div>
        </div>
      )}

      {tab==='architecture' && (
        <div className="bg-surface p-4 rounded border border-border space-y-3">
          <div className="text-sm">Architecture: <span className="font-semibold">{cfg.architecture || 'N/A'}</span></div>
          <div className="h-32 bg-muted border border-border rounded flex items-center justify-center">Architecture diagram placeholder</div>
          <div className="text-xs text-text/60">Future: layer breakdown, attention maps, feature maps, downloadable diagrams.</div>
        </div>
      )}

      {tab==='metrics' && (
        <div className="bg-surface p-4 rounded border border-border space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-text/70 mb-2">Metrics JSON</div>
              <pre className="bg-muted p-3 rounded text-xs overflow-auto">{JSON.stringify(metrics, null, 2)}</pre>
            </div>
            <div>
              <div className="text-sm text-text/70 mb-2">Hyperparameters (from config)</div>
              <pre className="bg-muted p-3 rounded text-xs overflow-auto">{JSON.stringify(cfg, null, 2)}</pre>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <MetricLine title="Train Loss" data={(metrics.train_loss_history||[]).map((y,i)=>({x:i,y}))} />
            <MetricLine title="Eval Accuracy" data={(metrics.eval_accuracy_history||[]).map((y,i)=>({x:i,y}))} />
          </div>
        </div>
      )}

      {tab==='files' && (
        <div className="bg-surface p-4 rounded border border-border space-y-3">
          <table className="w-full text-sm">
            <thead className="bg-muted border-b border-border"><tr><th className="text-left px-3 py-2">Path</th><th className="text-left px-3 py-2">Size</th></tr></thead>
            <tbody className="divide-y divide-border">
              {(info.files||[]).map((f,i)=>(<tr key={i}><td className="px-3 py-1">{f.path}</td><td className="px-3 py-1">{f.size}</td></tr>))}
            </tbody>
          </table>
        </div>
      )}

      {tab==='evals' && (
        <ModelEvals id={id} />
      )}

      {tab==='adapters' && (
        <div className="bg-surface p-4 rounded border border-border space-y-3">
          <div className="text-sm text-text/70">LoRA Adapters saved under this model</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted border-b border-border"><tr><th className="text-left px-3 py-2">Name</th><th className="text-left px-3 py-2">Path</th><th className="text-left px-3 py-2">Size</th><th className="px-3 py-2"></th></tr></thead>
              <tbody className="divide-y divide-border">
                {adapters.map((a)=> (
                  <tr key={a.name}>
                    <td className="px-3 py-2 font-semibold">{a.name}</td>
                    <td className="px-3 py-2 text-xs text-text/70">{a.path}</td>
                    <td className="px-3 py-2">{a.size_bytes != null ? (a.size_bytes/1e6).toFixed(2)+' MB' : '-'}</td>
                    <td className="px-3 py-2 text-right"><button className="px-3 py-1 border border-border rounded hover:bg-muted" onClick={()=>setMergeName(a.name)}>Merge…</button></td>
                  </tr>
                ))}
                {adapters.length===0 && (<tr><td className="px-3 py-2 text-sm text-text/60" colSpan={4}>No adapters found</td></tr>)}
              </tbody>
            </table>
          </div>
          <div className="flex justify-end"><button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={async()=>{ const res = await api.listModelAdapters(id); setAdapters(res.adapters||[]); }}>Refresh</button></div>
          {mergeName && (
            <div className="p-3 bg-muted rounded border border-border">
              <div className="text-sm mb-2">Merge adapter <span className="font-semibold">{mergeName}</span> into base model?</div>
              <div className="flex gap-2">
                <button disabled={merging} className={`px-3 py-2 rounded ${merging?'bg-muted text-text/60':'bg-primary text-on-primary hover:brightness-110'}`} onClick={async()=>{
                  setMerging(true);
                  try {
                    const res = await api.mergeModelAdapter(id, mergeName);
                    if (res && !res.error) {
                      toast.push({ type: 'success', title: 'Adapter merged successfully' });
                      setMergeName('');
                      const d = await api.getModel(id);
                      setInfo(d);
                    } else {
                      toast.push({ type: 'error', title: 'Merge failed', message: res.error || res.detail || 'Unknown error' });
                    }
                  } catch (e) {
                    toast.push({ type: 'error', title: 'Merge failed', message: e.message });
                  } finally {
                    setMerging(false);
                  }
                }}>Confirm Merge</button>
                <button className="px-3 py-2 border border-border rounded" onClick={()=>setMergeName('')}>Cancel</button>
              </div>
              <div className="text-xs text-text/60 mt-1">Merging overwrites base model weights under this model ID. Ensure you have exported or backed up if needed.</div>
            </div>
          )}
        </div>
      )}

      {tab==='card' && (
        <div className="bg-surface p-4 rounded border border-border space-y-3">
          <label className="text-sm font-semibold">Model Card (Markdown)</label>
          <textarea className="w-full border border-border rounded px-3 py-2 h-64 bg-surface" value={card} onChange={e=>setCard(e.target.value)} />
          <div className="flex justify-end"><button onClick={saveCard} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Save Card</button></div>
          <div className="text-xs text-text/60">Future: rich markdown editor, version history UI, export as PDF.</div>
        </div>
      )}

      {tab==='gallery' && (
        <div className="bg-surface p-4 rounded border border-border space-y-3">
          <div className="text-sm text-text/70">Screenshots / Media</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {(info.assets?.images||[]).map((p,i)=>(
              <div key={i} className="border border-border rounded overflow-hidden">
                <img src={`/api/models/${encodeURIComponent(id)}/file?path=${encodeURIComponent(p)}`} className="w-full h-32 object-cover" />
              </div>
            ))}
          </div>
          <div className="space-y-2">
            {(info.assets?.videos||[]).map((p,i)=>(
              <video key={i} src={`/api/models/${encodeURIComponent(id)}/file?path=${encodeURIComponent(p)}`} controls className="w-full max-w-xl rounded border border-border" />
            ))}
          </div>
        </div>
      )}

      {tab==='similar' && (
        <SimilarModels id={id} onOpen={onBack ? (mid)=>{} : undefined} />
      )}
    </div>
  );
}

function MetricLine({ title, data }){
  const width = 260, height = 120, pad = 24;
  if (!data || data.length===0) return <div className="text-xs text-text/60">No {title} data</div>;
  const minX = 0, maxX = Math.max(...data.map(d=>d.x), 1);
  const minY = Math.min(...data.map(d=>d.y)), maxY = Math.max(...data.map(d=>d.y));
  const sx = (x)=> pad + (x-minX)/(maxX-minX||1)*(width-2*pad);
  const sy = (y)=> height-pad - (y-minY)/(maxY-minY||1)*(height-2*pad);
  const path = data.map((d,i)=>`${i===0?'M':'L'} ${sx(d.x)} ${sy(d.y)}`).join(' ');
  return (
    <div>
      <div className="text-sm mb-1">{title}</div>
      <svg width={width} height={height} className="w-full">
        <rect x={0} y={0} width={width} height={height} fill="transparent" className="stroke-border" strokeWidth="1" />
        <path d={path} className="stroke-primary" strokeWidth={2} fill="none" />
      </svg>
    </div>
  );
}

function SimilarModels({ id }){
  const [items, setItems] = useState([]);
  useEffect(()=>{ (async ()=>{ try{ const r = await fetch(`/api/models/${encodeURIComponent(id)}/similar`); const j = await r.json(); setItems(j.similar||[]); } catch{} })(); }, [id]);
  if (!items.length) return <div className="text-xs text-text/60">No similar models found</div>;
  return (
    <div className="bg-surface p-4 rounded border border-border">
      <div className="text-sm font-semibold mb-2">Similar Models</div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {items.map((it,i)=>(<div key={i} className="text-sm">{it.id} <span className="text-xs text-text/60">({it.score})</span></div>))}
      </div>
    </div>
  );
}

function ModelEvals({ id }){
  const [items, setItems] = useState([]);
  const [name, setName] = useState('');
  const [metrics, setMetrics] = useState('{}');
  const [notes, setNotes] = useState('');
  const toast = useToast();

  const load = async () => { try{ const r = await fetch(`/api/models/${encodeURIComponent(id)}/evals`); const j = await r.json(); setItems(j.items||[]); }catch{} };
  useEffect(()=>{ load(); }, [id]);

  const save = async () => {
    try{
      await fetch(`/api/models/${encodeURIComponent(id)}/evals`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name, metrics: JSON.parse(metrics||'{}'), notes })});
      setName(''); setMetrics('{}'); setNotes(''); load();
      toast.push({ type: 'success', title: 'Evaluation saved' });
    } catch (e) {
      toast.push({ type: 'error', title: 'Save failed', message: e.message });
    }
  };
  return (
    <div className="bg-surface p-4 rounded border border-border space-y-3">
      <div className="text-sm font-semibold">Evaluations</div>
      <table className="w-full text-sm">
        <thead className="bg-muted border-b border-border"><tr><th className="text-left px-3 py-2">Time</th><th className="text-left px-3 py-2">Name</th><th className="text-left px-3 py-2">Metrics</th><th className="text-left px-3 py-2">Notes</th></tr></thead>
        <tbody className="divide-y divide-border">
          {items.map((it,i)=>(
            <tr key={i}><td className="px-3 py-2 text-xs">{new Date(it.ts).toLocaleString()}</td><td className="px-3 py-2">{it.name}</td><td className="px-3 py-2 text-xs"><code>{Object.entries(it.metrics||{}).slice(0,6).map(([k,v])=>`${k}: ${v}`).join(', ')}</code></td><td className="px-3 py-2 text-xs">{it.notes||''}</td></tr>
          ))}
          {items.length===0 && (<tr><td className="px-3 py-2 text-xs text-text/60" colSpan={4}>No evaluations</td></tr>)}
        </tbody>
      </table>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm">
        <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Benchmark name (e.g., GLUE)" value={name} onChange={e=>setName(e.target.value)} />
        <input className="border border-border rounded px-3 py-2 bg-surface" placeholder='Metrics JSON (e.g., {"acc":0.91})' value={metrics} onChange={e=>setMetrics(e.target.value)} />
        <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Notes" value={notes} onChange={e=>setNotes(e.target.value)} />
      </div>
      <div className="flex justify-end"><button className="px-3 py-2 border border-border rounded" onClick={save}>Add Evaluation</button></div>
    </div>
  );
}

export function ModelCompare({ ids, api, onBack }) {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    (async () => {
      const all = await Promise.all(ids.map(id => api.getModel(id)));
      setRows(all);
    })();
  }, [ids]);
  if (!rows.length) return <div>Loading...</div>;
  const cols = ids;
  const cell = (fn) => rows.map(fn);
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Compare Models</h1>
        <button onClick={onBack} className="px-3 py-2 border rounded">Back</button>
      </div>
      <table className="w-full text-sm">
        <tbody className="divide-y">
          <tr><th className="text-left px-3 py-2">ID</th>{cell((r,i)=>(<td key={i} className="px-3 py-2">{r.id || ids[i]}</td>))}</tr>
          <tr><th className="text-left px-3 py-2">Name</th>{cell((r,i)=>(<td key={i} className="px-3 py-2">{r.config?.name||'-'}</td>))}</tr>
          <tr><th className="text-left px-3 py-2">Framework</th>{cell((r,i)=>(<td key={i} className="px-3 py-2">{r.config?.framework||'-'}</td>))}</tr>
          <tr><th className="text-left px-3 py-2">Architecture</th>{cell((r,i)=>(<td key={i} className="px-3 py-2">{r.config?.architecture||'-'}</td>))}</tr>
          <tr><th className="text-left px-3 py-2">Parameters</th>{cell((r,i)=>(<td key={i} className="px-3 py-2">{(r.config?.parameters||0).toLocaleString()}</td>))}</tr>
          <tr><th className="text-left px-3 py-2">Accuracy</th>{cell((r,i)=>(<td key={i} className="px-3 py-2">{r.metrics?.eval_accuracy!=null ? r.metrics.eval_accuracy.toFixed(3) : '-'}</td>))}</tr>
          <tr><th className="text-left px-3 py-2">Eval Loss</th>{cell((r,i)=>(<td key={i} className="px-3 py-2">{r.metrics?.eval_loss!=null ? r.metrics.eval_loss.toFixed(4) : '-'}</td>))}</tr>
          <tr><th className="text-left px-3 py-2">Tags</th>{cell((r,i)=>(<td key={i} className="px-3 py-2">{(r.metadata?.tags||[]).join(', ')}</td>))}</tr>
        </tbody>
      </table>
    </div>
  );
}

// Icon mapping for template categories
const iconMap = {
  'mic': '🎤',
  'image': '🖼️',
  'eye': '👁️',
  'video': '🎥',
  'message-square': '💬',
  'layers': '📚',
  'wand': '✨',
  'type': '📝'
};

export function ModelTemplatesView({ api, toast, onNavigate }) {
  const [templates, setTemplates] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedTemplate, setSelectedTemplate] = useState(null);

  useEffect(() => {
    loadTemplates();
  }, []);

  const loadTemplates = async () => {
    try {
      const res = await fetch('/api/models/templates');
      if (res.ok) {
        const text = await res.text();
        if (!text) {
          console.error('Empty response from templates endpoint');
          setTemplates({ templates: {}, categories: {} });
          toast.push({ type: 'error', title: 'Empty response from server' });
          return;
        }
        try {
          const data = JSON.parse(text);
          setTemplates(data);
        } catch (parseError) {
          console.error('JSON parse error:', parseError, 'Response:', text);
          setTemplates({ templates: {}, categories: {} });
          toast.push({ type: 'error', title: 'Invalid response format' });
        }
      } else {
        try {
          const errorData = await res.json();
          toast.push({ type: 'error', title: 'Failed to load templates', message: errorData.error });
        } catch {
          toast.push({ type: 'error', title: 'Failed to load templates' });
        }
        // Set empty data to prevent crashes
        setTemplates({ templates: {}, categories: {} });
      }
    } catch (e) {
      console.error('Network error loading templates:', e);
      setTemplates({ templates: {}, categories: {} });
      toast.push({ type: 'error', title: 'Error loading templates', message: e.message });
    }
  };

  if (!templates) {
    return <div className="text-center py-8">Loading templates...</div>;
  }

  const categories = templates.categories || {};
  const templateList = Object.entries(templates.templates || {});
  const filteredTemplates = selectedCategory === 'all'
    ? templateList
    : templateList.filter(([_, t]) => t.category === selectedCategory);

  if (selectedTemplate) {
    return <TemplateDetail template={selectedTemplate} onBack={() => setSelectedTemplate(null)} toast={toast} onNavigate={onNavigate} />;
  }

  return (
    <div className="space-y-6">
      {/* Category Filter */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setSelectedCategory('all')}
          className={`px-4 py-2 rounded ${
            selectedCategory === 'all'
              ? 'bg-primary text-on-primary'
              : 'bg-surface border border-border hover:bg-muted'
          }`}
        >
          All Templates
        </button>
        {Object.entries(categories).map(([key, cat]) => (
          <button
            key={key}
            onClick={() => setSelectedCategory(key)}
            className={`px-4 py-2 rounded ${
              selectedCategory === key
                ? 'bg-primary text-on-primary'
                : 'bg-surface border border-border hover:bg-muted'
            }`}
          >
            {iconMap[cat.icon] || '📦'} {cat.name}
          </button>
        ))}
      </div>

      {/* Templates Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredTemplates.map(([id, template]) => (
          <TemplateCard
            key={id}
            id={id}
            template={template}
            onClick={() => setSelectedTemplate({ id, ...template })}
          />
        ))}
      </div>

      {filteredTemplates.length === 0 && (
        <div className="text-center py-12 text-text/60">
          No templates found in this category
        </div>
      )}
    </div>
  );
}

function TemplateCard({ id, template, onClick }) {
  const icon = iconMap[template.icon] || '📦';
  const category = template.category || 'general';

  return (
    <div
      onClick={onClick}
      className="border border-border rounded-lg bg-surface shadow-sm p-6 cursor-pointer hover:shadow-lg hover:border-primary/50 transition-all group"
    >
      <div className="flex items-start gap-4">
        <div className="text-4xl">{icon}</div>
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold mb-1 group-hover:text-primary transition-colors">
            {template.name}
          </h3>
          <p className="text-sm text-text/70 line-clamp-2 mb-3">
            {template.description}
          </p>
          <div className="flex flex-wrap gap-2 mb-3">
            {template.tags?.slice(0, 3).map((tag, idx) => (
              <span
                key={idx}
                className="px-2 py-1 text-xs bg-secondary/10 text-secondary border border-secondary/30 rounded"
              >
                {tag}
              </span>
            ))}
          </div>
          <div className="text-xs text-text/60 space-y-1">
            <div>Framework: <span className="font-semibold">{template.training?.framework || 'N/A'}</span></div>
            <div>Precision: <span className="font-semibold">{template.training?.precision || 'N/A'}</span></div>
            {template.resources?.min_gpu_memory && (
              <div>Min GPU: <span className="font-semibold">{template.resources.min_gpu_memory}</span></div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function TemplateDetail({ template, onBack, toast, onNavigate }) {
  const [selectedConfig, setSelectedConfig] = useState('training');

  const startTraining = () => {
    try {
      // Persist template for Job Wizard prefill
      localStorage.setItem('jobWizard.template', JSON.stringify(template));
    } catch {}
    toast.push({ type: 'success', title: 'Template loaded', message: 'Opening Job Wizard...' });
    if (onNavigate) onNavigate('wizard');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={onBack}
            className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted"
          >
            ← Back
          </button>
          <div>
            <h2 className="text-2xl font-bold">{template.name}</h2>
            <p className="text-text/70">{template.description}</p>
          </div>
        </div>
        <button
          onClick={startTraining}
          className="px-6 py-3 bg-primary text-on-primary rounded hover:brightness-110 font-semibold"
        >
          Use This Template
        </button>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-2">
        {template.tags?.map((tag, idx) => (
          <span
            key={idx}
            className="px-3 py-1 text-sm bg-secondary/10 text-secondary border border-secondary/30 rounded"
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Config Tabs */}
      <div className="border-b border-border">
        <div className="flex gap-4">
          {['overview', 'training', 'data', 'metrics', 'resources'].map(tab => (
            <button
              key={tab}
              onClick={() => setSelectedConfig(tab)}
              className={`px-4 py-2 font-medium capitalize transition-colors ${
                selectedConfig === tab
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-text/60 hover:text-text'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Config Content */}
      <div className="bg-surface border border-border rounded-lg p-6">
        {selectedConfig === 'overview' && (
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-semibold mb-2">Model Architecture</h3>
              <p className="text-sm text-text/70 mb-2">
                <span className="font-medium">Base Model:</span> {template.model?.base_model || 'Custom'}
              </p>
              <p className="text-sm text-text/70 mb-2">
                <span className="font-medium">Architecture:</span> {template.model?.architecture || 'N/A'}
              </p>
              <p className="text-sm text-text/70 mb-2">
                <span className="font-medium">Input Modalities:</span>{' '}
                {template.model?.input_modalities?.join(', ') || 'N/A'}
              </p>
              <p className="text-sm text-text/70">
                <span className="font-medium">Output Type:</span> {template.model?.output_type || 'N/A'}
              </p>
            </div>

            {template.model?.components && (
              <div>
                <h3 className="text-lg font-semibold mb-2">Components</h3>
                <pre className="bg-muted p-4 rounded text-xs overflow-x-auto">
                  {JSON.stringify(template.model.components, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {selectedConfig === 'training' && template.training && (
          <div>
            <h3 className="text-lg font-semibold mb-4">Training Configuration</h3>
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(template.training).map(([key, value]) => (
                <div key={key} className="text-sm">
                  <span className="font-medium capitalize text-text/70">
                    {key.replace(/_/g, ' ')}:
                  </span>{' '}
                  <span className="font-mono">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedConfig === 'data' && template.data && (
          <div>
            <h3 className="text-lg font-semibold mb-4">Data Configuration</h3>
            <pre className="bg-muted p-4 rounded text-xs overflow-x-auto">
              {JSON.stringify(template.data, null, 2)}
            </pre>
          </div>
        )}

        {selectedConfig === 'metrics' && template.metrics && (
          <div>
            <h3 className="text-lg font-semibold mb-4">Evaluation Metrics</h3>
            <div className="space-y-2">
              {Array.isArray(template.metrics) ? (
                template.metrics.map((metric, idx) => (
                  <div key={idx} className="px-3 py-2 bg-muted rounded border border-border">
                    {metric}
                  </div>
                ))
              ) : (
                Object.entries(template.metrics).map(([category, metrics]) => (
                  <div key={category}>
                    <h4 className="font-semibold capitalize mb-2">{category}</h4>
                    <div className="flex flex-wrap gap-2">
                      {metrics.map((metric, idx) => (
                        <span
                          key={idx}
                          className="px-3 py-1 bg-muted rounded border border-border text-sm"
                        >
                          {metric}
                        </span>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {selectedConfig === 'resources' && template.resources && (
          <div>
            <h3 className="text-lg font-semibold mb-4">Resource Requirements</h3>
            <div className="space-y-3">
              {Object.entries(template.resources).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="font-medium capitalize text-text/70">
                    {key.replace(/_/g, ' ')}:
                  </span>
                  <span>{value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Dataset Pipeline (for Apotheon) */}
      {template.dataset_pipeline && (
        <div className="bg-surface border border-border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Dataset Pipeline</h3>
          <div className="space-y-3">
            {template.dataset_pipeline.steps?.map((step, idx) => (
              <div key={idx} className="flex items-start gap-3 p-3 bg-muted rounded">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center text-sm font-semibold">
                  {idx + 1}
                </div>
                <div className="flex-1">
                  <div className="font-semibold">{step.name}</div>
                  <div className="text-xs text-text/60">{step.description}</div>
                  {step.params && (
                    <div className="text-xs mt-1 font-mono text-text/70">
                      {JSON.stringify(step.params)}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
