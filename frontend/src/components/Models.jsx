import React, { useEffect, useMemo, useState } from 'react';

export function ModelsPage({ api, onOpen, onCompare }) {
  const [items, setItems] = useState([]);
  const [view, setView] = useState('grid');
  const [q, setQ] = useState('');
  const [framework, setFramework] = useState('');
  const [architecture, setArchitecture] = useState('');
  const [sort, setSort] = useState('date');
  const [order, setOrder] = useState('desc');
  const [sel, setSel] = useState({});

  const load = async () => {
    const params = new URLSearchParams();
    if (q) params.set('q', q);
    if (framework) params.set('framework', framework);
    if (architecture) params.set('architecture', architecture);
    params.set('sort', sort); params.set('order', order);
    const data = await api.getModelsRaw(params.toString());
    setItems(data);
  };
  useEffect(() => { load(); }, [q, framework, architecture, sort, order]);

  const allSelected = useMemo(()=>items.length>0 && items.every(m => sel[m.id]), [items, sel]);
  const toggleAll = () => {
    if (allSelected) setSel({}); else setSel(Object.fromEntries(items.map(m => [m.id, true])));
  };
  const selectedIds = Object.keys(sel).filter(k => sel[k]);

  const bulkDelete = async () => {
    if (!selectedIds.length) return;
    if (!confirm(`Delete ${selectedIds.length} models? This cannot be undone.`)) return;
    await api.bulkDeleteModels(selectedIds);
    setSel({});
    load();
  };

  const bulkExport = async () => {
    if (!selectedIds.length) return;
    const url = api.exportModelsUrl(selectedIds);
    window.open(url, '_blank');
  };

  const bulkCompare = () => {
    if (selectedIds.length < 2) { alert('Select at least 2 models to compare'); return; }
    onCompare(selectedIds);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Models</h1>
        <div className="flex gap-2">
          <button onClick={()=>setView(view==='grid'?'list':'grid')} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">{view==='grid'?'List View':'Grid View'}</button>
          <button onClick={bulkCompare} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Compare</button>
          <button onClick={bulkExport} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Export</button>
          <button onClick={bulkDelete} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted text-danger">Delete</button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <input placeholder="Search" value={q} onChange={e=>setQ(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface"/>
        <select value={framework} onChange={e=>setFramework(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface">
          <option value="">All Frameworks</option>
          <option value="pytorch">PyTorch</option>
          <option value="huggingface">Hugging Face</option>
          <option value="tensorflow">TensorFlow</option>
        </select>
        <input placeholder="Architecture (e.g., resnet)" value={architecture} onChange={e=>setArchitecture(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface"/>
        <div className="flex gap-2">
          <select value={sort} onChange={e=>setSort(e.target.value)} className="border border-border rounded px-3 py-2 bg-surface">
            <option value="date">Date</option>
            <option value="size">Size</option>
            <option value="accuracy">Accuracy</option>
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
            <ModelCard key={m.id} m={m} selected={!!sel[m.id]} onSelect={v=>setSel(s=>({...s, [m.id]: v}))} onOpen={()=>onOpen(m.id)} />
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
                <td className="px-3 py-2 text-right"><button onClick={()=>onOpen(m.id)} className="text-blue-600 hover:text-blue-800">Open</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function ModelCard({ m, selected, onSelect, onOpen }) {
  const size = (m.size_bytes||0);
  const fmtSize = size >= (1<<30) ? (size/(1<<30)).toFixed(1)+' GB' : size >= (1<<20) ? (size/(1<<20)).toFixed(1)+' MB' : size+' B';
  return (
    <div className="border border-border rounded-lg bg-surface shadow-sm p-4 flex flex-col gap-2">
      <div className="flex justify-between items-start">
        <div>
          <div className="font-semibold text-lg">{m.name}</div>
          <div className="text-xs text-text/70">{m.framework} {m.architecture ? `• ${m.architecture}` : ''}</div>
        </div>
        <input type="checkbox" checked={selected} onChange={e=>onSelect(e.target.checked)} />
      </div>
      <div className="text-sm">
        {m.metrics?.eval_accuracy!=null && <div>Accuracy: <span className="font-semibold">{m.metrics.eval_accuracy.toFixed(3)}</span></div>}
        {m.parameters!=null && <div>Params: <span className="font-semibold">{m.parameters.toLocaleString()}</span></div>}
        <div>Size: <span className="font-semibold">{fmtSize}</span></div>
      </div>
      <div className="flex flex-wrap gap-2 text-xs">
        {(m.tags||[]).map((t,i)=> <span key={i} className="px-2 py-1 bg-muted rounded border border-border">{t}</span>)}
      </div>
      <div className="flex justify-end">
        <button onClick={onOpen} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Open</button>
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
  const load = async () => {
    const d = await api.getModel(id);
    setInfo(d);
    setTags((d.metadata?.tags||[]).join(','));
    setKv(JSON.stringify(d.metadata?.custom||{}, null, 2));
    setCard(d.card_md||'');
  };
  useEffect(()=>{ load(); }, [id]);

  const saveMeta = async () => {
    const payload = {
      tags: tags.split(',').map(s=>s.trim()).filter(Boolean),
      custom: (()=>{ try { return JSON.parse(kv||'{}'); } catch { return {}; } })(),
    };
    await api.updateModelMetadata(id, payload);
    load();
  };
  const saveCard = async () => {
    await api.updateModelCard(id, { content: card });
    load();
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
        {['overview','architecture','metrics','files','card'].map(t => (
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
          <div className="text-xs text-text/60">Future: loss curves, accuracy charts, confusion matrix for classification.</div>
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

      {tab==='card' && (
        <div className="bg-surface p-4 rounded border border-border space-y-3">
          <label className="text-sm font-semibold">Model Card (Markdown)</label>
          <textarea className="w-full border border-border rounded px-3 py-2 h-64 bg-surface" value={card} onChange={e=>setCard(e.target.value)} />
          <div className="flex justify-end"><button onClick={saveCard} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">Save Card</button></div>
          <div className="text-xs text-text/60">Future: rich markdown editor, version history UI, export as PDF.</div>
        </div>
      )}
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
