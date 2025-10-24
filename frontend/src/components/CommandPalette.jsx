import React, { useEffect, useMemo, useRef, useState } from 'react';

export default function CommandPalette({ open, onClose, api, onNavigate }) {
  const [q, setQ] = useState('');
  const [jobs, setJobs] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sel, setSel] = useState(0);
  const inputRef = useRef(null);

  const recentKey = 'palette.recent';
  const [recent, setRecent] = useState(() => {
    try { return JSON.parse(localStorage.getItem(recentKey) || '[]'); } catch { return []; }
  });

  useEffect(() => {
    if (!open) return;
    setQ(''); setSel(0);
    setLoading(true);
    (async () => {
      try {
        const [j, m, d] = await Promise.all([
          api.getJobs().catch(()=>[]),
          api.getModels().catch(()=>[]),
          api.getDatasets().catch(()=>[]),
        ]);
        setJobs(j || []); setModels(m || []); setDatasets(d || []);
      } finally { setLoading(false); }
    })();
    const t = setTimeout(() => { inputRef.current && inputRef.current.focus(); }, 50);
    return () => clearTimeout(t);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); }
      if (e.key === 'ArrowDown') { e.preventDefault(); setSel(s => Math.min(s + 1, items.length - 1)); }
      if (e.key === 'ArrowUp') { e.preventDefault(); setSel(s => Math.max(s - 1, 0)); }
      if (e.key === 'Enter') {
        e.preventDefault(); if (items[sel]) navigate(items[sel]);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, sel]);

  const items = useMemo(() => {
    const terms = (q || '').trim().toLowerCase().split(/\s+/).filter(Boolean);
    const match = (text) => terms.every(t => (text || '').toLowerCase().includes(t));
    const out = [];
    for (const j of jobs) {
      const text = `${j.name} ${j.framework} ${j.status}`;
      if (!terms.length || match(text)) out.push({ type: 'job', id: j.id, label: j.name, description: `${j.framework} • ${j.status}`, raw: j });
    }
    for (const m of models) {
      const text = `${m.name} ${m.framework} ${m.architecture}`;
      if (!terms.length || match(text)) out.push({ type: 'model', id: m.id, label: m.name || m.id, description: `${m.framework} • ${m.architecture || '-'}`, raw: m });
    }
    for (const d of datasets) {
      const text = `${d.name}`;
      if (!terms.length || match(text)) out.push({ type: 'dataset', id: d.name, label: d.name, description: `versions: ${d.versions}`, raw: d });
    }
    return out.slice(0, 100);
  }, [q, jobs, models, datasets]);

  const saveRecent = (entry) => {
    const list = [entry, ...recent.filter(r => r.key !== entry.key)].slice(0, 10);
    setRecent(list);
    try { localStorage.setItem(recentKey, JSON.stringify(list)); } catch {}
  };

  const navigate = (item) => {
    saveRecent({ key: `${item.type}:${item.id}`, label: item.label, description: item.description, type: item.type, id: item.id });
    if (item.type === 'job') onNavigate('jobs');
    if (item.type === 'model') onNavigate('models');
    if (item.type === 'dataset') onNavigate('datasets');
    onClose();
  };

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 bg-black/40 flex items-start justify-center p-6" onClick={onClose}>
      <div className="w-full max-w-2xl bg-surface border border-border rounded-lg shadow-lg" onClick={e=>e.stopPropagation()}>
        <div className="border-b border-border p-3">
          <input ref={inputRef} value={q} onChange={e=>{ setQ(e.target.value); setSel(0); }} placeholder="Search jobs, models, datasets…" className="w-full bg-transparent outline-none px-2 py-1" />
        </div>
        {loading ? (
          <div className="p-4 text-sm text-text/70">Loading…</div>
        ) : (
          <div className="max-h-80 overflow-auto">
            {(!q && recent.length>0) && (
              <div className="p-2 text-xs text-text/70">Recent</div>
            )}
            {(!q && recent.length>0) && recent.map((r, i) => (
              <div key={r.key} className="px-4 py-2 hover:bg-muted cursor-pointer" onMouseDown={()=>navigate(r)}>
                <div className="text-sm font-medium">{r.label}</div>
                <div className="text-xs text-text/70">{r.description}</div>
              </div>
            ))}
            {(q ? items : []).map((it, i) => (
              <div key={`${it.type}:${it.id}`} className={`px-4 py-2 hover:bg-muted cursor-pointer ${i===sel?'bg-muted':''}`} onMouseDown={()=>navigate(it)}>
                <div className="text-sm font-medium">{it.label}</div>
                <div className="text-xs text-text/70">{it.type} • {it.description}</div>
              </div>
            ))}
            {(q && items.length===0) && (<div className="p-4 text-sm text-text/70">No results</div>)}
          </div>
        )}
        <div className="border-t border-border p-2 text-xs text-text/60 flex justify-between">
          <div>Navigate with ↑ ↓, enter to open</div>
          <div>Close: Esc</div>
        </div>
      </div>
    </div>
  );
}

