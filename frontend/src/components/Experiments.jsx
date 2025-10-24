import React, { useEffect, useMemo, useState } from 'react';

export default function ExperimentsPage(){
  const [items, setItems] = useState([]);
  const [q, setQ] = useState('');
  const [favOnly, setFavOnly] = useState(false);
  const [sortBy, setSortBy] = useState('updated'); // updated|runs|name
  const [creating, setCreating] = useState('');
  const [detail, setDetail] = useState(null); // experiment object with runs
  const [edit, setEdit] = useState({ name:'', description:'', tags:'' });

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
    }catch{}
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

  const toggleStar = async (it) => {
    try{
      const res = await fetch(`/api/experiments/${encodeURIComponent(it.id)}/star`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ favorite: !it.favorite }) });
      if (res.ok) load();
    }catch{}
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Experiments</h1>
        <div className="flex gap-2 text-sm">
          <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Create new experiment" value={creating} onChange={e=>setCreating(e.target.value)} />
          <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={create}>Create</button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
        <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Search" value={q} onChange={e=>setQ(e.target.value)} />
        <label className="inline-flex items-center gap-2 text-sm"><input type="checkbox" checked={favOnly} onChange={e=>setFavOnly(e.target.checked)} /> Favorites</label>
        <select className="border border-border rounded px-3 py-2 bg-surface" value={sortBy} onChange={e=>setSortBy(e.target.value)}>
          <option value="updated">Recently Updated</option>
          <option value="runs">Run Count</option>
          <option value="name">Name</option>
        </select>
        <div />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {filtered.map(e => (
          <div key={e.id} className="bg-surface p-4 rounded border border-border">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold">{e.name}</div>
                <div className="text-xs text-text/60">{e.run_count||0} runs</div>
              </div>
              <div className="flex items-center gap-2">
                <button className={`text-sm px-2 py-1 rounded ${e.favorite?'bg-yellow-200 text-yellow-800 border border-yellow-300':'border border-border'}`} onClick={()=>toggleStar(e)}>{e.favorite?'★':'☆'}</button>
                <button className="text-sm px-2 py-1 border border-border rounded" onClick={()=>openDetail(e.id)}>Open</button>
              </div>
            </div>
            {e.description && <div className="text-xs text-text/70 mt-2 line-clamp-2">{e.description}</div>}
            {Array.isArray(e.tags) && e.tags.length>0 && (
              <div className="mt-2 flex flex-wrap gap-1 text-[10px] text-text/70">
                {e.tags.slice(0,6).map((t,i)=>(<span key={i} className="px-2 py-0.5 bg-muted rounded border border-border">{t}</span>))}
                {e.tags.length>6 && <span>+{e.tags.length-6}</span>}
              </div>
            )}
          </div>
        ))}
      </div>

      {detail && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-surface border border-border rounded max-w-3xl w-full max-h-[80vh] overflow-auto p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-2xl font-bold">{detail.name}</div>
              <button className="px-3 py-2 border border-border rounded" onClick={()=>setDetail(null)}>Close</button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
              <label className="md:col-span-1">Name<input className="w-full border border-border rounded px-2 py-1 bg-surface" value={edit.name} onChange={e=>setEdit({...edit, name:e.target.value})} /></label>
              <label className="md:col-span-2">Description<textarea className="w-full border border-border rounded px-2 py-1 bg-surface h-20" value={edit.description} onChange={e=>setEdit({...edit, description:e.target.value})} /></label>
              <label className="md:col-span-3">Tags<input className="w-full border border-border rounded px-2 py-1 bg-surface" value={edit.tags} onChange={e=>setEdit({...edit, tags:e.target.value})} placeholder="comma,separated" /></label>
              <div className="md:col-span-3 flex justify-end"><button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={saveDetail}>Save</button></div>
            </div>
            <div>
              <div className="text-sm font-semibold mb-1">Runs</div>
              <table className="w-full text-sm">
                <thead className="bg-muted border-b border-border"><tr><th className="text-left px-2 py-1">ID</th><th className="text-left px-2 py-1">Name</th><th className="text-left px-2 py-1">Status</th><th className="text-left px-2 py-1">Framework</th><th className="text-left px-2 py-1">Created</th></tr></thead>
                <tbody className="divide-y divide-border">
                  {(detail.runs||[]).map(r => (
                    <tr key={r.id}><td className="px-2 py-1 text-xs">{r.id.slice(0,8)}</td><td className="px-2 py-1">{r.name}</td><td className="px-2 py-1">{r.status}</td><td className="px-2 py-1">{r.framework}</td><td className="px-2 py-1 text-xs">{new Date(r.created).toLocaleString()}</td></tr>
                  ))}
                  {(!detail.runs || detail.runs.length===0) && (<tr><td className="px-2 py-1 text-xs text-text/60" colSpan={5}>No runs</td></tr>)}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

