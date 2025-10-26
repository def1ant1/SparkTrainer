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

export function DatasetsPage({ api }){
  const [items, setItems] = useState([]);
  const [q, setQ] = useState('');
  const [view, setView] = useState('grid');
  const [sortBy, setSortBy] = useState('date'); // date|size|name|files
  const [typeFilter, setTypeFilter] = useState('');
  const [tagFilter, setTagFilter] = useState('');
  const [sel, setSel] = useState(null); // selected dataset name
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [upName, setUpName] = useState('');
  const [upVer, setUpVer] = useState('');
  const [upFile, setUpFile] = useState(null);
  const [sync, setSync] = useState({ provider: 's3', direction: 'upload', bucket: '', prefix: '' });
  const [syncOut, setSyncOut] = useState('');
  // Ingest wizard
  const [ingType, setIngType] = useState('csv');
  const [ingText, setIngText] = useState('');
  const [ingPrev, setIngPrev] = useState(null);
  const [ingMap, setIngMap] = useState({ text: 'text', label: '' });
  const [ingName, setIngName] = useState('');
  const [ingVer, setIngVer] = useState('v1');
  // Streaming ingest (large files)
  const [streamName, setStreamName] = useState('');
  const [streamVer, setStreamVer] = useState('v1');
  const [streamType, setStreamType] = useState('csv');
  const [streamFile, setStreamFile] = useState(null);
  const [streamHeader, setStreamHeader] = useState('');
  const [streamMap, setStreamMap] = useState(''); // key:value per line
  const [streamProgress, setStreamProgress] = useState(0);
  const [samples, setSamples] = useState({ items: [], total: 0, page: 0 });
  const [quality, setQuality] = useState(null);
  const [metaEditing, setMetaEditing] = useState({ description: '', tags: '', categories: '', type: '' });
  const pageSize = 12;
  const [template, setTemplate] = useState({ name:'', template:'image_classification', version:'v1' });
  // Video processing
  const [videoPath, setVideoPath] = useState('');
  const [videoIndexJob, setVideoIndexJob] = useState(null);
  const toast = useToast();

  const load = async () => {
    setLoading(true);
    try{
      const data = await api.getDatasets();
      setItems(data);
    } finally { setLoading(false); }
  };
  useEffect(()=>{ load(); }, []);

  const filtered = useMemo(()=>{
    const qq = q.trim().toLowerCase();
    let arr = items.filter(d => !qq || d.name.toLowerCase().includes(qq) || (d.meta?.description||'').toLowerCase().includes(qq));
    if (typeFilter) arr = arr.filter(d => (d.meta?.type||'').toLowerCase() === typeFilter.toLowerCase());
    if (tagFilter) arr = arr.filter(d => (d.meta?.tags||[]).map(x=>String(x).toLowerCase()).includes(tagFilter.toLowerCase()));
    const key = (d) => sortBy==='size' ? (d.stats?.total_bytes||0) : sortBy==='files' ? (d.stats?.total_files||0) : sortBy==='name' ? d.name.toLowerCase() : (d.latest||'');
    return arr.sort((a,b)=> key(a) > key(b) ? -1 : 1);
  }, [items, q, typeFilter, tagFilter, sortBy]);

  const openDetail = async (name) => {
    setSel(name); setDetail(null);
    try{
      const d = await api.getDataset(name);
      setDetail(d);
      setMetaEditing({ description: d.meta?.description||'', tags: (d.meta?.tags||[]).join(','), categories: (d.meta?.categories||[]).join(','), type: d.meta?.type||'' });
      loadSamples(name, d.versions?.[0]?.version);
    } catch {}
  };

  const loadSamples = async (name, version, page=0) => {
    try{
      const res = await api.getDatasetSamples(name, { version, kind:'image', offset: page*pageSize, limit: pageSize });
      setSamples({ items: res.items||[], total: res.total||0, page });
    } catch { setSamples({ items:[], total:0, page:0 }); }
  };

  const upload = async () => {
    if (!upName || !upFile){ toast.push({type:'warning', title:'Missing fields', message:'Pick a name and a file'}); return; }
    try{
      await api.uploadDataset(upName, upFile, upVer);
      setUpFile(null); setUpName(''); setUpVer('');
      load();
      toast.push({type:'success', title:'Dataset uploaded'});
    } catch(e){ toast.push({type:'error', title:'Upload failed', message:e.message}); }
  };

  const deleteDataset = async (name) => {
    if (!confirm(`Delete dataset "${name}" and all its versions? This cannot be undone.`)) return;
    try {
      const res = await fetch(`/api/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Delete failed');
      }
      if (sel === name) {
        setSel(null);
        setDetail(null);
      }
      toast.push({ type: 'success', title: 'Dataset deleted' });
      load();
    } catch (e) {
      toast.push({ type: 'error', title: 'Delete failed', message: String(e.message || e) });
    }
  };

  const doSync = async () => {
    try{
      const out = await api.syncDatasets(sync);
      setSyncOut(JSON.stringify(out, null, 2));
    } catch(e){ setSyncOut(String(e)); }
  };

  const doStreamIngest = async () => {
    if (!streamName || !streamFile){ toast.push({type:'warning', title:'Missing fields', message:'Provide dataset name and choose a file'}); return; }
    try{
      setStreamProgress(0);
      const start = await api.ingestStreamStart({ name: streamName, version: streamVer, filename: streamFile.name, type: streamType });
      const session = start.session;
      const chunkSize = 4 * 1024 * 1024; // 4 MiB
      let offset = 0, index = 0;
      while (offset < streamFile.size){
        const chunk = streamFile.slice(offset, offset + chunkSize);
        const fd = new FormData();
        fd.append('session', session);
        fd.append('index', String(index));
        fd.append('file', chunk, `${streamFile.name}.part${index}`);
        await api.ingestStreamChunk(fd);
        offset += chunk.size;
        index += 1;
        setStreamProgress(Math.round((offset/streamFile.size)*100));
      }
      const mapping = {};
      (streamMap.split('\n')||[]).map(s=>s.trim()).filter(Boolean).forEach(line => {
        const [k, v] = line.split(':'); if (k && v) mapping[k.trim()] = v.trim();
      });
      const header = (streamHeader||'').split(',').map(s=>s.trim()).filter(Boolean);
      const fin = await api.ingestStreamFinalize({ session, mapping, header, type: streamType });
      if (fin.status === 'ok') { toast.push({type:'success', title:'Dataset ingested'}); load(); setStreamProgress(0); }
      else { toast.push({type:'error', title:'Ingest failed', message:JSON.stringify(fin)}); }
    } catch(e){ toast.push({type:'error', title:'Streaming ingest failed', message:e.message}); }
  };

  const indexVideos = async () => {
    if (!videoPath){ toast.push({type:'warning', title:'Missing path', message:'Provide video directory path'}); return; }
    try {
      const res = await fetch('/api/datasets/index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: upName || `video-dataset-${Date.now()}`,
          source_path: videoPath,
          version: upVer || 'v1',
          recursive: true,
          extract_metadata: true
        })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setVideoIndexJob(data.job_id);
        toast.push({type:'success', title:'Video indexing started', message:`Job ID: ${data.job_id}`});
      } else {
        toast.push({type:'error', title:'Index failed', message:data.error || 'Unknown error'});
      }
    } catch (e) {
      toast.push({type:'error', title:'Index failed', message:e.message});
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Datasets</h1>
        <div className="flex flex-wrap gap-2 items-center">
          <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Search" value={q} onChange={e=>setQ(e.target.value)} />
          <select className="border border-border rounded px-2 py-2 bg-surface" value={typeFilter} onChange={e=>setTypeFilter(e.target.value)}>
            <option value="">All Types</option>
            {['image_classification','text_generation','qa','instruction_tuning','conversational'].map(t=> <option key={t} value={t}>{t}</option>)}
          </select>
          <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Filter tag" value={tagFilter} onChange={e=>setTagFilter(e.target.value)} />
          <select className="border border-border rounded px-2 py-2 bg-surface" value={sortBy} onChange={e=>setSortBy(e.target.value)}>
            <option value="date">Sort: Date</option>
            <option value="size">Sort: Size</option>
            <option value="files">Sort: Files</option>
            <option value="name">Sort: Name</option>
          </select>
          <button onClick={()=>setView(view==='grid'?'list':'grid')} className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted">{view==='grid'?'List View':'Grid View'}</button>
        </div>
      </div>

      <div className="bg-surface p-4 rounded border border-border space-y-3">
        <div className="font-semibold text-text">Create / Upload</div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Dataset name" value={upName} onChange={e=>setUpName(e.target.value)} />
          <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Version (optional)" value={upVer} onChange={e=>setUpVer(e.target.value)} />
          <input type="file" className="bg-surface" onChange={e=>setUpFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)} />
          <button onClick={upload} className="px-3 py-2 bg-blue-600 text-white rounded">Upload</button>
        </div>
        <div className="text-xs text-text/60">Tip: upload a .zip to extract on the server; otherwise file is stored as-is inside the version folder.</div>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-2 mt-3 items-end">
          <div className="text-sm font-semibold md:col-span-5">Templates</div>
          <input className="border border-border rounded px-2 py-1 bg-surface" placeholder="Name" value={template.name} onChange={e=>setTemplate({...template, name:e.target.value})}/>
          <select className="border border-border rounded px-2 py-1 bg-surface" value={template.template} onChange={e=>setTemplate({...template, template:e.target.value})}>
            <option value="image_classification">Image classification</option>
            <option value="text_generation">Text generation</option>
            <option value="qa">Question answering</option>
            <option value="instruction_tuning">Instruction tuning</option>
            <option value="conversational">Conversational</option>
          </select>
          <input className="border border-border rounded px-2 py-1 bg-surface" placeholder="Version" value={template.version} onChange={e=>setTemplate({...template, version:e.target.value})}/>
          <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={async()=>{ const res = await api.createDatasetTemplate(template); if (res.status==='ok'){ load(); toast.push({type:'success', title:'Template created'}); } else toast.push({type:'error', title:'Failed', message:JSON.stringify(res)}); }}>Create from Template</button>
        </div>
      </div>

      <div className="bg-surface p-4 rounded border border-border space-y-3">
        <div className="font-semibold text-text">Video Dataset Indexing</div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Video directory path" value={videoPath} onChange={e=>setVideoPath(e.target.value)} />
          <button onClick={indexVideos} className="px-3 py-2 bg-primary text-on-primary rounded hover:brightness-110">Index Videos</button>
          {videoIndexJob && (
            <div className="md:col-span-2 text-sm text-text/70">
              Job ID: <span className="font-mono text-xs">{videoIndexJob}</span>
            </div>
          )}
        </div>
        <div className="text-xs text-text/60">Point to a directory containing video files. The system will scan, extract metadata, and create a manifest for training.</div>
      </div>

      <div className="bg-surface p-4 rounded border border-border space-y-3">
        <div className="font-semibold text-text">Large File Ingest (Streaming)</div>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-end">
          <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Dataset name" value={streamName} onChange={e=>setStreamName(e.target.value)} />
          <input className="border border-border rounded px-3 py-2 bg-surface" placeholder="Version" value={streamVer} onChange={e=>setStreamVer(e.target.value)} />
          <select className="border border-border rounded px-3 py-2 bg-surface" value={streamType} onChange={e=>setStreamType(e.target.value)}>
            <option value="csv">CSV</option>
            <option value="jsonl">JSONL</option>
          </select>
          <input type="file" className="bg-surface" onChange={e=>setStreamFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)} />
          <button className="px-3 py-2 border rounded bg-surface hover:bg-muted" onClick={doStreamIngest}>Upload & Map</button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="text-sm">CSV Header (comma-separated)</label>
            <input className="w-full border border-border rounded px-3 py-2 bg-surface font-mono text-xs" value={streamHeader} onChange={e=>setStreamHeader(e.target.value)} placeholder="id,text,label" />
            <div className="text-[11px] text-text/60 mt-1">For JSONL, leave header empty.</div>
          </div>
          <div>
            <label className="text-sm">Server-side Mapping (key:value per line)</label>
            <textarea className="w-full border border-border rounded px-3 py-2 bg-surface font-mono text-xs" rows={4} value={streamMap} onChange={e=>setStreamMap(e.target.value)} placeholder="text:text\nlabel:label" />
          </div>
        </div>
        <div className="text-xs text-text/70">Progress: {streamProgress}%</div>
        <div className="w-full h-2 bg-muted rounded"><div className="h-2 bg-primary rounded" style={{ width: `${streamProgress}%` }} /></div>
      </div>

      <div className="bg-surface p-4 rounded border border-border space-y-3">
        <div className="font-semibold text-text">Ingest Wizard (CSV / JSONL)</div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
          <div>
            <label className="text-sm">Dataset</label>
            <input className="w-full border border-border rounded px-3 py-2 bg-surface" placeholder="name" value={ingName} onChange={e=>setIngName(e.target.value)} />
          </div>
          <div>
            <label className="text-sm">Version</label>
            <input className="w-full border border-border rounded px-3 py-2 bg-surface" placeholder="v1" value={ingVer} onChange={e=>setIngVer(e.target.value)} />
          </div>
          <div>
            <label className="text-sm">Type</label>
            <select className="w-full border border-border rounded px-3 py-2 bg-surface" value={ingType} onChange={e=>setIngType(e.target.value)}>
              <option value="csv">CSV</option>
              <option value="jsonl">JSONL</option>
            </select>
          </div>
          <div className="flex gap-2">
            <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={async()=>{ const res = await api.ingestPreview({ type: ingType, text: ingText }); setIngPrev(res); if (res.columns) setIngMap(m=>({ ...m, text: res.columns[0]||'text' })); }}>Preview</button>
            <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={async()=>{
              if (!ingPrev) { toast.push({type:'warning', title:'Preview required', message:'Run preview first'}); return; }
              const payload = { name: ingName, version: ingVer, type: ingType, header: ingPrev.columns, mapping: ingMap, rows: ingPrev.rows };
              const out = await api.ingestApply(payload); if (out.status==='ok'){ toast.push({type:'success', title:'Dataset ingested'}); setIngText(''); load(); } else toast.push({type:'error', title:'Failed', message:JSON.stringify(out)});
            }}>Apply</button>
          </div>
        </div>
        <div>
          <label className="text-sm">Paste CSV or JSONL sample</label>
          <textarea className="w-full border border-border rounded px-3 py-2 bg-surface font-mono text-xs" rows={8} value={ingText} onChange={e=>setIngText(e.target.value)} placeholder="id,text,label\n1,hello,positive\n2,bye,negative" />
        </div>
        {ingPrev && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-sm font-semibold">Preview</div>
              <table className="w-full text-xs">
                <thead className="bg-muted border-b border-border"><tr>{(ingPrev.columns||[]).map((c,i)=>(<th key={i} className="text-left px-2 py-1">{c}</th>))}</tr></thead>
                <tbody className="divide-y divide-border">
                  {(ingPrev.rows||[]).map((r,i)=>(
                    <tr key={i}>{(Array.isArray(r) ? r : (ingPrev.columns||[]).map(k=>r[k])).map((v,j)=>(<td key={j} className="px-2 py-1">{String(v)}</td>))}</tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <div className="text-sm font-semibold">Mapping</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {['text','label','instruction','input','output'].map(k => (
                  <label key={k} className="block">
                    <div className="text-[11px] text-text/60">{k}</div>
                    <select className="w-full border border-border rounded px-2 py-1 bg-surface" value={ingMap[k]||''} onChange={e=>setIngMap({...ingMap, [k]: e.target.value})}>
                      <option value="">(none)</option>
                      {(ingPrev.columns||[]).map(c => (<option key={c} value={c}>{c}</option>))}
                    </select>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {loading ? <div>Loading...</div> : (
        view==='grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {filtered.map(d => (
              <div key={d.name} className="bg-surface p-4 rounded border border-border hover:shadow group relative">
                <div className="cursor-pointer" onClick={()=>openDetail(d.name)}>
                  <div className="font-semibold">{d.name}</div>
                  <div className="text-xs text-text/70">Versions: {d.versions}</div>
                  {Array.isArray(d.meta?.tags) && d.meta.tags.length>0 && (
                    <div className="mt-1 text-[10px] text-text/60">{d.meta.tags.slice(0,3).join(', ')}{d.meta.tags.length>3?' …':''}</div>
                  )}
                  <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                    <div className="p-2 bg-muted rounded border border-border"><div className="text-text/70">Files</div><div className="font-semibold">{d.stats.total_files}</div></div>
                    <div className="p-2 bg-muted rounded border border-border"><div className="text-text/70">Size</div><div className="font-semibold">{(d.stats.total_bytes/1e6).toFixed(1)} MB</div></div>
                    <div className="p-2 bg-muted rounded border border-border"><div className="text-text/70">Images</div><div className="font-semibold">{d.stats.image_count}</div></div>
                  </div>
                </div>
                <button
                  onClick={(e)=>{e.stopPropagation(); deleteDataset(d.name);}}
                  className="absolute top-2 right-2 px-2 py-1 text-xs bg-danger/10 text-danger border border-danger/30 rounded opacity-0 group-hover:opacity-100 transition-opacity hover:bg-danger/20"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        ) : (
          <table className="w-full">
            <thead className="bg-muted border-b border-border"><tr><th className="px-3 py-2 text-left">Name</th><th className="px-3 py-2 text-left">Versions</th><th className="px-3 py-2 text-left">Files</th><th className="px-3 py-2 text-left">Size</th><th className="px-3 py-2 text-left">Actions</th></tr></thead>
            <tbody className="divide-y">
              {filtered.map(d => (
                <tr key={d.name} className="hover:bg-muted">
                  <td className="px-3 py-2 font-semibold cursor-pointer" onClick={()=>openDetail(d.name)}>{d.name}</td>
                  <td className="px-3 py-2">{d.versions}</td>
                  <td className="px-3 py-2">{d.stats.total_files}</td>
                  <td className="px-3 py-2">{(d.stats.total_bytes/1e6).toFixed(1)} MB</td>
                  <td className="px-3 py-2">
                    <button onClick={()=>deleteDataset(d.name)} className="text-danger hover:brightness-110 text-sm">Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      )}

      {sel && detail && (
        <div className="bg-surface p-4 rounded border border-border space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold flex items-center gap-2">
                <span>{detail.name}</span>
                {detail.meta?.dvc_detected && (<span className="px-2 py-0.5 text-xs rounded bg-accent/10 text-accent border border-accent/30">DVC</span>)}
              </div>
              <div className="text-sm text-text/70">Versions: {detail.versions?.length || 0}</div>
            </div>
            <div className="flex gap-2">
              <button className="px-3 py-2 border border-danger/30 rounded bg-danger/10 text-danger hover:bg-danger/20" onClick={()=>deleteDataset(detail.name)}>Delete</button>
              <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={()=>{setSel(null); setDetail(null);}}>Close</button>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-text/70 mb-2">Versions</div>
              <table className="w-full text-sm">
                <thead className="bg-muted border-b border-border"><tr><th className="text-left px-2 py-1">Version</th><th className="text-left px-2 py-1">Created</th><th className="text-left px-2 py-1">Files</th><th className="text-left px-2 py-1">Size</th><th></th></tr></thead>
                <tbody className="divide-y divide-border">
                  {(detail.versions||[]).map(v => (
                    <tr key={v.version}>
                      <td className="px-2 py-1">{v.version}</td>
                      <td className="px-2 py-1">{v.created || '-'}</td>
                      <td className="px-2 py-1">{v.stats?.total_files ?? '-'}</td>
                      <td className="px-2 py-1">{v.stats ? (v.stats.total_bytes/1e6).toFixed(1)+' MB' : '-'}</td>
                      <td className="px-2 py-1 space-x-2">
                        <a className="text-primary" href={`/api/datasets/${encodeURIComponent(detail.name)}/download?version=${encodeURIComponent(v.version)}`}>Download</a>
                        <button className="text-secondary" onClick={async()=>{
                          const base = prompt('New version name based on '+v.version+':'); if (!base) return;
                          const res = await api.createDatasetVersion(detail.name, { base: v.version, new: base }); if (res.status==='ok') { load(); toast.push({type:'success', title:'Version created', message:base}); } else toast.push({type:'error', title:'Failed', message:JSON.stringify(res)});
                        }}>Branch</button>
                        <button className="text-warning" onClick={async()=>{
                          const target = v.version; if (!confirm('Rollback to '+target+'?')) return;
                          const res = await api.rollbackDatasetVersion(detail.name, { target }); if (res.status==='ok') { load(); toast.push({type:'success', title:'Rollback created', message:`Version ${res.version}`}); } else toast.push({type:'error', title:'Failed', message:JSON.stringify(res)});
                        }}>Rollback</button>
                        <button className="text-primary" onClick={async()=>{
                          const other = prompt('Diff against version:'); if (!other) return;
                          const res = await api.diffDatasetVersions(detail.name, { a: v.version, b: other }); if (res.added || res.removed || res.changed) {
                            toast.push({type:'info', title:'Diff result', message:`Added: ${res.added.length}, Removed: ${res.removed.length}, Changed: ${res.changed.length}`});
                          } else { toast.push({type:'info', title:'Diff result', message:JSON.stringify(res)}); }
                        }}>Diff</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <div className="text-sm text-text/70 mb-2">External Sync</div>
              <div className="grid grid-cols-2 gap-2">
                <select className="border border-border rounded px-2 py-1 bg-surface" value={sync.provider} onChange={e=>setSync({...sync, provider: e.target.value})}>
                  <option value="s3">AWS S3</option>
                  <option value="gcs">GCS</option>
                  <option value="azure">Azure</option>
                  <option value="minio">MinIO</option>
                </select>
                <select className="border border-border rounded px-2 py-1 bg-surface" value={sync.direction} onChange={e=>setSync({...sync, direction: e.target.value})}>
                  <option value="upload">Upload</option>
                  <option value="download">Download</option>
                </select>
                <input className="border border-border rounded px-2 py-1 bg-surface" placeholder="Bucket/Container or mc alias" value={sync.bucket} onChange={e=>setSync({...sync, bucket: e.target.value})} />
                <input className="border border-border rounded px-2 py-1 bg-surface" placeholder="Prefix/Path" value={sync.prefix} onChange={e=>setSync({...sync, prefix: e.target.value})} />
                <div className="col-span-2"><button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={()=>doSync()}>Run Sync</button></div>
                {syncOut && <pre className="col-span-2 bg-muted rounded border border-border p-2 text-xs overflow-auto max-h-48">{syncOut}</pre>}
              </div>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="text-sm font-semibold">Metadata</div>
              <label className="text-xs">Type<select className="w-full border border-border rounded px-2 py-1 bg-surface" value={metaEditing.type} onChange={e=>setMetaEditing({...metaEditing, type:e.target.value})}><option value="">-</option><option value="image_classification">image_classification</option><option value="text_generation">text_generation</option><option value="qa">qa</option><option value="instruction_tuning">instruction_tuning</option><option value="conversational">conversational</option></select></label>
              <label className="text-xs">Tags<input className="w-full border border-border rounded px-2 py-1 bg-surface" placeholder="comma-separated" value={metaEditing.tags} onChange={e=>setMetaEditing({...metaEditing, tags:e.target.value})}/></label>
              <label className="text-xs">Categories<input className="w-full border border-border rounded px-2 py-1 bg-surface" placeholder="comma-separated" value={metaEditing.categories} onChange={e=>setMetaEditing({...metaEditing, categories:e.target.value})}/></label>
              <label className="text-xs">Description<textarea className="w-full border border-border rounded px-2 py-1 bg-surface" rows={3} value={metaEditing.description} onChange={e=>setMetaEditing({...metaEditing, description:e.target.value})}/></label>
              <div><button className="px-3 py-1 border rounded" onClick={async()=>{ const payload={...metaEditing, tags: metaEditing.tags.split(',').map(s=>s.trim()).filter(Boolean), categories: metaEditing.categories.split(',').map(s=>s.trim()).filter(Boolean)}; const res = await api.updateDatasetMetadata(detail.name, payload); if (res.status==='ok'){ toast.push({type:'success', title:'Metadata saved'}); load(); } else toast.push({type:'error', title:'Failed', message:JSON.stringify(res)}); }}>Save</button></div>
            </div>
            <div className="space-y-2">
              <div className="text-sm font-semibold">Samples</div>
              <div className="grid grid-cols-3 gap-2">
                {samples.items.map((s,i) => (
                  <div key={i} className="border border-border rounded overflow-hidden bg-muted">
                    <img src={`/api/datasets/${encodeURIComponent(detail.name)}/file?version=${encodeURIComponent(detail.versions?.[0]?.version||'')}&path=${encodeURIComponent(s.path)}`} className="w-full h-24 object-cover" />
                    <div className="p-1 text-[10px] truncate">{s.path}</div>
                  </div>
                ))}
                {samples.items.length===0 && <div className="text-xs text-text/60">No samples</div>}
              </div>
              <div className="flex gap-2 text-xs items-center">
                <button className="px-2 py-1 border rounded" disabled={samples.page<=0} onClick={()=>loadSamples(detail.name, detail.versions?.[0]?.version, Math.max(0, samples.page-1))}>Prev</button>
                <span>Page {samples.page+1} of {Math.max(1, Math.ceil(samples.total/pageSize))}</span>
                <button className="px-2 py-1 border rounded" disabled={(samples.page+1)>=Math.ceil(samples.total/pageSize)} onClick={()=>loadSamples(detail.name, detail.versions?.[0]?.version, samples.page+1)}>Next</button>
              </div>
              <div className="mt-2">
                <button className="px-3 py-1 border rounded text-sm" onClick={async()=>{ try{ const v = detail.versions?.[0]?.version; const res = await fetch(`/api/datasets/${encodeURIComponent(detail.name)}/quality?version=${encodeURIComponent(v||'')}&near=1`); const q = await res.json(); setQuality(q); toast.push({type:'success', title:'Quality check complete'}); }catch(e){ toast.push({type:'error', title:'Quality check failed', message:e.message}); } }}>Run Quality Check</button>
                {quality && quality.suggestions && (
                  <button className="ml-2 px-3 py-1 border rounded text-sm" onClick={async()=>{
                    const v = detail.versions?.[0]?.version; const sugg = quality.suggestions || {};
                    const payload = { version: v, remove: sugg.duplicates_to_remove||[], balance: sugg.balance_plan||{} };
                    const out = await api.applyQuality(detail.name, payload);
                    if (out.status==='ok'){ toast.push({type:'success', title:'Suggestions applied'}); load(); } else toast.push({type:'error', title:'Failed', message:JSON.stringify(out)});
                  }}>Apply Suggestions</button>
                )}
              </div>
              {quality && (
                <div className="mt-2 space-y-2 text-xs">
                  <div className="font-semibold">Duplicates</div>
                  {Object.keys(quality.duplicates||{}).length===0 ? <div className="text-text/60">No duplicates detected</div> : (
                    <div className="max-h-32 overflow-auto border border-border rounded p-2 bg-muted">
                      {Object.entries(quality.duplicates).slice(0,20).map(([hash,paths]) => (
                        <div key={hash} className="mb-2">
                          <div className="text-[10px] text-text/60">{hash}</div>
                          {paths.map((p,i)=>(<div key={i}>{p}</div>))}
                        </div>
                      ))}
                      {Object.keys(quality.duplicates).length>20 && <div>… more</div>}
                    </div>
                  )}
                  <div className="font-semibold">Class Counts</div>
                  <div className="space-y-1">
                    {Object.entries(quality.class_counts||{}).map(([cls,cnt]) => (
                      <div key={cls} className="flex items-center gap-2">
                        <div className="w-32">{cls}</div>
                        <div className="flex-1 bg-muted h-2 rounded"><div className="bg-accent h-2 rounded" style={{ width: `${Math.min(100, (cnt/Math.max(1, Object.values(quality.class_counts||{}).reduce((a,b)=>a+b,0)))*100)}%` }} /></div>
                        <div className="w-10 text-right">{cnt}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="text-sm font-semibold">Curation (beta)</div>
              <div className="text-xs text-text/70">Integrates with local Ollama or OpenAI if configured on server.</div>
              <div className="flex gap-2">
                <button className="px-3 py-1 border rounded" onClick={async()=>{ const res = await fetch('/api/curation/ollama/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model:'llama3', task:'label', prompt:'Label samples' })}); alert(await res.text()); }}>Ollama Label</button>
                <button className="px-3 py-1 border rounded" onClick={async()=>{ const res = await fetch('/api/curation/openai/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model:'gpt-4o-mini', task:'synth' })}); alert(await res.text()); }}>OpenAI Synthesize</button>
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-sm font-semibold">Stats</div>
              <div className="text-xs">By extension</div>
              <div className="space-y-1">
                {Object.entries((detail.versions?.[0]?.stats?.by_extension)||{}).slice(0,8).map(([ext,count]) => (
                  <div key={ext} className="flex items-center gap-2 text-xs">
                    <div className="w-24">{ext||'(none)'}</div>
                    <div className="flex-1 bg-muted h-2 rounded"><div className="bg-blue-500 h-2 rounded" style={{ width: `${Math.min(100, count/((detail.versions?.[0]?.stats?.total_files||1))*100)}%` }} /></div>
                    <div className="w-10 text-right">{count}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
