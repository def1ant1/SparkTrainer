import React, { useEffect, useMemo, useState } from 'react';

export function DatasetsPage({ api }){
  const [items, setItems] = useState([]);
  const [q, setQ] = useState('');
  const [view, setView] = useState('grid');
  const [sel, setSel] = useState(null); // selected dataset name
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [upName, setUpName] = useState('');
  const [upVer, setUpVer] = useState('');
  const [upFile, setUpFile] = useState(null);
  const [sync, setSync] = useState({ provider: 's3', direction: 'upload', bucket: '', prefix: '' });
  const [syncOut, setSyncOut] = useState('');

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
    return items.filter(d => !qq || d.name.toLowerCase().includes(qq));
  }, [items, q]);

  const openDetail = async (name) => {
    setSel(name); setDetail(null);
    try{
      const d = await api.getDataset(name);
      setDetail(d);
    } catch {}
  };

  const upload = async () => {
    if (!upName || !upFile){ alert('Pick a name and a file'); return; }
    try{
      await api.uploadDataset(upName, upFile, upVer);
      setUpFile(null); setUpName(''); setUpVer('');
      load();
      alert('Uploaded');
    } catch(e){ alert('Upload failed: ' + e.message); }
  };

  const doSync = async () => {
    try{
      const out = await api.syncDatasets(sync);
      setSyncOut(JSON.stringify(out, null, 2));
    } catch(e){ setSyncOut(String(e)); }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Datasets</h1>
        <div className="flex gap-2">
          <input className="border rounded px-3 py-2" placeholder="Search" value={q} onChange={e=>setQ(e.target.value)} />
          <button onClick={()=>setView(view==='grid'?'list':'grid')} className="px-3 py-2 border rounded">{view==='grid'?'List View':'Grid View'}</button>
        </div>
      </div>

      <div className="bg-surface p-4 rounded border border-border space-y-3">
        <div className="font-semibold text-gray-700">Upload New Version</div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <input className="border rounded px-3 py-2" placeholder="Dataset name" value={upName} onChange={e=>setUpName(e.target.value)} />
          <input className="border rounded px-3 py-2" placeholder="Version (optional)" value={upVer} onChange={e=>setUpVer(e.target.value)} />
          <input type="file" onChange={e=>setUpFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)} />
          <button onClick={upload} className="px-3 py-2 bg-blue-600 text-white rounded">Upload</button>
        </div>
        <div className="text-xs text-gray-500">Tip: upload a .zip to extract on the server; otherwise file is stored as-is inside the version folder.</div>
      </div>

      {loading ? <div>Loading...</div> : (
        view==='grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {filtered.map(d => (
              <div key={d.name} className="bg-surface p-4 rounded border border-border hover:shadow cursor-pointer" onClick={()=>openDetail(d.name)}>
                <div className="font-semibold">{d.name}</div>
                <div className="text-xs text-text/70">Versions: {d.versions}</div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                  <div className="p-2 bg-muted rounded border border-border"><div className="text-text/70">Files</div><div className="font-semibold">{d.stats.total_files}</div></div>
                  <div className="p-2 bg-muted rounded border border-border"><div className="text-text/70">Size</div><div className="font-semibold">{(d.stats.total_bytes/1e6).toFixed(1)} MB</div></div>
                  <div className="p-2 bg-muted rounded border border-border"><div className="text-text/70">Images</div><div className="font-semibold">{d.stats.image_count}</div></div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <table className="w-full">
            <thead className="bg-muted border-b border-border"><tr><th className="px-3 py-2 text-left">Name</th><th className="px-3 py-2 text-left">Versions</th><th className="px-3 py-2 text-left">Files</th><th className="px-3 py-2 text-left">Size</th></tr></thead>
            <tbody className="divide-y">
              {filtered.map(d => (
                <tr key={d.name} className="hover:bg-muted" onClick={()=>openDetail(d.name)}>
                  <td className="px-3 py-2 font-semibold">{d.name}</td>
                  <td className="px-3 py-2">{d.versions}</td>
                  <td className="px-3 py-2">{d.stats.total_files}</td>
                  <td className="px-3 py-2">{(d.stats.total_bytes/1e6).toFixed(1)} MB</td>
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
              <div className="text-2xl font-bold">{detail.name}</div>
              <div className="text-sm text-text/70">Versions: {detail.versions?.length || 0}</div>
            </div>
            <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={()=>{setSel(null); setDetail(null);}}>Close</button>
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
                      <td className="px-2 py-1"><a className="text-primary" href={`/api/datasets/${encodeURIComponent(detail.name)}/download?version=${encodeURIComponent(v.version)}`}>Download</a></td>
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
        </div>
      )}
    </div>
  );
}
