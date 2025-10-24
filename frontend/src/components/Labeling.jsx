import React, { useEffect, useMemo, useState } from 'react';

export default function Labeling({ api }){
  const [datasets, setDatasets] = useState([]);
  const [name, setName] = useState('');
  const [version, setVersion] = useState('');
  const [detail, setDetail] = useState(null);
  const [mode, setMode] = useState('text_cls'); // text_cls | ner | image_bbox
  const [labels, setLabels] = useState('positive,negative');
  const [samples, setSamples] = useState([]);
  const [offset, setOffset] = useState(0);
  const [annos, setAnnos] = useState({}); // { path: { label: '...', spans: [{start,end,label}] } }
  const pageSize = 12;

  useEffect(() => { (async ()=>{ try{ setDatasets(await api.getDatasets()); }catch{}})(); }, []);

  useEffect(() => { (async ()=>{
    if (!name) return;
    const d = await api.getDataset(name); setDetail(d);
    if (!version && d.versions && d.versions[0]) setVersion(d.versions[0].version);
  })(); }, [name]);

  useEffect(() => { (async ()=>{
    if (!name || !version) return;
    const kind = mode==='image_bbox' ? 'image' : 'text';
    const s = await api.getDatasetSamples(name, { version, kind, offset, limit: pageSize });
    setSamples(s.items||[]);
    try { const a = await api.getAnnotations(name, { version }); const map={}; (a.items||[]).forEach(it => { map[it.path]=it; }); setAnnos(map); } catch {}
  })(); }, [name, version, offset, mode]);

  const labelList = useMemo(()=> labels.split(',').map(s=>s.trim()).filter(Boolean), [labels]);

  const updateAnno = (path, patch) => { setAnnos(prev => ({ ...prev, [path]: { ...(prev[path]||{ path }), ...patch } })); };

  const save = async () => {
    const items = Object.values(annos);
    const res = await api.saveAnnotations(name, { version, items });
    if (res.status==='ok') alert('Saved'); else alert(JSON.stringify(res));
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div>
          <label className="text-sm">Dataset</label>
          <select className="block border rounded px-3 py-2" value={name} onChange={e=>{ setName(e.target.value); setVersion(''); setOffset(0); }}>
            <option value="">Choose…</option>
            {(datasets||[]).map(d => (<option key={d.name} value={d.name}>{d.name}</option>))}
          </select>
        </div>
        <div>
          <label className="text-sm">Version</label>
          <select className="block border rounded px-3 py-2" value={version} onChange={e=>{ setVersion(e.target.value); setOffset(0); }}>
            {(detail?.versions||[]).map(v => (<option key={v.version} value={v.version}>{v.version}</option>))}
          </select>
        </div>
        <div>
          <label className="text-sm">Mode</label>
          <select className="block border rounded px-3 py-2" value={mode} onChange={e=>setMode(e.target.value)}>
            <option value="text_cls">Text Classification</option>
            <option value="ner">NER (beta)</option>
            <option value="image_bbox">Image Bounding Boxes</option>
          </select>
        </div>
        {mode==='text_cls' && (
          <div>
            <label className="text-sm">Labels</label>
            <input className="border rounded px-3 py-2" value={labels} onChange={e=>setLabels(e.target.value)} placeholder="comma-separated" />
          </div>
        )}
        <div className="ml-auto flex items-end gap-2">
          {mode==='image_bbox' && name && version && (
            <>
              <button className="px-3 py-2 border rounded" onClick={async()=>{
                try{
                  const res = await api.exportYolo(name, { version });
                  const blob = await res.blob();
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a'); a.href = url; a.download = `${name}_${version}_yolo.zip`; a.click(); URL.revokeObjectURL(url);
                }catch(e){ alert('YOLO export failed'); }
              }}>Export YOLO</button>
              <button className="px-3 py-2 border rounded" onClick={async()=>{
                try{
                  const coco = await api.exportCoco(name, { version });
                  const blob = new Blob([JSON.stringify(coco)], { type:'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a'); a.href = url; a.download = `${name}_${version}_coco.json`; a.click(); URL.revokeObjectURL(url);
                }catch(e){ alert('COCO export failed'); }
              }}>Export COCO</button>
            </>
          )}
          <button className="px-3 py-2 border rounded" onClick={()=> setOffset(Math.max(0, offset-pageSize))}>Prev</button>
          <button className="px-3 py-2 border rounded" onClick={()=> setOffset(offset+pageSize)}>Next</button>
          <button className="px-3 py-2 border rounded bg-primary text-on-primary" onClick={save}>Save</button>
        </div>
      </div>

      {name && version && (
        <div className="flex items-center gap-2 text-xs">
          <button className="px-2 py-1 border rounded" onClick={async()=>{
            const q = await api.queueAnnotations(name, { version, strategy:'missing', limit: pageSize });
            const items = (q.items||[]).map(p => ({ path: p }));
            setSamples(items);
          }}>Load Review Queue</button>
          <button className="px-2 py-1 border rounded" onClick={async()=>{
            const res = await api.prelabel(name, { version, provider:'stub', task: mode==='image_bbox' ? 'bbox' : (mode==='ner'?'ner':'classify')});
            alert((res && res.status) ? res.status : 'queued');
          }}>Prelabel (stub)</button>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {samples.map((s,i) => (
          <div key={i} className="p-3 border border-border rounded bg-surface">
            <div className="text-[11px] text-text/60 mb-1">{s.path}</div>
            {mode==='image_bbox' ? (
              <ImageBBoxAnnotator
                imageUrl={`/api/datasets/${encodeURIComponent(name)}/file?version=${encodeURIComponent(version)}&path=${encodeURIComponent(s.path)}`}
                anno={(annos[s.path]?.bboxes)||[]}
                onChange={(bboxes, size)=>updateAnno(s.path, { path: s.path, bboxes, image_size: size })}
                labels={labelList}
              />
            ) : (
              <>
                <TextPreview name={name} version={version} path={s.path} />
                {mode==='text_cls' ? (
                  <div className="mt-2">
                    <select className="border rounded px-2 py-1 text-sm" value={(annos[s.path]?.label)||''} onChange={e=>updateAnno(s.path, { path: s.path, label: e.target.value })}>
                      <option value="">unset</option>
                      {labelList.map(l => (<option key={l} value={l}>{l}</option>))}
                    </select>
                  </div>
                ) : (
                  <NERAnnotator textUrl={`/api/datasets/${encodeURIComponent(name)}/file?version=${encodeURIComponent(version)}&path=${encodeURIComponent(s.path)}`} anno={(annos[s.path]?.spans)||[]} onChange={(spans)=>updateAnno(s.path, { path: s.path, spans })} labels={labelList} />
                )}
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function TextPreview({ name, version, path }){
  const [text, setText] = useState('');
  useEffect(() => { (async ()=>{ try{ const res = await fetch(`/api/datasets/${encodeURIComponent(name)}/file?version=${encodeURIComponent(version)}&path=${encodeURIComponent(path)}`); setText(await res.text()); } catch {} })(); }, [name, version, path]);
  return <pre className="text-xs bg-muted p-2 rounded max-h-40 overflow-auto whitespace-pre-wrap">{text.slice(0, 4000)}</pre>;
}

function NERAnnotator({ textUrl, anno, onChange, labels }){
  const [text, setText] = useState('');
  const [spans, setSpans] = useState(anno||[]);
  useEffect(()=>{ setSpans(anno||[]); }, [anno]);
  useEffect(()=>{ (async()=>{ try{ const res = await fetch(textUrl); setText(await res.text()); } catch {} })(); }, [textUrl]);
  const addSpan = () => {
    const sel = window.getSelection && window.getSelection();
    if (!sel || sel.rangeCount===0) return;
    const start = sel.anchorOffset; const end = sel.focusOffset;
    if (start===end) return;
    const s = Math.min(start,end), e=Math.max(start,end);
    const label = labels[0] || 'ENT';
    const next = [...spans, { start: s, end: e, label }]; setSpans(next); onChange && onChange(next);
  };
  return (
    <div>
      <div className="text-xs text-text/60 mb-1">Select text then click Add Span.</div>
      <div className="border border-border rounded p-2 text-sm whitespace-pre-wrap" style={{ userSelect: 'text' }}>{text}</div>
      <div className="flex gap-2 mt-2">
        <button className="px-2 py-1 border rounded" onClick={addSpan}>Add Span</button>
        <button className="px-2 py-1 border rounded" onClick={()=>{ setSpans([]); onChange && onChange([]); }}>Clear</button>
      </div>
      <div className="mt-2 text-xs">
        {spans.map((sp,i)=>(<div key={i}>{sp.start}..{sp.end} — {sp.label}</div>))}
      </div>
    </div>
  );
}

function ImageBBoxAnnotator({ imageUrl, anno, onChange, labels }){
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 });
  const [boxes, setBoxes] = useState(anno||[]);
  const [draft, setDraft] = useState(null);
  const [label, setLabel] = useState(labels?.[0] || 'object');
  const imgRef = React.useRef(null);
  useEffect(()=>{ setBoxes(anno||[]); }, [anno]);
  const onImgLoad = (e) => {
    const img = e.target;
    setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    setDisplaySize({ w: img.clientWidth, h: img.clientHeight });
  };
  const toImageCoords = (x, y) => {
    // map from client coords over displayed image to natural image coords
    const rect = imgRef.current.getBoundingClientRect();
    const rx = Math.min(Math.max(0, x - rect.left), rect.width);
    const ry = Math.min(Math.max(0, y - rect.top), rect.height);
    const sx = imgSize.w / Math.max(1, rect.width);
    const sy = imgSize.h / Math.max(1, rect.height);
    return { x: Math.round(rx * sx), y: Math.round(ry * sy) };
  };
  const onMouseDown = (e) => {
    if (!imgRef.current) return;
    const p = toImageCoords(e.clientX, e.clientY);
    setDraft({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
  };
  const onMouseMove = (e) => {
    if (!draft || !imgRef.current) return;
    const p = toImageCoords(e.clientX, e.clientY);
    setDraft({ ...draft, x1: p.x, y1: p.y });
  };
  const onMouseUp = () => {
    if (!draft) return;
    const x = Math.min(draft.x0, draft.x1);
    const y = Math.min(draft.y0, draft.y1);
    const w = Math.abs(draft.x1 - draft.x0);
    const h = Math.abs(draft.y1 - draft.y0);
    if (w > 2 && h > 2) {
      const next = [...boxes, { x, y, w, h, label }];
      setBoxes(next);
      onChange && onChange(next, imgSize);
    }
    setDraft(null);
  };
  const removeBox = (idx) => {
    const next = boxes.filter((_,i)=>i!==idx); setBoxes(next); onChange && onChange(next, imgSize);
  };
  return (
    <div>
      <div className="flex items-center gap-2 text-xs mb-2">
        <label>Label<select className="border rounded px-2 py-1 ml-1" value={label} onChange={e=>setLabel(e.target.value)}>{(labels||['object']).map(l => <option key={l} value={l}>{l}</option>)}</select></label>
        <button className="px-2 py-1 border rounded" onClick={()=>{ setBoxes([]); onChange && onChange([], imgSize); }}>Clear</button>
      </div>
      <div
        style={{ position:'relative', display:'inline-block' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
      >
        <img ref={imgRef} src={imageUrl} onLoad={onImgLoad} className="max-h-64 rounded" />
        <svg style={{ position:'absolute', left:0, top:0, pointerEvents:'none' }} width={displaySize.w} height={displaySize.h}>
          {boxes.map((b,i) => (
            <rect key={i} x={(b.x/imgSize.w)*displaySize.w} y={(b.y/imgSize.h)*displaySize.h} width={(b.w/imgSize.w)*displaySize.w} height={(b.h/imgSize.h)*displaySize.h} fill="none" stroke="#22c55e" strokeWidth={2} />
          ))}
          {draft && (
            <rect x={(Math.min(draft.x0,draft.x1)/imgSize.w)*displaySize.w} y={(Math.min(draft.y0,draft.y1)/imgSize.h)*displaySize.h} width={(Math.abs(draft.x1-draft.x0)/imgSize.w)*displaySize.w} height={(Math.abs(draft.y1-draft.y0)/imgSize.h)*displaySize.h} fill="none" stroke="#3b82f6" strokeDasharray="4,3" strokeWidth={2} />
          )}
        </svg>
      </div>
      <div className="mt-2 text-xs">
        {boxes.map((b,i)=>(
          <div key={i} className="flex items-center gap-2">
            <span className="w-24 truncate">{b.label}</span>
            <span>({b.x},{b.y}) {b.w}x{b.h}</span>
            <button className="px-1 py-0.5 border rounded" onClick={()=>removeBox(i)}>Remove</button>
          </div>
        ))}
      </div>
    </div>
  );
}
