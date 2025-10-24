import React, { useEffect, useMemo, useRef, useState } from 'react';

// Simple DAG editor for pipelines
export default function PipelinesPage(){
  const [pipelines, setPipelines] = useState([]);
  const [current, setCurrent] = useState({ id:null, name:'', nodes:[], edges:[] });
  const [selNode, setSelNode] = useState(null);
  const [edgeFrom, setEdgeFrom] = useState('');
  const [edgeTo, setEdgeTo] = useState('');

  const load = async () => {
    try{ const r = await fetch('/api/pipelines'); const j = await r.json(); setPipelines(j.items||[]);}catch{}
  };
  useEffect(()=>{ load(); }, []);

  const addNode = () => {
    const id = `n${Date.now().toString(36)}`;
    const nn = { id, label:`Node ${current.nodes.length+1}`, type:'finetune', x: 80+current.nodes.length*30, y: 80+current.nodes.length*20, job: { name:`Job ${current.nodes.length+1}`, type:'finetune', framework:'huggingface', config:{ model_name:'bert-base-uncased', epochs:1, batch_size:8 } } };
    setCurrent(c => ({...c, nodes:[...c.nodes, nn]}));
    setSelNode(nn);
  };

  const addEdge = () => {
    if (!edgeFrom || !edgeTo || edgeFrom===edgeTo) return;
    setCurrent(c => ({...c, edges:[...c.edges, { from: edgeFrom, to: edgeTo }]}));
    setEdgeFrom(''); setEdgeTo('');
  };

  const onDrag = (id, dx, dy) => {
    setCurrent(c => ({...c, nodes: c.nodes.map(n => n.id===id ? ({...n, x: Math.max(0, n.x+dx), y: Math.max(0, n.y+dy)}) : n)}));
  };

  const save = async () => {
    const payload = { id: current.id || undefined, name: current.name||`pipeline-${Date.now()}`, nodes: current.nodes, edges: current.edges };
    const r = await fetch('/api/pipelines', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    if (!r.ok) { alert(await r.text()); return; }
    const j = await r.json();
    alert(`Saved pipeline ${j.id}`);
    load();
  };

  const run = async () => {
    if (!current.id){ alert('Load or save the pipeline first to get an id'); return; }
    const r = await fetch(`/api/pipelines/${encodeURIComponent(current.id)}/run`, { method:'POST' });
    const j = await r.json();
    if (j.error) alert(j.error); else alert('Pipeline started');
  };

  const loadPipeline = (p) => {
    setCurrent({ id: p.id, name: p.name||'', nodes: p.nodes||[], edges: p.edges||[] });
    setSelNode(null);
  };

  const template = (key) => {
    if (key === 'sweep'){
      const baseId = `train-${Date.now().toString(36)}`;
      const n1 = { id: baseId, label:'HPO: HF Sweep', type:'finetune', x:120, y:120, job:{ name:'HPO Sweep', type:'finetune', framework:'huggingface', config:{ model_name:'bert-base-uncased', epochs:1, batch_size:8, hpo: { enabled:true, metric:'eval_loss', direction:'minimize', max_trials:10, trial_epochs:1, space:[ { name:'learning_rate', type:'float', low:1e-6, high:1e-4, log:true }, { name:'batch_size', type:'int', low:4, high:32, step:4 } ] } } } };
      const n2 = { id: `${baseId}-eval`, label:'Eval Best', type:'eval', x:360, y:160, job:{ name:'Eval Best', type:'train', framework:'pytorch', config:{ epochs:1, batch_size:16 } } };
      setCurrent({ id:null, name:'HPO Sweep', nodes:[n1,n2], edges:[{from:n1.id,to:n2.id}] }); setSelNode(n1);
    } else if (key === 'cv'){
      const n1 = { id:'split', label:'CV Split', type:'split', x:120, y:80, job:{ name:'Split', type:'train', framework:'pytorch', config:{} } };
      const n2 = { id:'fold1', label:'Fold 1', type:'train', x:320, y:60, job:{ name:'Fold 1', type:'train', framework:'huggingface', config:{ model_name:'bert-base-uncased', epochs:1, batch_size:8 } } };
      const n3 = { id:'fold2', label:'Fold 2', type:'train', x:320, y:150, job:{ name:'Fold 2', type:'train', framework:'huggingface', config:{ model_name:'bert-base-uncased', epochs:1, batch_size:8 } } };
      const n4 = { id:'agg', label:'Aggregate', type:'eval', x:520, y:100, job:{ name:'Aggregate', type:'train', framework:'pytorch', config:{} } };
      setCurrent({ id:null, name:'Cross Validation', nodes:[n1,n2,n3,n4], edges:[{from:n1.id,to:n2.id},{from:n1.id,to:n3.id},{from:n2.id,to:n4.id},{from:n3.id,to:n4.id}] });
    } else if (key === 'ensemble'){
      const n1 = { id:'trA', label:'Train A', type:'train', x:100, y:80, job:{ name:'Train A', type:'train', framework:'huggingface', config:{ model_name:'bert-base-uncased', epochs:1, batch_size:8 } } };
      const n2 = { id:'trB', label:'Train B', type:'train', x:100, y:180, job:{ name:'Train B', type:'train', framework:'huggingface', config:{ model_name:'roberta-base', epochs:1, batch_size:8 } } };
      const n3 = { id:'ens', label:'Ensemble', type:'eval', x:360, y:130, job:{ name:'Ensemble', type:'train', framework:'pytorch', config:{} } };
      setCurrent({ id:null, name:'Ensemble', nodes:[n1,n2,n3], edges:[{from:n1.id,to:n3.id},{from:n2.id,to:n3.id}] });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Pipelines</h1>
        <div className="flex gap-2 text-sm">
          <select className="border border-border rounded px-3 py-2 bg-surface" onChange={e=>{ const id=e.target.value; if (!id) return; const p=pipelines.find(x=>x.id===id); if(p) loadPipeline(p);} }>
            <option value="">Load saved…</option>
            {pipelines.map(p => (<option key={p.id} value={p.id}>{p.name||p.id}</option>))}
          </select>
          <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={()=>template('sweep')}>Template: HPO Sweep</button>
          <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={()=>template('cv')}>Template: Cross‑Val</button>
          <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={()=>template('ensemble')}>Template: Ensemble</button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Editor */}
        <div className="lg:col-span-3 bg-surface border border-border rounded relative overflow-hidden" style={{height: 420}}>
          <Canvas nodes={current.nodes} edges={current.edges} onDrag={onDrag} onSelect={setSelNode} />
        </div>
        {/* Inspector */}
        <div className="bg-surface border border-border rounded p-3 space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-sm font-semibold">Inspector</div>
            <div className="flex gap-2"><button className="px-2 py-1 border border-border rounded" onClick={addNode}>Add Node</button></div>
          </div>
          <div className="text-xs text-text/70">Pipeline ID: {current.id || '-'}</div>
          <label className="text-xs">Name<input className="w-full border border-border rounded px-2 py-1 bg-surface" value={current.name} onChange={e=>setCurrent(c=>({...c, name:e.target.value}))} /></label>
          {selNode ? (
            <NodeEditor node={selNode} onChange={n=> setCurrent(c=>({...c, nodes: c.nodes.map(x=>x.id===n.id?n:x)}))} />
          ) : (
            <div className="text-xs text-text/60">Select a node to edit</div>
          )}
          <div className="border-t border-border pt-2">
            <div className="text-sm font-semibold mb-1">Edges</div>
            <div className="space-y-2 text-sm">
              {(current.edges||[]).map((e,i)=>(<div key={i} className="flex items-center justify-between"><span>{e.from} → {e.to}</span><button className="text-danger" onClick={()=>setCurrent(c=>({...c, edges:c.edges.filter((_,k)=>k!==i)}))}>Remove</button></div>))}
              <div className="flex gap-2">
                <select className="border border-border rounded px-2 py-1 bg-surface" value={edgeFrom} onChange={e=>setEdgeFrom(e.target.value)}><option value="">from…</option>{current.nodes.map(n=>(<option key={n.id} value={n.id}>{n.id}</option>))}</select>
                <select className="border border-border rounded px-2 py-1 bg-surface" value={edgeTo} onChange={e=>setEdgeTo(e.target.value)}><option value="">to…</option>{current.nodes.map(n=>(<option key={n.id} value={n.id}>{n.id}</option>))}</select>
                <button className="px-2 py-1 border border-border rounded" onClick={addEdge}>Add</button>
              </div>
            </div>
          </div>
          <div className="flex justify-between pt-2">
            <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={save}>Save</button>
            <button className="px-3 py-2 rounded bg-primary text-on-primary hover:brightness-110" onClick={run}>Run</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Canvas({ nodes, edges, onDrag, onSelect }){
  const ref = useRef(null);
  const [drag, setDrag] = useState(null);
  useEffect(()=>{
    const el = ref.current; if (!el) return;
    const onMove = (e) => {
      if (!drag) return;
      onDrag(drag.id, e.movementX, e.movementY);
    };
    const onUp = () => setDrag(null);
    el.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return ()=>{ el.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, [drag]);
  return (
    <div ref={ref} className="w-full h-full">
      <svg className="absolute inset-0 w-full h-full">
        {(edges||[]).map((e,i)=>{
          const a = nodes.find(n=>n.id===e.from); const b = nodes.find(n=>n.id===e.to);
          if (!a||!b) return null;
          const x1=a.x+60, y1=a.y+20, x2=b.x, y2=b.y+20; const mx=(x1+x2)/2; const d=`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
          return (<path key={i} d={d} stroke="#94a3b8" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />);
        })}
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
          </marker>
        </defs>
      </svg>
      {(nodes||[]).map(n => (
        <div key={n.id} className="absolute bg-white dark:bg-zinc-900 border border-border rounded shadow px-3 py-2 cursor-move select-none" style={{ left:n.x, top:n.y, width: 160 }} onMouseDown={()=>{ setDrag({id:n.id}); onSelect && onSelect(n); }}>
          <div className="font-semibold text-sm truncate">{n.label||n.id}</div>
          <div className="text-[10px] text-text/60">{n.type} • {n.job?.framework}</div>
        </div>
      ))}
    </div>
  );
}

function NodeEditor({ node, onChange }){
  const [n, setN] = useState(node);
  useEffect(()=>{ setN(node); }, [node?.id]);
  const update = (patch) => { const nn={...n, ...patch}; setN(nn); onChange && onChange(nn); };
  const updateJob = (patch) => { const j={...(n.job||{}) , ...patch}; update({ job: j }); };
  const updateCfg = (patch) => { const j={...(n.job||{}), config: { ...(n.job?.config||{}), ...patch } }; update({ job: j }); };
  return (
    <div className="space-y-2 text-sm">
      <div className="text-sm font-semibold">Node {n.id}</div>
      <label className="text-xs">Label<input className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.label||''} onChange={e=>update({label:e.target.value})} /></label>
      <label className="text-xs">Type<select className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.type||'train'} onChange={e=>update({type:e.target.value})}><option value="train">train</option><option value="finetune">finetune</option><option value="eval">eval</option></select></label>
      <div className="grid grid-cols-2 gap-2">
        <label className="text-xs">X<input type="number" className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.x||0} onChange={e=>update({x:parseInt(e.target.value)||0})} /></label>
        <label className="text-xs">Y<input type="number" className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.y||0} onChange={e=>update({y:parseInt(e.target.value)||0})} /></label>
      </div>
      <div className="border-t border-border pt-2">
        <div className="text-sm font-semibold">Job</div>
        <label className="text-xs">Name<input className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.job?.name||''} onChange={e=>updateJob({name:e.target.value})} /></label>
        <label className="text-xs">Framework<select className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.job?.framework||'pytorch'} onChange={e=>updateJob({framework:e.target.value})}><option value="pytorch">pytorch</option><option value="huggingface">huggingface</option><option value="tensorflow">tensorflow</option></select></label>
        <label className="text-xs">Job Type<select className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.job?.type||'train'} onChange={e=>updateJob({type:e.target.value})}><option value="train">train</option><option value="finetune">finetune</option></select></label>
        <div className="grid grid-cols-2 gap-2">
          <label className="text-xs">Epochs<input type="number" className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.job?.config?.epochs||1} onChange={e=>updateCfg({epochs: parseInt(e.target.value)||1})} /></label>
          <label className="text-xs">Batch<input type="number" className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.job?.config?.batch_size||8} onChange={e=>updateCfg({batch_size: parseInt(e.target.value)||8})} /></label>
        </div>
        {n.job?.framework==='huggingface' && (
          <>
            <label className="text-xs">Model<input className="w-full border border-border rounded px-2 py-1 bg-surface" value={n.job?.config?.model_name||''} onChange={e=>updateCfg({model_name:e.target.value})} /></label>
            <details>
              <summary className="cursor-pointer text-xs font-semibold">HPO Options</summary>
              <div className="mt-1 space-y-1 text-xs">
                <label className="inline-flex items-center gap-2"><input type="checkbox" checked={!!(n.job?.config?.hpo?.enabled)} onChange={e=>updateCfg({ hpo: { ...(n.job?.config?.hpo||{}), enabled: e.target.checked } })}/> Enable HPO (Optuna)</label>
                <label>Metric<input className="w-full border rounded px-2 py-1" value={(n.job?.config?.hpo?.metric)||'eval_loss'} onChange={e=>updateCfg({ hpo: { ...(n.job?.config?.hpo||{}), metric:e.target.value } })} /></label>
                <label>Direction<select className="w-full border rounded px-2 py-1" value={(n.job?.config?.hpo?.direction)||'minimize'} onChange={e=>updateCfg({ hpo: { ...(n.job?.config?.hpo||{}), direction:e.target.value } })}><option value="minimize">minimize</option><option value="maximize">maximize</option></select></label>
                <label>Max Trials<input type="number" className="w-full border rounded px-2 py-1" value={(n.job?.config?.hpo?.max_trials)||10} onChange={e=>updateCfg({ hpo: { ...(n.job?.config?.hpo||{}), max_trials: parseInt(e.target.value)||10 } })} /></label>
              </div>
            </details>
          </>
        )}
      </div>
    </div>
  );
}

