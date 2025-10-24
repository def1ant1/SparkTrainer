import React, { useEffect, useMemo, useState } from 'react';

export default function HPOViewer({ jobId, apiBase }){
  const [data, setData] = useState({ results: null, trials: [] });
  const [error, setError] = useState('');
  const API_BASE = apiBase || '/api';
  const [metricIndex, setMetricIndex] = useState(0);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/hpo/${jobId}`);
        if (!res.ok) throw new Error(await res.text());
        setData(await res.json());
      } catch (e) { setError(String(e.message||e)); }
    };
    if (jobId) load();
  }, [jobId]);

  const trials = data.trials || [];
  const results = data.results || {};
  const isMulti = Array.isArray(results.metrics) && results.metrics.length > 0;
  const getValue = (t) => isMulti ? ((t.values && t.values[metricIndex]) ?? null) : (t.value ?? null);
  const values = trials.map(t => getValue(t)).filter(v => v != null);
  const bestHistory = useMemo(() => {
    const dir = isMulti ? ((results.directions||[])[metricIndex]||'minimize').toLowerCase() : (results.direction || 'minimize').toLowerCase();
    let best = dir === 'minimize' ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
    return trials.map((t, i) => {
      const v = getValue(t);
      if (v == null) return null;
      if (dir === 'minimize') best = Math.min(best, v); else best = Math.max(best, v);
      return { i, best };
    }).filter(Boolean);
  }, [trials, results.direction, results.directions, metricIndex]);

  const paramNames = useMemo(() => {
    const set = new Set(); trials.forEach(t => Object.keys(t.params||{}).forEach(k => set.add(k))); return Array.from(set);
  }, [trials]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Hyperparameter Optimization</h1>
      </div>

      {error && <div className="p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="p-3 border border-border rounded bg-surface">
          <div className="text-sm font-semibold mb-1">Study</div>
          <div className="text-xs text-text/70">Name: {results.study_name || '-'}</div>
          {!isMulti ? (
            <>
              <div className="text-xs text-text/70">Direction: {results.direction || '-'}</div>
              <div className="text-xs text-text/70">Metric: {results.metric || '-'}</div>
            </>
          ) : (
            <>
              <div className="text-xs text-text/70">Directions: {(results.directions||[]).join(', ')}</div>
              <div className="text-xs text-text/70">Metrics:
                <select className="ml-2 border rounded px-1 py-0.5" value={metricIndex} onChange={e=>setMetricIndex(parseInt(e.target.value))}>
                  {(results.metrics||[]).map((m,i)=>(<option key={i} value={i}>{m}</option>))}
                </select>
              </div>
            </>
          )}
          <div className="text-xs text-text/70">Trials: {trials.length}</div>
          <div className="text-xs text-text/70">Best value: {(() => {
            const bv = isMulti ? (results.best_trial?.values ? results.best_trial.values[metricIndex] : null) : results.best_trial?.value;
            return bv != null ? Number(bv).toFixed(5) : '-';
          })()}</div>
        </div>

        <div className="p-3 border border-border rounded bg-surface lg:col-span-2">
          <div className="text-sm font-semibold mb-2">Optimization History</div>
          <LineChart data={bestHistory} width={640} height={180} />
        </div>

        <div className="p-3 border border-border rounded bg-surface lg:col-span-3">
          <div className="text-sm font-semibold mb-2">Parameter Importances</div>
          <ImportanceBars importances={results.param_importances || {}} />
        </div>

        <div className="p-3 border border-border rounded bg-surface lg:col-span-3">
          <div className="text-sm font-semibold mb-2">Parallel Coordinates</div>
          <ParallelCoords trials={trials} params={paramNames} metricName={isMulti ? (results.metrics||[])[metricIndex] : (results.metric || 'value')} metricIndex={isMulti ? metricIndex : null} />
        </div>

        <div className="p-3 border border-border rounded bg-surface lg:col-span-3">
          <div className="text-sm font-semibold mb-2">Parameter Slices</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {paramNames.slice(0,6).map(p => (
              <SlicePlot key={p} trials={trials} param={p} metricName={isMulti ? (results.metrics||[])[metricIndex] : (results.metric || 'value')} metricIndex={isMulti ? metricIndex : null} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function LineChart({ data, width, height }){
  const pad = 24;
  const xs = data.map(d => d.i);
  const ys = data.map(d => d.best);
  const minX = Math.min(...xs, 0), maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const scaleX = (x) => pad + (x-minX)/(maxX-minX||1) * (width-2*pad);
  const scaleY = (y) => height-pad - (y-minY)/(maxY-minY||1) * (height-2*pad);
  const path = data.map((d,i) => `${i===0?'M':'L'} ${scaleX(d.i)} ${scaleY(d.best)}`).join(' ');
  return (
    <svg width={width} height={height} className="w-full">
      <rect x={0} y={0} width={width} height={height} fill="transparent" stroke="#ddd" />
      <path d={path} stroke="#2563eb" strokeWidth={2} fill="none" />
    </svg>
  );
}

function ImportanceBars({ importances }){
  const entries = Object.entries(importances || {}).sort((a,b) => b[1]-a[1]);
  const maxV = Math.max(...entries.map(e=>e[1]), 1);
  return (
    <div className="space-y-2">
      {entries.length === 0 && <div className="text-xs text-text/60">No importances available</div>}
      {entries.map(([k,v]) => (
        <div key={k} className="flex items-center gap-2">
          <div className="w-48 text-xs truncate">{k}</div>
          <div className="flex-1 bg-muted h-3 rounded">
            <div className="h-3 bg-blue-500 rounded" style={{ width: `${100 * (v/maxV)}%` }} />
          </div>
          <div className="w-12 text-right text-xs">{v.toFixed(2)}</div>
        </div>
      ))}
    </div>
  );
}

function ParallelCoords({ trials, params, metricName, metricIndex }){
  const width = 800, height = 220, pad = 40;
  const axes = [...params, metricName];
  const scales = {};
  axes.forEach(ax => {
    const vals = trials.map(t => ax===metricName ? (metricIndex!=null ? (t.values?t.values[metricIndex]:null) : t.value) : t.params?.[ax]).filter(v => v != null && typeof v !== 'string');
    const min = Math.min(...vals, 0), max = Math.max(...vals, 1);
    scales[ax] = (v) => {
      const num = typeof v === 'string' ? 0 : Number(v);
      return height - pad - (num-min)/(max-min||1) * (height-2*pad);
    };
  });
  const xFor = (i) => pad + i * ((width-2*pad)/(axes.length-1));
  return (
    <svg width={width} height={height} className="w-full">
      <rect x={0} y={0} width={width} height={height} fill="transparent" stroke="#ddd" />
      {axes.map((ax,i) => (
        <g key={ax}>
          <line x1={xFor(i)} x2={xFor(i)} y1={pad} y2={height-pad} stroke="#ccc" />
          <text x={xFor(i)} y={pad-8} fontSize={10} textAnchor="middle" fill="#666">{ax}</text>
        </g>
      ))}
      {trials.map((t, idx) => {
        const pts = axes.map((ax,i) => {
          const v = ax===metricName ? (metricIndex!=null ? (t.values?t.values[metricIndex]:null) : t.value) : t.params?.[ax];
          return `${i===0?'M':'L'} ${xFor(i)} ${scales[ax](v)}`;
        }).join(' ');
        const col = t.state === 'TrialState.COMPLETE' ? '#10b981' : '#9ca3af';
        return <path key={idx} d={pts} stroke={col} strokeWidth={1} fill="none" opacity={0.6} />;
      })}
    </svg>
  );
}

function SlicePlot({ trials, param, metricName, metricIndex }){
  const width = 260, height = 160, pad = 28;
  const pts = trials.map(t => ({ x: t.params?.[param], y: (metricIndex!=null ? (t.values?t.values[metricIndex]:null) : t.value) })).filter(p => p.x != null && p.y != null && typeof p.x !== 'string');
  const minX = Math.min(...pts.map(p=>p.x), 0), maxX = Math.max(...pts.map(p=>p.x), 1);
  const minY = Math.min(...pts.map(p=>p.y), 0), maxY = Math.max(...pts.map(p=>p.y), 1);
  const scaleX = (x) => pad + (x-minX)/(maxX-minX||1) * (width-2*pad);
  const scaleY = (y) => height-pad - (y-minY)/(maxY-minY||1) * (height-2*pad);
  return (
    <div>
      <div className="text-xs text-text/70 mb-1">{param}</div>
      <svg width={width} height={height} className="w-full">
        <rect x={0} y={0} width={width} height={height} fill="transparent" stroke="#eee" />
        {pts.map((p,i) => <circle key={i} cx={scaleX(p.x)} cy={scaleY(p.y)} r={3} fill="#6366f1" opacity={0.7} />)}
      </svg>
    </div>
  );
}
