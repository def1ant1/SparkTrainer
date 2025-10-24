import React, { useState } from 'react';
import { cx } from './classNames';

export function Accordion({ items, defaultOpen = [] }){
  const [open, setOpen] = useState(new Set(defaultOpen));
  const toggle = (i) => setOpen(s => { const n = new Set(s); n.has(i) ? n.delete(i) : n.add(i); return n; });
  return (
    <div className="divide-y divide-border border border-border rounded-md">
      {items.map((it, i) => (
        <div key={i}>
          <button className="w-full text-left px-4 py-3 bg-surface hover:bg-muted flex items-center justify-between" onClick={()=>toggle(i)}>
            <span className="font-medium">{it.title}</span>
            <span className="text-text/60">{open.has(i)?'−':'+'}</span>
          </button>
          <div className={cx('px-4 overflow-hidden transition-[max-height] duration-300', open.has(i)?'max-h-96 py-3':'max-h-0')}>{it.content}</div>
        </div>
      ))}
    </div>
  );
}

