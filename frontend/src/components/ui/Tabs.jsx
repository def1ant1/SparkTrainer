import React from 'react';
import { cx } from './classNames';

export function Tabs({ value, onChange, tabs, className }){
  return (
    <div className={cx('border-b border-border', className)}>
      <div className="flex gap-2">
        {tabs.map(t => (
          <button key={t.value} onClick={()=>onChange && onChange(t.value)} className={cx('px-3 py-2 rounded-t-md border-b-2', value===t.value ? 'border-primary text-primary' : 'border-transparent text-text/70 hover:text-text hover:border-border')}>{t.label}</button>
        ))}
      </div>
    </div>
  );
}

