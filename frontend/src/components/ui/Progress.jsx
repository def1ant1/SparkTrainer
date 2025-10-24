import React from 'react';
import { cx } from './classNames';

export function LinearProgress({ value, max = 100, className }){
  const pct = Math.max(0, Math.min(100, (value ?? 0) / max * 100));
  return (
    <div className={cx('w-full h-2 bg-muted rounded', className)}>
      <div className="h-2 bg-primary rounded transition-[width] duration-300" style={{ width: pct + '%' }} />
    </div>
  );
}

export function CircularProgress({ value = 0, size = 40, stroke = 4, className }){
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value));
  const dash = c * (1 - pct / 100);
  return (
    <svg className={cx('block', className)} width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size/2} cy={size/2} r={r} stroke="rgb(var(--color-border))" strokeWidth={stroke} fill="none" />
      <circle cx={size/2} cy={size/2} r={r} stroke="rgb(var(--color-primary))" strokeWidth={stroke} fill="none" strokeDasharray={c} strokeDashoffset={dash} transform={`rotate(-90 ${size/2} ${size/2})`} style={{ transition: 'stroke-dashoffset var(--transition-normal) ease' }} />
    </svg>
  );
}

export function Skeleton({ className, circle }){
  return <div className={cx('bg-muted relative overflow-hidden rounded', circle ? 'rounded-full' : 'rounded', 'animate-shimmer', className)} />
}

