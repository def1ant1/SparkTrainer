import React from 'react';
import { cx } from './classNames';

export function Input({ label, helperText, error, success, className, ...props }){
  const state = error ? 'error' : success ? 'success' : 'default';
  const border = state === 'error' ? 'border-danger focus:ring-danger' : state === 'success' ? 'border-success focus:ring-success' : 'border-border focus:ring-primary';
  return (
    <label className={cx('block text-sm', className)}>
      {label && <span className="mb-1 inline-block text-text">{label}</span>}
      <input className={cx('w-full px-3 py-2 rounded-md bg-surface text-text placeholder:text-text/60 border outline-none focus:ring-2', border)} {...props} />
      {helperText && (
        <span className={cx('mt-1 block text-xs', error ? 'text-danger' : success ? 'text-success' : 'text-text/70')}>{helperText}</span>
      )}
    </label>
  );
}

