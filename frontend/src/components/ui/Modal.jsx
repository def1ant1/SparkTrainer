import React, { useEffect } from 'react';
import { cx } from './classNames';

export function Modal({ open, onClose, title, children, footer, className, closeOnOverlay = true }){
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose && onClose(); };
    if (open) document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/50 animate-fade-in" onClick={() => closeOnOverlay && onClose && onClose()} />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className={cx('w-full max-w-lg bg-surface text-text border border-border rounded-md shadow-lg animate-pop-in', className)}>
          {(title || onClose) && (
            <div className="px-4 py-3 border-b border-border flex items-center justify-between">
              <div className="font-semibold">{title}</div>
              {onClose && <button onClick={onClose} className="px-2 py-1 rounded hover:bg-muted">✕</button>}
            </div>
          )}
          <div className="p-4">{children}</div>
          {footer && <div className="px-4 py-3 border-t border-border">{footer}</div>}
        </div>
      </div>
    </div>
  );
}

