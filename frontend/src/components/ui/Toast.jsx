import React, { createContext, useContext, useMemo, useState, useCallback, useEffect } from 'react';
import { cx } from './classNames';

const ToastCtx = createContext(null);

export function ToastProvider({ children }){
  const [toasts, setToasts] = useState([]);
  const push = useCallback((t) => {
    const id = t.id || Math.random().toString(36).slice(2);
    const toast = { id, title: '', message: '', type: 'info', timeout: 3000, ...t };
    setToasts(list => [...list, toast]);
    if (toast.timeout) setTimeout(() => dismiss(id), toast.timeout);
    return id;
  }, []);
  const dismiss = useCallback((id) => setToasts((list) => list.filter(t => t.id !== id)), []);
  const api = useMemo(() => ({ push, dismiss }), [push, dismiss]);
  return (
    <ToastCtx.Provider value={api}>
      {children}
      <div className="fixed top-4 right-4 z-[60] space-y-2">
        {toasts.map(t => <ToastItem key={t.id} toast={t} onClose={()=>dismiss(t.id)} />)}
      </div>
    </ToastCtx.Provider>
  );
}

export function useToast(){
  const ctx = useContext(ToastCtx);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}

function ToastItem({ toast, onClose }){
  const color = toast.type==='success' ? 'bg-success text-on-primary' : toast.type==='error' ? 'bg-danger text-on-primary' : toast.type==='warning' ? 'bg-warning text-on-primary' : 'bg-surface text-text border border-border';
  return (
    <div className={cx('min-w-[240px] max-w-[360px] rounded-md shadow-lg overflow-hidden animate-slide-in', color)}>
      <div className="px-3 py-2 font-semibold">{toast.title || toast.type?.toUpperCase()}</div>
      {toast.message && <div className="px-3 pb-2 text-sm">{toast.message}</div>}
      <button className={cx('absolute top-1 right-1 px-2 py-1 rounded', toast.type==='info'? 'hover:bg-muted' : 'hover:brightness-110')} onClick={onClose}>✕</button>
    </div>
  );
}

