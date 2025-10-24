import React, { useEffect, useState } from 'react';

export default function QuickStart({ open, onClose, api, onDone }) {
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const steps = [
    { title: 'Welcome', body: 'This quick-start will launch an example training job, show key areas of the dashboard, and point to datasets/models. You can customize later.' },
    { title: 'Dashboard', body: 'Left: quick actions, Center: realtime metrics and widgets, Right: activity feed and notifications. Use the Customize button to tailor widgets.' },
    { title: 'Global Search', body: 'Press Cmd+K / Ctrl+K anytime to quickly search jobs, models, and datasets. Navigate with arrow keys and press Enter.' },
  ];

  useEffect(() => { if (!open) setStep(0); }, [open]);

  const launchExample = async () => {
    setRunning(true);
    try {
      const payload = {
        name: 'Example Training Job',
        type: 'train',
        framework: 'huggingface',
        config: {
          architecture: 'transformer',
          model_name: 'bert-base-uncased',
          epochs: 1,
          batch_size: 8,
          learning_rate: 5e-5,
          data: { source: 'huggingface', dataset_name: 'imdb', split: { train: 0.8, val: 0.1, test: 0.1 } },
        },
      };
      await api.createJob(payload);
      onDone && onDone();
      onClose();
    } catch (e) {
      alert('Failed to start example job: ' + String(e.message||e));
    } finally {
      setRunning(false);
    }
  };

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 bg-black/40 flex items-center justify-center p-6" onClick={onClose}>
      <div className="w-full max-w-xl bg-surface border border-border rounded-lg shadow-lg" onClick={e=>e.stopPropagation()}>
        <div className="p-6 space-y-4">
          <div className="text-2xl font-bold">Quick Start</div>
          <div className="space-y-1">
            <div className="text-lg font-semibold">{steps[step].title}</div>
            <div className="text-sm text-text/80">{steps[step].body}</div>
          </div>
          <div className="flex items-center justify-between pt-2">
            <div className="text-xs text-text/60">Step {step+1} of {steps.length}</div>
            <div className="flex gap-2">
              {step>0 && <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={()=>setStep(s=>s-1)}>Back</button>}
              {step<steps.length-1 && <button className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted" onClick={()=>setStep(s=>s+1)}>Next</button>}
              {step===steps.length-1 && <button className="px-3 py-2 rounded bg-primary text-on-primary hover:brightness-110" disabled={running} onClick={launchExample}>{running? 'Starting…' : 'One-click Example Job'}</button>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

