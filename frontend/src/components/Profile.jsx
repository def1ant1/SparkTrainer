import React, { useState, useEffect } from 'react';
import TransferQueue from './TransferQueue';

// Toast hook for notifications
const useToast = () => {
  return {
    push: (msg) => {
      const { type = 'info', title, message } = msg;
      const colors = { success: '#10b981', error: '#ef4444', info: '#3b82f6', warning: '#f59e0b' };
      const color = colors[type] || colors.info;
      const toast = document.createElement('div');
      toast.style.cssText = `position:fixed;top:20px;right:20px;background:${color};color:white;padding:12px 20px;border-radius:8px;z-index:9999;box-shadow:0 4px 6px rgba(0,0,0,0.2)`;
      toast.textContent = title + (message ? ': ' + message : '');
      document.body.appendChild(toast);
      setTimeout(() => toast.remove(), 3000);
    }
  };
};

export default function ProfilePage() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [settings, setSettings] = useState({
    name: '',
    email: '',
    organization: '',
    hf_token: '',
    openai_api_key: '',
    wandb_api_key: '',
    pinecone_api_key: '',
    default_framework: 'pytorch',
    auto_save_interval: 300,
    notification_enabled: true,
    theme: 'dark'
  });
  const [dashboard, setDashboard] = useState(null);
  const [persistentConfig, setPersistentConfig] = useState(null);
  const [environmentInfo, setEnvironmentInfo] = useState(null);
  const [hfModels, setHfModels] = useState([]);
  const [hfDatasets, setHfDatasets] = useState([]);
  const [hfSearchModel, setHfSearchModel] = useState('');
  const [hfSearchDataset, setHfSearchDataset] = useState('');
  const [downloading, setDownloading] = useState({});
  const [showTransferQueue, setShowTransferQueue] = useState(false);
  const toast = useToast();

  useEffect(() => {
    loadSettings();
    loadDashboard();
    loadPersistentConfig();
    loadEnvironmentInfo();
  }, []);

  const loadSettings = async () => {
    try {
      const res = await fetch('/api/user/settings');
      if (res.ok) {
        const data = await res.json();
        setSettings(prev => ({ ...prev, ...data }));
      }
    } catch (e) {
      console.error('Failed to load settings:', e);
    }
  };

  const loadDashboard = async () => {
    try {
      const res = await fetch('/api/user/dashboard');
      if (res.ok) {
        const text = await res.text();
        if (text) {
          try {
            const data = JSON.parse(text);
            setDashboard(data);
          } catch (parseError) {
            console.error('Failed to parse dashboard JSON:', parseError);
          }
        }
      }
    } catch (e) {
      console.error('Failed to load dashboard:', e);
    }
  };

  const loadPersistentConfig = async () => {
    try {
      const res = await fetch('/api/config/persistent');
      if (res.ok) {
        const text = await res.text();
        if (text) {
          try {
            const data = JSON.parse(text);
            setPersistentConfig(data);
          } catch (parseError) {
            console.error('Failed to parse config JSON:', parseError);
          }
        }
      }
    } catch (e) {
      console.error('Failed to load persistent config:', e);
    }
  };

  const loadEnvironmentInfo = async () => {
    try {
      const res = await fetch('/api/settings/environment');
      if (res.ok) {
        const text = await res.text();
        if (text) {
          try {
            const data = JSON.parse(text);
            setEnvironmentInfo(data.environment);
          } catch (parseError) {
            console.error('Failed to parse environment JSON:', parseError);
          }
        }
      }
    } catch (e) {
      console.error('Failed to load environment info:', e);
    }
  };

  const savePersistentConfig = async () => {
    try {
      const res = await fetch('/api/config/persistent', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(persistentConfig)
      });
      if (res.ok) {
        toast.push({ type: 'success', title: 'Persistent config saved' });
      } else {
        const err = await res.json();
        toast.push({ type: 'error', title: 'Save failed', message: err.error || 'Unknown error' });
      }
    } catch (e) {
      toast.push({ type: 'error', title: 'Save failed', message: e.message });
    }
  };

  const saveSettings = async () => {
    try {
      const res = await fetch('/api/user/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      if (res.ok) {
        toast.push({ type: 'success', title: 'Settings saved' });
      } else {
        const err = await res.json();
        toast.push({ type: 'error', title: 'Save failed', message: err.error || 'Unknown error' });
      }
    } catch (e) {
      toast.push({ type: 'error', title: 'Save failed', message: e.message });
    }
  };

  const searchHuggingFaceModels = async () => {
    if (!hfSearchModel.trim()) {
      toast.push({ type: 'warning', title: 'Enter search term' });
      return;
    }
    try {
      const res = await fetch(`https://huggingface.co/api/models?search=${encodeURIComponent(hfSearchModel)}&limit=10`);
      const data = await res.json();
      setHfModels(data);
      toast.push({ type: 'success', title: `Found ${data.length} models` });
    } catch (e) {
      toast.push({ type: 'error', title: 'Search failed', message: e.message });
    }
  };

  const searchHuggingFaceDatasets = async () => {
    if (!hfSearchDataset.trim()) {
      toast.push({ type: 'warning', title: 'Enter search term' });
      return;
    }
    try {
      const res = await fetch(`https://huggingface.co/api/datasets?search=${encodeURIComponent(hfSearchDataset)}&limit=10`);
      const data = await res.json();
      setHfDatasets(data);
      toast.push({ type: 'success', title: `Found ${data.length} datasets` });
    } catch (e) {
      toast.push({ type: 'error', title: 'Search failed', message: e.message });
    }
  };

  const downloadHuggingFaceModel = async (modelId) => {
    setDownloading(prev => ({ ...prev, [modelId]: true }));
    try {
      const res = await fetch('/api/huggingface/download-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          token: settings.hf_token || null
        })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        toast.push({
          type: 'success',
          title: 'Model queued for download',
          message: 'Check Transfer Queue for progress'
        });
        // Auto-open transfer queue after 1 second
        setTimeout(() => setShowTransferQueue(true), 1000);
      } else {
        toast.push({ type: 'error', title: 'Download failed', message: data.error || 'Unknown error' });
      }
    } catch (e) {
      toast.push({ type: 'error', title: 'Download failed', message: e.message });
    } finally {
      setDownloading(prev => ({ ...prev, [modelId]: false }));
    }
  };

  const downloadHuggingFaceDataset = async (datasetId) => {
    setDownloading(prev => ({ ...prev, [datasetId]: true }));
    try {
      const res = await fetch('/api/huggingface/download-dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          token: settings.hf_token || null
        })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        toast.push({
          type: 'success',
          title: 'Dataset queued for download',
          message: 'Check Transfer Queue for progress'
        });
        // Auto-open transfer queue after 1 second
        setTimeout(() => setShowTransferQueue(true), 1000);
      } else {
        toast.push({ type: 'error', title: 'Download failed', message: data.error || 'Unknown error' });
      }
    } catch (e) {
      toast.push({ type: 'error', title: 'Download failed', message: e.message });
    } finally {
      setDownloading(prev => ({ ...prev, [datasetId]: false }));
    }
  };

  const testHuggingFaceToken = async () => {
    if (!settings.hf_token) {
      toast.push({ type: 'warning', title: 'No token provided' });
      return;
    }
    try {
      const res = await fetch('https://huggingface.co/api/whoami', {
        headers: { 'Authorization': `Bearer ${settings.hf_token}` }
      });
      if (res.ok) {
        const data = await res.json();
        toast.push({ type: 'success', title: 'Token valid', message: `Logged in as ${data.name}` });
      } else {
        toast.push({ type: 'error', title: 'Token invalid' });
      }
    } catch (e) {
      toast.push({ type: 'error', title: 'Test failed', message: e.message });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Profile & Settings</h1>
        <button
          onClick={saveSettings}
          className="px-4 py-2 bg-primary text-on-primary rounded hover:brightness-110"
        >
          Save All Settings
        </button>
      </div>

      {/* Tabs */}
      <div className="border-b border-border">
        <div className="flex gap-4">
          {['dashboard', 'general', 'api-keys', 'huggingface', 'preferences'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === tab
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-text/60 hover:text-text'
              }`}
            >
              {tab.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && (
        <div className="space-y-6">
          {/* Statistics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-surface border border-border rounded-lg p-6">
              <div className="text-sm font-medium text-text/60 mb-2">Datasets</div>
              <div className="text-3xl font-bold">{dashboard?.datasets_count || 0}</div>
              <div className="text-xs text-text/50 mt-2">Available datasets</div>
            </div>

            <div className="bg-surface border border-border rounded-lg p-6">
              <div className="text-sm font-medium text-text/60 mb-2">Models</div>
              <div className="text-3xl font-bold">{dashboard?.models_count || 0}</div>
              <div className="text-xs text-text/50 mt-2">Trained models</div>
            </div>

            <div className="bg-surface border border-border rounded-lg p-6">
              <div className="text-sm font-medium text-text/60 mb-2">Total Runs</div>
              <div className="text-3xl font-bold">{dashboard?.job_stats?.total || 0}</div>
              <div className="text-xs text-text/50 mt-2">
                {dashboard?.job_stats?.running || 0} running, {dashboard?.job_stats?.queued || 0} queued
              </div>
            </div>
          </div>

          {/* Job Statistics */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Job Statistics</h2>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
              {dashboard?.job_stats && Object.entries(dashboard.job_stats).map(([key, value]) => (
                <div key={key}>
                  <div className="text-2xl font-bold">{value}</div>
                  <div className="text-xs text-text/60 capitalize">{key}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Environment Summary */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Environment Summary</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* System Information */}
              <div>
                <h3 className="text-sm font-semibold text-text/70 mb-3">System</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-text/60">Platform:</span>
                    <span className="font-mono text-xs">{environmentInfo?.system?.platform?.split(' ')[0] || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text/60">Architecture:</span>
                    <span className="font-mono">{environmentInfo?.system?.architecture || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text/60">Python:</span>
                    <span className="font-mono text-xs">{environmentInfo?.system?.python_version?.split(' ')[0] || dashboard?.environment?.python_version || 'N/A'}</span>
                  </div>
                </div>
              </div>

              {/* PyTorch & CUDA */}
              <div>
                <h3 className="text-sm font-semibold text-text/70 mb-3">Compute</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-text/60">PyTorch:</span>
                    <span className="font-mono">{environmentInfo?.pytorch?.version || dashboard?.environment?.pytorch_version || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text/60">CUDA:</span>
                    <span className="font-mono">
                      {environmentInfo?.pytorch?.cuda_available ? environmentInfo.pytorch.cuda_version : 'Not available'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text/60">cuDNN:</span>
                    <span className="font-mono">
                      {environmentInfo?.pytorch?.cudnn_version || 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text/60">GPUs:</span>
                    <span className="font-mono">{environmentInfo?.pytorch?.gpu_count || dashboard?.environment?.gpu_count || 0}</span>
                  </div>
                </div>
              </div>

              {/* Libraries */}
              <div>
                <h3 className="text-sm font-semibold text-text/70 mb-3">Libraries</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-text/60">Transformers:</span>
                    <span className="font-mono">{environmentInfo?.transformers?.version || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text/60">FFmpeg:</span>
                    <span className={`font-mono ${environmentInfo?.ffmpeg?.available ? 'text-success' : 'text-error'}`}>
                      {environmentInfo?.ffmpeg?.available ? 'Available' : 'Not installed'}
                    </span>
                  </div>
                  {environmentInfo?.spark_trainer && (
                    <div className="flex justify-between">
                      <span className="text-text/60">SparkTrainer:</span>
                      <span className="font-mono text-xs">{environmentInfo.spark_trainer.base_dir?.split('/').pop() || 'Installed'}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* GPU Details */}
            {(dashboard?.system?.gpus && dashboard.system.gpus.length > 0) || (environmentInfo?.gpus && environmentInfo.gpus.length > 0) ? (
              <div className="mt-6">
                <h3 className="text-sm font-semibold text-text/70 mb-3">GPU Details</h3>
                <div className="space-y-2">
                  {dashboard?.system?.gpus ? (
                    dashboard.system.gpus.map((gpu, idx) => (
                      <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded border border-border">
                        <div>
                          <div className="font-medium">{gpu.name || `GPU ${idx}`}</div>
                          <div className="text-xs text-text/60">
                            {(gpu.memory_used_mib / 1024).toFixed(1)} GB / {(gpu.memory_total_mib / 1024).toFixed(1)} GB
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-mono">{gpu.utilization || 0}%</div>
                          <div className="text-xs text-text/60">Utilization</div>
                        </div>
                      </div>
                    ))
                  ) : environmentInfo?.gpus?.map((gpu, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded border border-border">
                      <div>
                        <div className="font-medium">{gpu.name || `GPU ${idx}`}</div>
                        <div className="text-xs text-text/60">
                          Capability: {gpu.capability ? `${gpu.capability[0]}.${gpu.capability[1]}` : 'N/A'} |
                          Memory: {(gpu.total_memory / (1024**3)).toFixed(1)} GB
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-mono">GPU {gpu.id}</div>
                        <div className="text-xs text-text/60">Device ID</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {/* System Resources */}
            {dashboard?.system?.memory && (
              <div className="mt-6">
                <h3 className="text-sm font-semibold text-text/70 mb-3">System Resources</h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Memory Usage</span>
                      <span className="font-mono">{dashboard.system.memory.used_pct}%</span>
                    </div>
                    <div className="w-full bg-border rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all"
                        style={{ width: `${dashboard.system.memory.used_pct}%` }}
                      />
                    </div>
                    <div className="text-xs text-text/60 mt-1">
                      {dashboard.system.memory.used_mib} MB / {dashboard.system.memory.total_mib} MB
                    </div>
                  </div>

                  {dashboard.system.cpu?.load_avg && (
                    <div className="flex justify-between text-sm">
                      <span className="text-text/60">Load Average:</span>
                      <span className="font-mono">
                        {dashboard.system.cpu.load_avg.map(l => l.toFixed(2)).join(', ')}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Recent Runs */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Recent Runs</h2>
            {dashboard?.recent_jobs && dashboard.recent_jobs.length > 0 ? (
              <div className="space-y-2">
                {dashboard.recent_jobs.map(job => (
                  <div key={job.id} className="flex items-center justify-between p-3 bg-muted rounded border border-border hover:bg-muted/80">
                    <div className="flex-1 min-w-0">
                      <div className="font-medium truncate">{job.name}</div>
                      <div className="text-xs text-text/60">
                        {new Date(job.created).toLocaleString()}
                      </div>
                    </div>
                    <div className="ml-4">
                      <span className={`px-2 py-1 text-xs rounded ${
                        job.status === 'completed' ? 'bg-success/10 text-success border border-success/30' :
                        job.status === 'running' ? 'bg-primary/10 text-primary border border-primary/30' :
                        job.status === 'failed' ? 'bg-error/10 text-error border border-error/30' :
                        job.status === 'queued' ? 'bg-warning/10 text-warning border border-warning/30' :
                        'bg-muted text-text/60 border border-border'
                      }`}>
                        {job.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-text/60">
                No recent runs
              </div>
            )}
          </div>

          {/* Persistent Config */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Persistent Configuration</h2>
              <button
                onClick={savePersistentConfig}
                className="px-3 py-1 text-sm bg-primary text-on-primary rounded hover:brightness-110"
              >
                Save Config
              </button>
            </div>
            <div className="text-xs text-text/60 mb-3">
              Stored at: ~/.spark_trainer/config.json
            </div>
            <div className="space-y-4">
              {persistentConfig && (
                <div>
                  <label className="block text-sm font-medium mb-2">Default Framework</label>
                  <select
                    className="w-full border border-border rounded px-3 py-2 bg-surface"
                    value={persistentConfig.defaults?.framework || 'pytorch'}
                    onChange={e => setPersistentConfig({
                      ...persistentConfig,
                      defaults: { ...persistentConfig.defaults, framework: e.target.value }
                    })}
                  >
                    <option value="pytorch">PyTorch</option>
                    <option value="huggingface">HuggingFace Transformers</option>
                    <option value="tensorflow">TensorFlow</option>
                  </select>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* General Settings Tab */}
      {activeTab === 'general' && (
        <div className="bg-surface border border-border rounded-lg p-6 space-y-4">
          <h2 className="text-xl font-semibold mb-4">General Information</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Name</label>
              <input
                type="text"
                className="w-full border border-border rounded px-3 py-2 bg-surface"
                value={settings.name}
                onChange={e => setSettings({ ...settings, name: e.target.value })}
                placeholder="Your name"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Email</label>
              <input
                type="email"
                className="w-full border border-border rounded px-3 py-2 bg-surface"
                value={settings.email}
                onChange={e => setSettings({ ...settings, email: e.target.value })}
                placeholder="your.email@example.com"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Organization</label>
              <input
                type="text"
                className="w-full border border-border rounded px-3 py-2 bg-surface"
                value={settings.organization}
                onChange={e => setSettings({ ...settings, organization: e.target.value })}
                placeholder="Your organization"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Default Framework</label>
              <select
                className="w-full border border-border rounded px-3 py-2 bg-surface"
                value={settings.default_framework}
                onChange={e => setSettings({ ...settings, default_framework: e.target.value })}
              >
                <option value="pytorch">PyTorch</option>
                <option value="huggingface">HuggingFace Transformers</option>
                <option value="tensorflow">TensorFlow</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* API Keys Tab */}
      {activeTab === 'api-keys' && (
        <div className="bg-surface border border-border rounded-lg p-6 space-y-6">
          <h2 className="text-xl font-semibold mb-4">API Keys</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">HuggingFace Token</label>
              <div className="flex gap-2">
                <input
                  type="password"
                  className="flex-1 border border-border rounded px-3 py-2 bg-surface font-mono text-sm"
                  value={settings.hf_token}
                  onChange={e => setSettings({ ...settings, hf_token: e.target.value })}
                  placeholder="hf_..."
                />
                <button
                  onClick={testHuggingFaceToken}
                  className="px-3 py-2 border border-border rounded bg-surface hover:bg-muted"
                >
                  Test
                </button>
              </div>
              <p className="text-xs text-text/60 mt-1">
                Get your token from <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">huggingface.co/settings/tokens</a>
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">OpenAI API Key</label>
              <input
                type="password"
                className="w-full border border-border rounded px-3 py-2 bg-surface font-mono text-sm"
                value={settings.openai_api_key}
                onChange={e => setSettings({ ...settings, openai_api_key: e.target.value })}
                placeholder="sk-..."
              />
              <p className="text-xs text-text/60 mt-1">Used for GPT-based features and embeddings</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Weights & Biases API Key</label>
              <input
                type="password"
                className="w-full border border-border rounded px-3 py-2 bg-surface font-mono text-sm"
                value={settings.wandb_api_key}
                onChange={e => setSettings({ ...settings, wandb_api_key: e.target.value })}
                placeholder="..."
              />
              <p className="text-xs text-text/60 mt-1">Enable automatic experiment tracking with W&B</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Pinecone API Key</label>
              <input
                type="password"
                className="w-full border border-border rounded px-3 py-2 bg-surface font-mono text-sm"
                value={settings.pinecone_api_key || ''}
                onChange={e => setSettings({ ...settings, pinecone_api_key: e.target.value })}
                placeholder="..."
              />
              <p className="text-xs text-text/60 mt-1">Used for vector storage and similarity search</p>
            </div>
          </div>

          <div className="pt-4 border-t border-border">
            <p className="text-sm text-warning">
              ⚠️ API keys are stored securely on the server. Never share your keys with others.
            </p>
          </div>
        </div>
      )}

      {/* HuggingFace Integration Tab */}
      {activeTab === 'huggingface' && (
        <div className="space-y-6">
          {/* Transfer Queue Button */}
          <div className="flex justify-end">
            <button
              onClick={() => setShowTransferQueue(true)}
              className="px-4 py-2 bg-primary text-on-primary rounded-lg hover:brightness-110 flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              View Transfer Queue
            </button>
          </div>

          {/* Model Search */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Download HuggingFace Models</h2>

            <div className="flex gap-2 mb-4">
              <input
                type="text"
                className="flex-1 border border-border rounded px-3 py-2 bg-surface"
                value={hfSearchModel}
                onChange={e => setHfSearchModel(e.target.value)}
                placeholder="Search models (e.g., bert, gpt2, llama)"
                onKeyDown={e => e.key === 'Enter' && searchHuggingFaceModels()}
              />
              <button
                onClick={searchHuggingFaceModels}
                className="px-4 py-2 bg-primary text-on-primary rounded hover:brightness-110"
              >
                Search
              </button>
            </div>

            {hfModels.length > 0 && (
              <div className="space-y-2">
                <div className="text-sm font-medium mb-2">Results:</div>
                {hfModels.map(model => (
                  <div key={model.id} className="flex items-center justify-between border border-border rounded p-3 hover:bg-muted">
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold truncate">{model.id}</div>
                      <div className="text-xs text-text/60">
                        {model.downloads?.toLocaleString() || 0} downloads • {model.likes?.toLocaleString() || 0} likes
                      </div>
                      {model.pipeline_tag && (
                        <span className="inline-block mt-1 px-2 py-0.5 text-xs bg-accent/10 text-accent border border-accent/30 rounded">
                          {model.pipeline_tag}
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => downloadHuggingFaceModel(model.id)}
                      disabled={downloading[model.id]}
                      className="ml-4 px-3 py-1 bg-success/10 text-success border border-success/30 rounded hover:bg-success/20 disabled:opacity-50"
                    >
                      {downloading[model.id] ? 'Downloading...' : 'Download'}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Dataset Search */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Download HuggingFace Datasets</h2>

            <div className="flex gap-2 mb-4">
              <input
                type="text"
                className="flex-1 border border-border rounded px-3 py-2 bg-surface"
                value={hfSearchDataset}
                onChange={e => setHfSearchDataset(e.target.value)}
                placeholder="Search datasets (e.g., squad, imdb, coco)"
                onKeyDown={e => e.key === 'Enter' && searchHuggingFaceDatasets()}
              />
              <button
                onClick={searchHuggingFaceDatasets}
                className="px-4 py-2 bg-primary text-on-primary rounded hover:brightness-110"
              >
                Search
              </button>
            </div>

            {hfDatasets.length > 0 && (
              <div className="space-y-2">
                <div className="text-sm font-medium mb-2">Results:</div>
                {hfDatasets.map(dataset => (
                  <div key={dataset.id} className="flex items-center justify-between border border-border rounded p-3 hover:bg-muted">
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold truncate">{dataset.id}</div>
                      <div className="text-xs text-text/60">
                        {dataset.downloads?.toLocaleString() || 0} downloads • {dataset.likes?.toLocaleString() || 0} likes
                      </div>
                      {dataset.tags && dataset.tags.length > 0 && (
                        <div className="mt-1">
                          {dataset.tags.slice(0, 3).map(tag => (
                            <span key={tag} className="inline-block mr-1 px-2 py-0.5 text-xs bg-secondary/10 text-secondary border border-secondary/30 rounded">
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => downloadHuggingFaceDataset(dataset.id)}
                      disabled={downloading[dataset.id]}
                      className="ml-4 px-3 py-1 bg-success/10 text-success border border-success/30 rounded hover:bg-success/20 disabled:opacity-50"
                    >
                      {downloading[dataset.id] ? 'Downloading...' : 'Download'}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Preferences Tab */}
      {activeTab === 'preferences' && (
        <div className="bg-surface border border-border rounded-lg p-6 space-y-6">
          <h2 className="text-xl font-semibold mb-4">User Preferences</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Auto-save Interval (seconds)</label>
              <input
                type="number"
                className="w-full border border-border rounded px-3 py-2 bg-surface"
                value={settings.auto_save_interval}
                onChange={e => setSettings({ ...settings, auto_save_interval: parseInt(e.target.value) || 0 })}
                min="0"
                step="30"
              />
              <p className="text-xs text-text/60 mt-1">Set to 0 to disable auto-save</p>
            </div>

            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="notifications"
                checked={settings.notification_enabled}
                onChange={e => setSettings({ ...settings, notification_enabled: e.target.checked })}
                className="w-4 h-4"
              />
              <label htmlFor="notifications" className="text-sm font-medium">Enable desktop notifications</label>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Theme</label>
              <select
                className="w-full border border-border rounded px-3 py-2 bg-surface"
                value={settings.theme}
                onChange={e => setSettings({ ...settings, theme: e.target.value })}
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="auto">Auto (System)</option>
              </select>
            </div>
          </div>

          <div className="pt-4 border-t border-border">
            <h3 className="text-lg font-semibold mb-3">Danger Zone</h3>
            <button
              onClick={() => {
                if (confirm('Clear all cached data? This will not delete your datasets or models.')) {
                  toast.push({ type: 'info', title: 'Cache cleared' });
                }
              }}
              className="px-4 py-2 border border-warning text-warning rounded hover:bg-warning/10"
            >
              Clear Cache
            </button>
          </div>
        </div>
      )}

      {/* Transfer Queue Modal */}
      {showTransferQueue && (
        <TransferQueue onClose={() => setShowTransferQueue(false)} />
      )}
    </div>
  );
}
