import React, { useState, useEffect } from 'react';

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
  const [activeTab, setActiveTab] = useState('general');
  const [settings, setSettings] = useState({
    name: '',
    email: '',
    organization: '',
    hf_token: '',
    openai_api_key: '',
    wandb_api_key: '',
    default_framework: 'pytorch',
    auto_save_interval: 300,
    notification_enabled: true,
    theme: 'dark'
  });
  const [hfModels, setHfModels] = useState([]);
  const [hfDatasets, setHfDatasets] = useState([]);
  const [hfSearchModel, setHfSearchModel] = useState('');
  const [hfSearchDataset, setHfSearchDataset] = useState('');
  const [downloading, setDownloading] = useState({});
  const toast = useToast();

  useEffect(() => {
    loadSettings();
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
        toast.push({ type: 'success', title: 'Model download started', message: modelId });
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
        toast.push({ type: 'success', title: 'Dataset download started', message: datasetId });
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
          {['general', 'api-keys', 'huggingface', 'preferences'].map(tab => (
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
    </div>
  );
}
