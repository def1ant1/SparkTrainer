import React, { useState, useEffect, useCallback } from 'react';
import {
  Download,
  Upload,
  Pause,
  Play,
  X,
  Trash2,
  RefreshCw,
  Settings,
  CheckCircle,
  XCircle,
  Clock,
  AlertCircle
} from 'lucide-react';

/**
 * Transfer Queue component for HuggingFace model/dataset downloads and uploads
 * Shows active transfers with progress, bandwidth usage, and queue management
 */
export default function TransferQueue({ onClose }) {
  const [transfers, setTransfers] = useState([]);
  const [stats, setStats] = useState(null);
  const [settings, setSettings] = useState({
    global_limit_bps: 10 * 1024 * 1024, // 10 MB/s
    max_concurrent: 3
  });
  const [showSettings, setShowSettings] = useState(false);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('all'); // all, active, completed, failed

  // Fetch transfers and stats
  const fetchData = useCallback(async () => {
    try {
      // Fetch transfers
      const statusParam = filter !== 'all' ? `?status=${filter}` : '';
      const transfersRes = await fetch(`/api/transfers${statusParam}`);
      const transfersData = await transfersRes.json();
      setTransfers(transfersData.transfers || []);

      // Fetch stats
      const statsRes = await fetch('/api/transfers/stats');
      const statsData = await statsRes.json();
      setStats(statsData);

      // Fetch settings
      const settingsRes = await fetch('/api/transfers/settings');
      const settingsData = await settingsRes.json();
      setSettings(settingsData);
    } catch (error) {
      console.error('Error fetching transfer data:', error);
    }
  }, [filter]);

  useEffect(() => {
    fetchData();
    // Poll for updates every 2 seconds
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Transfer actions
  const pauseTransfer = async (transferId) => {
    try {
      await fetch(`/api/transfers/${transferId}/pause`, { method: 'POST' });
      fetchData();
    } catch (error) {
      console.error('Error pausing transfer:', error);
    }
  };

  const resumeTransfer = async (transferId) => {
    try {
      await fetch(`/api/transfers/${transferId}/resume`, { method: 'POST' });
      fetchData();
    } catch (error) {
      console.error('Error resuming transfer:', error);
    }
  };

  const cancelTransfer = async (transferId) => {
    if (!confirm('Are you sure you want to cancel this transfer?')) return;
    try {
      await fetch(`/api/transfers/${transferId}/cancel`, { method: 'POST' });
      fetchData();
    } catch (error) {
      console.error('Error canceling transfer:', error);
    }
  };

  const deleteTransfer = async (transferId) => {
    if (!confirm('Are you sure you want to delete this transfer record?')) return;
    try {
      await fetch(`/api/transfers/${transferId}`, { method: 'DELETE' });
      fetchData();
    } catch (error) {
      console.error('Error deleting transfer:', error);
    }
  };

  // Update settings
  const updateSettings = async () => {
    try {
      setLoading(true);
      await fetch('/api/transfers/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      setShowSettings(false);
      fetchData();
    } catch (error) {
      console.error('Error updating settings:', error);
    } finally {
      setLoading(false);
    }
  };

  // Format bytes to human readable
  const formatBytes = (bytes) => {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  // Format rate (bytes per second)
  const formatRate = (bps) => {
    if (!bps || bps === 0) return '0 B/s';
    return `${formatBytes(bps)}/s`;
  };

  // Format ETA
  const formatETA = (seconds) => {
    if (!seconds) return 'calculating...';
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
  };

  // Get status badge
  const getStatusBadge = (status) => {
    const badges = {
      queued: { icon: Clock, color: 'text-yellow-600 bg-yellow-50', label: 'Queued' },
      downloading: { icon: Download, color: 'text-blue-600 bg-blue-50', label: 'Downloading' },
      uploading: { icon: Upload, color: 'text-purple-600 bg-purple-50', label: 'Uploading' },
      paused: { icon: Pause, color: 'text-gray-600 bg-gray-50', label: 'Paused' },
      completed: { icon: CheckCircle, color: 'text-green-600 bg-green-50', label: 'Completed' },
      failed: { icon: XCircle, color: 'text-red-600 bg-red-50', label: 'Failed' },
      cancelled: { icon: X, color: 'text-gray-600 bg-gray-50', label: 'Cancelled' }
    };

    const badge = badges[status] || badges.queued;
    const Icon = badge.icon;

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${badge.color}`}>
        <Icon size={12} className="mr-1" />
        {badge.label}
      </span>
    );
  };

  // Get transfer type icon
  const getTypeIcon = (transferType) => {
    if (transferType.includes('download')) return <Download size={16} className="text-blue-600" />;
    if (transferType.includes('upload')) return <Upload size={16} className="text-purple-600" />;
    return <RefreshCw size={16} className="text-gray-600" />;
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Download size={24} className="text-blue-600" />
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Transfer Queue</h2>
              <p className="text-sm text-gray-500">
                {stats && `${stats.active_transfers} active • ${stats.queued_transfers} queued • ${stats.max_concurrent} max concurrent`}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              title="Settings"
            >
              <Settings size={20} className="text-gray-600" />
            </button>
            <button
              onClick={fetchData}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              title="Refresh"
            >
              <RefreshCw size={20} className="text-gray-600" />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              title="Close"
            >
              <X size={20} className="text-gray-600" />
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Bandwidth Settings</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-700 mb-1">
                  Global Bandwidth Limit (MB/s)
                </label>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={Math.round(settings.global_limit_bps / (1024 * 1024))}
                  onChange={(e) => setSettings({
                    ...settings,
                    global_limit_bps: parseInt(e.target.value) * 1024 * 1024
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="0 = unlimited"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Set to 0 for unlimited bandwidth
                </p>
              </div>
              <div>
                <label className="block text-sm text-gray-700 mb-1">
                  Max Concurrent Transfers
                </label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={settings.max_concurrent}
                  onChange={(e) => setSettings({
                    ...settings,
                    max_concurrent: parseInt(e.target.value)
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
            <div className="mt-3 flex justify-end space-x-2">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 text-sm text-gray-700 hover:bg-gray-200 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={updateSettings}
                disabled={loading}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                {loading ? 'Saving...' : 'Save Settings'}
              </button>
            </div>
          </div>
        )}

        {/* Filter Tabs */}
        <div className="px-6 py-3 border-b border-gray-200 flex space-x-4">
          {['all', 'active', 'completed', 'failed'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                filter === f
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>

        {/* Transfer List */}
        <div className="overflow-y-auto max-h-[calc(90vh-200px)]">
          {transfers.length === 0 ? (
            <div className="px-6 py-12 text-center">
              <Download size={48} className="mx-auto text-gray-400 mb-3" />
              <p className="text-gray-500">No transfers found</p>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {transfers.map((transfer) => (
                <div key={transfer.id} className="px-6 py-4 hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      {/* Transfer name and type */}
                      <div className="flex items-center space-x-2 mb-2">
                        {getTypeIcon(transfer.transfer_type)}
                        <h4 className="text-sm font-medium text-gray-900 truncate">
                          {transfer.name}
                        </h4>
                        {getStatusBadge(transfer.status)}
                      </div>

                      {/* Progress bar */}
                      {['downloading', 'uploading', 'paused'].includes(transfer.status) && (
                        <div className="mb-2">
                          <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
                            <span>{transfer.progress.toFixed(1)}%</span>
                            <span>
                              {formatBytes(transfer.bytes_transferred)} / {formatBytes(transfer.size_bytes)}
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${Math.min(transfer.progress, 100)}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Transfer stats */}
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        {transfer.average_rate > 0 && (
                          <span>Speed: {formatRate(transfer.average_rate)}</span>
                        )}
                        {transfer.eta_seconds && transfer.status === 'downloading' && (
                          <span>ETA: {formatETA(transfer.eta_seconds)}</span>
                        )}
                        {transfer.bandwidth_limit && (
                          <span>Limit: {formatRate(transfer.bandwidth_limit)}</span>
                        )}
                        {transfer.retries > 0 && (
                          <span className="text-orange-600">
                            Retries: {transfer.retries}/{transfer.max_retries}
                          </span>
                        )}
                      </div>

                      {/* Error message */}
                      {transfer.error_message && (
                        <div className="mt-2 flex items-start space-x-2 text-xs text-red-600">
                          <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
                          <span>{transfer.error_message}</span>
                        </div>
                      )}
                    </div>

                    {/* Action buttons */}
                    <div className="flex items-center space-x-2 ml-4">
                      {['downloading', 'uploading'].includes(transfer.status) && (
                        <button
                          onClick={() => pauseTransfer(transfer.id)}
                          className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                          title="Pause"
                        >
                          <Pause size={16} className="text-gray-600" />
                        </button>
                      )}
                      {transfer.status === 'paused' && (
                        <button
                          onClick={() => resumeTransfer(transfer.id)}
                          className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                          title="Resume"
                        >
                          <Play size={16} className="text-green-600" />
                        </button>
                      )}
                      {!['completed'].includes(transfer.status) && (
                        <button
                          onClick={() => cancelTransfer(transfer.id)}
                          className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                          title="Cancel"
                        >
                          <X size={16} className="text-red-600" />
                        </button>
                      )}
                      {['completed', 'failed', 'cancelled'].includes(transfer.status) && (
                        <button
                          onClick={() => deleteTransfer(transfer.id)}
                          className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                          title="Delete"
                        >
                          <Trash2 size={16} className="text-gray-600" />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer stats */}
        {stats && (
          <div className="px-6 py-3 border-t border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between text-xs text-gray-600">
              <span>
                Bandwidth: {stats.global_limit_bps
                  ? `${formatRate(stats.global_limit_bps)} limit`
                  : 'Unlimited'}
              </span>
              <span>
                Active: {stats.active_transfers_db} / {stats.max_concurrent}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
