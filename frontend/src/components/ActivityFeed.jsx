import React, { useState, useEffect, useCallback } from 'react';
import { Bell, Check, CheckCheck, Loader2, AlertCircle, CheckCircle2, Info, AlertTriangle, X } from 'lucide-react';

/**
 * ActivityFeed - Unified activity feed for tracking events
 *
 * Features:
 * - HF downloads/uploads (start, progress, completion/failure)
 * - Job lifecycle events
 * - Browser notifications (permission-gated)
 * - Mark as read functionality
 * - Real-time updates via polling
 */
const ActivityFeed = ({ userId, projectId, showUnreadOnly = false, className = "" }) => {
  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isOpen, setIsOpen] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);

  // Check and request notification permission
  useEffect(() => {
    if ('Notification' in window) {
      setNotificationsEnabled(Notification.permission === 'granted');
    }
  }, []);

  const requestNotificationPermission = async () => {
    if ('Notification' in window && Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      setNotificationsEnabled(permission === 'granted');
    }
  };

  // Show browser notification
  const showBrowserNotification = useCallback((activity) => {
    if (!notificationsEnabled || activity.read) return;

    const options = {
      body: activity.message,
      icon: '/icon.png',
      badge: '/badge.png',
      tag: activity.id,
      requireInteraction: activity.status === 'error',
    };

    const notification = new Notification(activity.title, options);

    notification.onclick = () => {
      window.focus();
      markAsRead(activity.id);
      notification.close();

      // Navigate to entity if possible
      if (activity.entity_type === 'job' && activity.entity_id) {
        window.location.href = `/jobs/${activity.entity_id}`;
      } else if (activity.entity_type === 'dataset' && activity.entity_id) {
        window.location.href = `/datasets`;
      }
    };
  }, [notificationsEnabled]);

  // Fetch activities
  const fetchActivities = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      params.append('limit', '50');
      if (userId) params.append('user_id', userId);
      if (projectId) params.append('project_id', projectId);
      if (showUnreadOnly) params.append('unread_only', 'true');

      const response = await fetch(`/api/activity?${params}`);
      if (!response.ok) throw new Error('Failed to fetch activities');

      const data = await response.json();
      const newActivities = data.activities || [];

      // Show notifications for new unread activities
      if (activities.length > 0) {
        const newUnread = newActivities.filter(a =>
          !a.read && !activities.find(old => old.id === a.id)
        );
        newUnread.forEach(showBrowserNotification);
      }

      setActivities(newActivities);
      setUnreadCount(newActivities.filter(a => !a.read).length);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [userId, projectId, showUnreadOnly, activities, showBrowserNotification]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    fetchActivities();
    const interval = setInterval(fetchActivities, 30000);
    return () => clearInterval(interval);
  }, [fetchActivities]);

  // Mark activity as read
  const markAsRead = async (activityId) => {
    try {
      const response = await fetch(`/api/activity/${activityId}/mark-read`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to mark as read');

      setActivities(prev => prev.map(a =>
        a.id === activityId ? { ...a, read: true } : a
      ));
      setUnreadCount(prev => Math.max(0, prev - 1));
    } catch (err) {
      console.error('Failed to mark activity as read:', err);
    }
  };

  // Mark all as read
  const markAllAsRead = async () => {
    try {
      const body = {};
      if (userId) body.user_id = userId;
      if (projectId) body.project_id = projectId;

      const response = await fetch('/api/activity/mark-all-read', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error('Failed to mark all as read');

      setActivities(prev => prev.map(a => ({ ...a, read: true })));
      setUnreadCount(0);
    } catch (err) {
      console.error('Failed to mark all as read:', err);
    }
  };

  // Get status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return <CheckCircle2 className="w-5 h-5 text-green-600" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
      default:
        return <Info className="w-5 h-5 text-blue-600" />;
    }
  };

  // Format time ago
  const timeAgo = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className={`relative ${className}`}>
      {/* Bell icon with badge */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
        aria-label="Activity feed"
      >
        <Bell className="w-6 h-6" />
        {unreadCount > 0 && (
          <span className="absolute top-0 right-0 inline-flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-red-600 rounded-full">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown panel */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 max-h-[32rem] overflow-hidden z-50">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Activity
              </h3>
              {unreadCount > 0 && (
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {unreadCount} unread
                </p>
              )}
            </div>
            <div className="flex items-center gap-2">
              {!notificationsEnabled && (
                <button
                  onClick={requestNotificationPermission}
                  className="text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
                  title="Enable browser notifications"
                >
                  Enable notifications
                </button>
              )}
              {unreadCount > 0 && (
                <button
                  onClick={markAllAsRead}
                  className="text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
                  title="Mark all as read"
                >
                  <CheckCheck className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Activities list */}
          <div className="overflow-y-auto max-h-96">
            {loading && activities.length === 0 && (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
              </div>
            )}

            {error && (
              <div className="p-4 text-center">
                <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-2" />
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}

            {!loading && !error && activities.length === 0 && (
              <div className="text-center py-12 px-4">
                <Bell className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  No activities yet
                </p>
              </div>
            )}

            {!loading && !error && activities.length > 0 && (
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {activities.map((activity) => (
                  <div
                    key={activity.id}
                    className={`p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors ${
                      !activity.read ? 'bg-blue-50/50 dark:bg-blue-900/10' : ''
                    }`}
                    onClick={() => markAsRead(activity.id)}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 mt-0.5">
                        {getStatusIcon(activity.status)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                          {activity.title}
                        </p>
                        {activity.message && (
                          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                            {activity.message}
                          </p>
                        )}
                        <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
                          {timeAgo(activity.created_at)}
                        </p>
                      </div>
                      {!activity.read && (
                        <div className="flex-shrink-0">
                          <div className="w-2 h-2 bg-blue-600 rounded-full" />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
            <a
              href="/activity"
              className="text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
            >
              View all activity →
            </a>
          </div>
        </div>
      )}

      {/* Click outside to close */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
};

export default ActivityFeed;
