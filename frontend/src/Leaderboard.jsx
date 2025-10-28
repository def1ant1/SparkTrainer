import React, { useState, useEffect } from 'react';
import { Trophy, TrendingUp, Medal, Star, Filter, Download } from 'lucide-react';
import { Card } from './components/Card';
import { Button } from './components/Button';
import { Tabs } from './components/Tabs';

const Leaderboard = () => {
  const [entries, setEntries] = useState([]);
  const [benchmark, setBenchmark] = useState('all');
  const [loading, setLoading] = useState(true);

  const benchmarks = [
    { id: 'all', name: 'All Benchmarks' },
    { id: 'mmlu', name: 'MMLU' },
    { id: 'coco', name: 'COCO Captioning' },
    { id: 'glue', name: 'GLUE' },
    { id: 'superglue', name: 'SuperGLUE' },
  ];

  useEffect(() => {
    fetchLeaderboard();
  }, [benchmark]);

  const fetchLeaderboard = async () => {
    try {
      const params = benchmark !== 'all' ? `?benchmark=${benchmark}` : '';
      const response = await fetch(`/api/leaderboard${params}`);
      const data = await response.json();
      setEntries(data);
    } catch (error) {
      console.error('Error fetching leaderboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRankBadge = (rank) => {
    if (rank === 1) return <Trophy className="text-yellow-500" size={24} />;
    if (rank === 2) return <Medal className="text-gray-400" size={24} />;
    if (rank === 3) return <Medal className="text-amber-600" size={24} />;
    return <span className="text-2xl font-bold text-gray-400">#{rank}</span>;
  };

  const getRankColor = (rank) => {
    if (rank === 1) return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500';
    if (rank === 2) return 'bg-gray-50 dark:bg-gray-800/50 border-gray-400';
    if (rank === 3) return 'bg-amber-50 dark:bg-amber-900/20 border-amber-600';
    return 'bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-700';
  };

  const exportLeaderboard = () => {
    const csv = [
      ['Rank', 'Model', 'Benchmark', 'Score', 'Type'].join(','),
      ...entries.map(e => [
        e.rank,
        e.model_name,
        e.benchmark_name,
        e.score.toFixed(4),
        e.model_type
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `leaderboard_${benchmark}_${Date.now()}.csv`;
    a.click();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Trophy className="text-yellow-500" size={32} />
            Leaderboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Compare model performance across benchmarks
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={exportLeaderboard}>
            <Download size={16} />
            Export CSV
          </Button>
        </div>
      </div>

      {/* Benchmark Filter */}
      <Card>
        <div className="p-4">
          <div className="flex items-center gap-4">
            <Filter size={20} className="text-gray-400" />
            <div className="flex gap-2 flex-wrap">
              {benchmarks.map((b) => (
                <button
                  key={b.id}
                  onClick={() => setBenchmark(b.id)}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    benchmark === b.id
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                  }`}
                >
                  {b.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Leaderboard Table */}
      {entries.length === 0 ? (
        <Card>
          <div className="p-12 text-center">
            <Trophy className="mx-auto mb-4 text-gray-400" size={64} />
            <h3 className="text-xl font-semibold mb-2">No entries yet</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Run evaluations on your models to see them here
            </p>
          </div>
        </Card>
      ) : (
        <div className="space-y-3">
          {entries.map((entry, index) => (
            <Card
              key={entry.id}
              className={`hover:shadow-lg transition-all border-2 ${getRankColor(entry.rank)}`}
            >
              <div className="p-6">
                <div className="flex items-center gap-6">
                  {/* Rank Badge */}
                  <div className="flex-shrink-0 w-16 text-center">
                    {getRankBadge(entry.rank)}
                  </div>

                  {/* Model Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-xl font-bold truncate">
                        {entry.model_name || `Experiment ${entry.experiment_id.slice(0, 8)}`}
                      </h3>
                      {entry.rank <= 3 && (
                        <Star className="text-yellow-500 fill-yellow-500" size={20} />
                      )}
                    </div>
                    <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                      <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
                        {entry.model_type || 'Unknown Type'}
                      </span>
                      <span>{entry.benchmark_name.toUpperCase()}</span>
                      <span>•</span>
                      <span>{new Date(entry.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>

                  {/* Score */}
                  <div className="flex-shrink-0 text-right">
                    <div className="text-4xl font-bold text-blue-600 dark:text-blue-400">
                      {(entry.score * 100).toFixed(2)}%
                    </div>
                    <div className="text-sm text-gray-500 mt-1">Score</div>
                  </div>

                  {/* Trend Indicator */}
                  <div className="flex-shrink-0">
                    {index === 0 ? (
                      <div className="flex items-center gap-2 text-green-500">
                        <TrendingUp size={20} />
                        <span className="text-sm font-medium">Top</span>
                      </div>
                    ) : entry.rank <= 10 ? (
                      <div className="text-sm font-medium text-gray-500">
                        Top {Math.ceil(entry.rank / 10) * 10}
                      </div>
                    ) : null}
                  </div>
                </div>

                {/* Additional Metrics */}
                {entry.metadata && Object.keys(entry.metadata).length > 0 && (
                  <div className="mt-4 pt-4 border-t dark:border-gray-700">
                    <div className="flex gap-6 text-sm">
                      {Object.entries(entry.metadata).slice(0, 4).map(([key, value]) => (
                        <div key={key}>
                          <span className="text-gray-500">{key}: </span>
                          <span className="font-medium">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Stats Summary */}
      {entries.length > 0 && (
        <Card>
          <div className="p-6">
            <h3 className="font-semibold mb-4">Leaderboard Statistics</h3>
            <div className="grid grid-cols-4 gap-6">
              <div>
                <div className="text-3xl font-bold text-blue-500">{entries.length}</div>
                <div className="text-sm text-gray-600">Total Entries</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-green-500">
                  {(entries[0]?.score * 100).toFixed(2)}%
                </div>
                <div className="text-sm text-gray-600">Top Score</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-purple-500">
                  {(entries.reduce((sum, e) => sum + e.score, 0) / entries.length * 100).toFixed(2)}%
                </div>
                <div className="text-sm text-gray-600">Average Score</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-orange-500">
                  {new Set(entries.map(e => e.model_type)).size}
                </div>
                <div className="text-sm text-gray-600">Model Types</div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default Leaderboard;
