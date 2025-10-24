import React, { useState, useEffect } from 'react';
import { Camera, Activity, Cpu, Database, Plus, List, Settings } from 'lucide-react';

// API Service
const API_BASE = 'http://localhost:5000/api';

const api = {
  getSystemInfo: () => fetch(`${API_BASE}/system/info`).then(r => r.json()),
  getFrameworks: () => fetch(`${API_BASE}/frameworks`).then(r => r.json()),
  getJobs: () => fetch(`${API_BASE}/jobs`).then(r => r.json()),
  getJob: (id) => fetch(`${API_BASE}/jobs/${id}`).then(r => r.json()),
  createJob: (data) => fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  }).then(r => r.json()),
  cancelJob: (id) => fetch(`${API_BASE}/jobs/${id}/cancel`, {
    method: 'POST'
  }).then(r => r.json()),
  getModels: () => fetch(`${API_BASE}/models`).then(r => r.json())
};

// Dashboard Component
const Dashboard = ({ onNavigate, systemInfo }) => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">DGX AI Trainer Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">GPUs Available</p>
              <p className="text-2xl font-bold text-blue-600">{systemInfo.gpus?.length || 0}</p>
            </div>
            <Cpu className="text-blue-500" size={32} />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Running Jobs</p>
              <p className="text-2xl font-bold text-green-600">{systemInfo.jobs_running || 0}</p>
            </div>
            <Activity className="text-green-500" size={32} />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Queued Jobs</p>
              <p className="text-2xl font-bold text-yellow-600">{systemInfo.jobs_queued || 0}</p>
            </div>
            <List className="text-yellow-500" size={32} />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Saved Models</p>
              <p className="text-2xl font-bold text-purple-600">{systemInfo.models_available || 0}</p>
            </div>
            <Database className="text-purple-500" size={32} />
          </div>
        </div>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
        <h2 className="text-xl font-semibold mb-4">GPU Status</h2>
        <div className="space-y-3">
          {systemInfo.gpus?.length > 0 ? (
            systemInfo.gpus.map((gpu, idx) => (
              <div key={idx} className="p-4 bg-gray-50 rounded-lg">
                <div className="flex justify-between items-center">
                  <div>
                    <p className="font-semibold">{gpu.name}</p>
                    <p className="text-sm text-gray-600">GPU {idx}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">Memory</p>
                    <p className="font-semibold">{gpu.memory_used} / {gpu.memory_total}</p>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <p className="text-gray-500">No GPU information available</p>
          )}
        </div>
      </div>
      
      <div className="flex gap-4">
        <button
          onClick={() => onNavigate('create')}
          className="flex-1 bg-blue-600 text-white py-4 px-6 rounded-lg hover:bg-blue-700 transition flex items-center justify-center gap-2 font-semibold"
        >
          <Plus size={20} />
          Create New Training Job
        </button>
        
        <button
          onClick={() => onNavigate('jobs')}
          className="flex-1 bg-gray-600 text-white py-4 px-6 rounded-lg hover:bg-gray-700 transition flex items-center justify-center gap-2 font-semibold"
        >
          <List size={20} />
          View All Jobs
        </button>
      </div>
    </div>
  );
};

// Create Job Component
const CreateJob = ({ onNavigate, frameworks }) => {
  const [jobType, setJobType] = useState('train');
  const [framework, setFramework] = useState('pytorch');
  const [jobName, setJobName] = useState('');
  const [config, setConfig] = useState({
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    architecture: 'custom',
    input_size: 784,
    output_size: 10,
    hidden_layers: [512, 256, 128],
    num_classes: 10
  });
  
  const handleSubmit = async () => {
    const jobData = {
      name: jobName || `Training Job ${Date.now()}`,
      type: jobType,
      framework: framework,
      config: config
    };
    
    try {
      const result = await api.createJob(jobData);
      alert(`Job created successfully! ID: ${result.id}`);
      onNavigate('jobs');
    } catch (error) {
      alert('Failed to create job: ' + error.message);
    }
  };
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">Create Training Job</h1>
        <button
          onClick={() => onNavigate('dashboard')}
          className="text-blue-600 hover:text-blue-800"
        >
          ← Back to Dashboard
        </button>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 space-y-6">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Job Name</label>
          <input
            type="text"
            value={jobName}
            onChange={(e) => setJobName(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="My Training Job"
          />
        </div>
        
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Job Type</label>
          <div className="flex gap-4">
            <button
              onClick={() => setJobType('train')}
              className={`flex-1 py-3 px-4 rounded-lg border-2 transition ${
                jobType === 'train'
                  ? 'border-blue-600 bg-blue-50 text-blue-700'
                  : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
              }`}
            >
              <div className="font-semibold">Train from Scratch</div>
              <div className="text-xs mt-1">Create a new model</div>
            </button>
            <button
              onClick={() => setJobType('finetune')}
              className={`flex-1 py-3 px-4 rounded-lg border-2 transition ${
                jobType === 'finetune'
                  ? 'border-blue-600 bg-blue-50 text-blue-700'
                  : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
              }`}
            >
              <div className="font-semibold">Fine-tune</div>
              <div className="text-xs mt-1">Fine-tune existing model</div>
            </button>
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Framework</label>
          <select
            value={framework}
            onChange={(e) => setFramework(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            {Object.entries(frameworks).map(([key, fw]) => (
              <option key={key} value={key}>{fw.name}</option>
            ))}
          </select>
        </div>
        
        {jobType === 'finetune' && (
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Base Model</label>
            <input
              type="text"
              value={config.model_name || ''}
              onChange={(e) => setConfig({...config, model_name: e.target.value})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              placeholder="resnet18, bert-base-uncased, etc."
            />
          </div>
        )}
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Epochs</label>
            <input
              type="number"
              value={config.epochs}
              onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Batch Size</label>
            <input
              type="number"
              value={config.batch_size}
              onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Learning Rate</label>
            <input
              type="number"
              step="0.0001"
              value={config.learning_rate}
              onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Number of Classes</label>
            <input
              type="number"
              value={config.num_classes}
              onChange={(e) => setConfig({...config, num_classes: parseInt(e.target.value)})}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
        
        <button
          onClick={handleSubmit}
          className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition font-semibold"
        >
          Create Training Job
        </button>
      </div>
    </div>
  );
};

// Jobs List Component
const JobsList = ({ onNavigate }) => {
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  
  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 5000);
    return () => clearInterval(interval);
  }, []);
  
  const loadJobs = async () => {
    try {
      const data = await api.getJobs();
      setJobs(data);
    } catch (error) {
      console.error('Failed to load jobs:', error);
    }
  };
  
  const handleCancelJob = async (jobId) => {
    if (confirm('Are you sure you want to cancel this job?')) {
      try {
        await api.cancelJob(jobId);
        loadJobs();
      } catch (error) {
        alert('Failed to cancel job: ' + error.message);
      }
    }
  };
  
  const getStatusColor = (status) => {
    const colors = {
      'queued': 'bg-yellow-100 text-yellow-800',
      'running': 'bg-blue-100 text-blue-800',
      'completed': 'bg-green-100 text-green-800',
      'failed': 'bg-red-100 text-red-800',
      'cancelled': 'bg-gray-100 text-gray-800'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">Training Jobs</h1>
        <div className="flex gap-2">
          <button
            onClick={() => onNavigate('create')}
            className="bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition flex items-center gap-2"
          >
            <Plus size={16} />
            New Job
          </button>
          <button
            onClick={() => onNavigate('dashboard')}
            className="text-blue-600 hover:text-blue-800"
          >
            ← Dashboard
          </button>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Name</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Type</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Framework</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Status</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Created</th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {jobs.length > 0 ? (
              jobs.map((job) => (
                <tr key={job.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">{job.name}</td>
                  <td className="px-6 py-4 text-sm text-gray-600 capitalize">{job.type}</td>
                  <td className="px-6 py-4 text-sm text-gray-600 capitalize">{job.framework}</td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(job.status)}`}>
                      {job.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600">
                    {new Date(job.created).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-sm">
                    <button
                      onClick={() => setSelectedJob(job)}
                      className="text-blue-600 hover:text-blue-800 mr-3"
                    >
                      View
                    </button>
                    {job.status === 'running' && (
                      <button
                        onClick={() => handleCancelJob(job.id)}
                        className="text-red-600 hover:text-red-800"
                      >
                        Cancel
                      </button>
                    )}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="6" className="px-6 py-8 text-center text-gray-500">
                  No jobs found. Create your first training job!
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      {selectedJob && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-3xl w-full max-h-[80vh] overflow-auto p-6">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-2xl font-bold">{selectedJob.name}</h2>
              <button
                onClick={() => setSelectedJob(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Status</p>
                  <p className="font-semibold capitalize">{selectedJob.status}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Framework</p>
                  <p className="font-semibold capitalize">{selectedJob.framework}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Type</p>
                  <p className="font-semibold capitalize">{selectedJob.type}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Created</p>
                  <p className="font-semibold">{new Date(selectedJob.created).toLocaleString()}</p>
                </div>
              </div>
              
              <div>
                <p className="text-sm text-gray-600 mb-2">Configuration</p>
                <pre className="bg-gray-50 p-4 rounded-lg text-xs overflow-auto">
                  {JSON.stringify(selectedJob.config, null, 2)}
                </pre>
              </div>
              
              {selectedJob.logs && (
                <div>
                  <p className="text-sm text-gray-600 mb-2">Logs</p>
                  <pre className="bg-gray-900 text-green-400 p-4 rounded-lg text-xs overflow-auto max-h-64">
                    {selectedJob.logs}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Main App Component
export default function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [systemInfo, setSystemInfo] = useState({});
  const [frameworks, setFrameworks] = useState({});
  
  useEffect(() => {
    loadSystemInfo();
    loadFrameworks();
    const interval = setInterval(loadSystemInfo, 10000);
    return () => clearInterval(interval);
  }, []);
  
  const loadSystemInfo = async () => {
    try {
      const data = await api.getSystemInfo();
      setSystemInfo(data);
    } catch (error) {
      console.error('Failed to load system info:', error);
    }
  };
  
  const loadFrameworks = async () => {
    try {
      const data = await api.getFrameworks();
      setFrameworks(data);
    } catch (error) {
      console.error('Failed to load frameworks:', error);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Database className="text-blue-600" size={32} />
              <h1 className="text-2xl font-bold text-gray-900">DGX AI Trainer</h1>
            </div>
            <div className="flex gap-4">
              <button
                onClick={() => setCurrentPage('dashboard')}
                className={`px-4 py-2 rounded-lg transition ${
                  currentPage === 'dashboard'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setCurrentPage('jobs')}
                className={`px-4 py-2 rounded-lg transition ${
                  currentPage === 'jobs'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Jobs
              </button>
            </div>
          </div>
        </div>
      </nav>
      
      <main className="max-w-7xl mx-auto px-4 py-8">
        {currentPage === 'dashboard' && (
          <Dashboard onNavigate={setCurrentPage} systemInfo={systemInfo} />
        )}
        {currentPage === 'create' && (
          <CreateJob onNavigate={setCurrentPage} frameworks={frameworks} />
        )}
        {currentPage === 'jobs' && (
          <JobsList onNavigate={setCurrentPage} />
        )}
      </main>
    </div>
  );
}
