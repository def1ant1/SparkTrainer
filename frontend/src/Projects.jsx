import React, { useState, useEffect } from 'react';
import { Folder, Plus, Trash2, Edit2, TrendingUp, Database, Flask } from 'lucide-react';
import { Button } from './components/Button';
import { Card } from './components/Card';
import { Input } from './components/Input';
import { Modal } from './components/Modal';

const Projects = () => {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProject, setNewProject] = useState({ name: '', description: '' });

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      const response = await fetch('/api/projects');
      const data = await response.json();
      setProjects(data);
    } catch (error) {
      console.error('Error fetching projects:', error);
    } finally {
      setLoading(false);
    }
  };

  const createProject = async () => {
    try {
      const response = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newProject),
      });
      const data = await response.json();
      setProjects([...projects, data]);
      setShowCreateModal(false);
      setNewProject({ name: '', description: '' });
    } catch (error) {
      console.error('Error creating project:', error);
    }
  };

  const deleteProject = async (id) => {
    if (!confirm('Are you sure you want to delete this project?')) return;

    try {
      await fetch(`/api/projects/${id}`, { method: 'DELETE' });
      setProjects(projects.filter(p => p.id !== id));
    } catch (error) {
      console.error('Error deleting project:', error);
    }
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
          <h1 className="text-3xl font-bold">Projects</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Organize your datasets, experiments, and models
          </p>
        </div>
        <Button onClick={() => setShowCreateModal(true)}>
          <Plus size={20} />
          New Project
        </Button>
      </div>

      {/* Project Grid */}
      {projects.length === 0 ? (
        <Card>
          <div className="p-12 text-center">
            <Folder className="mx-auto mb-4 text-gray-400" size={64} />
            <h3 className="text-xl font-semibold mb-2">No projects yet</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Create your first project to get started
            </p>
            <Button onClick={() => setShowCreateModal(true)}>
              <Plus size={20} />
              Create Project
            </Button>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((project) => (
            <Card key={project.id} className="hover:shadow-lg transition-shadow cursor-pointer">
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                      <Folder className="text-white" size={24} />
                    </div>
                    <div>
                      <h3 className="font-semibold text-lg">{project.name}</h3>
                      <p className="text-sm text-gray-500">
                        {new Date(project.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => window.location.href = `/projects/${project.id}`}
                      className="text-gray-400 hover:text-blue-500"
                    >
                      <Edit2 size={16} />
                    </button>
                    <button
                      onClick={() => deleteProject(project.id)}
                      className="text-gray-400 hover:text-red-500"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>

                <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-2">
                  {project.description || 'No description'}
                </p>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-4 pt-4 border-t dark:border-gray-700">
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-1">
                      <Flask size={16} className="text-blue-500" />
                    </div>
                    <div className="text-2xl font-bold">{project.experiments_count || 0}</div>
                    <div className="text-xs text-gray-500">Experiments</div>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-1">
                      <Database size={16} className="text-green-500" />
                    </div>
                    <div className="text-2xl font-bold">{project.datasets_count || 0}</div>
                    <div className="text-xs text-gray-500">Datasets</div>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-1">
                      <TrendingUp size={16} className="text-purple-500" />
                    </div>
                    <div className="text-2xl font-bold">{project.models_count || 0}</div>
                    <div className="text-xs text-gray-500">Models</div>
                  </div>
                </div>

                <Button
                  variant="primary"
                  className="w-full mt-4"
                  onClick={() => window.location.href = `/projects/${project.id}`}
                >
                  Open Project
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Create Project Modal */}
      <Modal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create New Project"
      >
        <div className="space-y-4">
          <Input
            label="Project Name"
            value={newProject.name}
            onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
            placeholder="My ML Project"
          />
          <div>
            <label className="block text-sm font-medium mb-2">Description</label>
            <textarea
              value={newProject.description}
              onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
              placeholder="Project description..."
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 min-h-[100px]"
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => setShowCreateModal(false)}>
              Cancel
            </Button>
            <Button onClick={createProject} disabled={!newProject.name}>
              Create Project
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default Projects;
