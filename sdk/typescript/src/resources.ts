/**
 * Resource clients for SparkTrainer API
 */

import { AxiosInstance } from 'axios';
import EventSource from 'eventsource';
import {
  Job,
  JobSubmit,
  Experiment,
  ExperimentCreate,
  Dataset,
  Model,
  GPU,
  Deployment,
  DeploymentCreate,
  HPOStudy,
  HPOStudyCreate,
} from './types';

export class JobsClient {
  constructor(private axios: AxiosInstance) {}

  async list(options?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<Job[]> {
    const response = await this.axios.get<Job[]>('/api/jobs', {
      params: options,
    });
    return response.data;
  }

  async create(job: JobSubmit): Promise<Job> {
    const response = await this.axios.post<Job>('/api/jobs', job);
    return response.data;
  }

  async get(jobId: string): Promise<Job> {
    const response = await this.axios.get<Job>(`/api/jobs/${jobId}`);
    return response.data;
  }

  async cancel(jobId: string): Promise<void> {
    await this.axios.post(`/api/jobs/${jobId}/cancel`);
  }

  async delete(jobId: string): Promise<void> {
    await this.axios.delete(`/api/jobs/${jobId}`);
  }

  async logs(jobId: string, stream: 'stdout' | 'stderr' = 'stdout'): Promise<string> {
    const response = await this.axios.get<string>(`/api/jobs/${jobId}/logs`, {
      params: { stream },
    });
    return response.data;
  }

  streamMetrics(jobId: string, callback: (metrics: any) => void): EventSource {
    const baseUrl = this.axios.defaults.baseURL || 'http://localhost:5001';
    const token = this.axios.defaults.headers.common['Authorization'];

    const eventSource = new EventSource(`${baseUrl}/api/jobs/${jobId}/stream`, {
      headers: token ? { Authorization: token as string } : {},
    });

    eventSource.onmessage = (event) => {
      const metrics = JSON.parse(event.data);
      callback(metrics);
    };

    return eventSource;
  }
}

export class ExperimentsClient {
  constructor(private axios: AxiosInstance) {}

  async list(): Promise<Experiment[]> {
    const response = await this.axios.get<Experiment[]>('/api/experiments');
    return response.data;
  }

  async create(experiment: ExperimentCreate): Promise<Experiment> {
    const response = await this.axios.post<Experiment>('/api/experiments', experiment);
    return response.data;
  }

  async get(experimentId: string): Promise<Experiment> {
    const response = await this.axios.get<Experiment>(`/api/experiments/${experimentId}`);
    return response.data;
  }
}

export class DatasetsClient {
  constructor(private axios: AxiosInstance) {}

  async list(): Promise<Dataset[]> {
    const response = await this.axios.get<Dataset[]>('/api/datasets');
    return response.data;
  }

  async create(name: string, file: File | Blob): Promise<Dataset> {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', file);

    const response = await this.axios.post<Dataset>('/api/datasets', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async get(datasetId: string): Promise<Dataset> {
    const response = await this.axios.get<Dataset>(`/api/datasets/${datasetId}`);
    return response.data;
  }
}

export class ModelsClient {
  constructor(private axios: AxiosInstance) {}

  async list(): Promise<Model[]> {
    const response = await this.axios.get<Model[]>('/api/models');
    return response.data;
  }

  async get(modelId: string): Promise<Model> {
    const response = await this.axios.get<Model>(`/api/models/${modelId}`);
    return response.data;
  }
}

export class GPUsClient {
  constructor(private axios: AxiosInstance) {}

  async list(): Promise<GPU[]> {
    const response = await this.axios.get<GPU[]>('/api/gpus');
    return response.data;
  }
}

export class DeploymentsClient {
  constructor(private axios: AxiosInstance) {}

  async list(): Promise<Deployment[]> {
    const response = await this.axios.get<Deployment[]>('/api/deployments');
    return response.data;
  }

  async create(deployment: DeploymentCreate): Promise<Deployment> {
    const response = await this.axios.post<Deployment>('/api/deployments', deployment);
    return response.data;
  }

  async get(deploymentId: string): Promise<Deployment> {
    const response = await this.axios.get<Deployment>(`/api/deployments/${deploymentId}`);
    return response.data;
  }

  async delete(deploymentId: string): Promise<void> {
    await this.axios.delete(`/api/deployments/${deploymentId}`);
  }
}

export class HPOClient {
  constructor(private axios: AxiosInstance) {}

  async listStudies(): Promise<HPOStudy[]> {
    const response = await this.axios.get<HPOStudy[]>('/api/hpo/studies');
    return response.data;
  }

  async createStudy(study: HPOStudyCreate): Promise<HPOStudy> {
    const response = await this.axios.post<HPOStudy>('/api/hpo/studies', study);
    return response.data;
  }

  async getStudy(studyId: string): Promise<HPOStudy> {
    const response = await this.axios.get<HPOStudy>(`/api/hpo/studies/${studyId}`);
    return response.data;
  }
}
