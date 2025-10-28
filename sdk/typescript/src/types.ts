/**
 * Type definitions for SparkTrainer SDK
 */

export interface Job {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  command: string;
  gpuCount: number;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  metrics?: Record<string, any>;
}

export interface JobSubmit {
  name: string;
  command: string;
  gpuCount?: number;
  priority?: number;
  environment?: Record<string, string>;
}

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  createdAt: Date;
  runs: any[];
}

export interface ExperimentCreate {
  name: string;
  description?: string;
  tags?: Record<string, string>;
}

export interface Dataset {
  id: string;
  name: string;
  size: number;
  format: string;
  createdAt: Date;
  version?: string;
}

export interface Model {
  id: string;
  name: string;
  framework: string;
  size: number;
  createdAt: Date;
  tags: string[];
}

export interface GPU {
  id: number;
  name: string;
  memoryTotal: number;
  memoryUsed: number;
  utilization: number;
  temperature: number;
  powerUsage: number;
}

export interface Deployment {
  id: string;
  name: string;
  modelId: string;
  backend: 'vllm' | 'tgi' | 'triton';
  status: string;
  endpoint: string;
}

export interface DeploymentCreate {
  name: string;
  modelId: string;
  backend: 'vllm' | 'tgi' | 'triton';
  replicas?: number;
  gpuCount?: number;
}

export interface HPOStudy {
  id: string;
  name: string;
  status: string;
  nTrials: number;
  bestValue?: number;
  bestParams?: Record<string, any>;
}

export interface HPOStudyCreate {
  name: string;
  objective: string;
  searchSpace: Record<string, any>;
  nTrials?: number;
  parallelism?: number;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  tokenType: string;
  expiresIn: number;
}

export interface ClientConfig {
  baseUrl?: string;
  apiKey?: string;
  username?: string;
  password?: string;
}
