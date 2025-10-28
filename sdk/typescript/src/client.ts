/**
 * SparkTrainer Client
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import {
  ClientConfig,
  AuthTokens,
} from './types';
import {
  SparkTrainerError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ValidationError,
  ServerError,
} from './errors';
import {
  JobsClient,
  ExperimentsClient,
  DatasetsClient,
  ModelsClient,
  GPUsClient,
  DeploymentsClient,
  HPOClient,
} from './resources';

export class SparkTrainerClient {
  private axiosInstance: AxiosInstance;

  public readonly jobs: JobsClient;
  public readonly experiments: ExperimentsClient;
  public readonly datasets: DatasetsClient;
  public readonly models: ModelsClient;
  public readonly gpus: GPUsClient;
  public readonly deployments: DeploymentsClient;
  public readonly hpo: HPOClient;

  constructor(config: ClientConfig = {}) {
    const baseUrl = config.baseUrl || 'http://localhost:5001';

    this.axiosInstance = axios.create({
      baseURL: baseUrl,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Setup error handling
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          const status = error.response.status;

          if (status === 401) {
            throw new AuthenticationError('Authentication failed');
          } else if (status === 404) {
            throw new NotFoundError('Resource not found');
          } else if (status === 429) {
            throw new RateLimitError('Rate limit exceeded');
          } else if (status === 400) {
            const message = (error.response.data as any)?.message || 'Validation error';
            throw new ValidationError(message);
          } else if (status >= 500) {
            throw new ServerError(`Server error: ${status}`);
          }
        }

        throw new SparkTrainerError(`Request failed: ${error.message}`);
      }
    );

    // Authenticate if credentials provided
    if (config.apiKey) {
      this.setAuthToken(config.apiKey);
    } else if (config.username && config.password) {
      this.login(config.username, config.password);
    }

    // Initialize resource clients
    this.jobs = new JobsClient(this.axiosInstance);
    this.experiments = new ExperimentsClient(this.axiosInstance);
    this.datasets = new DatasetsClient(this.axiosInstance);
    this.models = new ModelsClient(this.axiosInstance);
    this.gpus = new GPUsClient(this.axiosInstance);
    this.deployments = new DeploymentsClient(this.axiosInstance);
    this.hpo = new HPOClient(this.axiosInstance);
  }

  private setAuthToken(token: string): void {
    this.axiosInstance.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  }

  public async login(username: string, password: string): Promise<AuthTokens> {
    try {
      const response = await this.axiosInstance.post<AuthTokens>('/api/auth/login', {
        username,
        password,
      });

      const tokens = response.data;
      this.setAuthToken(tokens.accessToken);
      return tokens;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        throw new AuthenticationError('Invalid credentials');
      }
      throw error;
    }
  }

  public async refresh(refreshToken: string): Promise<AuthTokens> {
    const response = await this.axiosInstance.post<AuthTokens>('/api/auth/refresh', {
      refresh_token: refreshToken,
    });

    const tokens = response.data;
    this.setAuthToken(tokens.accessToken);
    return tokens;
  }

  public async logout(): Promise<void> {
    await this.axiosInstance.post('/api/auth/logout');
    delete this.axiosInstance.defaults.headers.common['Authorization'];
  }
}
