# SparkTrainer TypeScript SDK

Official TypeScript/JavaScript SDK for the SparkTrainer MLOps Platform.

## Installation

```bash
npm install @sparktrainer/sdk
```

## Quick Start

```typescript
import { SparkTrainerClient } from '@sparktrainer/sdk';

// Initialize client
const client = new SparkTrainerClient({
  baseUrl: 'http://localhost:5001',
  username: 'admin',
  password: 'your-password'
});

// Or use API key
const client = new SparkTrainerClient({
  baseUrl: 'http://localhost:5001',
  apiKey: 'your-api-key'
});

// Submit a training job
const job = await client.jobs.create({
  name: 'llama-finetuning',
  command: 'python -m spark_trainer.recipes.text_lora --config config.yaml',
  gpuCount: 4,
  priority: 10
});

console.log(`Job submitted: ${job.id}`);

// Stream real-time metrics
const eventSource = client.jobs.streamMetrics(job.id, (metrics) => {
  console.log(`Step ${metrics.step}: loss=${metrics.loss.toFixed(4)}`);
});

// Check job status
const updatedJob = await client.jobs.get(job.id);
console.log(`Status: ${updatedJob.status}`);

// Get logs
const logs = await client.jobs.logs(job.id, 'stdout');
console.log(logs);

// Close the stream when done
eventSource.close();
```

## Features

- **Job Management**: Submit, monitor, and manage training jobs
- **Real-time Streaming**: Stream metrics via Server-Sent Events
- **Experiment Tracking**: Create and manage experiments
- **Dataset Management**: Upload and version datasets
- **Model Registry**: Access trained models
- **GPU Monitoring**: Check GPU status and utilization
- **HPO**: Run hyperparameter optimization studies
- **Deployments**: Deploy models with vLLM, TGI, or Triton
- **TypeScript Support**: Full type definitions included

## Examples

### Create an Experiment

```typescript
const experiment = await client.experiments.create({
  name: 'llama-7b-finetuning',
  description: 'Fine-tuning Llama-7B on custom dataset',
  tags: { model: 'llama-7b', task: 'instruction-tuning' }
});
```

### Upload a Dataset

```typescript
const file = new File([fileContent], 'dataset.jsonl');
const dataset = await client.datasets.create('my-dataset', file);
```

### List GPUs

```typescript
const gpus = await client.gpus.list();
gpus.forEach(gpu => {
  console.log(`GPU ${gpu.id}: ${gpu.name} - ${gpu.utilization}% utilization`);
});
```

### Create HPO Study

```typescript
const study = await client.hpo.createStudy({
  name: 'llama-lr-search',
  objective: 'minimize',
  searchSpace: {
    learning_rate: {
      type: 'float',
      low: 1e-5,
      high: 1e-3,
      log: true
    },
    batch_size: {
      type: 'categorical',
      choices: [16, 32, 64, 128]
    }
  },
  nTrials: 100,
  parallelism: 4
});
```

### Deploy a Model

```typescript
const deployment = await client.deployments.create({
  name: 'llama-7b-prod',
  modelId: 'model-abc123',
  backend: 'vllm',
  replicas: 2,
  gpuCount: 1
});

console.log(`Deployment endpoint: ${deployment.endpoint}`);
```

## API Reference

### SparkTrainerClient

Main client class for interacting with the SparkTrainer API.

#### Constructor

```typescript
new SparkTrainerClient(config?: ClientConfig)
```

`ClientConfig`:
- `baseUrl?: string` - Base URL of the API (default: `http://localhost:5001`)
- `apiKey?: string` - API key for authentication
- `username?: string` - Username for password authentication
- `password?: string` - Password for password authentication

#### Properties

- `jobs`: Jobs API client
- `experiments`: Experiments API client
- `datasets`: Datasets API client
- `models`: Models API client
- `gpus`: GPUs API client
- `deployments`: Deployments API client
- `hpo`: HPO API client

#### Methods

- `login(username: string, password: string): Promise<AuthTokens>`
- `refresh(refreshToken: string): Promise<AuthTokens>`
- `logout(): Promise<void>`

### Jobs API

```typescript
client.jobs.list(options?: { status?: string; limit?: number; offset?: number }): Promise<Job[]>
client.jobs.create(job: JobSubmit): Promise<Job>
client.jobs.get(jobId: string): Promise<Job>
client.jobs.cancel(jobId: string): Promise<void>
client.jobs.delete(jobId: string): Promise<void>
client.jobs.logs(jobId: string, stream?: 'stdout' | 'stderr'): Promise<string>
client.jobs.streamMetrics(jobId: string, callback: (metrics: any) => void): EventSource
```

### Experiments API

```typescript
client.experiments.list(): Promise<Experiment[]>
client.experiments.create(experiment: ExperimentCreate): Promise<Experiment>
client.experiments.get(experimentId: string): Promise<Experiment>
```

### Datasets API

```typescript
client.datasets.list(): Promise<Dataset[]>
client.datasets.create(name: string, file: File | Blob): Promise<Dataset>
client.datasets.get(datasetId: string): Promise<Dataset>
```

### Models API

```typescript
client.models.list(): Promise<Model[]>
client.models.get(modelId: string): Promise<Model>
```

### GPUs API

```typescript
client.gpus.list(): Promise<GPU[]>
```

### Deployments API

```typescript
client.deployments.list(): Promise<Deployment[]>
client.deployments.create(deployment: DeploymentCreate): Promise<Deployment>
client.deployments.get(deploymentId: string): Promise<Deployment>
client.deployments.delete(deploymentId: string): Promise<void>
```

### HPO API

```typescript
client.hpo.listStudies(): Promise<HPOStudy[]>
client.hpo.createStudy(study: HPOStudyCreate): Promise<HPOStudy>
client.hpo.getStudy(studyId: string): Promise<HPOStudy>
```

## Error Handling

```typescript
import {
  SparkTrainerClient,
  AuthenticationError,
  NotFoundError,
  RateLimitError
} from '@sparktrainer/sdk';

try {
  const job = await client.jobs.get('invalid-id');
} catch (error) {
  if (error instanceof NotFoundError) {
    console.error('Job not found');
  } else if (error instanceof AuthenticationError) {
    console.error('Authentication failed');
  } else if (error instanceof RateLimitError) {
    console.error('Rate limit exceeded');
  }
}
```

## React Example

```tsx
import { useEffect, useState } from 'react';
import { SparkTrainerClient, Job } from '@sparktrainer/sdk';

const client = new SparkTrainerClient({ apiKey: 'your-api-key' });

function JobMonitor({ jobId }: { jobId: string }) {
  const [metrics, setMetrics] = useState<any>(null);

  useEffect(() => {
    const eventSource = client.jobs.streamMetrics(jobId, (data) => {
      setMetrics(data);
    });

    return () => eventSource.close();
  }, [jobId]);

  return (
    <div>
      {metrics && (
        <div>
          <p>Step: {metrics.step}</p>
          <p>Loss: {metrics.loss.toFixed(4)}</p>
        </div>
      )}
    </div>
  );
}
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Lint
npm run lint

# Format
npm run format
```

## License

MIT License - see LICENSE file for details
