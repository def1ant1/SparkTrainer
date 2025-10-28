/**
 * Error classes for SparkTrainer SDK
 */

export class SparkTrainerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SparkTrainerError';
  }
}

export class AuthenticationError extends SparkTrainerError {
  constructor(message: string = 'Authentication failed') {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class NotFoundError extends SparkTrainerError {
  constructor(message: string = 'Resource not found') {
    super(message);
    this.name = 'NotFoundError';
  }
}

export class RateLimitError extends SparkTrainerError {
  constructor(message: string = 'Rate limit exceeded') {
    super(message);
    this.name = 'RateLimitError';
  }
}

export class ValidationError extends SparkTrainerError {
  constructor(message: string = 'Validation error') {
    super(message);
    this.name = 'ValidationError';
  }
}

export class ServerError extends SparkTrainerError {
  constructor(message: string = 'Server error') {
    super(message);
    this.name = 'ServerError';
  }
}
