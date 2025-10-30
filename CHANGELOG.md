# Changelog

All notable changes to SparkTrainer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### 🚀 Major Features
- **Interactive API Documentation**: Flask-RESTX/Swagger integration for comprehensive API docs
  - Access at `/api/docs` for interactive endpoint exploration
  - Automatic request/response schema validation
  - Live API testing interface
- **Jupyter Notebook Tutorials**: Hands-on tutorials in `/examples/notebooks/`
  - Quick Start tutorial for beginners
  - Custom LoRA recipe development guide
  - Multimodal training examples
  - Advanced optimization techniques
- **Prometheus + Grafana Integration**: Production-grade monitoring stack
  - System metrics (GPU, CPU, memory, network, disk)
  - Training metrics (throughput, loss, accuracy)
  - Celery queue depth and worker health
  - Custom Grafana dashboards for visualization
- **Comprehensive Troubleshooting Guide**: Detailed solutions for common issues
  - GPU memory errors and optimization strategies
  - Whisper transcription failures
  - Dataset loading issues
  - Connection and networking problems
  - Performance optimization tips

#### 🔧 Performance & Optimization
- **Dask Integration**: Parallel data ingestion processing for faster dataset preparation
  - Distributed frame extraction for video datasets
  - Parallel image captioning
  - Concurrent transcription processing
- **Caching Layer**: Redis-based caching for repeated operations
  - Model caption caching to avoid redundant inference
  - Dataset manifest caching
  - System metrics caching
- **Enhanced Metrics**: 50+ Prometheus metrics for comprehensive monitoring
  - Per-job GPU memory tracking
  - Training throughput (samples/sec, tokens/sec)
  - API latency histograms
  - Dataset ingestion durations

#### 🏗️ Architecture & Extensibility
- **Plugin-based Recipe System**: Entry points for community-contributed recipes
  - Recipe discovery via setuptools entry points
  - Hot-reload recipe development workflow
  - Recipe validation and compatibility checking
- **DVC Configuration**: Dataset versioning for reproducible experiments
  - Automatic dataset tracking
  - Version control for large datasets
  - Remote storage support (S3, GCS, Azure)
- **Modular Data Ingestion**: Pluggable ingestion pipeline components
  - Custom quality gates
  - Extensible captioning models
  - Configurable preprocessing steps

#### 📚 Documentation & Guides
- **Hardware Benchmarks**: Comprehensive performance comparisons
  - A100 vs RTX 4090 training benchmarks
  - QLoRA memory requirements
  - Throughput analysis for common models
- **"How to Add a New Benchmark" Guide**: Developer guide for adding evaluation tasks
  - Benchmark interface documentation
  - Integration with MLflow
  - Best practices for reproducible benchmarks
- **Enhanced API Documentation**: 100+ documented endpoints with examples
  - Request/response schemas
  - Authentication flows
  - Rate limiting documentation
  - WebSocket protocol specification

#### 🎨 Frontend & UX
- **TanStack Query Integration**: Modern state management for React frontend
  - Automatic cache invalidation
  - Optimistic updates
  - Background refetching
- **Accessibility Improvements**: ARIA labels and keyboard navigation
  - Screen reader support
  - High contrast mode
  - Keyboard shortcuts for common actions
- **Offline Support**: Progressive Web App features
  - Service worker for offline caching
  - Queue operations for when connection restored
  - Offline-first architecture
- **Enhanced Progress Indicators**: Real-time job progress visualization
  - Training loss curves
  - Resource utilization graphs
  - Time remaining estimates

#### 🤝 Community & Engagement
- **Updated Code of Conduct**: Clear community guidelines and enforcement
- **GitHub Discussions Integration**: Links throughout documentation
- **Contributor Recognition**: Contributors section in README
- **Opt-in Telemetry**: Privacy-preserving usage statistics
  - Anonymous aggregate metrics
  - User-controlled data collection
  - Transparency dashboard

#### 🔐 Security & Compliance
- **Enhanced Authentication**: JWT with refresh tokens
- **API Rate Limiting**: Per-user and per-IP rate limits
- **Audit Logging**: Comprehensive activity tracking
- **Data Privacy**: PII redaction in datasets and logs

### Changed
- **Improved Error Messages**: More descriptive error messages with actionable solutions
- **Optimized Docker Images**: Reduced image sizes by 30%
- **Enhanced Database Migrations**: More robust Alembic migrations
- **Updated Dependencies**: Latest stable versions of core libraries
  - PyTorch 2.2.0
  - Transformers 4.35.0
  - Flask 3.0.0
  - React 18.2.0

### Fixed
- **GPU Memory Leaks**: Fixed memory accumulation in long-running jobs
- **Dataset Upload Reliability**: Improved handling of large file uploads
- **MLflow Tracking**: Fixed intermittent connection issues
- **WebSocket Stability**: Enhanced reconnection logic
- **Job Cancellation**: Proper cleanup of cancelled jobs

### Deprecated
- Legacy job submission API (will be removed in v2.0)
- Old-style recipe format (migration guide provided)

### Security
- Updated cryptography dependencies to address CVE-2024-XXXXX
- Enhanced input validation on all API endpoints
- Implemented content security policy headers

## [0.9.0] - 2024-10-15

### Added
- MLflow experiment tracking integration
- Video ingestion wizard with automated captioning
- LoRA and QLoRA recipe support
- Distributed training with DDP and FSDP
- Mixture-of-Experts gating mechanisms
- Safety and evaluation probes
- Model registry with lifecycle management
- Celery-based async job processing
- Real-time system metrics dashboard
- GPU resource monitoring
- Dataset versioning with lakeFS

### Changed
- Migrated frontend to React 18
- Updated backend to Flask 3.0
- Improved database schema with proper foreign keys
- Enhanced error handling and logging

### Fixed
- Job state machine transitions
- Database connection pooling
- Memory leaks in long-running workers

## [0.8.0] - 2024-08-01

### Added
- Initial public release
- Basic training job management
- Model and dataset browsers
- Web-based UI
- Docker Compose deployment
- PostgreSQL backend
- Redis message queue

## Version Numbering

- **Major version** (X.0.0): Breaking changes, major features
- **Minor version** (0.X.0): New features, backwards compatible
- **Patch version** (0.0.X): Bug fixes, minor improvements

## Upgrade Guides

### Upgrading to v1.0

1. **Database Migration**:
   ```bash
   docker-compose exec backend alembic upgrade head
   ```

2. **Update Docker Compose**:
   - New services added: Prometheus, Grafana
   - Update your `docker-compose.yml` from the template

3. **Recipe Updates**:
   - If you have custom recipes, migrate to the new plugin system
   - See the "How to Add a New Benchmark" guide for details

4. **Configuration Changes**:
   - New environment variables for Prometheus and DVC
   - Update your `.env` file

## Contributors

Special thanks to all contributors who made v1.0 possible:

- Community contributors for bug reports and feature requests
- Beta testers who provided valuable feedback
- Documentation reviewers

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the full list.

## Links

- [GitHub Repository](https://github.com/def1ant1/SparkTrainer)
- [Documentation](https://github.com/def1ant1/SparkTrainer/tree/main/docs)
- [Issue Tracker](https://github.com/def1ant1/SparkTrainer/issues)
- [Discussions](https://github.com/def1ant1/SparkTrainer/discussions)

---

For older versions, see the [full changelog on GitHub](https://github.com/def1ant1/SparkTrainer/releases).
