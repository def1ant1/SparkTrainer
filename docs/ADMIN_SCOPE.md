# Admin Dashboard - Scope Definition

## Overview
This document defines the complete scope of capabilities for the **Admin** role in SparkTrainer. The Admin Dashboard provides centralized control over users, infrastructure, security, and system health.

---

## 1. RBAC & Multi-Tenancy Management

### 1.1 User Management
**Capabilities:**
- ✅ **View all users** - List with filters (role, status, team, registration date)
- ✅ **Create/invite users** - Bulk invite via email or CSV upload
- ✅ **Edit user profiles** - Update role, status, contact info
- ✅ **Suspend/delete users** - Soft delete with data retention policy
- ✅ **Reset passwords** - Admin-initiated password reset
- ✅ **View user activity** - Last login, active sessions, resource usage
- ✅ **Impersonate users** (audit-logged) - Troubleshoot user issues

**API Endpoints:**
```
GET    /api/admin/users                 # List all users
POST   /api/admin/users                 # Create user
GET    /api/admin/users/{id}            # User details
PUT    /api/admin/users/{id}            # Update user
DELETE /api/admin/users/{id}            # Delete user
POST   /api/admin/users/{id}/suspend    # Suspend user
POST   /api/admin/users/{id}/reset-pwd  # Force password reset
POST   /api/admin/users/bulk-invite     # Bulk user invite
GET    /api/admin/users/{id}/activity   # User activity log
POST   /api/admin/users/{id}/impersonate # Impersonate user
```

### 1.2 Team Management
**Capabilities:**
- ✅ **View all teams** - List with member count, quota usage, billing
- ✅ **Create/edit teams** - Set team name, owner, description
- ✅ **Manage team members** - Add/remove, change roles (owner/admin/member)
- ✅ **Transfer team ownership** - Reassign owner with confirmation
- ✅ **Dissolve teams** - Archive team with data migration options
- ✅ **View team analytics** - Job success rate, GPU hours, cost trends

**API Endpoints:**
```
GET    /api/admin/teams                # List all teams
POST   /api/admin/teams                # Create team
GET    /api/admin/teams/{id}           # Team details
PUT    /api/admin/teams/{id}           # Update team
DELETE /api/admin/teams/{id}           # Delete team
PUT    /api/admin/teams/{id}/owner     # Transfer ownership
GET    /api/admin/teams/{id}/analytics # Team analytics
```

### 1.3 Role & Permission Management
**Current Roles:**
- `ADMIN` - Full system access
- `MAINTAINER` - Create/update/read operations
- `VIEWER` - Read-only access

**Enhanced Capabilities:**
- ✅ **View role matrix** - Visual permission grid (role × resource × action)
- ✅ **Assign roles** - Grant/revoke roles to users/teams
- ✅ **Custom permissions** - Override specific permissions per user
- ✅ **Project-level permissions** - Granular access control per project/experiment
- ✅ **API token management** - View, create, revoke API tokens for users

**Project-Level Permissions (New):**
```python
# Example: User can be VIEWER globally but MAINTAINER on specific projects
project_permissions = {
    "project_id": "proj-123",
    "user_id": "user-456",
    "role_override": "MAINTAINER",
    "expires_at": "2025-12-31T23:59:59Z"
}
```

**API Endpoints:**
```
GET    /api/admin/roles                     # List all roles
GET    /api/admin/roles/{role}/permissions  # Permission matrix
PUT    /api/admin/users/{id}/role           # Assign role
POST   /api/admin/permissions/override      # Grant custom permission
GET    /api/admin/projects/{id}/permissions # Project-level ACL
PUT    /api/admin/projects/{id}/permissions # Update project ACL
GET    /api/admin/tokens                    # All API tokens
DELETE /api/admin/tokens/{token_id}         # Revoke token
```

---

## 2. System Health & Infrastructure Monitoring

### 2.1 Service Health Dashboard
**Services to Monitor:**
- **Flask API** - Request rate, latency (p50/p95/p99), error rate
- **PostgreSQL** - Connection pool, query latency, slow queries, locks
- **Redis** - Memory usage, hit/miss ratio, connected clients, pub/sub channels
- **Celery Workers** - Active/idle/offline workers, task throughput
- **MLflow Tracking Server** - Experiment count, artifact size, API latency
- **Nginx** (if used) - Request rate, 5xx errors, upstream health

**Metrics to Display:**
```yaml
api_health:
  status: healthy | degraded | down
  requests_per_second: 120
  error_rate: 0.5%
  avg_latency_ms: 45
  endpoints_down: ["/api/models"]

postgres:
  status: healthy
  active_connections: 12 / 100
  slow_queries: 3
  replication_lag_ms: 50

redis:
  status: healthy
  memory_used_mb: 512 / 4096
  hit_rate: 98.5%
  connected_clients: 24

celery:
  status: degraded
  active_workers: 3 / 4  # 1 worker offline
  task_success_rate: 99.2%
  avg_task_duration_s: 120
```

**API Endpoints:**
```
GET /api/admin/health/overview       # All services status
GET /api/admin/health/api            # Flask API metrics
GET /api/admin/health/postgres       # Database health
GET /api/admin/health/redis          # Redis health
GET /api/admin/health/celery         # Celery cluster health
GET /api/admin/health/mlflow         # MLflow health
GET /api/admin/health/history        # Historical health trends (7d)
```

### 2.2 GPU Inventory & Capacity
**Capabilities:**
- ✅ **View all GPUs** - Device ID, model, memory, utilization, temperature
- ✅ **GPU allocation map** - Which jobs are on which GPUs
- ✅ **MIG instance status** - MIG partitions, enabled profiles
- ✅ **GPU health checks** - ECC errors, thermal throttling, power limits
- ✅ **Capacity planning** - Forecast GPU hours available vs. demand
- ✅ **Maintenance mode** - Mark GPUs offline for maintenance

**Metrics:**
```yaml
gpu_inventory:
  total_gpus: 8
  available: 2
  busy: 5
  maintenance: 1
  error: 0

gpus:
  - device_id: 0
    model: NVIDIA A100 80GB
    memory_used_gb: 72.4 / 80
    utilization: 95%
    temperature_c: 78
    power_draw_w: 350 / 400
    status: busy
    current_job: job-789
    mig_enabled: true
    mig_instances:
      - profile: 3g.20gb
        instance_id: 0
        job: job-790
```

**API Endpoints:**
```
GET  /api/admin/gpu/inventory         # All GPUs
GET  /api/admin/gpu/{device_id}       # GPU details
GET  /api/admin/gpu/allocation-map    # GPU → Job mapping
PUT  /api/admin/gpu/{device_id}/maintenance  # Toggle maintenance mode
GET  /api/admin/gpu/capacity-forecast # Predicted capacity vs. demand
```

### 2.3 Worker Node Management
**Capabilities:**
- ✅ **View worker nodes** - Hostname, IP, Celery queues subscribed
- ✅ **Worker heartbeats** - Last seen, health status
- ✅ **Worker capacity** - Max concurrent tasks, prefetch multiplier
- ✅ **Worker tasks** - Current task, task history
- ✅ **Drain workers** - Graceful shutdown (finish current tasks)
- ✅ **Scale workers** - Add/remove worker nodes (if using orchestration)

**API Endpoints:**
```
GET    /api/admin/workers              # List all workers
GET    /api/admin/workers/{worker_id}  # Worker details
POST   /api/admin/workers/{worker_id}/drain     # Drain worker
POST   /api/admin/workers/{worker_id}/terminate # Kill worker
GET    /api/admin/workers/{worker_id}/tasks     # Worker task history
```

---

## 3. Queue & Task Management

### 3.1 Celery Queue Monitoring
**Queues:**
- `training` (priority=10) - GPU-intensive training
- `preprocessing` (priority=5) - Data preparation
- `evaluation` (priority=3) - Model evaluation
- `default` - General tasks

**Metrics per Queue:**
```yaml
queue_stats:
  training:
    depth: 12              # Tasks waiting
    active: 5              # Tasks running
    scheduled: 3           # Tasks scheduled (ETA)
    retry_count: 2         # Retrying tasks
    failed_1h: 1           # Failed in last hour
    avg_wait_time_s: 45
    avg_execution_time_s: 240
```

**Capabilities:**
- ✅ **View queue depth** - Real-time task counts per queue
- ✅ **Task distribution** - Queue priority visualization
- ✅ **Queue pause/resume** - Temporarily stop new tasks
- ✅ **Queue purge** - Clear pending tasks (with confirmation)
- ✅ **Task inspection** - View task args, kwargs, state
- ✅ **Task cancellation** - Cancel pending/running tasks

**API Endpoints:**
```
GET    /api/admin/queues               # All queue stats
GET    /api/admin/queues/{name}        # Queue details
POST   /api/admin/queues/{name}/pause  # Pause queue
POST   /api/admin/queues/{name}/resume # Resume queue
POST   /api/admin/queues/{name}/purge  # Clear queue
GET    /api/admin/tasks/{task_id}      # Task details
POST   /api/admin/tasks/{task_id}/cancel # Cancel task
```

### 3.2 Retry & Dead Letter Queue (DLQ)
**Retry Policy (Current):**
- Max retries: 3
- Backoff: Exponential (60s, 120s, 240s)
- Retry on: `SoftTimeLimitExceeded`, transient errors

**DLQ Management:**
- ✅ **View DLQ** - Tasks that exhausted retries
- ✅ **Inspect failures** - Exception traceback, task args
- ✅ **Re-queue tasks** - Manual retry after fixing issue
- ✅ **Archive DLQ** - Move old failures to long-term storage
- ✅ **Failure analytics** - Common failure reasons, affected users

**API Endpoints:**
```
GET    /api/admin/dlq                  # Dead letter queue
GET    /api/admin/dlq/{task_id}        # DLQ task details
POST   /api/admin/dlq/{task_id}/retry  # Re-queue task
DELETE /api/admin/dlq/{task_id}        # Remove from DLQ
POST   /api/admin/dlq/archive          # Archive old failures
GET    /api/admin/dlq/analytics        # Failure statistics
```

---

## 4. Quotas & Resource Limits

### 4.1 Per-Tenant Quotas
**Current Quota Fields (Team Model):**
```python
quota = {
    "max_gpus": 4,                  # Max concurrent GPUs
    "max_storage_gb": 1000,         # Total storage
    "max_jobs_per_month": 100       # Job submission limit
}
```

**Enhanced Quotas:**
```python
enhanced_quota = {
    # GPU Limits
    "max_gpus_concurrent": 4,       # Max GPUs at once
    "max_gpu_hours_month": 500,     # Total GPU hours/month
    "max_gpu_memory_gb": 320,       # Total GPU memory (4x A100 80GB)

    # Storage Limits
    "max_storage_gb": 1000,
    "max_checkpoints": 50,
    "max_artifact_size_gb": 100,

    # Compute Limits
    "max_jobs_concurrent": 10,
    "max_jobs_per_month": 100,
    "max_job_duration_hours": 48,

    # API Limits
    "api_requests_per_minute": 1000,
    "api_requests_per_day": 100000,

    # MLflow Limits
    "max_experiments": 100,
    "max_runs_per_experiment": 1000
}
```

**Capabilities:**
- ✅ **View quota usage** - Current vs. limit (visual gauges)
- ✅ **Set/update quotas** - Per team or user
- ✅ **Quota templates** - Pre-defined tiers (Free, Pro, Enterprise)
- ✅ **Quota warnings** - Alert at 80%, 90%, 100% usage
- ✅ **Quota enforcement** - Reject submissions exceeding limits
- ✅ **Quota resets** - Monthly reset for time-based quotas
- ✅ **Quota exceptions** - Temporary quota increases

**API Endpoints:**
```
GET    /api/admin/quotas/teams/{team_id}        # Team quotas
PUT    /api/admin/quotas/teams/{team_id}        # Update team quota
GET    /api/admin/quotas/users/{user_id}        # User quotas
PUT    /api/admin/quotas/users/{user_id}        # Update user quota
GET    /api/admin/quotas/templates              # Quota templates
POST   /api/admin/quotas/teams/{team_id}/exception  # Temporary increase
GET    /api/admin/quotas/usage-report           # System-wide quota usage
```

### 4.2 Concurrency & Rate Limits
**Concurrency Limits:**
- Max concurrent jobs per user/team
- Max concurrent API requests
- Max concurrent MLflow runs

**Rate Limits:**
- API requests per minute/hour/day
- Job submissions per hour
- Model deployments per day

**API Endpoints:**
```
GET /api/admin/rate-limits              # Current rate limits
PUT /api/admin/rate-limits/{resource}   # Update rate limit
GET /api/admin/rate-limits/violations   # Recent violations
```

---

## 5. Feature Flags & Configuration

### 5.1 Feature Flag System (NEW)
**Purpose:** Enable/disable features dynamically without redeployment.

**Feature Flags:**
```python
feature_flags = {
    # Training Features
    "hpo_enabled": True,                    # Hyperparameter optimization
    "long_context_enabled": False,          # Long-context training (>8K tokens)
    "moe_enabled": True,                    # Mixture of Experts
    "moe_lora_enabled": True,               # MoE-LoRA
    "routerless_moe_enabled": False,        # DeepSeek-style MoE
    "mixture_of_depths_enabled": False,     # Dynamic layer selection

    # Infrastructure
    "mig_admin_enabled": True,              # MIG GPU partitioning
    "auto_resume_enabled": True,            # Auto-resume on failure
    "gpu_preemption_enabled": False,        # Job preemption

    # API Features
    "api_v2_enabled": False,                # New API version
    "graphql_enabled": False,               # GraphQL API
    "websocket_enabled": True,              # Real-time updates

    # Multi-Tenancy
    "team_resource_sharing": False,         # Share GPUs across teams
    "cross_team_experiments": False,        # Collab experiments

    # Billing
    "usage_based_billing": True,
    "prepaid_credits": False,

    # Security
    "mfa_required": False,                  # Multi-factor auth
    "ip_whitelist_enabled": False,
    "audit_logging_verbose": True
}
```

**Flag Scopes:**
- **Global** - Affects all users
- **Team** - Enabled for specific teams
- **User** - Enabled for specific users (beta testing)

**Capabilities:**
- ✅ **View all flags** - Current state, last modified, scope
- ✅ **Toggle flags** - Enable/disable (audit-logged)
- ✅ **Scheduled rollouts** - Enable flag at specific time
- ✅ **Gradual rollouts** - Enable for X% of users
- ✅ **A/B testing** - Compare metrics between enabled/disabled

**API Endpoints:**
```
GET    /api/admin/feature-flags                    # All flags
PUT    /api/admin/feature-flags/{flag}             # Toggle flag
POST   /api/admin/feature-flags/{flag}/schedule    # Schedule rollout
POST   /api/admin/feature-flags/{flag}/gradual     # Gradual rollout
GET    /api/admin/feature-flags/{flag}/users       # Users with flag enabled
```

### 5.2 System Configuration Management
**Configuration Categories:**
- **Training Defaults** - Default epochs, batch size, learning rate
- **Resource Limits** - Max job duration, GPU memory limits
- **Storage Settings** - Checkpoint retention, artifact expiry
- **Billing Rates** - GPU pricing per hour
- **Integration Keys** - HuggingFace, OpenAI, WandB (write-only display)

**Capabilities:**
- ✅ **View config** - Current system configuration
- ✅ **Update config** - Change settings (audit-logged)
- ✅ **Config validation** - Validate before applying
- ✅ **Config history** - Track changes over time
- ✅ **Rollback config** - Revert to previous version

**API Endpoints:**
```
GET    /api/admin/config                  # Current config
PUT    /api/admin/config/{key}            # Update config
GET    /api/admin/config/history          # Config change history
POST   /api/admin/config/rollback         # Rollback config
```

---

## 6. Audit Log & Security

### 6.1 Comprehensive Audit Log (ENHANCED)
**Current:** Only job status transitions tracked (`JobStatusTransition` table)

**Expanded Audit Events:**
```python
audit_events = [
    # Authentication
    "user.login", "user.logout", "user.login_failed",
    "user.password_reset", "user.mfa_enabled",

    # Authorization
    "role.assigned", "role.revoked",
    "permission.granted", "permission.revoked",
    "token.created", "token.revoked",

    # User Management
    "user.created", "user.updated", "user.deleted", "user.suspended",
    "user.impersonated",

    # Team Management
    "team.created", "team.updated", "team.deleted",
    "team.member_added", "team.member_removed", "team.owner_transferred",

    # Resource Operations
    "job.created", "job.cancelled", "job.status_changed",
    "experiment.created", "experiment.deleted",
    "model.tagged", "model.promoted", "model.deleted",
    "dataset.created", "dataset.updated", "dataset.deleted",

    # Quotas & Billing
    "quota.updated", "quota.exceeded", "quota.exception_granted",
    "billing.invoice_generated", "billing.payment_received",

    # Configuration
    "config.updated", "feature_flag.toggled",

    # Secrets
    "secret.viewed", "secret.created", "secret.rotated", "secret.deleted",

    # Infrastructure
    "gpu.maintenance_mode", "worker.terminated", "queue.purged"
]
```

**Audit Log Schema:**
```python
class AuditLog:
    id: uuid
    timestamp: datetime
    event_type: str                  # e.g., "role.assigned"
    actor_id: uuid                   # User who performed action
    actor_ip: str
    target_type: str                 # e.g., "user", "team", "job"
    target_id: uuid
    details: dict                    # Event-specific metadata
    severity: str                    # info, warning, critical
    immutable: bool = True           # Cannot be modified
```

**Capabilities:**
- ✅ **View audit log** - Searchable, filterable (date, user, event type)
- ✅ **Export audit log** - CSV/JSON export for compliance
- ✅ **Retention policy** - Configurable retention (e.g., 7 years)
- ✅ **Tamper detection** - Cryptographic signatures to detect tampering
- ✅ **Real-time alerts** - Slack/email on critical events
- ✅ **Compliance reports** - SOC 2, GDPR, HIPAA audit trails

**API Endpoints:**
```
GET    /api/admin/audit/logs              # Search audit logs
GET    /api/admin/audit/export            # Export logs (CSV/JSON)
GET    /api/admin/audit/users/{user_id}   # User-specific audit trail
GET    /api/admin/audit/events/{event_id} # Event details
POST   /api/admin/audit/alerts            # Configure alerts
```

### 6.2 Secrets & Credential Management
**Current Issues:**
- Credentials stored in plaintext JSON (`/jobs/users.json`)
- API keys stored in user profiles without encryption

**Enhanced Secrets Management:**
```python
secrets = {
    # System Secrets
    "jwt_secret_key": "*****",           # Redacted
    "database_url": "postgres://***",    # Redacted
    "redis_password": "*****",

    # Integration Secrets
    "huggingface_token": "hf_***",       # Show prefix only
    "openai_api_key": "sk-***",
    "wandb_api_key": "*****",
    "mlflow_tracking_token": "*****",

    # User Secrets (per-user)
    "user_api_tokens": "st_***",         # Show prefix
    "ssh_private_keys": "*****",
    "cloud_credentials": "*****"
}
```

**Capabilities:**
- ✅ **Encryption at rest** - AES-256 encryption for all secrets
- ✅ **Redaction in UI** - Show only prefixes/suffixes
- ✅ **Write-only fields** - API keys cannot be read back
- ✅ **Key rotation** - Rotate JWT secret, DB password
- ✅ **Rotation policies** - Auto-rotate every 90 days
- ✅ **Secrets expiry** - Time-limited secrets
- ✅ **Secrets audit** - Log all secret access
- ✅ **Integration with Vault** - Optional HashiCorp Vault backend

**Encryption Implementation:**
```python
# Use Fernet (symmetric encryption) with key derived from env var
from cryptography.fernet import Fernet

SECRET_KEY = os.environ.get("SECRET_ENCRYPTION_KEY")
cipher = Fernet(SECRET_KEY)

def encrypt_secret(plaintext: str) -> str:
    return cipher.encrypt(plaintext.encode()).decode()

def decrypt_secret(ciphertext: str) -> str:
    return cipher.decrypt(ciphertext.encode()).decode()
```

**API Endpoints:**
```
GET    /api/admin/secrets                      # List secrets (redacted)
POST   /api/admin/secrets                      # Create secret
PUT    /api/admin/secrets/{key}/rotate         # Rotate secret
DELETE /api/admin/secrets/{key}                # Delete secret
GET    /api/admin/secrets/rotation-policy      # View policies
PUT    /api/admin/secrets/rotation-policy      # Update policies
GET    /api/admin/secrets/audit                # Secret access audit
```

**Non-Sensitive Config Display:**
```python
# Show full value for non-sensitive config
safe_config = {
    "max_job_duration_hours": 48,
    "default_gpu_type": "A100",
    "checkpoint_retention_days": 30,
    "mlflow_url": "http://localhost:5001",  # OK to show
    "billing_currency": "USD",
    "gpu_pricing": {
        "A100": 3.0,
        "V100": 2.0,
        "T4": 0.5
    }
}
```

---

## 7. Admin Dashboard UI (Frontend)

### 7.1 Dashboard Layout
```
┌─────────────────────────────────────────────────────────┐
│  SparkTrainer Admin                [User] [Logout]      │
├─────────────────────────────────────────────────────────┤
│ [Overview] [Users] [Teams] [System] [Queues] [Audit]   │
├───────────┬─────────────────────────────────────────────┤
│           │                                             │
│  Sidebar  │           Main Content Area                │
│           │                                             │
│ • Overview│  - System health cards                     │
│ • Users   │  - GPU inventory visualization             │
│ • Teams   │  - Active jobs table                       │
│ • Roles   │  - Quota usage charts                      │
│ • System  │  - Recent audit events                     │
│   - Health│                                             │
│   - GPUs  │                                             │
│   - Workers                                             │
│ • Queues  │                                             │
│ • Quotas  │                                             │
│ • Billing │                                             │
│ • Audit   │                                             │
│ • Config  │                                             │
│ • Secrets │                                             │
└───────────┴─────────────────────────────────────────────┘
```

### 7.2 Key UI Components

**Overview Page:**
- System health status cards (API, DB, Redis, Celery, MLflow)
- GPU utilization heatmap
- Active jobs counter
- Queue depth charts
- Recent audit events feed

**Users Page:**
- Searchable user table (name, email, role, status, last login)
- Bulk actions (suspend, delete, export)
- User detail modal (profile, teams, quotas, activity)

**Teams Page:**
- Team cards with member count, quota usage
- Team detail view (members, quotas, billing, analytics)

**System Health Page:**
- Service status dashboard
- GPU inventory table
- Worker node list
- Capacity forecast chart

**Queues Page:**
- Queue depth gauges (training, preprocessing, evaluation)
- Active tasks table
- DLQ browser
- Queue controls (pause/resume/purge)

**Audit Log Page:**
- Searchable audit log table
- Filters (date range, event type, user, severity)
- Event detail modal with full JSON payload
- Export button

---

## 8. Implementation Priority (Phased Rollout)

### Phase 1: Security Hardening (HIGH PRIORITY)
1. ✅ Secrets encryption at rest
2. ✅ Audit log expansion
3. ✅ Password hashing upgrade (bcrypt/argon2)
4. ✅ API rate limiting
5. ✅ Input validation middleware
6. ✅ HTTPS enforcement

### Phase 2: Enhanced Monitoring (HIGH)
1. ✅ Comprehensive system health dashboard
2. ✅ Celery queue monitoring
3. ✅ Worker heartbeat tracking
4. ✅ DLQ management
5. ✅ GPU capacity forecasting

### Phase 3: Quota Enforcement (MEDIUM)
1. ✅ Enhanced quota schema
2. ✅ Quota enforcement at job submission
3. ✅ Quota warning system
4. ✅ Quota templates & exceptions

### Phase 4: Feature Flags (MEDIUM)
1. ✅ Feature flag system implementation
2. ✅ Flag management UI
3. ✅ Gradual rollouts
4. ✅ A/B testing framework

### Phase 5: Advanced Admin UI (LOWER)
1. ✅ Full admin dashboard UI
2. ✅ User/team management UI
3. ✅ System configuration UI
4. ✅ Audit log viewer

---

## 9. Security Considerations

### 9.1 Authentication & Authorization
- All admin endpoints require `Role.ADMIN`
- Admin actions logged to audit trail
- Impersonation requires separate permission
- MFA recommended for admin accounts

### 9.2 Data Protection
- Secrets encrypted with AES-256
- Audit logs immutable (append-only)
- PII handling compliant with GDPR/CCPA
- Data retention policies configurable

### 9.3 Network Security
- HTTPS required in production
- CORS restricted to specific origins
- Rate limiting on all endpoints
- DDoS protection (CloudFlare/AWS Shield)

---

## 10. Monitoring & Alerting

### 10.1 Admin Alerts
**Critical Alerts:**
- Service down (API, DB, Redis, Celery)
- GPU failure or thermal throttling
- Disk space >90% full
- DLQ depth >100 tasks
- Security event (failed logins >10/min)

**Warning Alerts:**
- Quota usage >90%
- Worker offline >5 minutes
- Queue depth >50 tasks
- Slow query >1 second

**Delivery Channels:**
- Email
- Slack webhook
- PagerDuty (for critical)

### 10.2 Metrics to Track
- API request rate, latency, error rate
- Database connection pool usage
- Celery task success rate, duration
- GPU utilization, memory
- Queue depth over time
- Quota usage trends
- Cost per team/user

---

## 11. API Summary

### Admin Endpoints Overview
```
# Users (11 endpoints)
GET    /api/admin/users
POST   /api/admin/users
GET    /api/admin/users/{id}
PUT    /api/admin/users/{id}
DELETE /api/admin/users/{id}
POST   /api/admin/users/{id}/suspend
POST   /api/admin/users/{id}/reset-pwd
POST   /api/admin/users/bulk-invite
GET    /api/admin/users/{id}/activity
POST   /api/admin/users/{id}/impersonate
PUT    /api/admin/users/{id}/role

# Teams (7 endpoints)
GET    /api/admin/teams
POST   /api/admin/teams
GET    /api/admin/teams/{id}
PUT    /api/admin/teams/{id}
DELETE /api/admin/teams/{id}
PUT    /api/admin/teams/{id}/owner
GET    /api/admin/teams/{id}/analytics

# Roles & Permissions (5 endpoints)
GET    /api/admin/roles
GET    /api/admin/roles/{role}/permissions
POST   /api/admin/permissions/override
GET    /api/admin/projects/{id}/permissions
PUT    /api/admin/projects/{id}/permissions

# System Health (8 endpoints)
GET    /api/admin/health/overview
GET    /api/admin/health/api
GET    /api/admin/health/postgres
GET    /api/admin/health/redis
GET    /api/admin/health/celery
GET    /api/admin/health/mlflow
GET    /api/admin/health/history
GET    /api/admin/health/alerts

# GPU Management (5 endpoints)
GET    /api/admin/gpu/inventory
GET    /api/admin/gpu/{device_id}
GET    /api/admin/gpu/allocation-map
PUT    /api/admin/gpu/{device_id}/maintenance
GET    /api/admin/gpu/capacity-forecast

# Workers (5 endpoints)
GET    /api/admin/workers
GET    /api/admin/workers/{worker_id}
POST   /api/admin/workers/{worker_id}/drain
POST   /api/admin/workers/{worker_id}/terminate
GET    /api/admin/workers/{worker_id}/tasks

# Queues & Tasks (8 endpoints)
GET    /api/admin/queues
GET    /api/admin/queues/{name}
POST   /api/admin/queues/{name}/pause
POST   /api/admin/queues/{name}/resume
POST   /api/admin/queues/{name}/purge
GET    /api/admin/tasks/{task_id}
POST   /api/admin/tasks/{task_id}/cancel
GET    /api/admin/tasks/{task_id}/logs

# DLQ (6 endpoints)
GET    /api/admin/dlq
GET    /api/admin/dlq/{task_id}
POST   /api/admin/dlq/{task_id}/retry
DELETE /api/admin/dlq/{task_id}
POST   /api/admin/dlq/archive
GET    /api/admin/dlq/analytics

# Quotas (8 endpoints)
GET    /api/admin/quotas/teams/{team_id}
PUT    /api/admin/quotas/teams/{team_id}
GET    /api/admin/quotas/users/{user_id}
PUT    /api/admin/quotas/users/{user_id}
GET    /api/admin/quotas/templates
POST   /api/admin/quotas/teams/{team_id}/exception
GET    /api/admin/quotas/usage-report
GET    /api/admin/rate-limits

# Feature Flags (5 endpoints)
GET    /api/admin/feature-flags
PUT    /api/admin/feature-flags/{flag}
POST   /api/admin/feature-flags/{flag}/schedule
POST   /api/admin/feature-flags/{flag}/gradual
GET    /api/admin/feature-flags/{flag}/users

# Config (4 endpoints)
GET    /api/admin/config
PUT    /api/admin/config/{key}
GET    /api/admin/config/history
POST   /api/admin/config/rollback

# Secrets (6 endpoints)
GET    /api/admin/secrets
POST   /api/admin/secrets
PUT    /api/admin/secrets/{key}/rotate
DELETE /api/admin/secrets/{key}
GET    /api/admin/secrets/rotation-policy
GET    /api/admin/secrets/audit

# Audit (4 endpoints)
GET    /api/admin/audit/logs
GET    /api/admin/audit/export
GET    /api/admin/audit/users/{user_id}
GET    /api/admin/audit/events/{event_id}

# Billing (already implemented)
GET    /api/admin/billing/overview
GET    /api/admin/billing/teams/{team_id}
GET    /api/admin/billing/cost-trends

# Tokens (2 endpoints)
GET    /api/admin/tokens
DELETE /api/admin/tokens/{token_id}

TOTAL: ~90 new/enhanced admin endpoints
```

---

## 12. Success Metrics

**Operational Metrics:**
- Mean Time to Detect (MTTD) incidents: <5 minutes
- Mean Time to Resolve (MTTR) incidents: <30 minutes
- Admin task completion time: <2 minutes average
- System uptime: >99.9%

**User Metrics:**
- Quota violation rate: <1% of jobs
- Admin action audit coverage: 100%
- Secret rotation compliance: 100%
- Failed job retry success rate: >80%

**Security Metrics:**
- Unauthorized access attempts: 0
- Unencrypted secrets: 0
- Audit log tampering: 0 (via cryptographic verification)
- MFA adoption for admins: >90%

---

## Conclusion

This comprehensive scope defines the full set of capabilities required for the SparkTrainer Admin Dashboard. The phased implementation approach prioritizes security hardening and monitoring, followed by quota management, feature flags, and finally the full admin UI.

**Current Implementation Status:**
- ✅ **60% Complete**: Basic RBAC, team management, GPU management, storage, billing
- ⚠️ **40% Remaining**: Security hardening, monitoring, feature flags, audit log, secrets management

**Estimated Implementation:**
- Phase 1 (Security): 2-3 weeks
- Phase 2 (Monitoring): 2-3 weeks
- Phase 3 (Quotas): 1-2 weeks
- Phase 4 (Feature Flags): 1-2 weeks
- Phase 5 (Admin UI): 3-4 weeks

**Total**: 9-14 weeks for full implementation
