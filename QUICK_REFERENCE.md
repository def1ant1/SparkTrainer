# SparkTrainer - Quick Reference Guide

## File Locations for Key Features

### Frontend Navigation & Pages
- **Main App**: `/frontend/src/App.jsx` (2,586 lines)
  - Page routing: lines 1288-1530 (currentPage state)
  - Sidebar: lines 2443-2486
  - Navigation items defined: lines 2444-2456

### Backend API
- **Flask App**: `/backend/app.py` (6,700+ lines)
  - Health/System: lines 422-3457
  - Jobs: lines 1114-1711
  - Models: lines 448-1074
  - Datasets: lines 2160-3349
  - Experiments: lines 145-422
  - Authentication: lines 4433-4666

### Database Models
- **SQLAlchemy ORM**: `/backend/models.py` (432 lines)
  - Project, Dataset, Experiment, Job, Artifact, Evaluation models
  - JobStatus state machine: lines 21-41
  - Foreign key relationships: throughout

### Authentication & Security
- **JWT + RBAC**: `/backend/auth.py` (282 lines)
  - TokenManager class: lines 98-158
  - PermissionManager class: lines 175-245
  - Role definitions: lines 27-40
  - Permission matrix: lines 51-95

### Component Pages
- **Models**: `/frontend/src/components/Models.jsx` - Model browser, detail, comparison
- **Datasets**: `/frontend/src/components/Datasets.jsx` - Data ingestion and versioning
- **Experiments**: `/frontend/src/components/Experiments.jsx` - Experiment tracking
- **Jobs**: Part of `App.jsx` - `JobsList` component
- **Profile**: `/frontend/src/components/Profile.jsx` - User settings and API keys
- **JobWizard**: `/frontend/src/components/JobWizard.jsx` - Advanced job creation

---

## Key Concepts & Patterns

### Navigation Flow
```
User clicks sidebar item
  → onNavigate(page_key)
  → setCurrentPage(key)
  → App renders matching page component
  → Breadcrumbs update automatically
```

### API Communication Pattern
```javascript
// Frontend defines centralized api object (lines 22-107)
api.getJobs() → fetch('/api/jobs')
api.createJob(data) → fetch('/api/jobs', {POST})

// Backend flask routes handle requests
@app.route('/api/jobs', methods=['GET','POST'])
def list_or_create_jobs():
  ...
  return jsonify({...})
```

### Database Access Pattern
```python
# Using SQLAlchemy ORM
from database import get_db

with get_db() as session:
  job = session.query(Job).filter_by(id=job_id).first()
  job.status = JobStatus.RUNNING
  session.commit()
```

### Job Lifecycle
```
PENDING → QUEUED → RUNNING → PAUSED → COMPLETED/FAILED/CANCELLED
```

### Credential Storage
```
Frontend (Profile component)
  → settings.hf_token, settings.openai_api_key, etc.
  → Send to /api/user/settings via PUT
Backend
  → Store in /jobs/users.json
  → users[user_id]['settings'] contains all credentials
  → No encryption (development mode)
```

---

## API Endpoint Categories

### Essential Endpoints to Know

**Jobs Management**:
- `POST /api/jobs` - Submit job
- `GET /api/jobs/<id>` - Get job status
- `POST /api/jobs/<id>/cancel` - Cancel job
- `GET /api/jobs/<id>/logs` - Stream logs
- `GET /api/jobs/<id>/metrics` - Get metrics

**Models**:
- `GET /api/models` - List models
- `GET /api/models/<id>` - Get model details
- `POST /api/models/save` - Save trained model
- `PUT /api/models/<id>/metadata` - Update metadata

**Datasets**:
- `GET /api/datasets` - List datasets
- `POST /api/datasets/upload` - Upload data
- `GET /api/datasets/<name>/samples` - Preview data
- `POST /api/datasets/ingest/stream_*` - Stream large files

**User & Settings**:
- `GET /api/user/settings` - Get user settings
- `PUT /api/user/settings` - Update settings (saves credentials)
- `POST /api/auth/login` - JWT login
- `POST /api/user/tokens` - Create API token

**System**:
- `GET /api/system/info` - System metrics
- `GET /api/gpu/partitions` - GPU configuration
- `GET /api/health` - Health check

---

## Component State Management

### Frontend State Examples

```javascript
// Dashboard page
const [currentPage, setCurrentPage] = useState('dashboard');
const [systemInfo, setSystemInfo] = useState({});
const [metrics, setMetrics] = useState({...});
const [jobs, setJobs] = useState([]);

// Models page
const [modelView, setModelView] = useState({ id: null, compareIds: [] });

// Profile page
const [settings, setSettings] = useState({
  name: '',
  hf_token: '',      // HuggingFace token
  openai_api_key: '',
  wandb_api_key: '',
  // ... other settings
});
```

### Update Intervals

```javascript
// Dashboard - system info every 10 seconds
useEffect(() => {
  const t = setInterval(load, 10000);
  return () => clearInterval(t);
}, []);

// Jobs list - every 5 seconds
useEffect(() => {
  const t = setInterval(load, 5000);
  return () => clearInterval(t);
}, []);
```

---

## Common Modifications & Extensions

### Adding a New Page

1. Create component in `/frontend/src/components/NewPage.jsx`
2. Import in `App.jsx` line 13-14
3. Add sidebar item in `Sidebar` function (line 2444+)
4. Add conditional render in main App (line 1418+)
5. Add to breadcrumbs logic (line 2507+)

### Adding a New API Endpoint

1. Define route in `/backend/app.py`:
   ```python
   @app.route('/api/new-feature', methods=['GET','POST'])
   def new_feature():
       data = request.json or {}
       # ... logic
       return jsonify({'status': 'ok', 'data': ...})
   ```

2. Add to frontend api object in `App.jsx` (line 22+):
   ```javascript
   api.newFeature = () => fetch(`/api/new-feature`).then(r => r.json());
   ```

3. Use in component:
   ```javascript
   const result = await api.newFeature();
   ```

### Adding a New Celery Task

1. Define in `/backend/celery_tasks.py`:
   ```python
   @celery.task(bind=True)
   def new_task(self, param1):
       # ... task logic
       return result
   ```

2. Route in `/backend/celery_app.py` if needed (line 49+):
   ```python
   task_routes={
       "celery_tasks.new_task": {"queue": "custom_queue"},
   }
   ```

3. Call from API endpoint:
   ```python
   from celery_app import celery
   task = celery.send_task('celery_tasks.new_task', args=[param])
   ```

---

## Database Connection & Sessions

```python
# Using the context manager
from database import get_db

with get_db() as session:
    # Query
    user = session.query(User).filter_by(id=user_id).first()
    
    # Create
    new_job = Job(id=uuid.uuid4(), status=JobStatus.PENDING)
    session.add(new_job)
    
    # Update
    job.status = JobStatus.RUNNING
    
    # Delete
    session.delete(artifact)
    
    # Auto-commits on context exit, rollsback on exception
```

---

## Testing the Platform

### Run Locally

```bash
# Terminal 1: Backend
cd backend
python app.py

# Terminal 2: Celery Worker
celery -A celery_app.celery worker --loglevel=info

# Terminal 3: Frontend
cd frontend
npm run dev

# Terminal 4: Database init (first time)
cd backend
python init_db.py
```

### Access Points

- Frontend: http://localhost:3000
- Backend API: http://localhost:5000/api
- MLflow: http://localhost:5001
- Flower (Celery): http://localhost:5555
- PostgreSQL: localhost:5432

---

## Performance Considerations

### Frontend
- Dashboard updates: 10s intervals (adjustable)
- Jobs list: 5s polling (could use WebSocket)
- Model comparison: Client-side only (efficient)
- Metrics history: Limited by deque size (1-hour window)

### Backend
- Job queries indexed by status, experiment, created_at
- Dataset queries indexed by project_id, name, version
- Model search has sorting/filtering
- Large file uploads support streaming

### Database
- Connection pooling: 10 pool_size + 20 max_overflow
- Indexes on: status, job_id, experiment_id, created_at, etc.
- Foreign key relationships with cascade deletes
- Event listeners for timestamp updates

---

## Security Notes for Production

Current state (Development):
- JWT tokens in memory
- Credentials in plain JSON files
- No HTTPS (development)
- No rate limiting
- No CORS restrictions enabled

For Production:
1. Enable HTTPS/TLS
2. Encrypt credentials with AES-256-GCM
3. Use proper secrets manager (AWS Secrets Manager, HashiCorp Vault)
4. Add rate limiting middleware
5. Implement request signing for API tokens
6. Add CORS whitelist
7. Enable SQL injection prevention (SQLAlchemy already handles this)
8. Add request/response logging and audit trails
9. Implement session timeout
10. Add 2FA support

---

## Debugging Tips

### Frontend Issues

```javascript
// Check state
console.log('currentPage:', currentPage);
console.log('systemInfo:', systemInfo);

// Check API response
const res = await fetch('/api/jobs');
const data = await res.json();
console.log('Jobs:', data);

// Check localStorage
console.log(localStorage.getItem('theme'));
```

### Backend Issues

```bash
# Check logs
tail -f /home/user/SparkTrainer/logs/*.log

# Check database
psql -U sparktrainer -d sparktrainer
SELECT * FROM jobs WHERE id = 'job_id';

# Check Celery tasks
celery -A celery_app inspect active

# Check Redis
redis-cli
KEYS *
GET user:settings:user_id
```

---

## Documentation Files in Repository

- `README.md` - Overview and quick start
- `FEATURES.md` - Detailed feature documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `ARCHITECTURE_ANALYSIS.md` - This detailed architecture analysis
- `CODEBASE_GUIDE.md` - Code organization guide
- `docs/` - Additional documentation
