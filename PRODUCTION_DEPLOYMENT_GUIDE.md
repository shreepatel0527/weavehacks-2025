# Production Deployment Guide for Ultimate Lab Assistant

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Pre-deployment Checklist](#pre-deployment-checklist)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Installation & Configuration](#installation--configuration)
5. [Security Configuration](#security-configuration)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring & Observability](#monitoring--observability)
8. [Backup & Recovery](#backup--recovery)
9. [Scaling Considerations](#scaling-considerations)
10. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 500GB SSD minimum (for data and video storage)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for ML models)
- **Network**: 1Gbps connection for cloud sync

### Software Requirements
- **OS**: Ubuntu 20.04 LTS or Windows Server 2019+
- **Python**: 3.9 or 3.10
- **Docker**: 20.10+
- **Redis**: 6.2+
- **PostgreSQL**: 13+
- **NVIDIA CUDA**: 11.8+ (for GPU acceleration)

## Pre-deployment Checklist

### 1. Environment Preparation
```bash
# Create dedicated user
sudo useradd -m -s /bin/bash labassistant
sudo usermod -aG docker labassistant

# Create directory structure
sudo mkdir -p /opt/lab-assistant/{app,data,logs,config,recordings}
sudo chown -R labassistant:labassistant /opt/lab-assistant
```

### 2. Security Audit
- [ ] SSL certificates obtained
- [ ] Firewall rules configured
- [ ] API keys secured in vault
- [ ] User authentication configured
- [ ] Network segmentation verified

### 3. Dependency Check
```bash
# Check Python version
python3 --version

# Check system dependencies
docker --version
redis-cli --version
psql --version

# Check GPU availability
nvidia-smi
```

## Infrastructure Setup

### 1. Database Setup
```bash
# PostgreSQL setup
sudo -u postgres createuser labassistant
sudo -u postgres createdb lab_assistant_prod
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE lab_assistant_prod TO labassistant;"

# Create schema
psql -U labassistant -d lab_assistant_prod -f schema/database_schema.sql
```

### 2. Redis Configuration
```bash
# Edit Redis configuration
sudo nano /etc/redis/redis.conf

# Add production settings
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000

# Restart Redis
sudo systemctl restart redis
```

### 3. Docker Setup
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  lab-assistant:
    build:
      context: .
      dockerfile: Dockerfile.prod
    image: lab-assistant:latest
    container_name: lab-assistant-prod
    restart: unless-stopped
    ports:
      - "8501:8501"  # Streamlit
      - "8080:8080"  # API
      - "8765:8765"  # WebSocket
    volumes:
      - /opt/lab-assistant/data:/app/data
      - /opt/lab-assistant/logs:/app/logs
      - /opt/lab-assistant/recordings:/app/recordings
      - /opt/lab-assistant/config:/app/config
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://labassistant:${DB_PASSWORD}@db:5432/lab_assistant_prod
      - REDIS_URL=redis://redis:6379
      - WEAVE_API_KEY=${WEAVE_API_KEY}
    depends_on:
      - db
      - redis
    deploy:
      resources:
        limits:
          cpus: '6'
          memory: 12G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  db:
    image: postgres:13-alpine
    container_name: lab-db-prod
    restart: unless-stopped
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=labassistant
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=lab_assistant_prod

  redis:
    image: redis:6.2-alpine
    container_name: lab-redis-prod
    restart: unless-stopped
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    container_name: lab-nginx-prod
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - lab-assistant

volumes:
  postgres_data:
  redis_data:
```

## Installation & Configuration

### 1. Clone and Setup
```bash
# Clone repository
git clone https://github.com/your-org/lab-assistant.git
cd lab-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-prod.txt
```

### 2. Environment Configuration
```bash
# Create .env.production file
cat > .env.production << EOF
# Application Settings
ENVIRONMENT=production
DEBUG=False
SECRET_KEY=$(openssl rand -hex 32)

# Database
DATABASE_URL=postgresql://labassistant:secure_password@localhost:5432/lab_assistant_prod

# Redis
REDIS_URL=redis://localhost:6379/0

# W&B Weave
WEAVE_API_KEY=your_weave_api_key
WEAVE_PROJECT=lab-assistant-prod

# Cloud Storage
STORAGE_BACKEND=azure
AZURE_CONNECTION_STRING=your_connection_string

# Security
JWT_SECRET_KEY=$(openssl rand -hex 32)
ALLOWED_HOSTS=lab.yourdomain.com
CORS_ORIGINS=https://lab.yourdomain.com

# Monitoring
SENTRY_DSN=your_sentry_dsn
DATADOG_API_KEY=your_datadog_key

# Email (for alerts)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=secure_password
EOF
```

### 3. Application Configuration
```python
# config/production.py
import os
from pathlib import Path

class ProductionConfig:
    # Base settings
    ENV = 'production'
    DEBUG = False
    TESTING = False
    
    # Security
    SECRET_KEY = os.environ['SECRET_KEY']
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # Database
    DATABASE_URL = os.environ['DATABASE_URL']
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Redis
    REDIS_URL = os.environ['REDIS_URL']
    
    # File storage
    UPLOAD_FOLDER = Path('/opt/lab-assistant/data/uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = '/opt/lab-assistant/logs/app.log'
    
    # Performance
    CACHE_TYPE = 'redis'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ['REDIS_URL']
    RATELIMIT_DEFAULT = "100 per hour"
    
    # Background jobs
    CELERY_BROKER_URL = os.environ['REDIS_URL']
    CELERY_RESULT_BACKEND = os.environ['REDIS_URL']
```

## Security Configuration

### 1. SSL/TLS Setup
```bash
# Generate SSL certificate (production should use Let's Encrypt)
sudo certbot certonly --standalone -d lab.yourdomain.com

# Configure nginx
server {
    listen 443 ssl http2;
    server_name lab.yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/lab.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/lab.yourdomain.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /ws {
        proxy_pass http://localhost:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 2. API Security
```python
# security/api_security.py
from functools import wraps
from flask import request, jsonify
import jwt

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def validate_api_key(api_key):
    # Implement API key validation
    return api_key in valid_api_keys
```

### 3. Data Encryption
```python
# security/encryption.py
from cryptography.fernet import Fernet
import os

class DataEncryption:
    def __init__(self):
        self.key = os.environ.get('ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.key)
    
    def encrypt_sensitive_data(self, data: bytes) -> bytes:
        return self.cipher.encrypt(data)
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> bytes:
        return self.cipher.decrypt(encrypted_data)
```

## Performance Optimization

### 1. Database Optimization
```sql
-- Create indexes
CREATE INDEX idx_experiments_user_id ON experiments(user_id);
CREATE INDEX idx_experiments_created_at ON experiments(created_at DESC);
CREATE INDEX idx_sensor_data_experiment_id ON sensor_data(experiment_id);
CREATE INDEX idx_sensor_data_timestamp ON sensor_data(timestamp DESC);

-- Partitioning for large tables
CREATE TABLE sensor_data_2024_01 PARTITION OF sensor_data
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### 2. Caching Strategy
```python
# caching/cache_config.py
from flask_caching import Cache
import redis

cache = Cache(config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ['REDIS_URL'],
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Use caching decorator
@cache.cached(timeout=60)
def get_experiment_statistics(experiment_id):
    # Expensive calculation
    return calculate_statistics(experiment_id)
```

### 3. Async Processing
```python
# background/celery_config.py
from celery import Celery

celery = Celery(
    'lab_assistant',
    broker=os.environ['REDIS_URL'],
    backend=os.environ['REDIS_URL']
)

celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
)

@celery.task
def process_video_async(video_path):
    # Process video in background
    return process_video(video_path)
```

## Monitoring & Observability

### 1. Application Monitoring
```python
# monitoring/app_monitoring.py
import sentry_sdk
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, Gauge

# Initialize Sentry
sentry_sdk.init(
    dsn=os.environ['SENTRY_DSN'],
    environment='production',
    traces_sample_rate=0.1,
)

# Prometheus metrics
request_count = Counter('app_requests_total', 'Total requests')
request_duration = Histogram('app_request_duration_seconds', 'Request duration')
active_experiments = Gauge('app_active_experiments', 'Active experiments')

# OpenTelemetry setup
tracer = trace.get_tracer(__name__)
```

### 2. System Monitoring
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: lab-prometheus
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:latest
    container_name: lab-grafana
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secure_password
  
  node_exporter:
    image: prom/node-exporter:latest
    container_name: lab-node-exporter
    ports:
      - "9100:9100"

volumes:
  prometheus_data:
  grafana_data:
```

### 3. Log Aggregation
```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /opt/lab-assistant/logs/*.log
  multiline.pattern: '^\d{4}-\d{2}-\d{2}'
  multiline.negate: true
  multiline.match: after

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "lab-assistant-%{+yyyy.MM.dd}"

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
```

## Backup & Recovery

### 1. Automated Backups
```bash
#!/bin/bash
# backup.sh

# Configuration
BACKUP_DIR="/opt/backups/lab-assistant"
DB_NAME="lab_assistant_prod"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR/$(date +%Y%m%d)

# Database backup
pg_dump -U labassistant -d $DB_NAME | gzip > $BACKUP_DIR/$(date +%Y%m%d)/database.sql.gz

# Redis backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb $BACKUP_DIR/$(date +%Y%m%d)/redis.rdb

# Application data
tar -czf $BACKUP_DIR/$(date +%Y%m%d)/app_data.tar.gz /opt/lab-assistant/data

# Upload to cloud storage
aws s3 sync $BACKUP_DIR/$(date +%Y%m%d) s3://lab-assistant-backups/$(date +%Y%m%d)

# Clean old backups
find $BACKUP_DIR -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;
```

### 2. Recovery Procedures
```bash
#!/bin/bash
# restore.sh

# Configuration
BACKUP_DATE=$1
BACKUP_DIR="/opt/backups/lab-assistant/$BACKUP_DATE"

# Stop services
docker-compose down

# Restore database
gunzip < $BACKUP_DIR/database.sql.gz | psql -U labassistant -d lab_assistant_prod

# Restore Redis
cp $BACKUP_DIR/redis.rdb /var/lib/redis/dump.rdb
chown redis:redis /var/lib/redis/dump.rdb

# Restore application data
tar -xzf $BACKUP_DIR/app_data.tar.gz -C /

# Restart services
docker-compose up -d
```

## Scaling Considerations

### 1. Horizontal Scaling
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lab-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lab-assistant
  template:
    metadata:
      labels:
        app: lab-assistant
    spec:
      containers:
      - name: lab-assistant
        image: lab-assistant:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: ENVIRONMENT
          value: "production"
```

### 2. Load Balancing
```yaml
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: lab-assistant-service
spec:
  selector:
    app: lab-assistant
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
  type: LoadBalancer
```

### 3. Auto-scaling
```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lab-assistant-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lab-assistant
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Troubleshooting

### Common Issues

#### 1. High Memory Usage
```bash
# Check memory usage
docker stats lab-assistant-prod

# Analyze memory profile
python -m memory_profiler app.py

# Solution: Increase memory limits or optimize code
```

#### 2. Slow Response Times
```bash
# Check database queries
psql -U labassistant -d lab_assistant_prod -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check Redis performance
redis-cli --latency

# Solution: Add indexes, optimize queries, increase cache
```

#### 3. WebSocket Connection Issues
```bash
# Check WebSocket connectivity
wscat -c wss://lab.yourdomain.com/ws

# Check nginx logs
tail -f /var/log/nginx/error.log

# Solution: Verify nginx configuration, check firewall rules
```

### Health Checks
```python
# health/health_checks.py
from flask import Blueprint, jsonify
import psutil
import redis

health_bp = Blueprint('health', __name__)

@health_bp.route('/health')
def health_check():
    checks = {
        'status': 'healthy',
        'database': check_database(),
        'redis': check_redis(),
        'disk_usage': psutil.disk_usage('/').percent,
        'memory_usage': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent()
    }
    
    # Determine overall health
    if any(not v for k, v in checks.items() if k.endswith('_check')):
        checks['status'] = 'unhealthy'
    
    return jsonify(checks)
```

## Maintenance Procedures

### 1. Regular Maintenance Tasks
- Daily: Check logs for errors, monitor disk usage
- Weekly: Review performance metrics, update dependencies
- Monthly: Security patches, database optimization
- Quarterly: Full system backup test, disaster recovery drill

### 2. Update Procedures
```bash
# Rolling update procedure
#!/bin/bash

# 1. Build new image
docker build -t lab-assistant:new .

# 2. Test new image
docker run --rm lab-assistant:new python -m pytest

# 3. Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# 4. Run smoke tests
python tests/smoke_tests.py

# 5. Deploy to production (rolling update)
kubectl set image deployment/lab-assistant lab-assistant=lab-assistant:new

# 6. Monitor deployment
kubectl rollout status deployment/lab-assistant
```

### 3. Emergency Procedures
```bash
# Emergency rollback
kubectl rollout undo deployment/lab-assistant

# Emergency shutdown
docker-compose stop

# Data preservation
pg_dump -U labassistant -d lab_assistant_prod > emergency_backup.sql
```

## Support & Documentation

### Contact Information
- **Technical Support**: support@lab-assistant.com
- **Emergency Hotline**: +1-555-LAB-HELP
- **Documentation**: https://docs.lab-assistant.com
- **Status Page**: https://status.lab-assistant.com

### Additional Resources
- API Documentation: `/api/docs`
- User Manual: `/docs/user-manual.pdf`
- Training Videos: https://training.lab-assistant.com
- Community Forum: https://community.lab-assistant.com

---

**Last Updated**: January 2024
**Version**: 3.0.0
**Maintained by**: Lab Assistant DevOps Team