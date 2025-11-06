# ScrollSafe Backend - Deployment Guide (Instance 1)

This instance runs 24/7 to serve the frontend Chrome extension and admin dashboard.

## Architecture

**Instance 1 (This):** Backend + Databases (always on)
- PostgreSQL (analysis results, admin labels)
- Redis (Celery broker + application cache)
- FastAPI Backend API
- Deep Scan Worker (for frontend-triggered scans)

**Instance 2 (Remote):** Doomscroller Workers (on-demand)
- Connects to this instance for database access
- Runs video discovery and batch analysis
- Can be stopped when not needed to save costs

## Prerequisites

1. AWS EC2 instance (t3.small recommended: 2 vCPU, 2GB RAM)
2. Docker and Docker Compose installed
3. Security groups configured:
   - Port 8000 (HTTP) - open to 0.0.0.0/0 (frontend access)
   - Port 5432 (PostgreSQL) - open to Instance 2 private IP only
   - Port 6379 (Redis) - open to Instance 2 private IP only
   - Port 22 (SSH) - your IP only

## Quick Start

### 1. Install Docker on AWS Instance

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io docker-compose-v2

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

### 2. Copy Project Files

```bash
# From your local machine
scp -r scrollsafe-backend ubuntu@<instance1-ip>:~/

# Or clone from git
# git clone <your-repo> ~/scrollsafe-backend
```

### 3. Configure Environment

```bash
cd ~/scrollsafe-backend

# Copy example config
cp .env.example .env

# Edit with your API keys
nano .env

# IMPORTANT: Generate a secure admin API key first
openssl rand -hex 32

# Update these values:
# ADMIN_API_KEY=<generated-secure-key-from-above>
# INFER_API_KEY=<your-inference-api-key>
# INFER_API_URL=<your-hf-endpoint-url>
# HUGGING_FACE_API_KEY=<your-hf-token>
```

### 4. Start All Services

**Note:** Database schema (videos, analyses, admin_labels tables) is automatically created on first startup via `schema.sql`.

```bash
# Start everything
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### 5. Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# Check Redis connection
docker compose exec redis redis-cli --pass postgres1997! ping
# Should return: PONG

# Check PostgreSQL and schema
docker compose exec postgres psql -U postgres -d doomscroller -c "\dt"
# Should show: videos, analyses, admin_labels tables
```

### 6. Test from External Network

```bash
# From your local machine
curl http://<instance1-public-ip>:8000/health
```

## Security Configuration

### Security Groups (AWS)

**Inbound Rules:**
```
Type        Protocol  Port Range  Source                Description
HTTP        TCP       8000        0.0.0.0/0            Frontend API access
PostgreSQL  TCP       5432        <Instance2-IP>/32   Doomscroller worker access
Redis       TCP       6379        <Instance2-IP>/32   Doomscroller worker access
SSH         TCP       22          <Your-IP>/32        Admin access
```

**Important:**
- PostgreSQL and Redis should NOT be open to 0.0.0.0/0
- Use VPC private IPs if both instances are in same VPC (free, more secure)

### Database Password

Redis and PostgreSQL both use password: `postgres1997!`

To change:
1. Update `docker-compose.yaml`:
   - PostgreSQL: `POSTGRES_PASSWORD`
   - Redis: `--requirepass` in command
2. Update `.env`:
   - `DATABASE_URL`
   - `REDIS_APP_URL`
   - `CELERY_BROKER_URL`
3. Update Instance 2's `.env` with same passwords

### Admin API Key

**All admin routes require API key authentication:**
- `/api/admin/metrics` (GET) - View dashboard metrics
- `/api/admin/labels` (POST) - Upload admin labels

**Generate a secure key:**
```bash
openssl rand -hex 32
# Example output: a1b2c3d4e5f6...
```

**Add to `.env`:**
```env
ADMIN_API_KEY=your_generated_key_here
```

**Using admin routes:**
```bash
# Get admin metrics
curl -H "X-API-Key: your_api_key_here" \
  http://<instance1-ip>:8000/api/admin/metrics

# Upload admin label
curl -X POST \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/shorts/abc123", "label": "verified", "notes": "Manually reviewed"}' \
  http://<instance1-ip>:8000/api/admin/labels
```

**Security notes:**
- Keep API key secret - don't commit to git
- Rotate key periodically
- Use environment variables, never hardcode
- 401 Unauthorized if key is missing or invalid

## Admin Dashboard Access

### Connect with pgAdmin

**Connection Settings:**
- Host: `<instance1-public-ip>` or `<instance1-private-ip>`
- Port: `5432`
- Database: `doomscroller`
- Username: `postgres`
- Password: `postgres1997!`

**Useful Queries:**

```sql
-- Recent analyses
SELECT platform, video_id, label, confidence, analyzed_at
FROM analyses
ORDER BY analyzed_at DESC
LIMIT 20;

-- Admin labels
SELECT * FROM admin_labels
ORDER BY created_at DESC;

-- Analysis stats by label
SELECT label, COUNT(*) as count, AVG(confidence) as avg_confidence
FROM analyses
GROUP BY label;

-- Today's activity
SELECT COUNT(*) as videos_analyzed
FROM analyses
WHERE analyzed_at > NOW() - INTERVAL '24 hours';
```

### Add Admin Labels via API

```bash
# Upload admin label for YouTube video
curl -X POST http://<instance1-ip>:8000/api/admin/upload-csv \
  -H "Content-Type: multipart/form-data" \
  -F "file=@labels.csv"

# CSV format:
# platform,video_id,label,notes
# youtube,dQw4w9WgXcQ,verified,Manually reviewed - real content
# instagram,ABC123,ai-detected,AI-generated CGI
```

## Frontend Configuration

Update your Chrome extension's `services/api.js` to point to this instance:

```javascript
const API_BASE_URL = 'http://<instance1-public-ip>:8000';
```

**Or use a domain name:**
1. Point DNS A record to Instance 1's elastic IP
2. Update API_BASE_URL to: `https://api.scrollsafe.com`
3. Add HTTPS with Nginx reverse proxy + Let's Encrypt

## Monitoring

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f deep-scan-worker
docker compose logs -f postgres
docker compose logs -f redis
```

### Resource Usage

```bash
# Real-time stats
docker stats

# Disk usage
docker system df

# Container status
docker compose ps
```

### API Metrics

Check backend logs for:
- Request counts
- Response times
- Error rates

Located in: `~/scrollsafe-backend/api_requests.log`

## Maintenance

### Backup Database

```bash
# Backup to file
docker compose exec postgres pg_dump -U postgres doomscroller > backup_$(date +%Y%m%d).sql

# Or use compressed backup
docker compose exec postgres pg_dump -U postgres -Fc doomscroller > backup_$(date +%Y%m%d).dump
```

### Restore Database

```bash
# From SQL file
docker compose exec -T postgres psql -U postgres -d doomscroller < backup_20250125.sql

# From compressed dump
docker compose exec -T postgres pg_restore -U postgres -d doomscroller < backup_20250125.dump
```

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker compose build --no-cache
docker compose up -d
```

### Clear Redis Cache

```bash
# Clear all app cache (db 1)
docker compose exec redis redis-cli --pass postgres1997! -n 1 FLUSHDB

# Clear Celery queue (db 0)
docker compose exec redis redis-cli --pass postgres1997! -n 0 FLUSHDB
```

## Troubleshooting

### Backend won't start

**Symptom:** Container restarts repeatedly

**Check logs:**
```bash
docker compose logs backend
```

**Common issues:**
- Missing .env file: Copy from .env.example
- Invalid database connection: Check DATABASE_URL
- Port 8000 already in use: Stop conflicting service

### Can't connect to database from Instance 2

**Symptom:** "Connection refused" from doomscroller workers

**Fix:**
1. Check security groups allow Instance 2 → Instance 1 on port 5432
2. Verify postgres container is running: `docker compose ps`
3. Test from Instance 2:
   ```bash
   telnet <instance1-ip> 5432
   ```
4. Check PostgreSQL is listening on all interfaces:
   ```bash
   docker compose logs postgres | grep "listening on"
   ```

### Deep scan worker stuck

**Symptom:** No progress on deep scan tasks

**Fix:**
```bash
# Check worker logs
docker compose logs deep-scan-worker

# Restart worker
docker compose restart deep-scan-worker

# Clear stuck jobs
docker compose exec redis redis-cli --pass postgres1997! -n 0 KEYS "deep:job:*"
```

## Cost Optimization

**Instance 1 (t3.small):**
- Hourly: ~$0.02/hour
- Monthly: ~$15/month
- Runs 24/7 to serve frontend

**Ways to reduce costs:**
1. Use reserved instance (1-year commitment: ~$10/month)
2. Use spot instance (risky for production, ~$6/month)
3. Use AWS Lightsail instead of EC2 (~$5/month for 1GB RAM)

## Connecting Instance 2

Once Instance 1 is running, configure Instance 2 (doomscroller workers) to connect:

**In Instance 2's .env:**
```env
# Use Instance 1's private IP (if in same VPC)
DATABASE_URL=postgresql://postgres:postgres1997!@10.0.1.50:5432/doomscroller
CELERY_BROKER_URL=redis://:postgres1997!@10.0.1.50:6379/0
REDIS_APP_URL=redis://:postgres1997!@10.0.1.50:6379/1
```

**Or use public IP (less secure, costs data transfer):**
```env
DATABASE_URL=postgresql://postgres:postgres1997!@<instance1-public-ip>:5432/doomscroller
```

See Instance 2's DEPLOYMENT.md for full setup.
