# Security & Bug Fix Changes

## Changes Made

### 1. Admin API Key Authentication ✅

**What changed:**
- Added API key authentication to admin routes
- Routes protected: `/api/admin/metrics` (GET) and `/api/admin/labels` (POST)

**Implementation:**
- New dependency function: `verify_admin_api_key()` validates `X-API-Key` header
- Requires `ADMIN_API_KEY` environment variable
- Returns 401 Unauthorized if key is invalid or missing
- Returns 500 if ADMIN_API_KEY not configured on server

**Files modified:**
- `main.py`:
  - Added `Header` and `Depends` imports
  - Added `ADMIN_API_KEY` environment variable loading
  - Added `verify_admin_api_key()` function
  - Updated `/api/admin/metrics` endpoint with `Depends(verify_admin_api_key)`
  - Updated `/api/admin/labels` endpoint with `Depends(verify_admin_api_key)`
  - Updated CORS to allow `X-API-Key` header

**Configuration:**
- `.env.example` updated with `ADMIN_API_KEY` placeholder
- `DEPLOYMENT.md` updated with:
  - Key generation instructions: `openssl rand -hex 32`
  - Usage examples with curl
  - Security best practices

**Usage:**
```bash
# Generate key
openssl rand -hex 32

# Add to .env
ADMIN_API_KEY=your_generated_key_here

# Use in requests
curl -H "X-API-Key: your_api_key_here" \
  http://localhost:8000/api/admin/metrics
```

---

### 2. Fixed Queue Depth Calculation ✅

**What was wrong:**
- Pattern `f"{queue_name}\x06\x16*"` was not properly matching Celery priority queue keys
- Function `_scan_priority_keys()` had incorrect Redis SCAN pattern
- Queue depth could be inaccurate for queues with priority tasks

**What changed:**
- Completely rewrote `queue_depth()` function in `services/admin_service.py`
- Removed separate `_scan_priority_keys()` function
- Improved pattern matching for Celery priority queues
- Added proper logging for debugging
- Better error handling

**Implementation details:**
1. Count main queue: `LLEN queue_name`
2. Scan for priority queues: `SCAN` with pattern `queue_name*`
3. Filter keys containing `\x06\x16` separator (Celery priority marker)
4. Skip main queue when counting priority keys
5. Handle both string and bytes key types
6. Added debug and info logging

**Files modified:**
- `services/admin_service.py`:
  - Replaced `_scan_priority_keys()` function
  - Rewrote `queue_depth()` function with accurate Celery queue detection
  - Added comprehensive logging

**Result:**
- Accurate queue depth counts for both main and priority queues
- Better visibility into queue state via logs
- Handles edge cases (empty queues, Redis errors)

---

## Testing

### Test Admin API Key

```bash
# Without API key (should fail with 401)
curl http://localhost:8000/api/admin/metrics
# Expected: {"detail":"Invalid API key"} or header missing error

# With valid API key (should succeed)
curl -H "X-API-Key: your_api_key_here" \
  http://localhost:8000/api/admin/metrics
# Expected: JSON with metrics data

# With invalid API key (should fail with 401)
curl -H "X-API-Key: wrong_key" \
  http://localhost:8000/api/admin/metrics
# Expected: {"detail":"Invalid API key"}
```

### Test Queue Depth

```bash
# View logs to see queue depth calculation
docker compose logs backend | grep "Queue"

# Check admin metrics (shows queue depths)
curl -H "X-API-Key: your_api_key_here" \
  http://localhost:8000/api/admin/metrics | jq '.queues'

# Expected output:
# {
#   "analyze": 5,
#   "deep_scan": 2
# }
```

---

## Migration Steps

### For Existing Deployments

1. **Update code:**
   ```bash
   cd ~/scrollsafe-backend
   git pull origin main
   ```

2. **Generate admin API key:**
   ```bash
   openssl rand -hex 32
   # Save the output
   ```

3. **Update .env:**
   ```bash
   nano .env
   # Add: ADMIN_API_KEY=your_generated_key_here
   ```

4. **Rebuild and restart:**
   ```bash
   docker compose build backend
   docker compose restart backend
   ```

5. **Update admin dashboard/scripts:**
   - Add `X-API-Key` header to all admin API calls
   - Update any scripts that call `/api/admin/*` endpoints

6. **Verify:**
   ```bash
   # Test with API key
   curl -H "X-API-Key: your_key" http://localhost:8000/api/admin/metrics

   # Check logs for queue depth
   docker compose logs backend | tail -20
   ```

---

## Security Considerations

### Admin API Key

**Best practices:**
- Use a strong, randomly generated key (32+ bytes)
- Never commit API key to git
- Store in environment variables or secrets manager
- Rotate key periodically (monthly recommended)
- Use different keys for dev/staging/production
- Revoke and regenerate if compromised

**Access control:**
- API key grants full admin access (metrics + label management)
- Consider implementing role-based keys if multiple admins
- Log all admin API access for audit trail

### Queue Depth Logging

- Logs include queue names and depths
- May reveal system load patterns
- Consider log rotation and retention policies
- INFO level logs in production, DEBUG in development

---

## Rollback

If issues occur:

```bash
# Revert to previous version
cd ~/scrollsafe-backend
git checkout <previous-commit-hash>
docker compose build backend
docker compose restart backend

# Or temporarily disable API key requirement:
# In main.py, remove `Depends(verify_admin_api_key)` from routes
```

---

## Files Changed

- ✅ `scrollsafe-backend/main.py` - Added API key authentication
- ✅ `scrollsafe-backend/services/admin_service.py` - Fixed queue depth calculation
- ✅ `scrollsafe-backend/.env.example` - Added ADMIN_API_KEY
- ✅ `scrollsafe-backend/DEPLOYMENT.md` - Documented API key usage
- ✅ `scrollsafe-backend/SECURITY_CHANGES.md` - This file

---

## Summary

**Admin routes are now secure with API key authentication.**
**Queue depth calculation is now accurate for Celery queues.**

Both changes are production-ready and safe to deploy.
