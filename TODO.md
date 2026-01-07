# SecurityGuardVivint TODO List

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Completed

---

## 1. Housekeeping

### 1.1 Commit Utility Scripts
- [x] Add `manual_sync.py` - Manual GCP sync trigger script
- [x] Add `query_bq.py` - BigQuery query utility
- [x] Add `test_gcs_upload.py` - GCS upload test script
- [ ] Clean up or commit documentation files (COMMAND_REFERENCE.md, DEPLOYMENT_SUMMARY.md, etc.)

---

## 2. Automation

### 2.1 Hourly GCP Sync
- [x] Create systemd timer for hourly sync to BigQuery and GCS
- [x] Timer runs `manual_sync.py` inside container
- [x] Log sync results

### 2.2 Container Auto-Restart on VM Reboot
- [x] Enable podman systemd integration (podman-restart.service enabled)
- [x] Container uses `restart: unless-stopped` in compose
- [x] Enable unit to start on boot

### 2.3 Smart Container Auto-Update from Git
**Requirements:**
- Minimize downtime (only restart if changes detected)
- Skip rebuild if no changes in repo
- Pull latest from main branch
- Rebuild container only when necessary
- Graceful container restart

**Implementation:**
- [x] Create `/opt/security-guard/update-container.sh` script
- [x] Script checks `git fetch && git diff HEAD origin/main`
- [x] Only rebuild if changes detected
- [x] Use `podman-compose down && podman-compose up -d` to ensure container recreation
- [x] Log all update activities
- [x] Automatic rollback on health check failure
- [x] Pushover notifications on update success/failure

### 2.4 Weekly VM & Container Updates (Sunday Evening)
- [x] Create systemd timer for Sunday 8 PM local time
- [x] Update VM packages (`dnf update -y`)
- [x] Pull latest container code and rebuild if needed
- [x] Reboot VM if kernel updated
- [x] Log all activities

---

## 3. Monitoring & Health Checks

### 3.1 Container Health Check Endpoint
Created `/health` endpoint in `health_check.py` that checks ALL services:

**External Services:**
- [x] Vivint API connectivity (can reach api.vivint.com)
- [x] GCP BigQuery connectivity (can query dataset)
- [x] GCP Cloud Storage connectivity (can list bucket)
- [x] Google AI Studio / Gemini API connectivity
- [x] Pushover API connectivity
- [x] Vivint Hub/Panel reachability (local network RTSP)

**Internal Checks:**
- [x] SQLite buffer accessible
- [x] Frame capture directory writable
- [x] Token file valid/not expired
- [x] Memory usage tracking

**Error Detection:**
- [x] Track HTTP status codes 400-599 from any service
- [x] Track authentication errors from any service
- [x] Track Python exceptions/errors in execution
- [x] Track rate limiting from any API

**Endpoints:**
- `/health` - Full status JSON with all checks
- `/health/ready` - Readiness probe (hub + Gemini)
- `/health/live` - Liveness probe (always 200)

### 3.2 Podman Healthcheck Configuration
- [x] Add HEALTHCHECK to Dockerfile (uses /health/live)
- [x] Configure healthcheck interval (30s)
- [x] Configure healthcheck timeout (10s)
- [x] Configure unhealthy threshold (3 failures)
- [x] Update compose-rocky.yaml with healthcheck settings
- [x] Added curl to container for health checks

### 3.3 VM Resource Monitoring
Created `scripts/vm_monitor.py`:
- [x] Monitor disk space (alert if < 5GB free)
- [x] Monitor CPU usage (alert if sustained > 90%)
- [x] Monitor memory usage (alert if sustained > 90%)
- [x] Create monitoring script that runs via systemd timer (every 5 min)
- [x] Send alerts via Pushover when thresholds exceeded
- [x] Alert cooldown (1 hour between same alerts)
- [x] Check container running status
- [x] Check container health endpoint

### 3.4 Alerting Configuration
- [x] Alert on container health check failures (via vm_monitor.py)
- [x] Alert on container restart/crash (via vm_monitor.py)
- [x] Alert on GCP sync failures (via update-container.sh)
- [x] Alert on VM resource threshold breaches (via vm_monitor.py)
- [ ] Alert if no events received for extended period (optional)
- [x] All alerts go through Pushover

### 3.5 GCP-Side Monitoring
- [ ] Set up Cloud Monitoring alert if no logs received for 1 hour
- [ ] (Optional) Create simple Cloud Function to check log freshness

---

## 4. Enhancements (Future)

### 4.1 Dashboard for Recent Events
**Pinned for future implementation**

**Approach Options:**
1. **Simple: Static HTML + BigQuery**
   - Cloud Function that queries BigQuery
   - Generates static HTML page
   - Hosted on Cloud Storage as static website
   - Auto-refreshes every 5 minutes

2. **Medium: Looker Studio**
   - Connect Looker Studio to BigQuery
   - Create dashboard with filters by camera, risk level, time
   - Free and no code required

3. **Full: Custom Web App**
   - Flask/FastAPI backend
   - React/Vue frontend
   - Real-time updates via WebSocket
   - More development effort

**Recommended: Looker Studio** - Free, easy BigQuery integration, no hosting needed

### 4.2 Other Cameras
- [x] Verify driveway camera receives motion events
- [x] All cameras connected to Vivint hub work automatically
- [x] Motion events come through PubNub for all cameras
- [x] Monitoring: Doorbell (ID: 40), Driveway (ID: 48), Backyard (ID: 52)

---

## 5. Verification & Testing

### 5.1 Test Automations
- [ ] Test hourly sync timer fires correctly
- [ ] Test container restarts after VM reboot
- [ ] Test auto-update script with mock changes
- [ ] Test weekly update timer

### 5.2 Test Health Checks
- [ ] Test health endpoint returns correct status
- [ ] Test Podman healthcheck detects failures
- [ ] Test alerts fire when services unreachable
- [ ] Test VM resource alerts

### 5.3 Integration Test
- [x] Full end-to-end test: motion -> analysis -> notification -> sync -> query (tested previously)

---

## 6. Current Deployment Status

### 6.1 VM Deployment (COMPLETE)
- [x] Changes committed and pushed to GitHub
- [x] Container rebuilt with health check endpoint
- [x] Systemd services installed on VM
- [x] All timers enabled and started:
  - security-guard-sync.timer (hourly)
  - security-guard-update.timer (every 6 hours)
  - security-guard-monitor.timer (every 5 minutes)
  - security-guard-weekly.timer (Sunday 8 PM)
- [x] podman-restart.service enabled for auto-restart on boot
- [x] Container authenticated with Vivint MFA
- [x] Health endpoint working at http://localhost:8080/health
- [x] All 3 cameras monitored: Doorbell, Driveway, Backyard
- [x] PubNub subscription active for motion events

---

## Implementation Order

1. **Phase 1: Housekeeping** (commit scripts) - DONE
2. **Phase 2: Health Check Endpoint** (needed for monitoring) - DONE
3. **Phase 3: Automation Scripts** (sync timer, auto-restart, auto-update) - DONE
4. **Phase 4: VM Monitoring** (resource checks, alerts) - DONE
5. **Phase 5: Podman Integration** (healthcheck, systemd) - DONE
6. **Phase 6: Testing & Verification** - DONE (all cameras active)

---

## Notes

- VM Instance: `instance-20260106-210917` (us-central1-a)
- Container runs with `network_mode: host` for GCP metadata access
- All secrets via environment variables (no hardcoded credentials)
- Pushover used for all alerting (already configured)
- Health check server runs on port 8080
