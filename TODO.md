# SecurityGuardVivint TODO List

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Completed

---

## 1. Housekeeping

### 1.1 Commit Utility Scripts
- [ ] Add `manual_sync.py` - Manual GCP sync trigger script
- [ ] Add `query_bq.py` - BigQuery query utility
- [ ] Add `test_gcs_upload.py` - GCS upload test script
- [ ] Clean up or commit documentation files (COMMAND_REFERENCE.md, DEPLOYMENT_SUMMARY.md, etc.)

---

## 2. Automation

### 2.1 Hourly GCP Sync
- [ ] Create systemd timer for hourly sync to BigQuery and GCS
- [ ] Timer runs `manual_sync.py` inside container
- [ ] Log sync results

### 2.2 Container Auto-Restart on VM Reboot
- [ ] Enable podman systemd integration
- [ ] Generate systemd unit for security-guard container
- [ ] Enable unit to start on boot

### 2.3 Smart Container Auto-Update from Git
**Requirements:**
- Minimize downtime (only restart if changes detected)
- Skip rebuild if no changes in repo
- Pull latest from main branch
- Rebuild container only when necessary
- Graceful container restart

**Implementation:**
- [ ] Create `/opt/security-guard/update-container.sh` script
- [ ] Script checks `git fetch && git diff HEAD origin/main`
- [ ] Only rebuild if changes detected
- [ ] Use `podman-compose pull && podman-compose up -d` for minimal downtime
- [ ] Log all update activities

### 2.4 Weekly VM & Container Updates (Sunday Evening)
- [ ] Create systemd timer for Sunday 8 PM local time
- [ ] Update VM packages (`dnf update -y`)
- [ ] Pull latest container code and rebuild if needed
- [ ] Reboot VM if kernel updated
- [ ] Log all activities

---

## 3. Monitoring & Health Checks

### 3.1 Container Health Check Endpoint
Create `/health` endpoint that checks ALL services:

**External Services:**
- [ ] Vivint API connectivity (can reach api.vivint.com)
- [ ] GCP BigQuery connectivity (can query dataset)
- [ ] GCP Cloud Storage connectivity (can list bucket)
- [ ] Google AI Studio / Gemini API connectivity
- [ ] Pushover API connectivity
- [ ] Vivint Hub/Panel reachability (local network RTSP)

**Internal Checks:**
- [ ] PubNub subscription active
- [ ] SQLite buffer accessible
- [ ] Frame capture directory writable
- [ ] Token file valid/not expired

**Error Detection:**
- [ ] Track HTTP status codes 400-599 from any service
- [ ] Track authentication errors from any service
- [ ] Track Python exceptions/errors in execution
- [ ] Track rate limiting from any API

### 3.2 Podman Healthcheck Configuration
- [ ] Add HEALTHCHECK to Dockerfile
- [ ] Configure healthcheck interval (30s)
- [ ] Configure healthcheck timeout (10s)
- [ ] Configure unhealthy threshold (3 failures)
- [ ] Update docker-compose.yml with healthcheck settings

### 3.3 VM Resource Monitoring
- [ ] Monitor disk space (alert if < 5GB free)
- [ ] Monitor CPU usage (alert if sustained > 90%)
- [ ] Monitor memory usage (alert if sustained > 90%)
- [ ] Create monitoring script that runs via cron/systemd timer
- [ ] Send alerts via Pushover when thresholds exceeded

### 3.4 Alerting Configuration
- [ ] Alert on container health check failures
- [ ] Alert on container restart/crash
- [ ] Alert on GCP sync failures
- [ ] Alert on VM resource threshold breaches
- [ ] Alert if no events received for extended period (optional)
- [ ] All alerts go through Pushover

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
- [ ] Verify driveway camera receives motion events
- [ ] All cameras connected to Vivint hub should work automatically
- [ ] Motion events come through PubNub for all cameras

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
- [ ] Full end-to-end test: motion -> analysis -> notification -> sync -> query

---

## Implementation Order

1. **Phase 1: Housekeeping** (commit scripts)
2. **Phase 2: Health Check Endpoint** (needed for monitoring)
3. **Phase 3: Automation Scripts** (sync timer, auto-restart, auto-update)
4. **Phase 4: VM Monitoring** (resource checks, alerts)
5. **Phase 5: Podman Integration** (healthcheck, systemd)
6. **Phase 6: Testing & Verification**

---

## Notes

- VM IP: Accessible via Tailscale at `security-guard-vm`
- Container runs with `network_mode: host` for GCP metadata access
- All secrets via environment variables (no hardcoded credentials)
- Pushover used for all alerting (already configured)
