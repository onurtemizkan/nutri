# Task ID: 33

**Title:** Configure Hetzner Server Snapshots

**Status:** pending

**Dependencies:** 28 ✓

**Priority:** medium

**Description:** Enable and configure Hetzner automatic server snapshots for disaster recovery with documentation for restore procedures.

**Details:**

**Enable Hetzner Backups via Console:**

1. Log into [Hetzner Cloud Console](https://console.hetzner.cloud/)
2. Select your server
3. Go to "Backups" tab
4. Click "Enable Backups"
   - Cost: ~20% of server price (€1.10/mo for CX32)
   - Schedule: Daily automatic backups
   - Retention: Last 7 daily + 4 weekly backups

**Or via Hetzner CLI/API:**
```bash
# Install hcloud CLI
brew install hcloud

# Configure
hcloud context create nutri
# Enter API token from Hetzner Console > Security > API Tokens

# Enable backups on server
hcloud server enable-backup <server-name>
```

**Create Manual Snapshot Script `scripts/deploy/create-snapshot.sh`:**
```bash
#!/bin/bash
set -e

# =============================================================================
# Create Hetzner Server Snapshot
# Use before major deployments or configuration changes
# =============================================================================

SERVER_NAME="${SERVER_NAME:-nutri-production}"
SNAPSHOT_DESC="${1:-Manual snapshot $(date +%Y-%m-%d_%H%M)}"

echo "=== Creating Hetzner Snapshot ==="
echo "Server: $SERVER_NAME"
echo "Description: $SNAPSHOT_DESC"
echo ""

# Check if hcloud is installed
if ! command -v hcloud &> /dev/null; then
    echo "Error: hcloud CLI not installed"
    echo "Install with: brew install hcloud"
    exit 1
fi

# Create snapshot
echo "Creating snapshot (this may take a few minutes)..."
hcloud server create-image --type snapshot --description "$SNAPSHOT_DESC" "$SERVER_NAME"

echo ""
echo "=== Snapshot Created ==="
echo ""
echo "Available snapshots:"
hcloud image list --type snapshot
```

**Document Restore Procedure in `docs/deployment/DISASTER-RECOVERY.md`:**
```markdown
# Disaster Recovery

## Server Snapshot Recovery

### When to Use
- Server is unrecoverable
- Corrupted system files
- Major misconfiguration

### Restore Steps

1. **Via Hetzner Console:**
   - Go to Images > Snapshots
   - Select snapshot to restore
   - Click "Create Server from Image"
   - Configure same specs as original server
   - Update DNS to point to new server IP

2. **Via CLI:**
   ```bash
   # List available snapshots
   hcloud image list --type snapshot
   
   # Create new server from snapshot
   hcloud server create \
     --name nutri-production-restored \
     --type cx32 \
     --image <snapshot-id> \
     --location nbg1
   
   # Update DNS
   # ... update A record to new IP
   ```

3. **Post-Restore Checklist:**
   - [ ] Verify server accessible via SSH
   - [ ] Check Coolify dashboard
   - [ ] Verify all containers running
   - [ ] Test health endpoints
   - [ ] Update DNS if IP changed
   - [ ] Update GitHub secrets with new webhook URLs

### Recovery Time Estimate

| Step | Duration |
|------|----------|
| Create server from snapshot | 5-10 min |
| DNS propagation | 5-30 min |
| Verify services | 5 min |
| **Total** | **15-45 min** |

## Monthly Recovery Test

Schedule monthly recovery drill:
1. Create snapshot of production
2. Spin up test server from snapshot
3. Verify services work
4. Document any issues
5. Delete test server
```

**Test Strategy:**

1. Enable backups in Hetzner Console
2. Create manual snapshot using script
3. Verify snapshot appears in console
4. Test restore by creating new server from snapshot
5. Verify restored server has correct configuration
6. Document actual recovery time
7. Delete test server after verification
8. Set up monthly recovery test reminder
