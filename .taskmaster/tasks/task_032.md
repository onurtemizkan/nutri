# Task ID: 32

**Title:** Create Database Backup Script

**Status:** pending

**Dependencies:** None

**Priority:** medium

**Description:** Create a backup script for database exports with support for S3-compatible storage, retention policies, and restore procedures for the Supabase free tier.

**Details:**

**Note:** Supabase Pro tier includes automatic backups. This script is for free tier users.

**Create `scripts/deploy/backup-database.sh`:**
```bash
#!/bin/bash
set -e

# =============================================================================
# Nutri Database Backup Script
# For use with Supabase free tier (no automatic backups)
# =============================================================================

echo "=== Nutri Database Backup ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
S3_BUCKET="${S3_BUCKET:-}"
S3_ENDPOINT="${S3_ENDPOINT:-}"

# Validate required environment variables
if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL is required"
    exit 1
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate backup filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="nutri_backup_${TIMESTAMP}.sql.gz"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILE"

# Extract connection details from DATABASE_URL
# Format: postgresql://user:password@host:port/database
proto="$(echo $DATABASE_URL | grep :// | sed -e's,^\(.*://\).*,\1,g')"
url="$(echo ${DATABASE_URL/$proto/})"
user="$(echo $url | grep @ | cut -d@ -f1 | cut -d: -f1)"
password="$(echo $url | grep @ | cut -d@ -f1 | cut -d: -f2)"
hostport="$(echo ${url/$user:$password@/} | cut -d/ -f1)"
host="$(echo $hostport | cut -d: -f1)"
port="$(echo $hostport | cut -d: -f2)"
database="$(echo $url | grep / | cut -d/ -f2-)"

echo "Backing up database: $database on $host"

# Create backup with pg_dump
export PGPASSWORD="$password"
pg_dump -h "$host" -p "$port" -U "$user" -d "$database" \
    --format=custom \
    --no-owner \
    --no-acl \
    --verbose 2>&1 | gzip > "$BACKUP_PATH"

BACKUP_SIZE=$(ls -lh "$BACKUP_PATH" | awk '{print $5}')
echo "Backup created: $BACKUP_PATH ($BACKUP_SIZE)"

# Upload to S3 if configured
if [ -n "$S3_BUCKET" ]; then
    echo "Uploading to S3: $S3_BUCKET"
    
    S3_PATH="s3://$S3_BUCKET/nutri/backups/$BACKUP_FILE"
    
    if [ -n "$S3_ENDPOINT" ]; then
        # For S3-compatible storage (Backblaze B2, MinIO, etc.)
        aws s3 cp "$BACKUP_PATH" "$S3_PATH" --endpoint-url "$S3_ENDPOINT"
    else
        aws s3 cp "$BACKUP_PATH" "$S3_PATH"
    fi
    
    echo "Uploaded to: $S3_PATH"
    
    # Remove local file after S3 upload
    rm "$BACKUP_PATH"
    echo "Local backup removed (stored in S3)"
fi

# Clean up old backups
echo "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "nutri_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true

# S3 lifecycle should handle S3 cleanup, but we can list what's there
if [ -n "$S3_BUCKET" ]; then
    echo "Current S3 backups:"
    aws s3 ls "s3://$S3_BUCKET/nutri/backups/" ${S3_ENDPOINT:+--endpoint-url $S3_ENDPOINT} | tail -5
fi

echo ""
echo "=== Backup Complete ==="
```

**Create `scripts/deploy/restore-database.sh`:**
```bash
#!/bin/bash
set -e

# =============================================================================
# Nutri Database Restore Script
# =============================================================================

echo "=== Nutri Database Restore ==="
echo "WARNING: This will overwrite the target database!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled."
    exit 1
fi

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file.sql.gz>"
    echo ""
    echo "Available backups:"
    ls -lh /backups/nutri_backup_*.sql.gz 2>/dev/null || echo "No local backups found"
    exit 1
fi

if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL is required"
    exit 1
fi

# Parse DATABASE_URL (same as backup script)
# ... [parsing code] ...

echo "Restoring from: $BACKUP_FILE"
echo "Target database: $database on $host"

export PGPASSWORD="$password"

# Restore
gunzip -c "$BACKUP_FILE" | pg_restore \
    -h "$host" -p "$port" -U "$user" -d "$database" \
    --no-owner \
    --no-acl \
    --clean \
    --if-exists \
    --verbose

echo "=== Restore Complete ==="
```

**Create cron job for automated backups:**
```bash
# Add to server crontab
0 3 * * * /opt/nutri/scripts/backup-database.sh >> /var/log/nutri-backup.log 2>&1
```

**Document in `docs/deployment/BACKUPS.md`:**
```markdown
# Backup and Recovery

## Automatic Backups

- **Supabase Pro:** Automatic daily backups included
- **Supabase Free:** Use `scripts/deploy/backup-database.sh` with cron

## Manual Backup

```bash
export DATABASE_URL="postgresql://..."
./scripts/deploy/backup-database.sh
```

## Restore Procedure

1. Download backup file
2. Run restore script:
   ```bash
   ./scripts/deploy/restore-database.sh /path/to/backup.sql.gz
   ```
3. Verify data integrity
4. Run migrations if needed: `npx prisma migrate deploy`
```

**Test Strategy:**

1. Run backup script against dev database
2. Verify backup file created and valid: `gunzip -t backup.sql.gz`
3. Test restore to empty test database
4. Verify data integrity after restore
5. Test S3 upload with Backblaze B2 or MinIO
6. Verify retention policy deletes old files
7. Test cron job execution
8. Document recovery time (RTO target: <30 min)

## Subtasks

### 32.1. Create backup-database.sh script with pg_dump and compression

**Status:** pending  
**Dependencies:** None  

Create the main database backup script at scripts/deploy/backup-database.sh with pg_dump, gzip compression, and configurable retention policy

**Details:**

Create scripts/deploy/backup-database.sh implementing:

1. Environment variable parsing:
   - DATABASE_URL (required) - PostgreSQL connection string
   - BACKUP_DIR (default: /backups) - Local backup storage
   - RETENTION_DAYS (default: 7) - Days to keep backups

2. Database URL parsing to extract host, port, user, password, database from postgresql://user:password@host:port/database format

3. Core backup functionality:
   - Use pg_dump with --format=custom for efficient restore
   - Add --no-owner and --no-acl flags (Supabase uses pooler roles)
   - Pipe through gzip for compression
   - Generate timestamped filename: nutri_backup_YYYYMMDD_HHMMSS.sql.gz

4. Local retention cleanup:
   - Use find -mtime +$RETENTION_DAYS to delete old backups
   - Display backup size after creation

5. Error handling:
   - Exit on error (set -e)
   - Validate required DATABASE_URL
   - Show helpful error messages

6. Follow existing script patterns from scripts/deploy/migrate.sh:
   - Color-coded output (RED/GREEN/YELLOW/BLUE)
   - Timestamp header
   - Clear section organization

### 32.2. Add S3/B2 cloud storage upload support to backup script

**Status:** pending  
**Dependencies:** 32.1  

Extend backup-database.sh to support uploading backups to S3-compatible storage providers (AWS S3, Backblaze B2, MinIO)

**Details:**

Extend scripts/deploy/backup-database.sh with cloud upload:

1. Add environment variables:
   - S3_BUCKET - Bucket name (enables upload when set)
   - S3_ENDPOINT - Custom endpoint for B2/MinIO (optional)
   - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY - Standard AWS env vars

2. Implement upload logic:
   - Upload to s3://$S3_BUCKET/nutri/backups/$BACKUP_FILE
   - Use --endpoint-url flag for S3-compatible providers
   - Remove local file after successful S3 upload
   - List recent S3 backups (tail -5) after upload

3. Provider-specific notes:
   - AWS S3: Use default endpoint
   - Backblaze B2: Set S3_ENDPOINT=https://s3.us-west-000.backblazeb2.com
   - MinIO: Set S3_ENDPOINT to MinIO server URL

4. S3 lifecycle policy recommendation:
   - Document that S3 lifecycle rules should handle cloud retention
   - Suggest 30-day retention policy in S3 bucket settings

5. Error handling:
   - Verify aws CLI is available before upload
   - Keep local backup if S3 upload fails
   - Clear error messages for S3 configuration issues

### 32.3. Implement restore-database.sh script with safety checks

**Status:** pending  
**Dependencies:** 32.1  

Create the database restore script at scripts/deploy/restore-database.sh with interactive confirmation, validation, and clear restore process

**Details:**

Create scripts/deploy/restore-database.sh implementing:

1. Safety features:
   - Interactive confirmation prompt (yes/no)
   - Clear warning about data overwrite
   - Display target database name before restore
   - Option to download from S3 if backup file path starts with s3://

2. Command-line interface:
   - Usage: ./restore-database.sh <backup-file.sql.gz>
   - Show available local backups when no argument provided
   - Support both .sql.gz (compressed) and .sql (uncompressed) files

3. Restore implementation:
   - Parse DATABASE_URL same as backup script
   - Use pg_restore for --format=custom backups
   - Add --clean --if-exists for safe overwrite
   - Add --no-owner --no-acl for Supabase compatibility
   - Use --verbose for progress output

4. S3 restore support:
   - Detect s3:// paths and download first
   - Use aws s3 cp with --endpoint-url if S3_ENDPOINT set
   - Clean up downloaded file after restore

5. Post-restore guidance:
   - Remind to verify data integrity
   - Suggest running migrations: npx prisma migrate deploy
   - Show basic verification queries

### 32.4. Document backup procedures and set up cron job configuration

**Status:** pending  
**Dependencies:** 32.1, 32.2, 32.3  

Create comprehensive backup documentation at docs/deployment/BACKUPS.md and provide cron job examples for automated backups

**Details:**

Create docs/deployment/BACKUPS.md with:

1. Overview section:
   - Explain Supabase Pro vs Free tier backup differences
   - Link to Supabase backup documentation
   - When to use this script (free tier, custom requirements)

2. Prerequisites:
   - Required tools: pg_dump, gzip, aws CLI (for S3)
   - Environment variables reference
   - S3 bucket setup for different providers

3. Manual backup guide:
   - Step-by-step commands
   - Example with local storage
   - Example with S3/B2 upload

4. Automated backup setup:
   - Cron job example: 0 3 * * * /opt/nutri/scripts/deploy/backup-database.sh >> /var/log/nutri-backup.log 2>&1
   - Systemd timer alternative (more modern)
   - Log rotation configuration

5. Restore procedures:
   - Emergency restore checklist
   - Restore from local backup
   - Restore from S3
   - Post-restore verification steps
   - Migration after restore

6. Update existing docs:
   - Add link from docs/deployment/RUNBOOK.md Backup & Recovery section
   - Update docs/deployment/README.md quick links

7. Backup verification:
   - Monthly backup test procedure
   - Data integrity verification queries
