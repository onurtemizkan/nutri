# Task ID: 25

**Title:** Create Database Migration Workflow

**Status:** done

**Dependencies:** 21

**Priority:** high

**Description:** Implement a safe database migration process that runs Prisma migrations as part of deployment with backup and rollback capabilities.

**Details:**

**Create `.github/workflows/migrate.yml`:**
```yaml
name: Database Migration

on:
  workflow_dispatch:
    inputs:
      dry_run:
        description: 'Dry run (show changes without applying)'
        required: true
        default: 'true'
        type: choice
        options:
          - 'true'
          - 'false'

jobs:
  migrate:
    name: Run Migrations
    runs-on: ubuntu-latest
    environment: production  # Requires approval

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: cd server && npm ci

      - name: Generate Prisma client
        run: cd server && npx prisma generate

      - name: Show pending migrations (Dry Run)
        if: inputs.dry_run == 'true'
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          cd server
          npx prisma migrate status
          echo "---"
          echo "Pending migrations preview:"
          npx prisma migrate diff --from-migrations ./prisma/migrations --to-schema-datamodel ./prisma/schema.prisma

      - name: Run migrations
        if: inputs.dry_run == 'false'
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          cd server
          echo "Running migrations..."
          npx prisma migrate deploy
          echo "Migration complete!"
          npx prisma migrate status

      - name: Notify on success
        if: success() && inputs.dry_run == 'false'
        run: |
          curl -X POST "${{ secrets.DISCORD_WEBHOOK_URL }}" \
            -H "Content-Type: application/json" \
            -d '{"content": "✅ Database migration completed successfully"}' || true

      - name: Notify on failure
        if: failure()
        run: |
          curl -X POST "${{ secrets.DISCORD_WEBHOOK_URL }}" \
            -H "Content-Type: application/json" \
            -d '{"content": "❌ Database migration failed! Check GitHub Actions logs."}' || true
```

**Create migration helper script `scripts/deploy/migrate.sh`:**
```bash
#!/bin/bash
set -e

echo "=== Nutri Database Migration ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL environment variable is not set"
    exit 1
fi

# Navigate to server directory
cd "$(dirname "$0")/../../server"

# Show current status
echo "Current migration status:"
npx prisma migrate status
echo ""

# Parse arguments
DRY_RUN=true
if [ "$1" = "--apply" ]; then
    DRY_RUN=false
fi

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - showing changes without applying"
    echo "Run with --apply to execute migrations"
    echo ""
    echo "Pending changes:"
    npx prisma migrate diff --from-migrations ./prisma/migrations --to-schema-datamodel ./prisma/schema.prisma || true
else
    echo "APPLYING MIGRATIONS..."
    npx prisma migrate deploy
    echo ""
    echo "Migration complete. New status:"
    npx prisma migrate status
fi
```

**Update deploy workflow to include migration step:**
Add before deployment triggers:
```yaml
      - name: Run database migrations
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          cd server
          npm ci
          npx prisma generate
          npx prisma migrate deploy
```

**Test Strategy:**

1. Test dry run mode shows pending migrations without applying
2. Test migrate deploy in staging environment
3. Verify migration rollback with Prisma migrate down (manual)
4. Test migration failure handling (create intentionally failing migration)
5. Verify environment protection requires approval
6. Test notification webhooks
7. Run migration with no pending changes (should be no-op)

## Subtasks

### 25.1. Create GitHub Actions migrate.yml workflow file

**Status:** pending  
**Dependencies:** None  

Create `.github/workflows/migrate.yml` with workflow_dispatch trigger, dry_run input parameter (choice: true/false), and production environment configuration.

**Details:**

Set up the workflow structure with:
- `workflow_dispatch` trigger with `dry_run` input (boolean choice, default: 'true')
- Job named `migrate` running on `ubuntu-latest`
- Environment set to `production` for approval requirement
- Steps for checkout, Node.js 20 setup with npm cache
- Dependencies installation: `cd server && npm ci`
- Prisma client generation: `cd server && npx prisma generate`

Reference existing workflows in `.github/workflows/` for consistency with project patterns. Ensure DATABASE_URL is read from secrets.

### 25.2. Implement dry run mode in migrate workflow

**Status:** pending  
**Dependencies:** 25.1  

Add workflow steps that show pending migrations without applying them when dry_run=true using Prisma migrate status and migrate diff commands.

**Details:**

Add conditional step with `if: inputs.dry_run == 'true'`:
- Run `npx prisma migrate status` to show current migration state
- Run `npx prisma migrate diff --from-migrations ./prisma/migrations --to-schema-datamodel ./prisma/schema.prisma` to preview pending changes
- Include clear output formatting with echo statements
- Set DATABASE_URL from secrets: `${{ secrets.DATABASE_URL }}`

This step must execute in the `server` directory and handle cases where no migrations are pending gracefully.

### 25.3. Implement migration execution step with error handling

**Status:** pending  
**Dependencies:** 25.2  

Add workflow step that runs actual migrations when dry_run=false using Prisma migrate deploy with proper error handling and status reporting.

**Details:**

Add conditional step with `if: inputs.dry_run == 'false'`:
- Run `npx prisma migrate deploy` to apply migrations
- Include pre-execution echo: "Running migrations..."
- Post-execution confirmation: "Migration complete!"
- Run `npx prisma migrate status` after deployment to verify
- Set DATABASE_URL from secrets
- Ensure proper error propagation (set -e behavior)

Execute in `server` directory. The step should fail the workflow if migration fails, triggering failure notifications.

### 25.4. Add Discord notification steps to migrate workflow

**Status:** pending  
**Dependencies:** 25.3  

Implement success and failure notification steps that send messages to Discord webhook for migration results.

**Details:**

Add two notification steps:

**Success notification** (`if: success() && inputs.dry_run == 'false'`):
- POST to `${{ secrets.DISCORD_WEBHOOK_URL }}`
- Message: "✅ Database migration completed successfully"
- Include `|| true` to prevent notification failures from failing workflow

**Failure notification** (`if: failure()`):
- POST to `${{ secrets.DISCORD_WEBHOOK_URL }}`
- Message: "❌ Database migration failed! Check GitHub Actions logs."
- Include workflow run URL if possible
- Include `|| true` for graceful degradation

Use curl with Content-Type: application/json.

### 25.5. Create migration helper script scripts/deploy/migrate.sh

**Status:** pending  
**Dependencies:** None  

Create bash script for manual migration execution with dry-run mode, status checking, and proper error handling for local and CI use.

**Details:**

Create `scripts/deploy/migrate.sh`:
- Shebang: `#!/bin/bash` with `set -e`
- Header with timestamp: `date -u +%Y-%m-%dT%H:%M:%SZ`
- Check DATABASE_URL is set (exit 1 if missing)
- Navigate to server directory: `cd "$(dirname "$0")/../../server"`
- Show current status: `npx prisma migrate status`
- Parse `--apply` argument (default: dry run)
- Dry run: show pending changes with `prisma migrate diff`
- Apply mode: run `npx prisma migrate deploy` and show final status
- Make executable: `chmod +x scripts/deploy/migrate.sh`

Follows pattern from deployment-infrastructure-prd.md example.

### 25.6. Integrate migration step into deploy workflow

**Status:** pending  
**Dependencies:** 25.1, 25.3, 25.5  

Add database migration execution to the deployment workflow (.github/workflows/deploy.yml) before application deployment triggers.

**Details:**

Update `.github/workflows/deploy.yml` (created in task 24):
- Add migration step after checkout/setup, before Coolify webhook
- Position: after Node.js setup, before deployment trigger
- Step name: "Run database migrations"
- Set DATABASE_URL from secrets
- Commands:
  ```
  cd server
  npm ci
  npx prisma generate
  npx prisma migrate deploy
  ```
- Ensure this step fails the workflow if migrations fail
- Add comment explaining this runs before deployment

Reference task 24 implementation for integration point.
