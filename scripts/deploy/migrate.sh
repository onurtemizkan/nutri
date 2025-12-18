#!/bin/bash
# =============================================================================
# Nutri Database Migration Script
# Safe database migration helper for manual execution
# =============================================================================
#
# Usage:
#   ./scripts/deploy/migrate.sh              # Dry run (show pending changes)
#   ./scripts/deploy/migrate.sh --apply      # Apply migrations
#   ./scripts/deploy/migrate.sh --status     # Show migration status only
#
# Environment Variables:
#   DATABASE_URL    Required. PostgreSQL connection string.
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Header
echo ""
echo -e "${BLUE}=== Nutri Database Migration ===${NC}"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo -e "${RED}Error: DATABASE_URL environment variable is not set${NC}"
    echo ""
    echo "Set it with:"
    echo "  export DATABASE_URL=\"postgresql://user:password@host:port/database\""
    echo ""
    exit 1
fi

# Navigate to server directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/../../server"

if [ ! -d "$SERVER_DIR" ]; then
    echo -e "${RED}Error: Server directory not found at $SERVER_DIR${NC}"
    exit 1
fi

cd "$SERVER_DIR"

# Parse arguments
MODE="dry_run"
if [ "$1" = "--apply" ]; then
    MODE="apply"
elif [ "$1" = "--status" ]; then
    MODE="status"
elif [ -n "$1" ] && [ "$1" != "--dry-run" ]; then
    echo -e "${YELLOW}Unknown option: $1${NC}"
    echo ""
    echo "Usage:"
    echo "  ./migrate.sh              # Dry run (default)"
    echo "  ./migrate.sh --apply      # Apply migrations"
    echo "  ./migrate.sh --status     # Show status only"
    exit 1
fi

# Show current status
echo -e "${BLUE}Current migration status:${NC}"
echo "---"
npx prisma migrate status 2>&1 || true
echo "---"
echo ""

# Execute based on mode
case $MODE in
    "status")
        echo -e "${GREEN}Status check complete.${NC}"
        ;;

    "dry_run")
        echo -e "${YELLOW}DRY RUN MODE${NC} - showing changes without applying"
        echo "Run with --apply to execute migrations"
        echo ""
        echo -e "${BLUE}Pending schema changes:${NC}"
        echo "---"
        npx prisma migrate diff \
            --from-migrations ./prisma/migrations \
            --to-schema-datamodel ./prisma/schema.prisma 2>&1 || {
                exit_code=$?
                if [ $exit_code -eq 2 ]; then
                    echo -e "${GREEN}No pending migrations - schema is up to date${NC}"
                fi
            }
        echo "---"
        echo ""
        echo -e "${YELLOW}To apply these changes, run:${NC}"
        echo "  ./scripts/deploy/migrate.sh --apply"
        ;;

    "apply")
        echo -e "${RED}⚠️  APPLYING MIGRATIONS...${NC}"
        echo ""
        read -p "Are you sure you want to apply migrations? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
        echo ""
        echo "Running migrations..."
        npx prisma migrate deploy

        echo ""
        echo -e "${GREEN}Migration complete!${NC}"
        echo ""
        echo -e "${BLUE}Updated migration status:${NC}"
        npx prisma migrate status
        ;;
esac

echo ""
echo -e "${GREEN}Done.${NC}"
