# Docker Development Environment

This guide explains how to use Docker for local development of the Nutri application.

## Overview

The Docker development environment provides:

- **PostgreSQL 16** - Primary database
- **Redis 7** - Cache and session store
- **Backend API** - Express/Node.js with hot reload
- **ML Service** - FastAPI/Python with hot reload
- **Adminer** - Database admin UI
- **Redis Commander** - Redis admin UI
- **Prisma Studio** - ORM database viewer

All services have **hot reloading enabled** - changes to your code are automatically reflected without restarting containers.

## Quick Start

```bash
# Start all development services
npm run docker:dev:start

# View logs
npm run docker:dev:logs

# Stop all services
npm run docker:dev:stop
```

## Prerequisites

- **Docker Desktop** - [Download](https://www.docker.com/products/docker-desktop/)
- Minimum 4GB RAM allocated to Docker
- Ports 3000, 5432, 6379, 8000, 8080, 8081, 5555 available

## Commands

### NPM Scripts

| Command | Description |
|---------|-------------|
| `npm run docker:dev:start` | Start all services |
| `npm run docker:dev:stop` | Stop all services |
| `npm run docker:dev:logs` | View logs (Ctrl+C to exit) |
| `npm run docker:dev:build` | Rebuild and start services |
| `npm run docker:dev:clean` | Stop and remove all data |
| `npm run docker:dev:status` | Show container status |
| `npm run docker:dev:shell:backend` | Shell into backend container |
| `npm run docker:dev:shell:ml` | Shell into ML service container |

### Direct Script Usage

```bash
./scripts/docker-dev.sh start
./scripts/docker-dev.sh stop
./scripts/docker-dev.sh logs backend
./scripts/docker-dev.sh shell ml
./scripts/docker-dev.sh migrate
./scripts/docker-dev.sh clean
```

## Service URLs

After running `npm run docker:dev:start`:

| Service | URL | Purpose |
|---------|-----|---------|
| Backend API | http://localhost:3000 | REST API |
| ML Service | http://localhost:8000 | ML predictions |
| Adminer | http://localhost:8080 | Database admin |
| Redis Commander | http://localhost:8081 | Redis admin |
| Prisma Studio | http://localhost:5555 | ORM viewer |

### Database Connection (Adminer)

- **System**: PostgreSQL
- **Server**: postgres
- **Username**: postgres
- **Password**: postgres
- **Database**: nutri_db

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                           │
│                   (nutri-dev-network)                       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Backend    │  │  ML Service  │  │    Prisma    │      │
│  │  (Node.js)   │  │   (Python)   │  │    Studio    │      │
│  │   :3000      │  │    :8000     │  │    :5555     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
│         │                 │                                 │
│         │                 │                                 │
│  ┌──────▼─────────────────▼──────┐  ┌──────────────┐       │
│  │         PostgreSQL            │  │    Redis     │       │
│  │           :5432               │  │    :6379     │       │
│  └───────────────────────────────┘  └──────────────┘       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │   Adminer    │  │    Redis     │                        │
│  │   :8080      │  │  Commander   │                        │
│  └──────────────┘  │    :8081     │                        │
│                    └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Hot Reloading

### Backend (Node.js)

Code changes in `server/src/` are automatically detected and the server restarts:

```bash
# Edit a file
vim server/src/routes/health.ts

# Changes are reflected immediately
curl http://localhost:3000/health
```

The backend uses `tsx watch` for hot reloading.

### ML Service (Python)

Code changes in `ml-service/app/` are automatically detected:

```bash
# Edit a file
vim ml-service/app/main.py

# Changes are reflected immediately
curl http://localhost:8000/health
```

The ML service uses `uvicorn --reload` for hot reloading.

## Database Operations

### Run Migrations

```bash
# Using npm script
npm run docker:dev -- migrate

# Or directly
./scripts/docker-dev.sh migrate

# Or inside container
npm run docker:dev:shell:backend
npx prisma migrate deploy
```

### Reset Database

```bash
npm run docker:dev:shell:backend
npm run db:reset
```

### Open Prisma Studio

Prisma Studio is automatically started at http://localhost:5555

Or run manually:

```bash
npm run docker:dev:shell:backend
npm run db:studio
```

## Debugging

### Backend Debugging (VS Code)

1. The backend exposes port `9229` for Node.js debugging
2. Add this to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Docker: Attach to Backend",
      "type": "node",
      "request": "attach",
      "port": 9229,
      "address": "localhost",
      "localRoot": "${workspaceFolder}/server",
      "remoteRoot": "/app",
      "sourceMaps": true
    }
  ]
}
```

3. Start the backend with debug mode (already configured in docker-compose)
4. Attach the debugger in VS Code

### ML Service Debugging (VS Code)

1. The ML service exposes port `5678` for Python debugging
2. Add this to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Docker: Attach to ML Service",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}/ml-service",
          "remoteRoot": "/app"
        }
      ]
    }
  ]
}
```

### View Logs

```bash
# All services
npm run docker:dev:logs

# Specific service
./scripts/docker-dev.sh logs backend
./scripts/docker-dev.sh logs ml-service
./scripts/docker-dev.sh logs postgres
```

## Volume Management

The development environment uses Docker volumes for persistence:

| Volume | Purpose |
|--------|---------|
| `nutri_dev_postgres_data` | PostgreSQL data |
| `nutri_dev_redis_data` | Redis data |
| `nutri_dev_backend_node_modules` | Backend dependencies |
| `nutri_dev_backend_prisma_client` | Prisma client |

### Clean Volumes

```bash
# Remove all dev volumes (will delete data!)
npm run docker:dev:clean
```

## Troubleshooting

### Port Already in Use

```bash
# Find what's using the port
lsof -i :3000

# Kill the process
kill -9 <PID>

# Or stop all Docker containers
docker stop $(docker ps -aq)
```

### Container Won't Start

```bash
# Check logs
npm run docker:dev:logs

# Rebuild containers
npm run docker:dev:build

# Full clean and restart
npm run docker:dev:clean
npm run docker:dev:start
```

### Database Connection Issues

```bash
# Check if postgres is running
docker ps | grep postgres

# Check postgres logs
./scripts/docker-dev.sh logs postgres

# Connect directly
./scripts/docker-dev.sh shell postgres
# This opens psql
```

### Node Modules Issues

If the backend has dependency issues:

```bash
# Remove node_modules volume
docker volume rm nutri_dev_backend_node_modules

# Rebuild
npm run docker:dev:build
```

### Hot Reload Not Working

Check if the volume mounts are correct:

```bash
docker compose -f docker-compose.dev.yml config
```

Ensure your IDE isn't blocking file system events.

## Comparison: Docker vs Local Development

| Aspect | Docker Dev | Local Dev |
|--------|------------|-----------|
| Setup | Single command | Install each service |
| Consistency | Identical for all devs | Varies by machine |
| Resources | Higher memory usage | Lower overhead |
| Hot Reload | Via volume mounts | Native |
| Debugging | Needs port forwarding | Native |
| Database | Isolated | Shared/local |

### When to Use Docker Dev

- First time setup
- Consistent environment across team
- Testing production-like setup
- Running all services together

### When to Use Local Dev

- Lower resource usage needed
- Faster hot reload
- Native debugging
- Working on single service

## Tips

### Running Alongside Local Services

You can run Docker services alongside local ones:

```bash
# Start only database services
docker compose -f docker-compose.dev.yml up -d postgres redis

# Run backend locally
cd server && npm run dev

# Run ML service locally
cd ml-service && uvicorn app.main:app --reload
```

### IDE Integration

For VS Code, install:
- **Docker** extension
- **Remote - Containers** extension (for in-container development)

### Performance on macOS

Enable these Docker Desktop settings for better performance:
- Use gRPC FUSE for file sharing
- Enable VirtioFS (Docker Desktop 4.6+)

---

*Last updated: December 2025*
