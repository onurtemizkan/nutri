# Task ID: 18

**Title:** Create Backend API Dockerfile

**Status:** done

**Dependencies:** None

**Priority:** high

**Description:** Create an optimized multi-stage Dockerfile for the Express.js backend server with proper caching, security configurations, and health check support.

**Details:**

Create `server/Dockerfile` with the following implementation:

**Stage 1 - Dependencies:**
```dockerfile
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production
```

**Stage 2 - Builder:**
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY prisma ./prisma/
RUN npx prisma generate
COPY tsconfig.json ./
COPY src ./src/
RUN npm run build
```

**Stage 3 - Production:**
```dockerfile
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 expressjs

# Copy production dependencies and built assets
COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma
COPY --from=builder /app/prisma ./prisma
COPY package.json ./

USER expressjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

CMD ["node", "dist/index.js"]
```

**Also create `server/.dockerignore`:**
```
node_modules
dist
.env*
*.log
coverage
__tests__
*.md
.git
```

**Key requirements:**
- Target image size under 200MB
- Run as non-root user (expressjs)
- Include Prisma client generation
- HEALTHCHECK instruction pointing to /health endpoint
- Proper layer caching for node_modules

**Test Strategy:**

1. Build image locally: `docker build -t nutri-backend:test ./server`
2. Verify image size: `docker images nutri-backend:test` (should be <200MB)
3. Run container: `docker run -d -p 3000:3000 --name test-backend nutri-backend:test`
4. Check health: `docker inspect --format='{{.State.Health.Status}}' test-backend`
5. Verify non-root: `docker exec test-backend whoami` (should output 'expressjs')
6. Test API: `curl http://localhost:3000/health`

## Subtasks

### 18.1. Create multi-stage Dockerfile with dependency caching

**Status:** pending  
**Dependencies:** None  

Set up the base multi-stage Dockerfile structure with three stages (deps, builder, runner) using node:20-alpine base images and proper layer caching for node_modules.

**Details:**

Create `server/Dockerfile` with:

**Stage 1 (deps):** Install production dependencies only using `npm ci --only=production`. This stage caches node_modules for production.

**Stage 2 (builder):** Install all dependencies (including devDependencies) with `npm ci`. This stage will be used for TypeScript compilation and Prisma generation.

**Stage 3 (runner):** Create the final production image by copying production node_modules from deps stage. Use node:20-alpine as base and set NODE_ENV=production.

Each stage should have proper WORKDIR /app and copy package files first to leverage Docker layer caching when dependencies don't change.

### 18.2. Configure Prisma generation and TypeScript build in builder stage

**Status:** pending  
**Dependencies:** 18.1  

Add Prisma client generation and TypeScript compilation steps to the builder stage, ensuring the compiled dist folder and .prisma client are available for production.

**Details:**

In the builder stage of `server/Dockerfile`:

1. Copy prisma schema: `COPY prisma ./prisma/`
2. Generate Prisma client: `RUN npx prisma generate`
3. Copy TypeScript configuration: `COPY tsconfig.json ./`
4. Copy source code: `COPY src ./src/`
5. Build TypeScript: `RUN npm run build`

This produces:
- `dist/` folder with compiled JavaScript
- `node_modules/.prisma/` with generated Prisma client

Both will be copied to the production stage in the next subtask.

### 18.3. Create production runtime with security hardening and health checks

**Status:** pending  
**Dependencies:** 18.2  

Configure the final production stage with non-root user, copy built artifacts from previous stages, add health check, and create .dockerignore file for optimal image size.

**Details:**

Complete the runner stage in `server/Dockerfile`:

1. Create non-root user: `RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 expressjs`
2. Copy artifacts:
   - `COPY --from=deps /app/node_modules ./node_modules`
   - `COPY --from=builder /app/dist ./dist`
   - `COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma`
   - `COPY --from=builder /app/prisma ./prisma`
   - `COPY package.json ./`
3. Switch user: `USER expressjs`
4. Expose port: `EXPOSE 3000`
5. Add HEALTHCHECK: `HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"`
6. Set entrypoint: `CMD ["node", "dist/index.js"]`

**Create `server/.dockerignore`:**
```
node_modules
dist
.env*
*.log
coverage
__tests__
*.md
.git
.gitignore
npm-debug.log*
.DS_Store
```
