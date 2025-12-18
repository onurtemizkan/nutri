# =============================================================================
# Nutri Backend API - Production Dockerfile
# Multi-stage build for optimized image size and security
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Dependencies
# Install production dependencies only (for smaller final image)
# -----------------------------------------------------------------------------
FROM node:20-alpine AS deps

WORKDIR /app

# Copy package files first to leverage Docker layer caching
COPY package.json package-lock.json ./

# Install production dependencies only
# Using --omit=dev instead of deprecated --only=production
RUN npm ci --omit=dev

# -----------------------------------------------------------------------------
# Stage 2: Builder
# Install all dependencies, generate Prisma client, and compile TypeScript
# -----------------------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install ALL dependencies (including devDependencies for building)
RUN npm ci

# Copy Prisma schema and generate client
COPY prisma ./prisma/
RUN npx prisma generate

# Copy TypeScript configuration
COPY tsconfig.json ./

# Copy source code
COPY src ./src/

# Build TypeScript to JavaScript
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 3: Production Runner
# Minimal production image with security hardening
# -----------------------------------------------------------------------------
FROM node:20-alpine AS runner

WORKDIR /app

# Set production environment
ENV NODE_ENV=production

# Create non-root user for security
# Using numeric IDs for consistency across systems
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 expressjs

# Copy production dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy compiled JavaScript from builder stage
COPY --from=builder /app/dist ./dist

# Copy generated Prisma client from builder stage
COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma

# Copy Prisma schema (needed for migrations in production)
COPY --from=builder /app/prisma ./prisma

# Copy package.json (needed for version info and scripts)
COPY package.json ./

# Set ownership to non-root user
RUN chown -R expressjs:nodejs /app

# Switch to non-root user
USER expressjs

# Expose the application port
EXPOSE 3000

# Health check - verify the /health endpoint responds
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

# Start the application
CMD ["node", "dist/index.js"]
