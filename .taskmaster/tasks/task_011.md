# Task ID: 11

**Title:** Generate OpenAPI Documentation and Polish Production Readiness

**Status:** pending

**Dependencies:** 2 ✓, 3 ✓, 4 ✓, 6

**Priority:** low

**Description:** Add comprehensive API documentation, perform security audit, and optimize performance for production deployment.

**Details:**

1. Generate OpenAPI/Swagger documentation:
   - Backend (Express): Add swagger-jsdoc and swagger-ui-express
   - ML Service (FastAPI): Already has built-in docs at /docs
   - Document all endpoints with request/response schemas
   - Add authentication requirements
   - Include example requests and responses

2. Express API documentation setup:
```javascript
import swaggerJsdoc from 'swagger-jsdoc';
import swaggerUi from 'swagger-ui-express';

const options = {
  definition: {
    openapi: '3.0.0',
    info: { title: 'Nutri API', version: '1.0.0' },
    servers: [{ url: '/api' }],
    components: {
      securitySchemes: {
        bearerAuth: { type: 'http', scheme: 'bearer' }
      }
    }
  },
  apis: ['./src/routes/*.ts'],
};

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerJsdoc(options)));
```

3. Performance optimization:
   - Add database query logging to identify slow queries
   - Implement connection pooling for PostgreSQL
   - Add Redis caching for frequently accessed data (user profile, daily summary)
   - Compress API responses with compression middleware
   - Optimize Prisma queries with select/include

4. Security audit checklist:
   - Review all authentication flows
   - Verify rate limiting is effective
   - Check for SQL injection (Prisma handles this)
   - Verify XSS prevention in sanitize middleware
   - Review CORS configuration
   - Ensure sensitive data not logged
   - Check JWT secret rotation capability

5. Production configuration:
   - Environment variable validation on startup
   - Health check endpoints for load balancers
   - Graceful shutdown handling
   - Error tracking integration (Sentry ready)
   - Logging configuration (structured JSON logs)

6. Mobile app optimization:
   - Review bundle size
   - Implement proper loading states
   - Add offline detection and handling
   - Optimize image handling

7. Create deployment documentation:
   - Docker setup for backend and ML service
   - Environment variables reference
   - Database migration guide
   - Monitoring recommendations

**Test Strategy:**

1. Validate OpenAPI spec with swagger-cli validate
2. Load testing with k6 or artillery (100 concurrent users)
3. Security scan with npm audit and OWASP ZAP
4. Test rate limiting triggers correctly
5. Test graceful shutdown
6. Verify logging output format
7. Test health check endpoints
8. Performance benchmark for critical endpoints
