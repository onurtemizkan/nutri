/**
 * Swagger/OpenAPI Configuration
 *
 * Generates API documentation for the Nutri backend
 */

import swaggerJsdoc from 'swagger-jsdoc';
import { version } from '../../package.json';

const options: swaggerJsdoc.Options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Nutri API',
      version,
      description: `
# Nutri API Documentation

Full-stack nutrition tracking API with ML-powered insights.

## Features
- **Authentication**: JWT-based auth with Apple Sign-In support
- **Meals**: Track meals with nutritional information
- **Health Metrics**: Log and analyze health data from multiple sources
- **Activities**: Track physical activities and workouts
- **Goals**: Set and track nutrition and fitness goals
- **ML Predictions**: Get AI-powered health insights

## Authentication
Most endpoints require a valid JWT token in the Authorization header:
\`\`\`
Authorization: Bearer <your-jwt-token>
\`\`\`

## Rate Limiting
- Standard endpoints: 100 requests per 15 minutes
- Auth endpoints: 5 requests per 15 minutes
- Password reset: 3 requests per hour
      `,
      contact: {
        name: 'Nutri API Support',
        email: 'support@nutri.app',
      },
      license: {
        name: 'Proprietary',
      },
    },
    servers: [
      {
        url: '/api',
        description: 'API Base URL',
      },
    ],
    components: {
      securitySchemes: {
        bearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT',
          description: 'JWT token from /api/auth/login or /api/auth/register',
        },
      },
      schemas: {
        Error: {
          type: 'object',
          properties: {
            error: {
              type: 'string',
              description: 'Error message',
            },
            details: {
              type: 'array',
              items: {
                type: 'object',
              },
              description: 'Validation error details (if applicable)',
            },
          },
        },
        PaginatedResponse: {
          type: 'object',
          properties: {
            data: {
              type: 'array',
              items: {},
            },
            total: {
              type: 'integer',
              description: 'Total number of records',
            },
            limit: {
              type: 'integer',
              description: 'Records per page',
            },
            offset: {
              type: 'integer',
              description: 'Number of records skipped',
            },
            hasMore: {
              type: 'boolean',
              description: 'Whether more records exist',
            },
          },
        },
        User: {
          type: 'object',
          properties: {
            id: { type: 'string', format: 'cuid' },
            email: { type: 'string', format: 'email' },
            name: { type: 'string' },
            avatarUrl: { type: 'string', nullable: true },
            goalCalories: { type: 'integer', nullable: true },
            goalProtein: { type: 'integer', nullable: true },
            goalCarbs: { type: 'integer', nullable: true },
            goalFat: { type: 'integer', nullable: true },
            goalFiber: { type: 'integer', nullable: true },
            goalWater: { type: 'integer', nullable: true, description: 'Daily water goal in ml' },
            currentWeight: { type: 'number', nullable: true },
            goalWeight: { type: 'number', nullable: true },
            height: { type: 'number', nullable: true },
            activityLevel: {
              type: 'string',
              enum: ['sedentary', 'light', 'moderate', 'active', 'veryActive'],
              nullable: true,
            },
            subscriptionTier: {
              type: 'string',
              enum: ['FREE', 'PRO_TRIAL', 'PRO'],
            },
            createdAt: { type: 'string', format: 'date-time' },
            updatedAt: { type: 'string', format: 'date-time' },
          },
        },
        Meal: {
          type: 'object',
          properties: {
            id: { type: 'string', format: 'cuid' },
            userId: { type: 'string' },
            name: { type: 'string' },
            mealType: {
              type: 'string',
              enum: ['breakfast', 'lunch', 'dinner', 'snack'],
            },
            calories: { type: 'number' },
            protein: { type: 'number' },
            carbs: { type: 'number' },
            fat: { type: 'number' },
            fiber: { type: 'number', nullable: true },
            sugar: { type: 'number', nullable: true },
            sodium: { type: 'number', nullable: true },
            notes: { type: 'string', nullable: true },
            consumedAt: { type: 'string', format: 'date-time' },
            createdAt: { type: 'string', format: 'date-time' },
            updatedAt: { type: 'string', format: 'date-time' },
          },
        },
        HealthMetric: {
          type: 'object',
          properties: {
            id: { type: 'string', format: 'cuid' },
            userId: { type: 'string' },
            metricType: {
              type: 'string',
              description:
                'Type of health metric (e.g., RESTING_HEART_RATE, HEART_RATE_VARIABILITY_SDNN)',
            },
            value: { type: 'number' },
            unit: { type: 'string' },
            source: {
              type: 'string',
              enum: ['APPLE_HEALTH', 'FITBIT', 'GARMIN', 'OURA', 'WHOOP', 'MANUAL', 'CGM'],
            },
            measuredAt: { type: 'string', format: 'date-time' },
            metadata: { type: 'object', nullable: true },
            createdAt: { type: 'string', format: 'date-time' },
          },
        },
        Activity: {
          type: 'object',
          properties: {
            id: { type: 'string', format: 'cuid' },
            userId: { type: 'string' },
            activityType: {
              type: 'string',
              enum: [
                'RUNNING',
                'CYCLING',
                'SWIMMING',
                'WEIGHTLIFTING',
                'YOGA',
                'WALKING',
                'HIKING',
                'OTHER',
              ],
            },
            duration: { type: 'integer', description: 'Duration in minutes' },
            caloriesBurned: { type: 'number', nullable: true },
            distance: { type: 'number', nullable: true, description: 'Distance in meters' },
            intensity: {
              type: 'string',
              enum: ['LOW', 'MODERATE', 'HIGH', 'VERY_HIGH'],
            },
            notes: { type: 'string', nullable: true },
            startedAt: { type: 'string', format: 'date-time' },
            createdAt: { type: 'string', format: 'date-time' },
          },
        },
        WaterIntake: {
          type: 'object',
          properties: {
            id: { type: 'string', format: 'cuid' },
            userId: { type: 'string' },
            amount: { type: 'integer', description: 'Amount in ml' },
            recordedAt: { type: 'string', format: 'date-time' },
            createdAt: { type: 'string', format: 'date-time' },
          },
        },
        WeightRecord: {
          type: 'object',
          properties: {
            id: { type: 'string', format: 'cuid' },
            userId: { type: 'string' },
            weight: { type: 'number', description: 'Weight in kg' },
            notes: { type: 'string', nullable: true },
            recordedAt: { type: 'string', format: 'date-time' },
            createdAt: { type: 'string', format: 'date-time' },
          },
        },
        Supplement: {
          type: 'object',
          properties: {
            id: { type: 'string', format: 'cuid' },
            userId: { type: 'string' },
            name: { type: 'string' },
            brand: { type: 'string', nullable: true },
            dosage: { type: 'string' },
            frequency: { type: 'string' },
            timeOfDay: { type: 'string', nullable: true },
            notes: { type: 'string', nullable: true },
            isActive: { type: 'boolean' },
            createdAt: { type: 'string', format: 'date-time' },
          },
        },
      },
      parameters: {
        limitParam: {
          name: 'limit',
          in: 'query',
          description: 'Number of records to return (max 100)',
          schema: {
            type: 'integer',
            minimum: 1,
            maximum: 100,
            default: 20,
          },
        },
        offsetParam: {
          name: 'offset',
          in: 'query',
          description: 'Number of records to skip',
          schema: {
            type: 'integer',
            minimum: 0,
            default: 0,
          },
        },
        dateParam: {
          name: 'date',
          in: 'query',
          description: 'Date in YYYY-MM-DD format',
          schema: {
            type: 'string',
            format: 'date',
          },
        },
      },
      responses: {
        UnauthorizedError: {
          description: 'Missing or invalid authentication token',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                error: 'Unauthorized access',
              },
            },
          },
        },
        NotFoundError: {
          description: 'Resource not found',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                error: 'Resource not found',
              },
            },
          },
        },
        ValidationError: {
          description: 'Invalid request data',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                error: 'Validation failed',
                details: [
                  {
                    path: ['email'],
                    message: 'Invalid email format',
                  },
                ],
              },
            },
          },
        },
      },
    },
    security: [
      {
        bearerAuth: [],
      },
    ],
    tags: [
      {
        name: 'Auth',
        description: 'Authentication endpoints - register, login, profile management',
      },
      {
        name: 'Meals',
        description: 'Meal tracking and nutrition logging',
      },
      {
        name: 'Health Metrics',
        description: 'Health data from various sources',
      },
      {
        name: 'Activities',
        description: 'Physical activity and workout tracking',
      },
      {
        name: 'Water',
        description: 'Water intake tracking',
      },
      {
        name: 'Weight',
        description: 'Weight tracking and progress',
      },
      {
        name: 'Goals',
        description: 'Goal progress and achievements',
      },
      {
        name: 'Supplements',
        description: 'Supplement tracking and reminders',
      },
      {
        name: 'Foods',
        description: 'Food database and analysis',
      },
      {
        name: 'Reports',
        description: 'Weekly and monthly nutrition reports',
      },
      {
        name: 'Notifications',
        description: 'Push notification management',
      },
      {
        name: 'Email',
        description: 'Email preferences and webhooks',
      },
    ],
  },
  apis: ['./src/routes/*.ts', './src/routes/**/*.ts'],
};

export const swaggerSpec = swaggerJsdoc(options);
