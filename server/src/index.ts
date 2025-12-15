import express from 'express';
import cors from 'cors';
import { config } from './config/env';
import { errorHandler } from './middleware/errorHandler';
import authRoutes from './routes/authRoutes';
import mealRoutes from './routes/mealRoutes';
import healthMetricRoutes from './routes/healthMetricRoutes';
import activityRoutes from './routes/activityRoutes';
import foodAnalysisRoutes from './routes/foodAnalysisRoutes';
import supplementRoutes from './routes/supplementRoutes';

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health check
app.get('/health', (_req, res) => {
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/meals', mealRoutes);
app.use('/api/health-metrics', healthMetricRoutes);
app.use('/api/activities', activityRoutes);
app.use('/api/food', foodAnalysisRoutes);
app.use('/api/supplements', supplementRoutes);

// Error handler (must be last)
app.use(errorHandler);

// Only start server if not in test mode
// In test mode, supertest will handle server lifecycle
if (process.env.NODE_ENV !== 'test') {
  const PORT = config.port;

  app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`📝 Environment: ${config.nodeEnv}`);
    console.log(`🏥 Health check: http://localhost:${PORT}/health`);
  });
}

export default app;
