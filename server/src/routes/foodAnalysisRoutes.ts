/**
 * Food Analysis Routes
 * Proxies requests to the ML service for food image analysis.
 */
import { Router, Request, Response, NextFunction } from 'express';
import axios, { AxiosError } from 'axios';
import FormData from 'form-data';
import multer from 'multer';

const router = Router();

// ML Service configuration
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// Configure multer for file uploads (10MB limit)
const upload = multer({
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype === 'image/jpeg' || file.mimetype === 'image/png') {
      cb(null, true);
    } else {
      cb(new Error('Only JPEG and PNG images are allowed'));
    }
  },
});

/**
 * POST /api/food/analyze
 * Analyze food image and estimate nutrition
 */
router.post(
  '/analyze',
  upload.single('image'),
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    console.log('📸 Food analysis request received');
    console.log('  - File:', req.file ? `${req.file.originalname} (${req.file.size} bytes)` : 'NO FILE');
    console.log('  - Body keys:', Object.keys(req.body));

    try {
      if (!req.file) {
        console.log('❌ No image file provided');
        res.status(400).json({ error: 'No image file provided' });
        return;
      }

      // Create form data for ML service
      const formData = new FormData();
      formData.append('image', req.file.buffer, {
        filename: req.file.originalname || 'food.jpg',
        contentType: req.file.mimetype,
      });

      // Add optional parameters
      if (req.body.dimensions) {
        formData.append('dimensions', req.body.dimensions);
      }
      if (req.body.cooking_method) {
        formData.append('cooking_method', req.body.cooking_method);
      }

      // Forward request to ML service
      const response = await axios.post(
        `${ML_SERVICE_URL}/api/food/analyze`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
          },
          timeout: 30000, // 30 second timeout
        }
      );

      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;
        if (axiosError.response) {
          // ML service returned an error
          res
            .status(axiosError.response.status)
            .json(axiosError.response.data);
          return;
        }
        if (axiosError.code === 'ECONNREFUSED') {
          res.status(503).json({
            error: 'ML service unavailable',
            message: 'The food analysis service is temporarily unavailable',
          });
          return;
        }
        if (axiosError.code === 'ECONNABORTED') {
          res.status(504).json({
            error: 'timeout',
            message: 'Food analysis request timed out',
          });
          return;
        }
      }
      next(error);
    }
  }
);

/**
 * GET /api/food/models/info
 * Get information about available food classification models
 */
router.get(
  '/models/info',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const response = await axios.get(`${ML_SERVICE_URL}/api/food/models/info`, {
        timeout: 5000,
      });
      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error) && error.code === 'ECONNREFUSED') {
        res.status(503).json({
          error: 'ML service unavailable',
          message: 'The food analysis service is temporarily unavailable',
        });
        return;
      }
      next(error);
    }
  }
);

/**
 * GET /api/food/nutrition-db/search
 * Search nutrition database by food name
 */
router.get(
  '/nutrition-db/search',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { q } = req.query;
      if (!q || typeof q !== 'string' || q.length < 2) {
        res.status(400).json({
          error: 'Invalid query',
          message: 'Query must be at least 2 characters long',
        });
        return;
      }

      const response = await axios.get(
        `${ML_SERVICE_URL}/api/food/nutrition-db/search`,
        {
          params: { q },
          timeout: 10000,
        }
      );
      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error) && error.code === 'ECONNREFUSED') {
        res.status(503).json({
          error: 'ML service unavailable',
          message: 'The food analysis service is temporarily unavailable',
        });
        return;
      }
      next(error);
    }
  }
);

/**
 * GET /api/food/cooking-methods
 * Get list of supported cooking methods
 */
router.get(
  '/cooking-methods',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const response = await axios.get(
        `${ML_SERVICE_URL}/api/food/cooking-methods`,
        {
          timeout: 5000,
        }
      );
      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error) && error.code === 'ECONNREFUSED') {
        res.status(503).json({
          error: 'ML service unavailable',
          message: 'The food analysis service is temporarily unavailable',
        });
        return;
      }
      next(error);
    }
  }
);

/**
 * GET /api/food/health
 * Health check for food analysis service
 */
router.get(
  '/health',
  async (_req: Request, res: Response): Promise<void> => {
    try {
      const response = await axios.get(`${ML_SERVICE_URL}/api/food/health`, {
        timeout: 5000,
      });
      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          res.status(503).json({
            status: 'unhealthy',
            service: 'food-analysis',
            error: 'ML service not reachable',
          });
          return;
        }
        if (error.response) {
          res.status(error.response.status).json(error.response.data);
          return;
        }
      }
      res.status(503).json({
        status: 'unhealthy',
        service: 'food-analysis',
        error: 'Unknown error',
      });
    }
  }
);

// ============================================================================
// FEEDBACK ROUTES - Proxy to ML service
// ============================================================================

/**
 * POST /api/food/feedback
 * Submit feedback when food classification is incorrect
 */
router.post(
  '/feedback',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const response = await axios.post(
        `${ML_SERVICE_URL}/api/food/feedback`,
        req.body,
        { timeout: 10000 }
      );
      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          res.status(503).json({
            error: 'ML service unavailable',
            message: 'The feedback service is temporarily unavailable',
          });
          return;
        }
        if (error.response) {
          res.status(error.response.status).json(error.response.data);
          return;
        }
      }
      next(error);
    }
  }
);

/**
 * GET /api/food/feedback/stats
 * Get feedback statistics
 */
router.get(
  '/feedback/stats',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const response = await axios.get(
        `${ML_SERVICE_URL}/api/food/feedback/stats`,
        { timeout: 10000 }
      );
      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          res.status(503).json({
            error: 'ML service unavailable',
            message: 'The feedback service is temporarily unavailable',
          });
          return;
        }
        if (error.response) {
          res.status(error.response.status).json(error.response.data);
          return;
        }
      }
      next(error);
    }
  }
);

/**
 * GET /api/food/feedback/suggestions/:foodKey
 * Get prompt suggestions for a specific food
 */
router.get(
  '/feedback/suggestions/:foodKey',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const response = await axios.get(
        `${ML_SERVICE_URL}/api/food/feedback/suggestions/${encodeURIComponent(req.params.foodKey)}`,
        { timeout: 10000 }
      );
      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          res.status(503).json({
            error: 'ML service unavailable',
            message: 'The feedback service is temporarily unavailable',
          });
          return;
        }
        if (error.response) {
          res.status(error.response.status).json(error.response.data);
          return;
        }
      }
      next(error);
    }
  }
);

export default router;
