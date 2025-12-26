/**
 * Food Analysis Routes
 * Proxies requests to the ML service for food image analysis.
 */
import { Router, Request, Response, NextFunction } from 'express';
import axios, { AxiosError } from 'axios';
import FormData from 'form-data';
import multer from 'multer';
import { logger } from '../config/logger';

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
    logger.debug(
      {
        hasFile: !!req.file,
        fileSize: req.file?.size,
        mimeType: req.file?.mimetype,
        bodyKeys: Object.keys(req.body),
      },
      'Food analysis request received'
    );

    try {
      if (!req.file) {
        logger.warn('Food analysis request missing image file');
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
      const response = await axios.post(`${ML_SERVICE_URL}/api/food/analyze`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 30000, // 30 second timeout
      });

      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;
        if (axiosError.response) {
          // ML service returned an error
          res.status(axiosError.response.status).json(axiosError.response.data);
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

      const response = await axios.get(`${ML_SERVICE_URL}/api/food/nutrition-db/search`, {
        params: { q },
        timeout: 10000,
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
 * GET /api/food/cooking-methods
 * Get list of supported cooking methods
 */
router.get(
  '/cooking-methods',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const response = await axios.get(`${ML_SERVICE_URL}/api/food/cooking-methods`, {
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
 * GET /api/food/health
 * Health check for food analysis service
 */
router.get('/health', async (_req: Request, res: Response): Promise<void> => {
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
});

// ============================================================================
// FEEDBACK ROUTES - Proxy to ML service
// ============================================================================

/**
 * POST /api/food/feedback
 * Submit feedback when food classification is incorrect
 */
router.post('/feedback', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/api/food/feedback`, req.body, {
      timeout: 10000,
    });
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
});

/**
 * GET /api/food/feedback/stats
 * Get feedback statistics
 */
router.get(
  '/feedback/stats',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const response = await axios.get(`${ML_SERVICE_URL}/api/food/feedback/stats`, {
        timeout: 10000,
      });
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

// ============================================================================
// CLASSIFICATION-ASSISTED SEARCH - Combines ML classification with USDA search
// ============================================================================

/**
 * POST /api/food/classify-and-search
 * Classify food image and return USDA search results filtered by classification
 *
 * This endpoint implements the hybrid approach:
 * 1. Classify image using ML coarse classifier (25-30 food categories)
 * 2. Use classification to filter/rank USDA search results
 * 3. Return both classification and search results
 */
router.post(
  '/classify-and-search',
  upload.single('image'),
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    logger.debug('Classify-and-search request received');

    try {
      if (!req.file) {
        res.status(400).json({ error: 'No image file provided' });
        return;
      }

      // 1. Call ML service for coarse classification
      const classifyFormData = new FormData();
      classifyFormData.append('image', req.file.buffer, {
        filename: req.file.originalname || 'food.jpg',
        contentType: req.file.mimetype,
      });

      // Add optional query for enhanced classification
      if (req.body.query) {
        classifyFormData.append('query', req.body.query);
      }

      let classification;
      try {
        const classifyResponse = await axios.post(
          `${ML_SERVICE_URL}/api/food/coarse-classify`,
          classifyFormData,
          {
            headers: {
              ...classifyFormData.getHeaders(),
            },
            timeout: 15000, // 15 second timeout for classification
          }
        );
        classification = classifyResponse.data;
      } catch (classifyError) {
        if (axios.isAxiosError(classifyError)) {
          if (classifyError.code === 'ECONNREFUSED') {
            // ML service unavailable - return search-only results
            logger.warn('ML service unavailable, falling back to search-only');
            classification = null;
          } else if (classifyError.response) {
            logger.warn({ responseData: classifyError.response.data }, 'Classification failed');
            classification = null;
          } else {
            classification = null;
          }
        } else {
          classification = null;
        }
      }

      // 2. Build search query from classification hints
      let searchQuery = req.body.query || '';
      let dataTypes: string[] = [];

      if (classification) {
        // Use classification hints to enhance query
        if (classification.search_hints?.suggested_query_enhancement) {
          searchQuery = classification.search_hints.suggested_query_enhancement;
        }

        // Use recommended USDA data types
        dataTypes = classification.usda_datatypes || [];
      }

      // 3. Import and use food database service for USDA search
      const { foodDatabaseService } = await import('../services/foodDatabaseService');

      let searchResults = null;
      if (searchQuery.length >= 2) {
        try {
          searchResults = await foodDatabaseService.searchFoods({
            query: searchQuery,
            dataType: dataTypes.length > 0 ? dataTypes : undefined,
            limit: 10,
          });
        } catch (searchError) {
          logger.error({ err: searchError }, 'USDA search failed');
          // Continue without search results
        }
      }

      // 4. Parse AR dimensions if provided
      let portionEstimate = null;
      if (req.body.dimensions) {
        try {
          const dims = JSON.parse(req.body.dimensions);
          // Simple volume-based estimation (cm³ to grams, assuming ~1g/cm³ average)
          const volumeCm3 = dims.width * dims.height * dims.depth;
          // Apply a shape factor of 0.7 (most foods aren't perfect cuboids)
          portionEstimate = {
            estimated_grams: Math.round(volumeCm3 * 0.7),
            dimensions: dims,
            quality: volumeCm3 > 10 && volumeCm3 < 2000 ? 'medium' : 'low',
          };
        } catch {
          // Invalid dimensions JSON, ignore
        }
      }

      // 5. Return combined results
      res.json({
        classification: classification || {
          category: 'unknown',
          confidence: 0,
          usda_datatypes: ['Foundation', 'SR Legacy', 'Survey (FNDDS)', 'Branded'],
          search_hints: {
            subcategory_hints: [],
            suggested_query_enhancement: searchQuery,
          },
          alternatives: [],
        },
        searchResults: searchResults || {
          foods: [],
          pagination: {
            page: 1,
            limit: 10,
            total: 0,
            totalPages: 0,
            hasNextPage: false,
            hasPrevPage: false,
          },
        },
        portionEstimate,
        query: searchQuery,
      });
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;
        if (axiosError.response) {
          res.status(axiosError.response.status).json(axiosError.response.data);
          return;
        }
        if (axiosError.code === 'ECONNREFUSED') {
          res.status(503).json({
            error: 'Service unavailable',
            message: 'The classification service is temporarily unavailable',
          });
          return;
        }
      }
      next(error);
    }
  }
);

/**
 * GET /api/food/coarse-classify/categories
 * Get list of coarse classification categories and their USDA mappings
 */
router.get(
  '/coarse-classify/categories',
  async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const response = await axios.get(`${ML_SERVICE_URL}/api/food/coarse-classify/categories`, {
        timeout: 5000,
      });
      res.json(response.data);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNREFUSED') {
          res.status(503).json({
            error: 'ML service unavailable',
            message: 'The classification service is temporarily unavailable',
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
