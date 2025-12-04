/**
 * Sanitization Middleware Tests
 *
 * Unit tests for the XSS protection and input sanitization middleware.
 * These tests focus on the sanitization logic without requiring database operations.
 */

import { Request, Response, NextFunction } from 'express';
import {
  sanitizeInput,
  validateContentType,
  preventParameterPollution,
  addSecurityHeaders,
} from '../middleware/sanitize';

// Helper to create mock request
function createMockRequest(overrides: Partial<Request> = {}): Request {
  const defaultGet = jest.fn().mockReturnValue(undefined);
  return {
    body: {},
    query: {},
    params: {},
    method: 'GET',
    get: defaultGet,
    ...overrides,
  } as unknown as Request;
}

// Helper to create mock get function with specific header values
function createMockGet(headers: Record<string, string | undefined>): Request['get'] {
  return jest.fn((header: string) => headers[header]) as unknown as Request['get'];
}

// Helper to create mock response
function createMockResponse(): Response {
  const res: Partial<Response> = {
    status: jest.fn().mockReturnThis(),
    json: jest.fn().mockReturnThis(),
    setHeader: jest.fn().mockReturnThis(),
  };
  return res as Response;
}

// Helper to create mock next function
function createMockNext(): NextFunction {
  return jest.fn() as NextFunction;
}

describe('Sanitization Middleware', () => {
  // ============================================================================
  // XSS Protection Tests (sanitizeInput)
  // ============================================================================

  describe('sanitizeInput - XSS Protection', () => {
    it('should strip script tags from body input', () => {
      const req = createMockRequest({
        body: {
          name: 'Test <script>alert("xss")</script> Meal',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.name).not.toContain('<script>');
      expect(req.body.name).not.toContain('</script>');
      expect(req.body.name).toContain('Test');
      expect(req.body.name).toContain('Meal');
      expect(next).toHaveBeenCalled();
    });

    it('should strip iframe tags from input', () => {
      const req = createMockRequest({
        body: {
          name: 'Test <iframe src="evil.com"></iframe> Meal',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.name).not.toContain('<iframe');
      expect(req.body.name).not.toContain('</iframe>');
      expect(next).toHaveBeenCalled();
    });

    it('should strip javascript: URIs from input', () => {
      const req = createMockRequest({
        body: {
          name: 'Test javascript:alert(1) Meal',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.name).not.toContain('javascript:');
      expect(next).toHaveBeenCalled();
    });

    it('should strip event handlers from input', () => {
      const req = createMockRequest({
        body: {
          name: 'Test onclick=alert(1) Meal',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.name).not.toContain('onclick=');
      expect(next).toHaveBeenCalled();
    });

    it('should strip data: URIs from input', () => {
      const req = createMockRequest({
        body: {
          name: 'Test data:text/html,<script>alert(1)</script> Meal',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.name).not.toContain('data:');
      expect(req.body.name).not.toContain('<script>');
      expect(next).toHaveBeenCalled();
    });

    it('should handle HTML entity encoded XSS attempts', () => {
      const req = createMockRequest({
        body: {
          name: 'Test &#60;script&#62;alert(1)&#60;/script&#62; Meal',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      // Should decode and then sanitize
      expect(req.body.name).not.toContain('<script>');
      expect(req.body.name).not.toContain('&#60;script');
      expect(next).toHaveBeenCalled();
    });

    it('should handle hex-encoded XSS attempts', () => {
      const req = createMockRequest({
        body: {
          name: 'Test &#x3c;script&#x3e;alert(1)&#x3c;/script&#x3e; Meal',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.name).not.toContain('<script>');
      expect(next).toHaveBeenCalled();
    });

    it('should preserve legitimate text content', () => {
      const req = createMockRequest({
        body: {
          name: 'Grilled Chicken & Vegetables',
          notes: 'Delicious meal with herbs & spices!',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.name).toBe('Grilled Chicken & Vegetables');
      expect(req.body.notes).toBe('Delicious meal with herbs & spices!');
      expect(next).toHaveBeenCalled();
    });

    it('should handle nested objects correctly', () => {
      const req = createMockRequest({
        body: {
          name: 'Normal <script>bad</script> Meal',
          metadata: {
            description: '<img src=x onerror=alert(1)> Some description',
          },
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.name).not.toContain('<script>');
      expect(req.body.metadata.description).not.toContain('onerror=');
      expect(next).toHaveBeenCalled();
    });

    it('should sanitize query parameters', () => {
      const req = createMockRequest({
        query: {
          search: '<script>alert(1)</script>chicken',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.query.search).not.toContain('<script>');
      expect(req.query.search).toContain('chicken');
      expect(next).toHaveBeenCalled();
    });

    it('should sanitize URL params', () => {
      const req = createMockRequest({
        params: {
          id: '<script>alert(1)</script>123',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.params.id).not.toContain('<script>');
      expect(req.params.id).toContain('123');
      expect(next).toHaveBeenCalled();
    });

    it('should handle arrays in body', () => {
      const req = createMockRequest({
        body: {
          items: [
            '<script>bad</script>item1',
            'item2<img onerror=alert(1)>',
          ],
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.items[0]).not.toContain('<script>');
      expect(req.body.items[1]).not.toContain('onerror=');
      expect(next).toHaveBeenCalled();
    });

    it('should preserve numbers and booleans', () => {
      const req = createMockRequest({
        body: {
          calories: 500,
          isVegan: true,
          protein: 30.5,
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.calories).toBe(500);
      expect(req.body.isVegan).toBe(true);
      expect(req.body.protein).toBe(30.5);
      expect(next).toHaveBeenCalled();
    });

    it('should handle null and undefined values', () => {
      const req = createMockRequest({
        body: {
          name: 'Test',
          notes: null,
          description: undefined,
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      sanitizeInput(req, res, next);

      expect(req.body.name).toBe('Test');
      expect(req.body.notes).toBeNull();
      expect(req.body.description).toBeUndefined();
      expect(next).toHaveBeenCalled();
    });

    it('should call next even on errors', () => {
      const req = createMockRequest({
        body: null, // This could cause issues but should be handled
      });
      const res = createMockResponse();
      const next = createMockNext();

      // Should not throw
      sanitizeInput(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });

  // ============================================================================
  // Parameter Pollution Prevention Tests
  // ============================================================================

  describe('preventParameterPollution', () => {
    it('should reject arrays exceeding max size', () => {
      const largeArray = Array(101).fill({ name: 'item' });
      const req = createMockRequest({
        body: {
          items: largeArray,
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      preventParameterPollution(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: expect.stringContaining('Array size exceeds maximum'),
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should accept arrays within limit', () => {
      const validArray = Array(50).fill({ name: 'item' });
      const req = createMockRequest({
        body: {
          items: validArray,
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      preventParameterPollution(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });

    it('should reject deeply nested objects', () => {
      // Create a deeply nested object (depth > 10)
      let nested: Record<string, unknown> = { value: 'deep' };
      for (let i = 0; i < 12; i++) {
        nested = { inner: nested };
      }

      const req = createMockRequest({
        body: nested,
      });
      const res = createMockResponse();
      const next = createMockNext();

      preventParameterPollution(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: expect.stringContaining('nesting exceeds maximum depth'),
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should accept objects within nesting limit', () => {
      const req = createMockRequest({
        body: {
          level1: {
            level2: {
              level3: {
                value: 'ok',
              },
            },
          },
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      preventParameterPollution(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });

    it('should reject objects with too many keys', () => {
      const manyKeys: Record<string, string> = {};
      for (let i = 0; i < 101; i++) {
        manyKeys[`key${i}`] = `value${i}`;
      }

      const req = createMockRequest({
        body: manyKeys,
      });
      const res = createMockResponse();
      const next = createMockNext();

      preventParameterPollution(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: expect.stringContaining('too many keys'),
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should also check query parameters', () => {
      const largeArray = Array(101).fill('item');
      const req = createMockRequest({
        query: {
          ids: largeArray,
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      preventParameterPollution(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(next).not.toHaveBeenCalled();
    });
  });

  // ============================================================================
  // Content-Type Validation Tests
  // ============================================================================

  describe('validateContentType', () => {
    it('should accept application/json content type', () => {
      const req = createMockRequest({
        method: 'POST',
        get: createMockGet({
          'Content-Type': 'application/json',
          'Content-Length': '100',
        }),
      });
      const res = createMockResponse();
      const next = createMockNext();

      validateContentType(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });

    it('should accept multipart/form-data content type', () => {
      const req = createMockRequest({
        method: 'POST',
        get: createMockGet({
          'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundary',
          'Content-Length': '100',
        }),
      });
      const res = createMockResponse();
      const next = createMockNext();

      validateContentType(req, res, next);

      expect(next).toHaveBeenCalled();
    });

    it('should reject invalid content type for POST', () => {
      const req = createMockRequest({
        method: 'POST',
        get: createMockGet({
          'Content-Type': 'text/plain',
          'Content-Length': '100',
        }),
      });
      const res = createMockResponse();
      const next = createMockNext();

      validateContentType(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: expect.stringContaining('Invalid Content-Type'),
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should reject missing content type for POST with body', () => {
      const req = createMockRequest({
        method: 'POST',
        get: createMockGet({
          'Content-Length': '100',
        }),
      });
      const res = createMockResponse();
      const next = createMockNext();

      validateContentType(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: expect.stringContaining('Content-Type header is required'),
      });
    });

    it('should allow GET requests without content type check', () => {
      const req = createMockRequest({
        method: 'GET',
        get: createMockGet({}),
      });
      const res = createMockResponse();
      const next = createMockNext();

      validateContentType(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });

    it('should allow POST requests with empty body (content-length: 0)', () => {
      const req = createMockRequest({
        method: 'POST',
        get: createMockGet({
          'Content-Length': '0',
        }),
      });
      const res = createMockResponse();
      const next = createMockNext();

      validateContentType(req, res, next);

      expect(next).toHaveBeenCalled();
    });

    it('should validate PUT requests', () => {
      const req = createMockRequest({
        method: 'PUT',
        get: createMockGet({
          'Content-Type': 'text/html',
          'Content-Length': '100',
        }),
      });
      const res = createMockResponse();
      const next = createMockNext();

      validateContentType(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should validate PATCH requests', () => {
      const req = createMockRequest({
        method: 'PATCH',
        get: createMockGet({
          'Content-Type': 'application/json',
          'Content-Length': '100',
        }),
      });
      const res = createMockResponse();
      const next = createMockNext();

      validateContentType(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });

  // ============================================================================
  // Security Headers Tests
  // ============================================================================

  describe('addSecurityHeaders', () => {
    it('should set X-Content-Type-Options header', () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      addSecurityHeaders(req, res, next);

      expect(res.setHeader).toHaveBeenCalledWith('X-Content-Type-Options', 'nosniff');
    });

    it('should set X-XSS-Protection header', () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      addSecurityHeaders(req, res, next);

      expect(res.setHeader).toHaveBeenCalledWith('X-XSS-Protection', '1; mode=block');
    });

    it('should set X-Frame-Options header', () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      addSecurityHeaders(req, res, next);

      expect(res.setHeader).toHaveBeenCalledWith('X-Frame-Options', 'DENY');
    });

    it('should set Referrer-Policy header', () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      addSecurityHeaders(req, res, next);

      expect(res.setHeader).toHaveBeenCalledWith('Referrer-Policy', 'strict-origin-when-cross-origin');
    });

    it('should call next after setting headers', () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      addSecurityHeaders(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });
});
