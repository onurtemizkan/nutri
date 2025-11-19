import { Request, Response, NextFunction } from 'express';

/**
 * Input sanitization middleware
 * Prevents XSS attacks by sanitizing user input
 */

/**
 * Sanitize a value by removing dangerous characters
 */
function sanitizeValue(value: unknown): unknown {
  if (typeof value === 'string') {
    // Remove script tags and dangerous characters
    return value
      .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
      .replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '')
      .replace(/javascript:/gi, '')
      .replace(/on\w+\s*=/gi, '');
  }

  if (Array.isArray(value)) {
    return value.map(sanitizeValue);
  }

  if (value && typeof value === 'object') {
    const sanitized: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(value)) {
      sanitized[key] = sanitizeValue(val);
    }
    return sanitized;
  }

  return value;
}

/**
 * Sanitize request body, query, and params
 */
export function sanitizeInput(req: Request, _res: Response, next: NextFunction): void {
  if (req.body) {
    req.body = sanitizeValue(req.body);
  }

  if (req.query) {
    req.query = sanitizeValue(req.query) as Request['query'];
  }

  if (req.params) {
    req.params = sanitizeValue(req.params) as Request['params'];
  }

  next();
}

/**
 * Validate content type for POST/PUT requests
 */
export function validateContentType(req: Request, res: Response, next: NextFunction): void {
  if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
    const contentType = req.get('Content-Type');

    if (!contentType || (!contentType.includes('application/json') && !contentType.includes('multipart/form-data'))) {
      res.status(400).json({
        error: 'Invalid Content-Type. Expected application/json or multipart/form-data',
      });
      return;
    }
  }

  next();
}

/**
 * Prevent parameter pollution by limiting array size
 */
export function preventParameterPollution(req: Request, res: Response, next: NextFunction): void {
  const MAX_ARRAY_SIZE = 100;

  function checkArraySize(obj: unknown): boolean {
    if (Array.isArray(obj)) {
      if (obj.length > MAX_ARRAY_SIZE) return false;
      return obj.every(checkArraySize);
    }

    if (obj && typeof obj === 'object') {
      return Object.values(obj).every(checkArraySize);
    }

    return true;
  }

  if (req.body && !checkArraySize(req.body)) {
    res.status(400).json({
      error: 'Request contains arrays that exceed maximum size limit',
    });
    return;
  }

  next();
}
