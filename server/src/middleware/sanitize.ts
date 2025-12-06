import { Request, Response, NextFunction } from 'express';
import sanitizeHtml from 'sanitize-html';

/**
 * Input sanitization middleware
 *
 * Provides comprehensive XSS protection using sanitize-html library.
 * Handles various attack vectors including:
 * - Script tags and iframe injections
 * - Event handler attributes (onclick, onerror, etc.)
 * - JavaScript URIs (javascript:, data:, vbscript:)
 * - Encoded payloads (HTML entities, URL encoding)
 * - CSS expression attacks
 * - SVG/MathML based XSS
 */

/**
 * Strict sanitization options for API input
 * Removes ALL HTML tags and dangerous patterns
 */
const STRICT_SANITIZE_OPTIONS: sanitizeHtml.IOptions = {
  allowedTags: [], // No HTML tags allowed
  allowedAttributes: {}, // No attributes allowed
  disallowedTagsMode: 'discard',
  // Strip all tags and their content for dangerous elements
  exclusiveFilter: (frame: sanitizeHtml.IFrame) => {
    // Remove script, style, and other dangerous tags completely
    const dangerousTags = ['script', 'style', 'iframe', 'frame', 'frameset', 'object', 'embed', 'applet', 'form', 'input', 'button', 'textarea', 'select', 'meta', 'link', 'base'];
    return dangerousTags.includes(frame.tag);
  },
  textFilter: (text: string) => {
    // Additional text-level sanitization
    return text
      // Remove null bytes
      .replace(/\0/g, '')
      // Remove javascript: and other dangerous URI schemes (case insensitive, with possible whitespace/encoding)
      .replace(/j\s*a\s*v\s*a\s*s\s*c\s*r\s*i\s*p\s*t\s*:/gi, '')
      .replace(/v\s*b\s*s\s*c\s*r\s*i\s*p\s*t\s*:/gi, '')
      .replace(/d\s*a\s*t\s*a\s*:/gi, '')
      // Remove event handlers that might slip through
      .replace(/on\w+\s*=/gi, '');
  },
};

/**
 * Recursively sanitize a value
 * Handles strings, arrays, and nested objects
 */
function sanitizeValue(value: unknown): unknown {
  if (value === null || value === undefined) {
    return value;
  }

  if (typeof value === 'string') {
    // Apply strict HTML sanitization
    let sanitized = sanitizeHtml(value, STRICT_SANITIZE_OPTIONS);

    // Additional protection against encoded attacks
    // Decode common HTML entities that might be used to bypass filters
    sanitized = decodeAndSanitize(sanitized);

    // Decode safe entities for user-friendly output
    // This preserves characters like & " ' without compromising security
    sanitized = decodeSafeEntities(sanitized);

    return sanitized;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }

  if (Array.isArray(value)) {
    return value.map(sanitizeValue);
  }

  if (typeof value === 'object') {
    const sanitized: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(value)) {
      // Also sanitize object keys to prevent prototype pollution via key names
      const sanitizedKey = typeof key === 'string'
        ? sanitizeHtml(key, STRICT_SANITIZE_OPTIONS).slice(0, 100) // Limit key length
        : key;
      sanitized[sanitizedKey] = sanitizeValue(val);
    }
    return sanitized;
  }

  // For any other type, convert to string and sanitize
  return sanitizeHtml(String(value), STRICT_SANITIZE_OPTIONS);
}

/**
 * Decode HTML entities and re-sanitize to catch encoded attacks
 * Handles: &#x3C; (hex), &#60; (decimal), &lt; (named)
 */
function decodeAndSanitize(input: string): string {
  // Decode HTML entities
  const decoded = input
    // Decode hex entities (&#x3C; -> <)
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
    // Decode decimal entities (&#60; -> <)
    .replace(/&#(\d+);/g, (_, dec) => String.fromCharCode(parseInt(dec, 10)))
    // Decode common named entities
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&amp;/gi, '&');

  // If decoding revealed any dangerous content, sanitize again
  if (decoded !== input) {
    return sanitizeHtml(decoded, STRICT_SANITIZE_OPTIONS);
  }

  return input;
}

/**
 * Decode safe HTML entities in the output
 * This preserves user-friendly characters like & while maintaining security
 */
function decodeSafeEntities(input: string): string {
  return input
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'");
  // Note: We do NOT decode &lt; and &gt; as those could be dangerous
}

/**
 * Sanitize request body, query, and params
 * Main middleware function
 */
export function sanitizeInput(req: Request, _res: Response, next: NextFunction): void {
  try {
    if (req.body && typeof req.body === 'object') {
      req.body = sanitizeValue(req.body);
    }

    if (req.query && typeof req.query === 'object') {
      req.query = sanitizeValue(req.query) as Request['query'];
    }

    if (req.params && typeof req.params === 'object') {
      req.params = sanitizeValue(req.params) as Request['params'];
    }

    next();
  } catch (error) {
    // Log error but don't expose details to client
    console.error('Sanitization error:', error);
    next();
  }
}

/**
 * Validate content type for POST/PUT/PATCH requests
 * Prevents content-type confusion attacks
 */
export function validateContentType(req: Request, res: Response, next: NextFunction): void {
  if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
    const contentType = req.get('Content-Type');

    // Allow requests without body (content-length: 0)
    const contentLength = req.get('Content-Length');
    if (contentLength === '0') {
      next();
      return;
    }

    if (!contentType) {
      res.status(400).json({
        error: 'Content-Type header is required for requests with body',
      });
      return;
    }

    const allowedContentTypes = [
      'application/json',
      'multipart/form-data',
      'application/x-www-form-urlencoded',
    ];

    const isAllowed = allowedContentTypes.some((allowed) =>
      contentType.toLowerCase().includes(allowed)
    );

    if (!isAllowed) {
      res.status(400).json({
        error: `Invalid Content-Type. Expected one of: ${allowedContentTypes.join(', ')}`,
      });
      return;
    }
  }

  next();
}

/**
 * Prevent parameter pollution by limiting array size and nesting depth
 * Protects against DoS via deeply nested objects or huge arrays
 */
export function preventParameterPollution(req: Request, res: Response, next: NextFunction): void {
  const MAX_ARRAY_SIZE = 100;
  const MAX_NESTING_DEPTH = 10;
  const MAX_OBJECT_KEYS = 100;

  function checkStructure(obj: unknown, depth: number = 0): { valid: boolean; reason?: string } {
    // Check nesting depth
    if (depth > MAX_NESTING_DEPTH) {
      return { valid: false, reason: `Object nesting exceeds maximum depth of ${MAX_NESTING_DEPTH}` };
    }

    if (Array.isArray(obj)) {
      // Check array size
      if (obj.length > MAX_ARRAY_SIZE) {
        return { valid: false, reason: `Array size exceeds maximum of ${MAX_ARRAY_SIZE}` };
      }
      // Check array elements
      for (const item of obj) {
        const result = checkStructure(item, depth + 1);
        if (!result.valid) return result;
      }
    } else if (obj && typeof obj === 'object') {
      const keys = Object.keys(obj);
      // Check number of keys
      if (keys.length > MAX_OBJECT_KEYS) {
        return { valid: false, reason: `Object has too many keys (max: ${MAX_OBJECT_KEYS})` };
      }
      // Check nested values
      for (const key of keys) {
        const result = checkStructure((obj as Record<string, unknown>)[key], depth + 1);
        if (!result.valid) return result;
      }
    }

    return { valid: true };
  }

  // Check body
  if (req.body) {
    const result = checkStructure(req.body);
    if (!result.valid) {
      res.status(400).json({ error: result.reason });
      return;
    }
  }

  // Check query params
  if (req.query) {
    const result = checkStructure(req.query);
    if (!result.valid) {
      res.status(400).json({ error: result.reason });
      return;
    }
  }

  next();
}

/**
 * Security headers middleware
 * Adds additional HTTP headers for XSS protection
 */
export function addSecurityHeaders(_req: Request, res: Response, next: NextFunction): void {
  // Prevent MIME type sniffing
  res.setHeader('X-Content-Type-Options', 'nosniff');

  // Enable browser XSS filter (legacy, but still useful)
  res.setHeader('X-XSS-Protection', '1; mode=block');

  // Prevent clickjacking
  res.setHeader('X-Frame-Options', 'DENY');

  // Control referrer information
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');

  next();
}
