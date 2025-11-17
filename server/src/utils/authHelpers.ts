import { Response } from 'express';
import { AuthenticatedRequest } from '../types';

/**
 * Validate that the request has a userId from authentication
 * @param req - Authenticated request
 * @param res - Response object
 * @returns userId if authenticated, undefined otherwise (and sends 401 response)
 */
export function requireAuth(
  req: AuthenticatedRequest,
  res: Response
): string | undefined {
  if (!req.userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return undefined;
  }
  return req.userId;
}
