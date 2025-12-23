import { Response, NextFunction } from 'express';
import { AdminRole } from '@prisma/client';
import { verifyAdminToken } from '../services/adminAuthService';
import prisma from '../config/database';
import { HTTP_STATUS } from '../config/constants';
import { AdminAuthenticatedRequest } from '../types';
import { logger } from '../config/logger';

/**
 * Middleware to require admin authentication
 * Optionally restricts access to specific roles
 */
export function requireAdmin(allowedRoles?: AdminRole[]) {
  return async (
    req: AdminAuthenticatedRequest,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    try {
      // Extract token from Authorization header
      const authHeader = req.headers.authorization;
      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        res.status(HTTP_STATUS.UNAUTHORIZED).json({
          error: 'Authorization header required',
        });
        return;
      }

      const token = authHeader.substring(7);

      // Verify JWT token
      const payload = verifyAdminToken(token);
      if (!payload) {
        res.status(HTTP_STATUS.UNAUTHORIZED).json({
          error: 'Invalid or expired token',
        });
        return;
      }

      // Load admin user from database
      const adminUser = await prisma.adminUser.findUnique({
        where: { id: payload.adminUserId },
        select: {
          id: true,
          email: true,
          name: true,
          role: true,
          isActive: true,
        },
      });

      if (!adminUser) {
        res.status(HTTP_STATUS.UNAUTHORIZED).json({
          error: 'Admin user not found',
        });
        return;
      }

      if (!adminUser.isActive) {
        res.status(HTTP_STATUS.FORBIDDEN).json({
          error: 'Account is disabled',
        });
        return;
      }

      // Check role authorization if roles specified
      if (allowedRoles && allowedRoles.length > 0) {
        if (!allowedRoles.includes(adminUser.role)) {
          logger.warn(
            { adminUserId: adminUser.id, role: adminUser.role, requiredRoles: allowedRoles },
            'Admin access denied: insufficient role'
          );
          res.status(HTTP_STATUS.FORBIDDEN).json({
            error: 'Insufficient permissions',
          });
          return;
        }
      }

      // Attach admin user to request
      req.adminUser = {
        id: adminUser.id,
        email: adminUser.email,
        name: adminUser.name,
        role: adminUser.role,
      };

      next();
    } catch (error) {
      logger.error({ error }, 'Admin auth middleware error');
      res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
        error: 'Authentication failed',
      });
    }
  };
}

/**
 * Convenience middleware for SUPER_ADMIN only endpoints
 */
export const requireSuperAdmin = requireAdmin(['SUPER_ADMIN']);

/**
 * Convenience middleware for any admin (no role restriction)
 */
export const requireAnyAdmin = requireAdmin();
