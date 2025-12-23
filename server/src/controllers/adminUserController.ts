import { Response } from 'express';
import {
  getUserList,
  getUserDetail,
  exportUserData,
  deleteUserAccount,
} from '../services/adminUserService';
import {
  listUsersQuerySchema,
  userIdParamSchema,
  deleteUserSchema,
} from '../validation/adminSchemas';
import { logger } from '../config/logger';
import { HTTP_STATUS } from '../config/constants';
import { AdminAuthenticatedRequest } from '../types';

/**
 * GET /api/admin/users
 * List users with pagination, search, and filters
 */
export async function listUsers(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    // Validate query parameters
    const parseResult = listUsersQuerySchema.safeParse(req.query);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid query parameters',
        details: parseResult.error.errors,
      });
      return;
    }

    const result = await getUserList(parseResult.data);

    res.status(HTTP_STATUS.OK).json(result);
  } catch (error) {
    logger.error({ error }, 'Error listing users');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to list users',
    });
  }
}

/**
 * GET /api/admin/users/:id
 * Get detailed user information
 */
export async function getUser(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    // Validate ID parameter
    const parseResult = userIdParamSchema.safeParse(req.params);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid user ID',
        details: parseResult.error.errors,
      });
      return;
    }

    const { id } = parseResult.data;
    const user = await getUserDetail(id);

    if (!user) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'User not found',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json(user);
  } catch (error) {
    logger.error({ error }, 'Error getting user detail');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to get user details',
    });
  }
}

/**
 * POST /api/admin/users/:id/export
 * Export all user data for GDPR compliance
 */
export async function exportUser(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    // Validate ID parameter
    const parseResult = userIdParamSchema.safeParse(req.params);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid user ID',
        details: parseResult.error.errors,
      });
      return;
    }

    const { id } = parseResult.data;
    const data = await exportUserData(id);

    if (!data) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'User not found',
      });
      return;
    }

    // Set headers for file download
    res.setHeader('Content-Type', 'application/json');
    res.setHeader(
      'Content-Disposition',
      `attachment; filename=user-${id}-export-${Date.now()}.json`
    );

    res.status(HTTP_STATUS.OK).json(data);
  } catch (error) {
    logger.error({ error }, 'Error exporting user data');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to export user data',
    });
  }
}

/**
 * DELETE /api/admin/users/:id
 * Delete user account for GDPR compliance
 * Requires SUPER_ADMIN role (enforced by middleware)
 */
export async function deleteUser(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    // Validate ID parameter
    const idParseResult = userIdParamSchema.safeParse(req.params);
    if (!idParseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid user ID',
        details: idParseResult.error.errors,
      });
      return;
    }

    // Validate body (reason required)
    const bodyParseResult = deleteUserSchema.safeParse(req.body);
    if (!bodyParseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Deletion reason required',
        details: bodyParseResult.error.errors,
      });
      return;
    }

    const { id } = idParseResult.data;
    const { reason } = bodyParseResult.data;
    const adminUserId = req.adminUser?.id || 'unknown';

    const deleted = await deleteUserAccount(id, adminUserId, reason);

    if (!deleted) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'User not found',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json({
      message: 'User account deleted successfully',
    });
  } catch (error) {
    logger.error({ error }, 'Error deleting user');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to delete user',
    });
  }
}
