import { Request, Response } from 'express';
import {
  loginAdmin,
  verifyMFA,
  getAdminUser,
  createAdminUser,
  hasAdminUsers,
} from '../services/adminAuthService';
import {
  adminLoginSchema,
  adminMFAVerifySchema,
} from '../validation/adminSchemas';
import { logger } from '../config/logger';
import { HTTP_STATUS } from '../config/constants';

/**
 * POST /api/admin/auth/login
 * Admin login endpoint
 */
export async function login(req: Request, res: Response): Promise<void> {
  try {
    // Validate input
    const parseResult = adminLoginSchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Validation failed',
        details: parseResult.error.errors,
      });
      return;
    }

    const { email, password } = parseResult.data;
    const ipAddress = req.ip || req.socket.remoteAddress || 'unknown';

    const result = await loginAdmin(email, password, ipAddress);

    if (result.requiresMFA) {
      // MFA required - return pending token
      res.status(HTTP_STATUS.OK).json({
        requiresMFA: true,
        mfaSetupRequired: result.mfaSetupRequired || false,
        pendingToken: result.pendingToken,
        qrCode: result.qrCode, // Only present if MFA setup is required
      });
      return;
    }

    // Login successful without MFA
    res.status(HTTP_STATUS.OK).json({
      token: result.token,
      requiresMFA: false,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Login failed';
    logger.warn({ error: message }, 'Admin login error');

    // Don't reveal whether email exists or password is wrong
    res.status(HTTP_STATUS.UNAUTHORIZED).json({
      error: 'Invalid credentials',
    });
  }
}

/**
 * POST /api/admin/auth/mfa/verify
 * Verify MFA code and complete login
 */
export async function verifyMFACode(req: Request, res: Response): Promise<void> {
  try {
    // Validate input
    const parseResult = adminMFAVerifySchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Validation failed',
        details: parseResult.error.errors,
      });
      return;
    }

    const { pendingToken, code } = parseResult.data;
    const ipAddress = req.ip || req.socket.remoteAddress || 'unknown';

    const result = await verifyMFA(pendingToken, code, ipAddress);

    res.status(HTTP_STATUS.OK).json({
      token: result.token,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'MFA verification failed';
    logger.warn({ error: message }, 'MFA verification error');

    res.status(HTTP_STATUS.UNAUTHORIZED).json({
      error: message,
    });
  }
}

/**
 * POST /api/admin/auth/logout
 * Admin logout endpoint (invalidate session on client side)
 */
export async function logout(_req: Request, res: Response): Promise<void> {
  // In a JWT-based system, we just tell the client to clear the token
  // For enhanced security, you could maintain a token blacklist in Redis
  res.status(HTTP_STATUS.OK).json({
    message: 'Logged out successfully',
  });
}

/**
 * GET /api/admin/auth/me
 * Get current admin user info
 */
export async function getMe(req: Request, res: Response): Promise<void> {
  try {
    // req.adminUser is set by the adminAuth middleware
    const adminUserId = (req as Request & { adminUser?: { id: string } }).adminUser?.id;

    if (!adminUserId) {
      res.status(HTTP_STATUS.UNAUTHORIZED).json({
        error: 'Not authenticated',
      });
      return;
    }

    const adminUser = await getAdminUser(adminUserId);

    if (!adminUser) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'Admin user not found',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json(adminUser);
  } catch (error) {
    logger.error({ error }, 'Error fetching admin user info');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to fetch user info',
    });
  }
}

/**
 * POST /api/admin/auth/setup
 * Create initial admin user (only works if no admin users exist)
 */
export async function setupInitialAdmin(req: Request, res: Response): Promise<void> {
  try {
    // Check if any admin users exist
    const adminExists = await hasAdminUsers();
    if (adminExists) {
      res.status(HTTP_STATUS.FORBIDDEN).json({
        error: 'Admin users already exist. Contact an existing admin.',
      });
      return;
    }

    // Validate input
    const parseResult = adminLoginSchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Validation failed',
        details: parseResult.error.errors,
      });
      return;
    }

    const { email, password } = parseResult.data;
    const name = req.body.name || 'Admin';

    // Create SUPER_ADMIN user
    const adminUser = await createAdminUser(email, password, name, 'SUPER_ADMIN');

    logger.info({ email }, 'Initial admin user created');

    res.status(HTTP_STATUS.CREATED).json({
      message: 'Admin user created successfully',
      user: adminUser,
    });
  } catch (error) {
    logger.error({ error }, 'Error creating initial admin');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to create admin user',
    });
  }
}
