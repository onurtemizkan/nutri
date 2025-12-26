import { Request, Response } from 'express';
import { authService } from '../services/authService';
import { AuthenticatedRequest, UpdateUserProfileInput } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  appleSignInSchema,
  registerSchema,
  loginSchema,
  forgotPasswordSchema,
  resetPasswordSchema,
} from '../validation/schemas';
import { withErrorHandling, ErrorHandlers } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { z } from 'zod';

const verifyTokenSchema = z.object({
  token: z.string().min(1, 'Reset token is required'),
});

export class AuthController {
  register = withErrorHandling(async (req: Request, res: Response) => {
    const validatedData = registerSchema.parse(req.body);
    const result = await authService.register(validatedData);

    res.status(HTTP_STATUS.CREATED).json(result);
  });

  login = ErrorHandlers.withUnauthorized(async (req: Request, res: Response) => {
    const validatedData = loginSchema.parse(req.body);
    const result = await authService.login(validatedData);

    res.status(HTTP_STATUS.OK).json(result);
  });

  getProfile = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const user = await authService.getUserProfile(userId);
    res.status(HTTP_STATUS.OK).json(user);
  });

  updateProfile = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data: UpdateUserProfileInput = req.body;
    const user = await authService.updateUserProfile(userId, data);

    res.status(HTTP_STATUS.OK).json(user);
  });

  forgotPassword = withErrorHandling(async (req: Request, res: Response) => {
    const validatedData = forgotPasswordSchema.parse(req.body);
    const result = await authService.requestPasswordReset(validatedData.email);

    res.status(HTTP_STATUS.OK).json(result);
  });

  resetPassword = withErrorHandling(async (req: Request, res: Response) => {
    const validatedData = resetPasswordSchema.parse(req.body);
    const result = await authService.resetPassword(validatedData.token, validatedData.newPassword);

    res.status(HTTP_STATUS.OK).json(result);
  });

  verifyResetToken = withErrorHandling(async (req: Request, res: Response) => {
    const validatedData = verifyTokenSchema.parse(req.body);
    const result = await authService.verifyResetToken(validatedData.token);

    res.status(HTTP_STATUS.OK).json(result);
  });

  appleSignIn = withErrorHandling(async (req: Request, res: Response) => {
    const validatedData = appleSignInSchema.parse(req.body);
    const result = await authService.appleSignIn(validatedData);

    res.status(HTTP_STATUS.OK).json(result);
  });

  deleteAccount = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await authService.deleteAccount(userId);
    res.status(HTTP_STATUS.OK).json(result);
  });
}

export const authController = new AuthController();
