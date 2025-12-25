import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';
import prisma from '../config/database';
import { config } from '../config/env';
import { RegisterInput, LoginInput } from '../types';
import { logger } from '../config/logger';

export class AuthService {
  async register(data: RegisterInput) {
    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email: data.email },
    });

    if (existingUser) {
      throw new Error('User with this email already exists');
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(data.password, 10);

    // Create user
    const user = await prisma.user.create({
      data: {
        email: data.email,
        password: hashedPassword,
        name: data.name,
      },
      select: {
        id: true,
        email: true,
        name: true,
        goalCalories: true,
        goalProtein: true,
        goalCarbs: true,
        goalFat: true,
        currentWeight: true,
        goalWeight: true,
        height: true,
        activityLevel: true,
        createdAt: true,
      },
    });

    // Generate JWT token
    const token = jwt.sign({ userId: user.id }, config.jwt.secret, {
      expiresIn: config.jwt.expiresIn,
    } as jwt.SignOptions);

    return { user, token };
  }

  async login(data: LoginInput) {
    // Find user
    const user = await prisma.user.findUnique({
      where: { email: data.email },
    });

    if (!user) {
      throw new Error('Invalid email or password');
    }

    // Check if user has a password (might be null for Apple Sign In users)
    if (!user.password) {
      throw new Error('This account uses Apple Sign In. Please sign in with Apple.');
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(data.password, user.password);

    if (!isPasswordValid) {
      throw new Error('Invalid email or password');
    }

    // Generate JWT token
    const token = jwt.sign({ userId: user.id }, config.jwt.secret, {
      expiresIn: config.jwt.expiresIn,
    } as jwt.SignOptions);

    return {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        goalCalories: user.goalCalories,
        goalProtein: user.goalProtein,
        goalCarbs: user.goalCarbs,
        goalFat: user.goalFat,
        currentWeight: user.currentWeight,
        goalWeight: user.goalWeight,
        height: user.height,
        activityLevel: user.activityLevel,
      },
      token,
    };
  }

  async getUserProfile(userId: string) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        name: true,
        goalCalories: true,
        goalProtein: true,
        goalCarbs: true,
        goalFat: true,
        currentWeight: true,
        goalWeight: true,
        height: true,
        activityLevel: true,
        createdAt: true,
      },
    });

    if (!user) {
      throw new Error('User not found');
    }

    return user;
  }

  async requestPasswordReset(email: string) {
    // Find user
    const user = await prisma.user.findUnique({
      where: { email },
    });

    // Don't reveal if user exists or not for security
    if (!user) {
      return {
        message: 'If an account with that email exists, a password reset link has been sent.',
      };
    }

    // Generate reset token
    const resetToken = crypto.randomBytes(32).toString('hex');
    const hashedToken = crypto.createHash('sha256').update(resetToken).digest('hex');

    // Token expires in 1 hour
    const resetTokenExpiresAt = new Date(Date.now() + 60 * 60 * 1000);

    // Save token to database
    await prisma.user.update({
      where: { id: user.id },
      data: {
        resetToken: hashedToken,
        resetTokenExpiresAt,
      },
    });

    // In a real app, you would send an email here with the reset link
    // For now, we'll return the token (in production, this would be sent via email)
    if (process.env.NODE_ENV === 'development') {
      // Log only a short hash prefix for debugging - never log the actual token
      const tokenHashPrefix = crypto
        .createHash('sha256')
        .update(resetToken)
        .digest('hex')
        .substring(0, 8);
      logger.debug(
        {
          tokenHashPrefix,
          email: user.email,
          expiresAt: resetTokenExpiresAt.toISOString(),
        },
        'Password reset token generated (hash prefix for debugging only)'
      );
    }

    return {
      message: 'If an account with that email exists, a password reset link has been sent.',
      // In development/test, return the token so we can test
      ...((process.env.NODE_ENV === 'development' || process.env.NODE_ENV === 'test') && {
        resetToken,
      }),
    };
  }

  async resetPassword(token: string, newPassword: string) {
    // Hash the token to compare with database
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    // Find user with valid token
    const user = await prisma.user.findFirst({
      where: {
        resetToken: hashedToken,
        resetTokenExpiresAt: {
          gt: new Date(), // Token must not be expired
        },
      },
    });

    if (!user) {
      throw new Error('Invalid or expired reset token');
    }

    // Hash new password
    const hashedPassword = await bcrypt.hash(newPassword, 10);

    // Update password and clear reset token
    await prisma.user.update({
      where: { id: user.id },
      data: {
        password: hashedPassword,
        resetToken: null,
        resetTokenExpiresAt: null,
      },
    });

    return { message: 'Password has been reset successfully' };
  }

  async verifyResetToken(token: string) {
    // Hash the token to compare with database
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    // Find user with valid token
    const user = await prisma.user.findFirst({
      where: {
        resetToken: hashedToken,
        resetTokenExpiresAt: {
          gt: new Date(),
        },
      },
    });

    if (!user) {
      throw new Error('Invalid or expired reset token');
    }

    return { valid: true, email: user.email };
  }

  async deleteAccount(userId: string) {
    // Verify user exists
    const user = await prisma.user.findUnique({
      where: { id: userId },
    });

    if (!user) {
      throw new Error('User not found');
    }

    // Delete user - Prisma will cascade delete all related data
    // (meals, health metrics, activities, etc.) due to onDelete: Cascade
    await prisma.user.delete({
      where: { id: userId },
    });

    return { message: 'Account deleted successfully' };
  }

  async appleSignIn(data: {
    identityToken: string;
    authorizationCode: string;
    user?: {
      email?: string;
      name?: {
        firstName?: string;
        lastName?: string;
      };
    };
  }) {
    // Note: In production, you should verify the identityToken with Apple's servers
    // For now, we'll decode it without verification (development only)
    // See: https://developer.apple.com/documentation/sign_in_with_apple/sign_in_with_apple_rest_api/verifying_a_user

    let appleId: string;
    let email: string | undefined;

    try {
      // Decode the identity token (without verification)
      // The token is a JWT with three parts: header.payload.signature
      const tokenParts = data.identityToken.split('.');
      if (tokenParts.length !== 3) {
        throw new Error('Invalid identity token format');
      }

      const payload = JSON.parse(Buffer.from(tokenParts[1], 'base64').toString('utf8'));
      appleId = payload.sub; // Apple user ID
      email = payload.email || data.user?.email;

      if (!appleId) {
        throw new Error('Invalid identity token: missing user ID');
      }
    } catch {
      throw new Error('Failed to decode Apple identity token');
    }

    // Check if user already exists with this Apple ID
    let user = await prisma.user.findUnique({
      where: { appleId },
    });

    if (!user) {
      // Check if user exists with this email
      if (email) {
        user = await prisma.user.findUnique({
          where: { email },
        });

        if (user) {
          // Link Apple ID to existing account
          user = await prisma.user.update({
            where: { id: user.id },
            data: { appleId },
          });
        }
      }
    }

    if (!user) {
      // Create new user
      if (!email) {
        throw new Error('Email is required for new users');
      }

      const firstName = data.user?.name?.firstName || '';
      const lastName = data.user?.name?.lastName || '';
      const name = `${firstName} ${lastName}`.trim() || '';

      user = await prisma.user.create({
        data: {
          email,
          appleId,
          name,
          password: null, // No password for OAuth users
        },
      });
    }

    // Generate JWT token
    const token = jwt.sign({ userId: user.id }, config.jwt.secret, {
      expiresIn: config.jwt.expiresIn,
    } as jwt.SignOptions);

    return {
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        goalCalories: user.goalCalories,
        goalProtein: user.goalProtein,
        goalCarbs: user.goalCarbs,
        goalFat: user.goalFat,
        currentWeight: user.currentWeight,
        goalWeight: user.goalWeight,
        height: user.height,
        activityLevel: user.activityLevel,
      },
      token,
    };
  }
}

export const authService = new AuthService();
