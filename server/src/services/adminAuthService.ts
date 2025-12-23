import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import speakeasy from 'speakeasy';
import QRCode from 'qrcode';
import prisma from '../config/database';
import { AdminRole } from '@prisma/client';
import { logger } from '../config/logger';

// Constants
const ADMIN_SESSION_EXPIRY = '8h';
const SALT_ROUNDS = 12;

/**
 * Get JWT secret with production validation
 * Throws an error if JWT_SECRET is not set in production environment
 */
function getJwtSecret(): string {
  const secret = process.env.JWT_SECRET;

  if (!secret) {
    if (process.env.NODE_ENV === 'production') {
      throw new Error(
        'FATAL: JWT_SECRET environment variable is not set. ' +
        'This is required for secure admin authentication in production.'
      );
    }
    // Development fallback with warning
    logger.warn(
      'JWT_SECRET not set - using insecure default. DO NOT use in production!'
    );
    return 'admin-secret-change-in-production-INSECURE';
  }

  return secret;
}

const JWT_SECRET = getJwtSecret();

// Types
export interface AdminSessionPayload {
  adminUserId: string;
  email: string;
  role: AdminRole;
  sessionId: string;
  type: 'admin';
}

export interface LoginResult {
  token?: string;
  requiresMFA: boolean;
  mfaSetupRequired?: boolean;
  qrCode?: string;
  pendingToken?: string;
}

export interface AdminUserInfo {
  id: string;
  email: string;
  name: string;
  role: AdminRole;
  mfaEnabled: boolean;
}

/**
 * Generate a secure session ID
 */
function generateSessionId(): string {
  return `sess_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
}

/**
 * Hash a password using bcrypt
 */
export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, SALT_ROUNDS);
}

/**
 * Verify a password against a hash
 */
export async function verifyPassword(
  password: string,
  hash: string
): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

/**
 * Generate a JWT token for admin session
 */
export function generateAdminToken(payload: AdminSessionPayload): string {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: ADMIN_SESSION_EXPIRY });
}

/**
 * Verify and decode an admin JWT token
 */
export function verifyAdminToken(token: string): AdminSessionPayload | null {
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as AdminSessionPayload;
    if (decoded.type !== 'admin') {
      return null;
    }
    return decoded;
  } catch {
    return null;
  }
}

/**
 * Generate a pending MFA token (short-lived, for MFA verification step)
 */
function generatePendingMFAToken(adminUserId: string): string {
  return jwt.sign(
    { adminUserId, type: 'mfa_pending' },
    JWT_SECRET,
    { expiresIn: '5m' } // 5 minutes to complete MFA
  );
}

/**
 * Verify pending MFA token
 */
function verifyPendingMFAToken(
  token: string
): { adminUserId: string } | null {
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as {
      adminUserId: string;
      type: string;
    };
    if (decoded.type !== 'mfa_pending') {
      return null;
    }
    return { adminUserId: decoded.adminUserId };
  } catch {
    return null;
  }
}

/**
 * Login an admin user
 */
export async function loginAdmin(
  email: string,
  password: string,
  ipAddress: string
): Promise<LoginResult> {
  // Find admin user
  const adminUser = await prisma.adminUser.findUnique({
    where: { email: email.toLowerCase() },
  });

  if (!adminUser) {
    logger.warn({ email }, 'Admin login failed: user not found');
    throw new Error('Invalid credentials');
  }

  if (!adminUser.isActive) {
    logger.warn({ email }, 'Admin login failed: account disabled');
    throw new Error('Account is disabled');
  }

  // Verify password
  const passwordValid = await verifyPassword(password, adminUser.passwordHash);
  if (!passwordValid) {
    logger.warn({ email }, 'Admin login failed: invalid password');
    throw new Error('Invalid credentials');
  }

  // Check if MFA is enabled
  if (adminUser.mfaEnabled && adminUser.mfaSecret) {
    // MFA required - return pending token
    const pendingToken = generatePendingMFAToken(adminUser.id);
    return {
      requiresMFA: true,
      pendingToken,
    };
  }

  // Check if MFA setup is required (first login for SUPER_ADMIN)
  if (adminUser.role === 'SUPER_ADMIN' && !adminUser.mfaEnabled) {
    const pendingToken = generatePendingMFAToken(adminUser.id);
    const mfaSetup = await setupMFA(adminUser.id);
    return {
      requiresMFA: true,
      mfaSetupRequired: true,
      pendingToken,
      qrCode: mfaSetup.qrCode,
    };
  }

  // No MFA - generate session token directly
  const sessionId = generateSessionId();
  const token = generateAdminToken({
    adminUserId: adminUser.id,
    email: adminUser.email,
    role: adminUser.role,
    sessionId,
    type: 'admin',
  });

  // Update last login
  await prisma.adminUser.update({
    where: { id: adminUser.id },
    data: {
      lastLoginAt: new Date(),
      lastLoginIp: ipAddress,
    },
  });

  logger.info({ email, adminUserId: adminUser.id }, 'Admin login successful');

  return {
    token,
    requiresMFA: false,
  };
}

/**
 * Set up MFA for an admin user
 */
export async function setupMFA(
  adminUserId: string
): Promise<{ secret: string; qrCode: string }> {
  const adminUser = await prisma.adminUser.findUnique({
    where: { id: adminUserId },
  });

  if (!adminUser) {
    throw new Error('Admin user not found');
  }

  // Generate TOTP secret
  const secret = speakeasy.generateSecret({
    name: `Nutri Admin (${adminUser.email})`,
    issuer: 'Nutri Admin Panel',
    length: 20,
  });

  // Store the secret (but don't enable MFA yet - that happens after first verification)
  await prisma.adminUser.update({
    where: { id: adminUserId },
    data: { mfaSecret: secret.base32 },
  });

  // Generate QR code
  const otpauthUrl = secret.otpauth_url;
  if (!otpauthUrl) {
    throw new Error('Failed to generate OTP auth URL');
  }

  const qrCode = await QRCode.toDataURL(otpauthUrl);

  logger.info({ adminUserId }, 'MFA setup initiated');

  return {
    secret: secret.base32,
    qrCode,
  };
}

/**
 * Verify MFA token and complete login
 */
export async function verifyMFA(
  pendingToken: string,
  mfaCode: string,
  ipAddress: string
): Promise<{ token: string }> {
  // Verify pending token
  const pending = verifyPendingMFAToken(pendingToken);
  if (!pending) {
    throw new Error('Invalid or expired MFA session');
  }

  const adminUser = await prisma.adminUser.findUnique({
    where: { id: pending.adminUserId },
  });

  if (!adminUser || !adminUser.mfaSecret) {
    throw new Error('MFA not configured');
  }

  if (!adminUser.isActive) {
    throw new Error('Account is disabled');
  }

  // Verify TOTP code
  const verified = speakeasy.totp.verify({
    secret: adminUser.mfaSecret,
    encoding: 'base32',
    token: mfaCode,
    window: 1, // Allow 1 step before/after for clock drift
  });

  if (!verified) {
    logger.warn({ adminUserId: adminUser.id }, 'MFA verification failed');
    throw new Error('Invalid MFA code');
  }

  // If this is the first successful MFA verification, enable MFA
  if (!adminUser.mfaEnabled) {
    await prisma.adminUser.update({
      where: { id: adminUser.id },
      data: { mfaEnabled: true },
    });
    logger.info({ adminUserId: adminUser.id }, 'MFA enabled for admin user');
  }

  // Generate session token
  const sessionId = generateSessionId();
  const token = generateAdminToken({
    adminUserId: adminUser.id,
    email: adminUser.email,
    role: adminUser.role,
    sessionId,
    type: 'admin',
  });

  // Update last login
  await prisma.adminUser.update({
    where: { id: adminUser.id },
    data: {
      lastLoginAt: new Date(),
      lastLoginIp: ipAddress,
    },
  });

  logger.info({ adminUserId: adminUser.id }, 'Admin MFA verification successful');

  return { token };
}

/**
 * Get admin user info from token
 */
export async function getAdminUser(
  adminUserId: string
): Promise<AdminUserInfo | null> {
  const adminUser = await prisma.adminUser.findUnique({
    where: { id: adminUserId },
    select: {
      id: true,
      email: true,
      name: true,
      role: true,
      mfaEnabled: true,
    },
  });

  return adminUser;
}

/**
 * Create a new admin user (for seeding or admin management)
 */
export async function createAdminUser(
  email: string,
  password: string,
  name: string,
  role: AdminRole = 'SUPPORT'
): Promise<AdminUserInfo> {
  const passwordHash = await hashPassword(password);

  const adminUser = await prisma.adminUser.create({
    data: {
      email: email.toLowerCase(),
      passwordHash,
      name,
      role,
    },
    select: {
      id: true,
      email: true,
      name: true,
      role: true,
      mfaEnabled: true,
    },
  });

  logger.info({ email, role }, 'Admin user created');

  return adminUser;
}

/**
 * Check if any admin users exist (for initial setup)
 */
export async function hasAdminUsers(): Promise<boolean> {
  const count = await prisma.adminUser.count();
  return count > 0;
}
