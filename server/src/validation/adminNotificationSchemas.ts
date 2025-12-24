/**
 * Validation schemas for admin notification endpoints
 */

import { z } from 'zod';

/**
 * Create campaign schema
 */
export const createCampaignSchema = z.object({
  name: z.string().min(1, 'Campaign name is required').max(100),
  title: z.string().min(1, 'Notification title is required').max(100),
  body: z.string().min(1, 'Notification body is required').max(500),
  targetSegment: z
    .enum(['ALL', 'PREMIUM', 'FREE', 'INACTIVE', 'NEW_USERS', 'ENGAGED'])
    .default('ALL'),
  scheduledFor: z.string().datetime().optional().nullable(),
  metadata: z.record(z.unknown()).optional(),
});

export type CreateCampaignInput = z.infer<typeof createCampaignSchema>;

/**
 * Update campaign schema
 */
export const updateCampaignSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  title: z.string().min(1).max(100).optional(),
  body: z.string().min(1).max(500).optional(),
  targetSegment: z
    .enum(['ALL', 'PREMIUM', 'FREE', 'INACTIVE', 'NEW_USERS', 'ENGAGED'])
    .optional(),
  scheduledFor: z.string().datetime().optional().nullable(),
  metadata: z.record(z.unknown()).optional(),
});

export type UpdateCampaignInput = z.infer<typeof updateCampaignSchema>;

/**
 * Send test notification schema
 */
export const sendTestNotificationSchema = z.object({
  title: z.string().min(1, 'Title is required').max(100),
  body: z.string().min(1, 'Body is required').max(500),
});

export type SendTestNotificationInput = z.infer<typeof sendTestNotificationSchema>;
