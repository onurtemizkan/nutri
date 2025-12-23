import prisma from '../config/database';
import { Prisma, WebhookEventStatus } from '@prisma/client';
import { logger } from '../config/logger';

// Types
export interface ListWebhooksParams {
  notificationType?: string;
  status?: WebhookEventStatus;
  startDate?: Date;
  endDate?: Date;
  originalTransactionId?: string;
  userId?: string;
  page: number;
  limit: number;
}

export interface WebhookEventListItem {
  id: string;
  notificationType: string;
  subtype: string | null;
  originalTransactionId: string | null;
  status: WebhookEventStatus;
  userId: string | null;
  receivedAt: Date;
  processedAt: Date | null;
}

export interface PaginatedWebhooksResponse {
  events: WebhookEventListItem[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export interface WebhookEventDetail {
  id: string;
  notificationType: string;
  subtype: string | null;
  originalTransactionId: string | null;
  transactionId: string | null;
  bundleId: string | null;
  payload: Prisma.JsonValue;
  status: WebhookEventStatus;
  errorMessage: string | null;
  retryCount: number;
  lastRetryAt: Date | null;
  userId: string | null;
  receivedAt: Date;
  processedAt: Date | null;
  createdAt: Date;
  user?: {
    id: string;
    email: string;
    name: string;
  } | null;
}

/**
 * Get paginated list of webhook events with filters
 */
export async function getWebhookList(
  params: ListWebhooksParams
): Promise<PaginatedWebhooksResponse> {
  const {
    notificationType,
    status,
    startDate,
    endDate,
    originalTransactionId,
    userId,
    page,
    limit,
  } = params;

  // Build where clause
  const where: Prisma.AppStoreWebhookEventWhereInput = {};

  if (notificationType) {
    where.notificationType = notificationType;
  }

  if (status) {
    where.status = status;
  }

  if (originalTransactionId) {
    where.originalTransactionId = {
      contains: originalTransactionId,
      mode: 'insensitive',
    };
  }

  if (userId) {
    where.userId = userId;
  }

  if (startDate || endDate) {
    where.receivedAt = {};
    if (startDate) {
      where.receivedAt.gte = startDate;
    }
    if (endDate) {
      where.receivedAt.lte = endDate;
    }
  }

  const skip = (page - 1) * limit;

  const [events, total] = await Promise.all([
    prisma.appStoreWebhookEvent.findMany({
      where,
      skip,
      take: limit,
      orderBy: { receivedAt: 'desc' },
      select: {
        id: true,
        notificationType: true,
        subtype: true,
        originalTransactionId: true,
        status: true,
        userId: true,
        receivedAt: true,
        processedAt: true,
      },
    }),
    prisma.appStoreWebhookEvent.count({ where }),
  ]);

  const totalPages = Math.ceil(total / limit);

  return {
    events,
    pagination: {
      page,
      limit,
      total,
      totalPages,
    },
  };
}

/**
 * Get detailed webhook event by ID
 */
export async function getWebhookDetail(
  eventId: string
): Promise<WebhookEventDetail | null> {
  const event = await prisma.appStoreWebhookEvent.findUnique({
    where: { id: eventId },
  });

  if (!event) {
    return null;
  }

  // If we have a userId, fetch user details
  let user = null;
  if (event.userId) {
    user = await prisma.user.findUnique({
      where: { id: event.userId },
      select: {
        id: true,
        email: true,
        name: true,
      },
    });
  }

  return {
    ...event,
    user,
  };
}

/**
 * Search webhook events by originalTransactionId
 */
export async function searchWebhooksByTransactionId(
  originalTransactionId: string
): Promise<WebhookEventListItem[]> {
  const events = await prisma.appStoreWebhookEvent.findMany({
    where: {
      originalTransactionId: {
        contains: originalTransactionId,
        mode: 'insensitive',
      },
    },
    orderBy: { receivedAt: 'desc' },
    take: 50,
    select: {
      id: true,
      notificationType: true,
      subtype: true,
      originalTransactionId: true,
      status: true,
      userId: true,
      receivedAt: true,
      processedAt: true,
    },
  });

  return events;
}

/**
 * Retry processing a failed webhook event
 *
 * This function marks the webhook event as PENDING for retry processing.
 * In a production environment, a job queue system (e.g., Bull with Redis)
 * should be used to handle the actual retry processing asynchronously.
 *
 * TODO: Integrate with a proper job queue system for webhook retry processing
 * Current implementation only updates the status - actual reprocessing should
 * be handled by a background worker/queue consumer.
 */
export async function retryWebhookEvent(
  eventId: string,
  adminUserId: string
): Promise<WebhookEventDetail | null> {
  const event = await prisma.appStoreWebhookEvent.findUnique({
    where: { id: eventId },
  });

  if (!event) {
    return null;
  }

  // Update retry metadata - marks the event as PENDING for retry
  const updated = await prisma.appStoreWebhookEvent.update({
    where: { id: eventId },
    data: {
      status: 'PENDING',
      errorMessage: null,
      retryCount: { increment: 1 },
      lastRetryAt: new Date(),
    },
  });

  logger.info(
    {
      eventId,
      adminUserId,
      retryCount: updated.retryCount,
      notificationType: event.notificationType,
    },
    'Webhook event retry initiated - event marked as PENDING'
  );

  // NOTE: In production, this should trigger a job queue to process the retry.
  // The queue worker would:
  // 1. Fetch the event payload from the database
  // 2. Call the appropriate webhook handler based on notificationType
  // 3. Update the status to SUCCESS or FAILED based on the result
  //
  // Example with Bull queue:
  // await webhookRetryQueue.add('process-retry', { eventId }, { attempts: 3 });

  return getWebhookDetail(eventId);
}

/**
 * Get webhook event statistics
 */
export async function getWebhookStats(): Promise<{
  total: number;
  pending: number;
  success: number;
  failed: number;
  byType: { type: string; count: number }[];
}> {
  const [total, pending, success, failed, byType] = await Promise.all([
    prisma.appStoreWebhookEvent.count(),
    prisma.appStoreWebhookEvent.count({ where: { status: 'PENDING' } }),
    prisma.appStoreWebhookEvent.count({ where: { status: 'SUCCESS' } }),
    prisma.appStoreWebhookEvent.count({ where: { status: 'FAILED' } }),
    prisma.appStoreWebhookEvent.groupBy({
      by: ['notificationType'],
      _count: { notificationType: true },
      orderBy: { _count: { notificationType: 'desc' } },
    }),
  ]);

  return {
    total,
    pending,
    success,
    failed,
    byType: byType.map((item) => ({
      type: item.notificationType,
      count: item._count.notificationType,
    })),
  };
}
