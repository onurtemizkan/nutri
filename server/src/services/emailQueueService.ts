/**
 * Email Queue Service - Async email sending via Bull queues
 *
 * Features:
 * - Queue-based email processing for reliability
 * - Rate limiting to respect provider limits
 * - Retry logic for transient failures
 * - Campaign bulk sending
 * - Email sequence step processing
 */

import Bull from 'bull';
import { PrismaClient, EmailCampaignStatus, EmailSequenceStatus, Prisma } from '@prisma/client';
import { sendTemplateEmail, sendEmail } from './emailService';
import { logger } from '../config/logger';

const prisma = new PrismaClient();

// Queue configuration
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';

// Create email queues
export const emailQueue = new Bull('email-send', REDIS_URL, {
  defaultJobOptions: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 5000, // 5 seconds initial delay
    },
    removeOnComplete: 100, // Keep last 100 completed jobs
    removeOnFail: 500, // Keep last 500 failed jobs
  },
  limiter: {
    max: 50, // 50 emails per second (Resend limit is 100/sec for Pro)
    duration: 1000,
  },
});

export const campaignQueue = new Bull('email-campaign', REDIS_URL, {
  defaultJobOptions: {
    attempts: 1,
    removeOnComplete: 10,
    removeOnFail: 50,
  },
});

export const sequenceQueue = new Bull('email-sequence', REDIS_URL, {
  defaultJobOptions: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 60000, // 1 minute initial delay
    },
    removeOnComplete: 100,
    removeOnFail: 100,
  },
});

// Job types
interface SendEmailJob {
  type: 'template' | 'raw';
  userId: string;
  email: string;
  templateSlug?: string;
  variables?: Record<string, string | number | boolean>;
  campaignId?: string;
  sequenceEnrollmentId?: string;
  // For raw emails
  subject?: string;
  html?: string;
  text?: string;
}

interface CampaignJob {
  campaignId: string;
}

interface SequenceStepJob {
  enrollmentId: string;
}

// Process email send jobs
emailQueue.process(async (job) => {
  const data = job.data as SendEmailJob;

  logger.info(
    { jobId: job.id, type: data.type, userId: data.userId, templateSlug: data.templateSlug },
    'Processing email job'
  );

  if (data.type === 'template') {
    const result = await sendTemplateEmail({
      userId: data.userId,
      email: data.email,
      templateSlug: data.templateSlug!,
      variables: data.variables,
      campaignId: data.campaignId,
      sequenceEnrollmentId: data.sequenceEnrollmentId,
    });

    if (!result.success) {
      throw new Error(result.error || 'Failed to send template email');
    }

    return result;
  } else {
    // Raw email
    const result = await sendEmail({
      to: data.email,
      subject: data.subject!,
      html: data.html!,
      text: data.text,
    });

    if (!result.success) {
      throw new Error(result.error || 'Failed to send raw email');
    }

    return result;
  }
});

// Process campaign jobs
campaignQueue.process(async (job) => {
  const { campaignId } = job.data as CampaignJob;

  logger.info({ campaignId }, 'Processing campaign job');

  // Get campaign with template
  const campaign = await prisma.emailCampaign.findUnique({
    where: { id: campaignId },
    include: { template: true },
  });

  if (!campaign) {
    throw new Error(`Campaign not found: ${campaignId}`);
  }

  if (campaign.status !== EmailCampaignStatus.SCHEDULED) {
    logger.warn({ campaignId, status: campaign.status }, 'Campaign not in scheduled status');
    return;
  }

  // Update campaign status to SENDING
  await prisma.emailCampaign.update({
    where: { id: campaignId },
    data: {
      status: EmailCampaignStatus.SENDING,
      sentAt: new Date(),
    },
  });

  try {
    // Build audience query based on segment criteria
    const segmentCriteria = (campaign.segmentCriteria as Record<string, unknown>) || {};
    const userQuery = buildUserQuery(segmentCriteria);

    // Get users matching segment
    const users = await prisma.user.findMany({
      where: {
        ...userQuery,
        isActive: true,
      },
      select: {
        id: true,
        email: true,
        name: true,
      },
    });

    logger.info({ campaignId, userCount: users.length }, 'Sending campaign to users');

    // Queue individual emails
    let queuedCount = 0;
    for (const user of users) {
      // Check if user can receive marketing emails
      const preferences = await prisma.emailPreference.findUnique({
        where: { userId: user.id },
      });

      if (
        preferences?.isSuppressed ||
        preferences?.globalUnsubscribedAt ||
        !preferences?.marketingOptIn
      ) {
        continue;
      }

      await emailQueue.add({
        type: 'template',
        userId: user.id,
        email: user.email,
        templateSlug: campaign.template.slug,
        variables: {
          userName: user.name,
        },
        campaignId,
      });

      queuedCount++;
    }

    // Update campaign with actual sent count
    await prisma.emailCampaign.update({
      where: { id: campaignId },
      data: {
        status: EmailCampaignStatus.COMPLETED,
        completedAt: new Date(),
        actualSent: queuedCount,
      },
    });

    logger.info({ campaignId, queuedCount }, 'Campaign completed');

    return { campaignId, queuedCount };
  } catch (error) {
    // Mark campaign as failed
    await prisma.emailCampaign.update({
      where: { id: campaignId },
      data: {
        status: EmailCampaignStatus.FAILED,
      },
    });

    throw error;
  }
});

// Process sequence step jobs
sequenceQueue.process(async (job) => {
  const { enrollmentId } = job.data as SequenceStepJob;

  logger.info({ enrollmentId }, 'Processing sequence step');

  // Get enrollment with sequence
  const enrollment = await prisma.emailSequenceEnrollment.findUnique({
    where: { id: enrollmentId },
    include: { sequence: true },
  });

  if (!enrollment) {
    throw new Error(`Enrollment not found: ${enrollmentId}`);
  }

  if (enrollment.status !== EmailSequenceStatus.ACTIVE) {
    logger.info({ enrollmentId, status: enrollment.status }, 'Enrollment not active, skipping');
    return;
  }

  const steps = enrollment.sequence.steps as Array<{
    stepNumber: number;
    delayHours: number;
    templateSlug: string;
    condition?: Record<string, unknown>;
  }>;

  const currentStep = steps.find((s) => s.stepNumber === enrollment.currentStep);

  if (!currentStep) {
    // No more steps, complete the sequence
    await prisma.emailSequenceEnrollment.update({
      where: { id: enrollmentId },
      data: {
        status: EmailSequenceStatus.COMPLETED,
        completedAt: new Date(),
      },
    });

    // Update sequence analytics
    await prisma.emailSequence.update({
      where: { id: enrollment.sequenceId },
      data: {
        activeEnrollments: { decrement: 1 },
        completedEnrollments: { increment: 1 },
      },
    });

    return { enrollmentId, completed: true };
  }

  // Get user info
  const user = await prisma.user.findUnique({
    where: { id: enrollment.userId },
    select: { email: true, name: true },
  });

  if (!user) {
    throw new Error(`User not found: ${enrollment.userId}`);
  }

  // Check step condition if any
  if (currentStep.condition) {
    const conditionMet = await evaluateCondition(enrollment.userId, currentStep.condition);
    if (!conditionMet) {
      // Skip this step and move to next
      await scheduleNextStep(enrollmentId, enrollment.currentStep + 1, steps);
      return { enrollmentId, skipped: true, stepNumber: currentStep.stepNumber };
    }
  }

  // Send the email
  await emailQueue.add({
    type: 'template',
    userId: enrollment.userId,
    email: user.email,
    templateSlug: currentStep.templateSlug,
    variables: {
      userName: user.name,
    },
    sequenceEnrollmentId: enrollmentId,
  });

  // Schedule next step
  await scheduleNextStep(enrollmentId, enrollment.currentStep + 1, steps);

  return { enrollmentId, stepNumber: currentStep.stepNumber };
});

/**
 * Build Prisma where clause from segment criteria
 */
function buildUserQuery(criteria: Record<string, unknown>): Record<string, unknown> {
  const where: Record<string, unknown> = {};

  if (criteria.subscriptionTier) {
    where.subscriptionTier = criteria.subscriptionTier;
  }

  if (criteria.activityLevel) {
    where.activityLevel = criteria.activityLevel;
  }

  if (criteria.primaryGoal) {
    where.primaryGoal = criteria.primaryGoal;
  }

  if (criteria.createdAfter) {
    where.createdAt = { gte: new Date(criteria.createdAfter as string) };
  }

  if (criteria.createdBefore) {
    where.createdAt = {
      ...((where.createdAt as object) || {}),
      lte: new Date(criteria.createdBefore as string),
    };
  }

  return where;
}

/**
 * Evaluate a condition for sequence step
 */
async function evaluateCondition(
  userId: string,
  condition: Record<string, unknown>
): Promise<boolean> {
  // Simple condition evaluation
  if (condition.hasLoggedMeal) {
    const mealCount = await prisma.meal.count({
      where: { userId },
    });
    return mealCount > 0;
  }

  if (condition.hasActiveSubscription) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { subscriptionTier: true },
    });
    return user?.subscriptionTier === 'PRO';
  }

  // Default: condition met
  return true;
}

/**
 * Schedule the next step in a sequence
 */
async function scheduleNextStep(
  enrollmentId: string,
  nextStepNumber: number,
  steps: Array<{ stepNumber: number; delayHours: number; templateSlug: string }>
): Promise<void> {
  const nextStep = steps.find((s) => s.stepNumber === nextStepNumber);

  if (!nextStep) {
    // No more steps
    return;
  }

  const delayMs = nextStep.delayHours * 60 * 60 * 1000;
  const scheduledAt = new Date(Date.now() + delayMs);

  // Update enrollment
  await prisma.emailSequenceEnrollment.update({
    where: { id: enrollmentId },
    data: {
      currentStep: nextStepNumber,
      nextStepScheduledAt: scheduledAt,
    },
  });

  // Queue the job with delay
  await sequenceQueue.add({ enrollmentId }, { delay: delayMs });
}

// Queue helper functions

/**
 * Queue a template email for sending
 */
export async function queueTemplateEmail(
  userId: string,
  email: string,
  templateSlug: string,
  variables?: Record<string, string | number | boolean>,
  options?: { priority?: number; delay?: number }
): Promise<Bull.Job<SendEmailJob>> {
  return emailQueue.add(
    {
      type: 'template',
      userId,
      email,
      templateSlug,
      variables,
    },
    {
      priority: options?.priority,
      delay: options?.delay,
    }
  );
}

/**
 * Queue a campaign for sending
 */
export async function queueCampaign(
  campaignId: string,
  scheduledAt?: Date
): Promise<Bull.Job<CampaignJob>> {
  const delay = scheduledAt ? scheduledAt.getTime() - Date.now() : 0;

  return campaignQueue.add({ campaignId }, { delay: Math.max(0, delay) });
}

/**
 * Enroll a user in an email sequence
 */
export async function enrollInSequence(
  userId: string,
  sequenceId: string,
  metadata?: Record<string, unknown>
): Promise<string | null> {
  // Check if sequence is active
  const sequence = await prisma.emailSequence.findUnique({
    where: { id: sequenceId },
  });

  if (!sequence || !sequence.isActive) {
    logger.warn({ userId, sequenceId }, 'Cannot enroll in inactive sequence');
    return null;
  }

  // Check if already enrolled
  const existing = await prisma.emailSequenceEnrollment.findUnique({
    where: { userId_sequenceId: { userId, sequenceId } },
  });

  if (existing) {
    logger.info({ userId, sequenceId }, 'User already enrolled in sequence');
    return existing.id;
  }

  // Create enrollment
  const enrollment = await prisma.emailSequenceEnrollment.create({
    data: {
      userId,
      sequenceId,
      currentStep: 1,
      status: EmailSequenceStatus.ACTIVE,
      metadata: (metadata as Prisma.InputJsonValue) ?? Prisma.JsonNull,
    },
  });

  // Update sequence analytics
  await prisma.emailSequence.update({
    where: { id: sequenceId },
    data: {
      totalEnrollments: { increment: 1 },
      activeEnrollments: { increment: 1 },
    },
  });

  // Get first step delay
  const steps = sequence.steps as Array<{ stepNumber: number; delayHours: number }>;
  const firstStep = steps.find((s) => s.stepNumber === 1);

  if (firstStep) {
    const delayMs = firstStep.delayHours * 60 * 60 * 1000;

    await sequenceQueue.add({ enrollmentId: enrollment.id }, { delay: delayMs });

    await prisma.emailSequenceEnrollment.update({
      where: { id: enrollment.id },
      data: {
        nextStepScheduledAt: new Date(Date.now() + delayMs),
      },
    });
  }

  logger.info({ userId, sequenceId, enrollmentId: enrollment.id }, 'User enrolled in sequence');

  return enrollment.id;
}

/**
 * Trigger sequences for a specific event
 */
export async function triggerSequenceEvent(userId: string, eventType: string): Promise<void> {
  // Find active sequences for this trigger
  const sequences = await prisma.emailSequence.findMany({
    where: {
      triggerEvent: eventType as any,
      isActive: true,
    },
  });

  for (const sequence of sequences) {
    await enrollInSequence(userId, sequence.id);
  }
}

// Queue event handlers for monitoring
emailQueue.on('completed', (job, result) => {
  logger.info({ jobId: job.id, result }, 'Email job completed');
});

emailQueue.on('failed', (job, err) => {
  logger.error({ jobId: job.id, error: err.message }, 'Email job failed');
});

campaignQueue.on('completed', (job, result) => {
  logger.info({ jobId: job.id, result }, 'Campaign job completed');
});

campaignQueue.on('failed', (job, err) => {
  logger.error({ jobId: job.id, error: err.message }, 'Campaign job failed');
});

sequenceQueue.on('completed', (job, result) => {
  logger.info({ jobId: job.id, result }, 'Sequence job completed');
});

sequenceQueue.on('failed', (job, err) => {
  logger.error({ jobId: job.id, error: err.message }, 'Sequence job failed');
});

export default {
  emailQueue,
  campaignQueue,
  sequenceQueue,
  queueTemplateEmail,
  queueCampaign,
  enrollInSequence,
  triggerSequenceEvent,
};
