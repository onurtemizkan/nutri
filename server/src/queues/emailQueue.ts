/**
 * Email Queue
 *
 * Bull queue for processing email sends with:
 * - Rate limiting
 * - Retry logic with exponential backoff
 * - Job prioritization
 */

import Bull, { Job, Queue } from 'bull';
import { config } from '../config/env';
import { logger } from '../config/logger';

/**
 * Email job types
 */
export enum EmailJobType {
  TRANSACTIONAL = 'transactional',
  CAMPAIGN_BATCH = 'campaign_batch',
  SEQUENCE_STEP = 'sequence_step',
}

/**
 * Email job data interface
 */
export interface EmailJobData {
  type: EmailJobType;
  userId?: string;
  email: string;
  templateSlug: string;
  variables: Record<string, unknown>;
  campaignId?: string;
  sequenceEnrollmentId?: string;
  correlationId: string;
}

/**
 * Batch email job data
 */
export interface BatchEmailJobData {
  type: EmailJobType.CAMPAIGN_BATCH;
  campaignId: string;
  emails: Array<{
    userId: string;
    email: string;
    variables: Record<string, unknown>;
  }>;
  templateSlug: string;
  correlationId: string;
}

/**
 * Sequence step job data
 */
export interface SequenceStepJobData {
  type: EmailJobType.SEQUENCE_STEP;
  enrollmentId: string;
  correlationId: string;
}

export type AnyEmailJobData = EmailJobData | BatchEmailJobData | SequenceStepJobData;

/**
 * Email queue instance
 */
let emailQueue: Queue<AnyEmailJobData> | null = null;

/**
 * Get or create the email queue
 */
export function getEmailQueue(): Queue<AnyEmailJobData> {
  if (!emailQueue) {
    emailQueue = new Bull<AnyEmailJobData>('email-queue', config.redis.url, {
      defaultJobOptions: {
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 2000, // Start with 2 seconds
        },
        removeOnComplete: 100, // Keep last 100 completed jobs
        removeOnFail: 500, // Keep last 500 failed jobs for debugging
      },
      limiter: {
        max: config.email.rateLimitPerSecond,
        duration: 1000, // Per second
      },
      settings: {
        stalledInterval: 30000, // Check for stalled jobs every 30 seconds
        maxStalledCount: 2, // Retry stalled jobs twice before failing
      },
    });

    // Set up event handlers
    setupQueueEventHandlers(emailQueue);
  }

  return emailQueue;
}

/**
 * Set up queue event handlers for logging and monitoring
 */
function setupQueueEventHandlers(queue: Queue<AnyEmailJobData>): void {
  queue.on('completed', (job: Job<AnyEmailJobData>, result: unknown) => {
    logger.info(
      {
        jobId: job.id,
        type: job.data.type,
        correlationId: job.data.correlationId,
        result,
      },
      'Email job completed'
    );
  });

  queue.on('failed', (job: Job<AnyEmailJobData>, err: Error) => {
    logger.error(
      {
        jobId: job.id,
        type: job.data.type,
        correlationId: job.data.correlationId,
        error: err.message,
        attemptsMade: job.attemptsMade,
      },
      'Email job failed'
    );
  });

  queue.on('stalled', (job: Job<AnyEmailJobData>) => {
    logger.warn(
      {
        jobId: job.id,
        type: job.data.type,
        correlationId: job.data.correlationId,
      },
      'Email job stalled'
    );
  });

  queue.on('error', (error: Error) => {
    logger.error({ error: error.message }, 'Email queue error');
  });
}

/**
 * Add an email job to the queue
 */
export async function addEmailJob(
  data: EmailJobData,
  options?: Bull.JobOptions
): Promise<Job<AnyEmailJobData>> {
  const queue = getEmailQueue();

  // Transactional emails get higher priority
  const priority = data.type === EmailJobType.TRANSACTIONAL ? 1 : 2;

  return queue.add(data, {
    priority,
    ...options,
  });
}

/**
 * Add a batch email job to the queue
 */
export async function addBatchEmailJob(
  data: BatchEmailJobData,
  options?: Bull.JobOptions
): Promise<Job<AnyEmailJobData>> {
  const queue = getEmailQueue();

  return queue.add(data, {
    priority: 3, // Lower priority than transactional
    ...options,
  });
}

/**
 * Add a sequence step job to the queue
 */
export async function addSequenceStepJob(
  data: SequenceStepJobData,
  options?: Bull.JobOptions
): Promise<Job<AnyEmailJobData>> {
  const queue = getEmailQueue();

  return queue.add(data, {
    priority: 2,
    ...options,
  });
}

/**
 * Get queue statistics
 */
export async function getQueueStats(): Promise<{
  waiting: number;
  active: number;
  completed: number;
  failed: number;
  delayed: number;
}> {
  const queue = getEmailQueue();
  const [waiting, active, completed, failed, delayed] = await Promise.all([
    queue.getWaitingCount(),
    queue.getActiveCount(),
    queue.getCompletedCount(),
    queue.getFailedCount(),
    queue.getDelayedCount(),
  ]);

  return { waiting, active, completed, failed, delayed };
}

/**
 * Clean up old jobs
 */
export async function cleanupOldJobs(olderThanMs: number = 7 * 24 * 60 * 60 * 1000): Promise<void> {
  const queue = getEmailQueue();
  await queue.clean(olderThanMs, 'completed');
  await queue.clean(olderThanMs, 'failed');
  logger.info({ olderThanMs }, 'Cleaned up old email jobs');
}

/**
 * Close the queue connection
 */
export async function closeEmailQueue(): Promise<void> {
  if (emailQueue) {
    await emailQueue.close();
    emailQueue = null;
    logger.info('Email queue closed');
  }
}

export default getEmailQueue;
