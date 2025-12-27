/**
 * Email Service
 *
 * Centralized service for sending emails with:
 * - Template rendering (MJML + Handlebars)
 * - Queue-based sending with retries
 * - Preference checking
 * - Analytics logging
 */

import mjml2html from 'mjml';
import Handlebars from 'handlebars';
import { convert as htmlToText } from 'html-to-text';
import { CreateEmailOptions } from 'resend';

import prisma from '../config/database';
import { logger } from '../config/logger';
import { getResend, EMAIL_CONFIG, isMarketingCategory, isEmailEnabled } from '../config/resend';
import {
  EmailJobType,
  addEmailJob,
  addBatchEmailJob,
  addSequenceStepJob,
  type EmailJobData,
  type BatchEmailJobData,
  type SequenceStepJobData,
} from '../queues/emailQueue';
import {
  generateListUnsubscribeHeaders,
  generateUnsubscribeUrl,
  generatePreferenceCenterUrl,
  generateEmailCorrelationId,
  sanitizeEmailHtml,
} from '../utils/emailHelpers';

/**
 * Compiled template cache
 */
const templateCache = new Map<
  string,
  {
    subject: Handlebars.TemplateDelegate;
    html: string;
    compiledMjml: Handlebars.TemplateDelegate;
  }
>();

/**
 * Register Handlebars helpers
 */
Handlebars.registerHelper('formatDate', (date: Date | string) => {
  if (!date) return '';
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
});

Handlebars.registerHelper('formatNumber', (num: number) => {
  if (typeof num !== 'number') return '0';
  return num.toLocaleString();
});

Handlebars.registerHelper('formatCurrency', (amount: number, currency = 'USD') => {
  if (typeof amount !== 'number') return '$0.00';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(amount);
});

Handlebars.registerHelper('eq', (a: unknown, b: unknown) => a === b);
Handlebars.registerHelper('gt', (a: number, b: number) => a > b);
Handlebars.registerHelper('lt', (a: number, b: number) => a < b);

/**
 * Render an email template with variables
 */
export async function renderTemplate(
  templateSlug: string,
  variables: Record<string, unknown>
): Promise<{ subject: string; html: string; plainText: string }> {
  // Check cache first
  let cached = templateCache.get(templateSlug);

  if (!cached) {
    // Fetch template from database
    const template = await prisma.emailTemplate.findUnique({
      where: { slug: templateSlug },
    });

    if (!template) {
      throw new Error(`Email template not found: ${templateSlug}`);
    }

    if (!template.isActive) {
      throw new Error(`Email template is inactive: ${templateSlug}`);
    }

    // Compile subject template
    const subjectTemplate = Handlebars.compile(template.subject);

    // Compile MJML template (we store MJML, compile to HTML on render)
    const mjmlTemplate = Handlebars.compile(template.mjmlContent);

    cached = {
      subject: subjectTemplate,
      html: template.htmlContent || '', // Use cached HTML if available
      compiledMjml: mjmlTemplate,
    };

    templateCache.set(templateSlug, cached);
  }

  // Render subject with variables
  const subject = cached.subject(variables);

  // Render MJML with variables, then compile to HTML
  const mjmlWithVariables = cached.compiledMjml(variables);
  const mjmlResult = mjml2html(mjmlWithVariables, {
    validationLevel: 'soft',
    minify: true,
  });

  if (mjmlResult.errors.length > 0) {
    logger.warn({ errors: mjmlResult.errors, templateSlug }, 'MJML compilation warnings');
  }

  const html = sanitizeEmailHtml(mjmlResult.html);

  // Generate plain text version
  const plainText = htmlToText(html, {
    wordwrap: 80,
    selectors: [
      { selector: 'img', format: 'skip' },
      { selector: 'a', options: { hideLinkHrefIfSameAsText: true } },
    ],
  });

  return { subject, html, plainText };
}

/**
 * Clear template cache (call when templates are updated)
 */
export function clearTemplateCache(templateSlug?: string): void {
  if (templateSlug) {
    templateCache.delete(templateSlug);
  } else {
    templateCache.clear();
  }
}

/**
 * Check if user can receive email based on preferences
 */
export async function canReceiveEmail(
  userId: string,
  category: string
): Promise<{ allowed: boolean; reason?: string }> {
  const preference = await prisma.emailPreference.findUnique({
    where: { userId },
  });

  // If no preferences, allow transactional only
  if (!preference) {
    if (isMarketingCategory(category)) {
      return { allowed: false, reason: 'Marketing opt-in required' };
    }
    return { allowed: true };
  }

  // Check global unsubscribe
  if (preference.globalUnsubscribedAt) {
    if (isMarketingCategory(category)) {
      return { allowed: false, reason: 'Globally unsubscribed from marketing' };
    }
    // Still allow transactional emails
    return { allowed: true };
  }

  // Check marketing opt-in
  if (isMarketingCategory(category) && !preference.marketingOptIn) {
    return { allowed: false, reason: 'Marketing opt-in not confirmed' };
  }

  // Check category preference
  const categories = preference.categories as Record<string, boolean>;
  if (categories && categories[category] === false) {
    return { allowed: false, reason: `Category ${category} disabled` };
  }

  return { allowed: true };
}

/**
 * Send a transactional email
 * These are always sent (password reset, receipts, etc.)
 */
export async function sendTransactional(
  userId: string,
  templateSlug: string,
  variables: Record<string, unknown> = {}
): Promise<string> {
  // Get user email
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { email: true, name: true },
  });

  if (!user) {
    throw new Error(`User not found: ${userId}`);
  }

  const correlationId = generateEmailCorrelationId();

  // Add job to queue
  const jobData: EmailJobData = {
    type: EmailJobType.TRANSACTIONAL,
    userId,
    email: user.email,
    templateSlug,
    variables: {
      ...variables,
      userName: user.name,
      userEmail: user.email,
      unsubscribeUrl: generateUnsubscribeUrl(userId),
      preferenceCenterUrl: generatePreferenceCenterUrl(userId),
    },
    correlationId,
  };

  await addEmailJob(jobData);

  // Create email log entry
  await prisma.emailLog.create({
    data: {
      userId,
      email: user.email,
      templateSlug,
      status: 'QUEUED',
      metadata: { correlationId },
    },
  });

  logger.info({ correlationId, userId, templateSlug }, 'Transactional email queued');

  return correlationId;
}

/**
 * Send a marketing email to multiple users
 */
export async function sendMarketing(
  userIds: string[],
  campaignId: string
): Promise<{ queued: number; skipped: number }> {
  // Get campaign with template
  const campaign = await prisma.emailCampaign.findUnique({
    where: { id: campaignId },
    include: { template: true },
  });

  if (!campaign) {
    throw new Error(`Campaign not found: ${campaignId}`);
  }

  // Get users with their preferences
  const users = await prisma.user.findMany({
    where: { id: { in: userIds } },
    select: {
      id: true,
      email: true,
      name: true,
      emailPreference: true,
    },
  });

  // Filter out users who can't receive this email
  const eligibleUsers: Array<{
    userId: string;
    email: string;
    variables: Record<string, unknown>;
  }> = [];

  for (const user of users) {
    const canReceive = await canReceiveEmail(user.id, 'marketing');
    if (canReceive.allowed) {
      eligibleUsers.push({
        userId: user.id,
        email: user.email,
        variables: {
          userName: user.name,
          userEmail: user.email,
          unsubscribeUrl: generateUnsubscribeUrl(user.id, campaignId),
          preferenceCenterUrl: generatePreferenceCenterUrl(user.id),
        },
      });
    }
  }

  const skipped = users.length - eligibleUsers.length;

  // Split into batches
  const batchSize = EMAIL_CONFIG.rateLimit.batchSize;
  for (let i = 0; i < eligibleUsers.length; i += batchSize) {
    const batch = eligibleUsers.slice(i, i + batchSize);
    const correlationId = generateEmailCorrelationId();

    const jobData: BatchEmailJobData = {
      type: EmailJobType.CAMPAIGN_BATCH,
      campaignId,
      emails: batch,
      templateSlug: campaign.template.slug,
      correlationId,
    };

    await addBatchEmailJob(jobData);
  }

  // Update campaign status
  await prisma.emailCampaign.update({
    where: { id: campaignId },
    data: {
      status: 'SENDING',
      sentAt: new Date(),
      estimatedAudience: eligibleUsers.length,
    },
  });

  logger.info({ campaignId, queued: eligibleUsers.length, skipped }, 'Marketing campaign queued');

  return { queued: eligibleUsers.length, skipped };
}

/**
 * Process a single email job (called by queue processor)
 */
export async function processEmailJob(jobData: EmailJobData): Promise<void> {
  if (!isEmailEnabled()) {
    logger.warn({ correlationId: jobData.correlationId }, 'Email sending disabled - skipping');
    return;
  }

  const { email, templateSlug, variables, userId, campaignId, correlationId } = jobData;

  try {
    // Render template
    const { subject, html, plainText } = await renderTemplate(templateSlug, variables);

    // Prepare headers
    const headers = userId ? generateListUnsubscribeHeaders(userId, campaignId) : {};

    // Determine from address
    const isMarketing = jobData.type === EmailJobType.CAMPAIGN_BATCH;
    const from = isMarketing ? EMAIL_CONFIG.from.marketing : EMAIL_CONFIG.from.transactional;

    // Send via Resend
    const resend = getResend();
    const emailOptions: CreateEmailOptions = {
      from,
      to: email,
      subject,
      html,
      text: plainText,
      replyTo: EMAIL_CONFIG.replyTo,
      headers,
    };

    const result = await resend.emails.send(emailOptions);

    // Update email log
    await prisma.emailLog.updateMany({
      where: {
        email,
        templateSlug,
        status: 'QUEUED',
      },
      data: {
        status: 'SENT',
        sentAt: new Date(),
        providerId: result.data?.id,
      },
    });

    logger.info(
      { correlationId, email, templateSlug, providerId: result.data?.id },
      'Email sent successfully'
    );
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';

    // Update email log with error
    await prisma.emailLog.updateMany({
      where: {
        email,
        templateSlug,
        status: 'QUEUED',
      },
      data: {
        status: 'BOUNCED',
        providerError: errorMessage,
      },
    });

    logger.error(
      { correlationId, email, templateSlug, error: errorMessage },
      'Failed to send email'
    );

    throw error; // Re-throw to trigger retry
  }
}

/**
 * Process a batch email job (called by queue processor)
 */
export async function processBatchEmailJob(jobData: BatchEmailJobData): Promise<void> {
  const { campaignId, emails, templateSlug, correlationId } = jobData;

  let successCount = 0;
  let failureCount = 0;

  for (const { userId, email, variables } of emails) {
    try {
      await processEmailJob({
        type: EmailJobType.TRANSACTIONAL, // Individual sends are processed same way
        userId,
        email,
        templateSlug,
        variables,
        campaignId,
        correlationId,
      });
      successCount++;
    } catch {
      failureCount++;
    }
  }

  // Update campaign analytics
  await prisma.emailCampaign.update({
    where: { id: campaignId },
    data: {
      actualSent: { increment: successCount },
    },
  });

  logger.info(
    { correlationId, campaignId, successCount, failureCount },
    'Batch email processing complete'
  );
}

/**
 * Process a sequence step job
 */
export async function processSequenceStepJob(jobData: SequenceStepJobData): Promise<void> {
  const { enrollmentId, correlationId } = jobData;

  // Get enrollment with sequence
  const enrollment = await prisma.emailSequenceEnrollment.findUnique({
    where: { id: enrollmentId },
    include: { sequence: true },
  });

  if (!enrollment) {
    logger.warn({ enrollmentId }, 'Sequence enrollment not found');
    return;
  }

  if (enrollment.status !== 'ACTIVE') {
    logger.info({ enrollmentId, status: enrollment.status }, 'Sequence enrollment not active');
    return;
  }

  // Get user
  const user = await prisma.user.findUnique({
    where: { id: enrollment.userId },
    select: { email: true, name: true },
  });

  if (!user) {
    logger.warn({ userId: enrollment.userId }, 'User not found for sequence');
    return;
  }

  // Get current step from sequence steps
  const steps = enrollment.sequence.steps as Array<{
    stepNumber: number;
    delayHours: number;
    templateId: string;
  }>;

  const currentStepIndex = enrollment.currentStep;
  const step = steps[currentStepIndex];

  if (!step) {
    // No more steps, complete the sequence
    await prisma.emailSequenceEnrollment.update({
      where: { id: enrollmentId },
      data: {
        status: 'COMPLETED',
        completedAt: new Date(),
      },
    });

    await prisma.emailSequence.update({
      where: { id: enrollment.sequenceId },
      data: {
        completedEnrollments: { increment: 1 },
        activeEnrollments: { decrement: 1 },
      },
    });

    logger.info({ enrollmentId }, 'Sequence completed');
    return;
  }

  // Get template for this step
  const template = await prisma.emailTemplate.findUnique({
    where: { id: step.templateId },
  });

  if (!template) {
    logger.error({ templateId: step.templateId }, 'Sequence step template not found');
    return;
  }

  // Send the email
  await processEmailJob({
    type: EmailJobType.SEQUENCE_STEP,
    userId: enrollment.userId,
    email: user.email,
    templateSlug: template.slug,
    variables: {
      userName: user.name,
      userEmail: user.email,
      unsubscribeUrl: generateUnsubscribeUrl(enrollment.userId),
      preferenceCenterUrl: generatePreferenceCenterUrl(enrollment.userId),
    },
    sequenceEnrollmentId: enrollmentId,
    correlationId,
  });

  // Schedule next step if exists
  const nextStepIndex = currentStepIndex + 1;
  const nextStep = steps[nextStepIndex];

  if (nextStep) {
    const nextStepTime = new Date(Date.now() + nextStep.delayHours * 60 * 60 * 1000);

    await prisma.emailSequenceEnrollment.update({
      where: { id: enrollmentId },
      data: {
        currentStep: nextStepIndex,
        nextStepScheduledAt: nextStepTime,
      },
    });

    // Queue the next step
    await addSequenceStepJob(
      {
        type: EmailJobType.SEQUENCE_STEP,
        enrollmentId,
        correlationId: generateEmailCorrelationId(),
      },
      { delay: nextStep.delayHours * 60 * 60 * 1000 }
    );

    logger.info(
      { enrollmentId, nextStep: nextStepIndex, scheduledAt: nextStepTime },
      'Next sequence step scheduled'
    );
  } else {
    // Complete the sequence
    await prisma.emailSequenceEnrollment.update({
      where: { id: enrollmentId },
      data: {
        status: 'COMPLETED',
        completedAt: new Date(),
        currentStep: nextStepIndex,
      },
    });

    await prisma.emailSequence.update({
      where: { id: enrollment.sequenceId },
      data: {
        completedEnrollments: { increment: 1 },
        activeEnrollments: { decrement: 1 },
      },
    });

    logger.info({ enrollmentId }, 'Sequence completed');
  }
}

/**
 * Enroll user in an email sequence
 */
export async function enrollInSequence(
  userId: string,
  triggerEvent: string
): Promise<string | null> {
  // Find active sequence for this trigger
  const sequence = await prisma.emailSequence.findFirst({
    where: {
      triggerEvent: triggerEvent as never,
      isActive: true,
    },
  });

  if (!sequence) {
    return null;
  }

  // Check if already enrolled
  const existing = await prisma.emailSequenceEnrollment.findUnique({
    where: {
      userId_sequenceId: {
        userId,
        sequenceId: sequence.id,
      },
    },
  });

  if (existing) {
    logger.info({ userId, sequenceId: sequence.id }, 'User already enrolled in sequence');
    return existing.id;
  }

  // Check enrollment criteria
  const canReceive = await canReceiveEmail(userId, 'marketing');
  if (!canReceive.allowed) {
    logger.info({ userId, reason: canReceive.reason }, 'User cannot receive sequence emails');
    return null;
  }

  // Create enrollment
  const enrollment = await prisma.emailSequenceEnrollment.create({
    data: {
      userId,
      sequenceId: sequence.id,
      currentStep: 0,
      status: 'ACTIVE',
      nextStepScheduledAt: new Date(), // Start immediately
    },
  });

  // Update sequence stats
  await prisma.emailSequence.update({
    where: { id: sequence.id },
    data: {
      totalEnrollments: { increment: 1 },
      activeEnrollments: { increment: 1 },
    },
  });

  // Queue first step
  await addSequenceStepJob({
    type: EmailJobType.SEQUENCE_STEP,
    enrollmentId: enrollment.id,
    correlationId: generateEmailCorrelationId(),
  });

  logger.info(
    { userId, sequenceId: sequence.id, enrollmentId: enrollment.id },
    'User enrolled in sequence'
  );

  return enrollment.id;
}

/**
 * Track email event (called by webhook handler)
 */
export async function trackEvent(
  providerId: string,
  event: string,
  metadata?: Record<string, unknown>
): Promise<void> {
  // Find email log by provider ID
  const emailLog = await prisma.emailLog.findFirst({
    where: { providerId },
  });

  if (!emailLog) {
    logger.warn({ providerId, event }, 'Email log not found for event');
    return;
  }

  // Update based on event type
  const updateData: Record<string, unknown> = {};

  switch (event) {
    case 'delivered':
      updateData.status = 'DELIVERED';
      updateData.deliveredAt = new Date();
      break;
    case 'opened':
      updateData.status = 'OPENED';
      updateData.openedAt = new Date();
      // Check for Apple Mail Privacy Protection
      if (metadata?.userAgent?.toString().includes('Apple Mail')) {
        updateData.isAppleProxyOpen = true;
      }
      break;
    case 'clicked':
      updateData.status = 'CLICKED';
      updateData.clickedAt = new Date();
      if (metadata?.url) {
        const existingClicks = (emailLog.clickData as string[]) || [];
        updateData.clickData = [...existingClicks, metadata.url];
      }
      break;
    case 'bounced':
      updateData.status = 'BOUNCED';
      updateData.bouncedAt = new Date();
      updateData.bounceType = metadata?.bounceType?.toString() || 'unknown';
      break;
    case 'complained':
      updateData.status = 'COMPLAINED';
      // Auto-unsubscribe from all marketing
      if (emailLog.userId) {
        await prisma.emailPreference.upsert({
          where: { userId: emailLog.userId },
          update: { globalUnsubscribedAt: new Date() },
          create: {
            userId: emailLog.userId,
            globalUnsubscribedAt: new Date(),
          },
        });
      }
      break;
    case 'unsubscribed':
      updateData.status = 'UNSUBSCRIBED';
      updateData.unsubscribedAt = new Date();
      break;
  }

  if (Object.keys(updateData).length > 0) {
    await prisma.emailLog.update({
      where: { id: emailLog.id },
      data: updateData,
    });

    // Update campaign analytics if applicable
    if (emailLog.campaignId) {
      const analyticsField = getAnalyticsField(event);
      if (analyticsField) {
        await prisma.emailCampaign.update({
          where: { id: emailLog.campaignId },
          data: { [analyticsField]: { increment: 1 } },
        });
      }
    }

    logger.info({ emailLogId: emailLog.id, event, providerId }, 'Email event tracked');
  }
}

/**
 * Map event to campaign analytics field
 */
function getAnalyticsField(event: string): string | null {
  const mapping: Record<string, string> = {
    delivered: 'deliveredCount',
    opened: 'openedCount',
    clicked: 'clickedCount',
    bounced: 'bouncedCount',
    complained: 'complainedCount',
    unsubscribed: 'unsubscribedCount',
  };
  return mapping[event] || null;
}

export default {
  renderTemplate,
  clearTemplateCache,
  canReceiveEmail,
  sendTransactional,
  sendMarketing,
  processEmailJob,
  processBatchEmailJob,
  processSequenceStepJob,
  enrollInSequence,
  trackEvent,
};
