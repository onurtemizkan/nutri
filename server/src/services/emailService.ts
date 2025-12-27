/**
 * Email Service - Handles email sending via Resend with MJML template compilation
 *
 * Features:
 * - MJML template compilation to HTML
 * - Variable substitution in templates
 * - Resend API integration
 * - Email logging for analytics
 * - Bounce/complaint handling
 */

import { Resend } from 'resend';
import mjml2html from 'mjml';
import { PrismaClient, EmailLogStatus, EmailCategory } from '@prisma/client';
import { logger } from '../config/logger';

const prisma = new PrismaClient();

// Initialize Resend client
const resendApiKey = process.env.RESEND_API_KEY;
const resend = resendApiKey ? new Resend(resendApiKey) : null;

// Default sender configuration
const DEFAULT_FROM_EMAIL = process.env.EMAIL_FROM || 'Nutri <noreply@nutri.app>';
const DEFAULT_REPLY_TO = process.env.EMAIL_REPLY_TO || 'support@nutri.app';

// Types
interface SendEmailOptions {
  to: string | string[];
  subject: string;
  html: string;
  text?: string;
  from?: string;
  replyTo?: string;
  tags?: { name: string; value: string }[];
  headers?: Record<string, string>;
}

interface EmailResult {
  success: boolean;
  messageId?: string;
  error?: string;
}

interface TemplateVariables {
  [key: string]: string | number | boolean | undefined;
}

interface SendTemplateEmailOptions {
  userId: string;
  email: string;
  templateSlug: string;
  variables?: TemplateVariables;
  campaignId?: string;
  sequenceEnrollmentId?: string;
}

/**
 * Compile MJML template to HTML
 */
export function compileMjml(mjmlContent: string): { html: string; errors: string[] } {
  try {
    const result = mjml2html(mjmlContent, {
      validationLevel: 'soft', // Allow some errors
      minify: true,
      keepComments: false,
    });

    const errors = result.errors?.map((e) => e.message) || [];

    return {
      html: result.html,
      errors,
    };
  } catch (error) {
    logger.error({ error }, 'MJML compilation failed');
    throw new Error(`MJML compilation failed: ${error}`);
  }
}

/**
 * Substitute variables in template content
 * Variables are in format: {{variableName}}
 */
export function substituteVariables(content: string, variables: TemplateVariables): string {
  let result = content;

  for (const [key, value] of Object.entries(variables)) {
    const placeholder = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
    result = result.replace(placeholder, String(value ?? ''));
  }

  // Remove any remaining unsubstituted variables
  result = result.replace(/{{\s*\w+\s*}}/g, '');

  return result;
}

/**
 * Send an email via Resend
 */
export async function sendEmail(options: SendEmailOptions): Promise<EmailResult> {
  if (!resend) {
    logger.warn(
      { to: options.to, subject: options.subject },
      'Resend API key not configured, email not sent'
    );
    return {
      success: false,
      error: 'Email provider not configured',
    };
  }

  try {
    const result = await resend.emails.send({
      from: options.from || DEFAULT_FROM_EMAIL,
      to: Array.isArray(options.to) ? options.to : [options.to],
      subject: options.subject,
      html: options.html,
      text: options.text,
      replyTo: options.replyTo || DEFAULT_REPLY_TO,
      tags: options.tags,
      headers: options.headers,
    });

    if (result.error) {
      logger.error({ error: result.error }, 'Resend API error');
      return {
        success: false,
        error: result.error.message,
      };
    }

    logger.info(
      { messageId: result.data?.id, to: options.to, subject: options.subject },
      'Email sent successfully'
    );

    return {
      success: true,
      messageId: result.data?.id,
    };
  } catch (error) {
    logger.error({ error, to: options.to }, 'Failed to send email');
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Send email using a stored template
 */
export async function sendTemplateEmail(
  options: SendTemplateEmailOptions
): Promise<{ success: boolean; emailLogId?: string; error?: string }> {
  const { userId, email, templateSlug, variables = {}, campaignId, sequenceEnrollmentId } = options;

  // Check user preferences and suppression
  const preferences = await prisma.emailPreference.findUnique({
    where: { userId },
  });

  if (preferences?.isSuppressed) {
    logger.info({ userId, email, templateSlug }, 'Email not sent - user is suppressed');
    return {
      success: false,
      error: 'User is suppressed',
    };
  }

  // Get template
  const template = await prisma.emailTemplate.findUnique({
    where: { slug: templateSlug },
  });

  if (!template) {
    logger.error({ templateSlug }, 'Email template not found');
    return {
      success: false,
      error: 'Template not found',
    };
  }

  if (!template.isActive) {
    logger.warn({ templateSlug }, 'Email template is inactive');
    return {
      success: false,
      error: 'Template is inactive',
    };
  }

  // Check marketing consent for marketing emails
  if (template.category === EmailCategory.MARKETING) {
    if (!preferences?.marketingOptIn || preferences?.globalUnsubscribedAt) {
      logger.info({ userId, email, templateSlug }, 'Marketing email not sent - user not opted in');
      return {
        success: false,
        error: 'User not opted in to marketing emails',
      };
    }
  }

  // Add unsubscribe link variable
  const unsubscribeToken = preferences?.unsubscribeToken || '';
  const enhancedVariables = {
    ...variables,
    unsubscribeUrl: `${process.env.APP_URL}/unsubscribe/${unsubscribeToken}`,
    preferencesUrl: `${process.env.APP_URL}/email-preferences/${unsubscribeToken}`,
  };

  // Compile content
  let htmlContent = template.htmlContent;

  if (!htmlContent && template.mjmlContent) {
    const compiled = compileMjml(template.mjmlContent);
    htmlContent = compiled.html;

    // Cache the compiled HTML
    await prisma.emailTemplate.update({
      where: { id: template.id },
      data: { htmlContent },
    });
  }

  if (!htmlContent) {
    return {
      success: false,
      error: 'No HTML content available',
    };
  }

  // Substitute variables
  const subject = substituteVariables(template.subject, enhancedVariables);
  const html = substituteVariables(htmlContent, enhancedVariables);
  const text = template.plainTextContent
    ? substituteVariables(template.plainTextContent, enhancedVariables)
    : undefined;

  // Create email log entry
  const emailLog = await prisma.emailLog.create({
    data: {
      userId,
      email,
      templateSlug,
      campaignId,
      sequenceEnrollmentId,
      status: EmailLogStatus.QUEUED,
    },
  });

  // Send the email
  const result = await sendEmail({
    to: email,
    subject,
    html,
    text,
    tags: [
      { name: 'template', value: templateSlug },
      { name: 'userId', value: userId },
      ...(campaignId ? [{ name: 'campaign', value: campaignId }] : []),
    ],
    headers: {
      'X-Email-Log-Id': emailLog.id,
      'List-Unsubscribe': `<${process.env.APP_URL}/unsubscribe/${unsubscribeToken}>`,
      'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
    },
  });

  // Update email log
  if (result.success) {
    await prisma.emailLog.update({
      where: { id: emailLog.id },
      data: {
        status: EmailLogStatus.SENT,
        sentAt: new Date(),
        providerMessageId: result.messageId,
      },
    });
  } else {
    await prisma.emailLog.update({
      where: { id: emailLog.id },
      data: {
        status: EmailLogStatus.BOUNCED,
        bounceReason: result.error,
      },
    });
  }

  return {
    success: result.success,
    emailLogId: emailLog.id,
    error: result.error,
  };
}

/**
 * Handle email delivery webhook events from Resend
 */
export async function handleEmailWebhook(
  eventType: string,
  data: {
    email_id: string;
    email?: string;
    timestamp?: string;
    reason?: string;
    bounce_type?: string;
    link?: { url: string };
  }
): Promise<void> {
  // Find the email log by provider message ID
  const emailLog = await prisma.emailLog.findFirst({
    where: { providerMessageId: data.email_id },
  });

  if (!emailLog) {
    logger.warn({ eventType, emailId: data.email_id }, 'Email log not found for webhook event');
    return;
  }

  const now = new Date();

  switch (eventType) {
    case 'email.delivered':
      await prisma.emailLog.update({
        where: { id: emailLog.id },
        data: {
          status: EmailLogStatus.DELIVERED,
          deliveredAt: now,
        },
      });
      break;

    case 'email.opened':
      await prisma.emailLog.update({
        where: { id: emailLog.id },
        data: {
          status: EmailLogStatus.OPENED,
          openedAt: emailLog.openedAt || now, // Only set first open time
          openCount: { increment: 1 },
        },
      });

      // Update user engagement
      await updateUserEngagement(emailLog.userId, 'open');
      break;

    case 'email.clicked':
      const clickData = (emailLog.clickData as Array<{ url: string; clickedAt: string }>) || [];
      clickData.push({
        url: data.link?.url || 'unknown',
        clickedAt: now.toISOString(),
      });

      await prisma.emailLog.update({
        where: { id: emailLog.id },
        data: {
          status: EmailLogStatus.CLICKED,
          clickedAt: emailLog.clickedAt || now,
          clickData,
        },
      });

      // Update user engagement
      await updateUserEngagement(emailLog.userId, 'click');
      break;

    case 'email.bounced':
      await prisma.emailLog.update({
        where: { id: emailLog.id },
        data: {
          status: EmailLogStatus.BOUNCED,
          bouncedAt: now,
          bounceType: data.bounce_type,
          bounceReason: data.reason,
        },
      });

      // Suppress user on hard bounce
      if (data.bounce_type === 'hard') {
        await suppressUser(emailLog.userId, 'hard_bounce');
      }
      break;

    case 'email.complained':
      await prisma.emailLog.update({
        where: { id: emailLog.id },
        data: {
          status: EmailLogStatus.COMPLAINED,
          complainedAt: now,
        },
      });

      // Immediately suppress user on complaint
      await suppressUser(emailLog.userId, 'complaint');
      break;

    default:
      logger.info({ eventType, data }, 'Unhandled email webhook event');
  }

  // Update campaign analytics if applicable
  if (emailLog.campaignId) {
    await updateCampaignAnalytics(emailLog.campaignId, eventType);
  }
}

/**
 * Update user engagement score
 */
async function updateUserEngagement(userId: string, action: 'open' | 'click'): Promise<void> {
  await prisma.emailPreference.upsert({
    where: { userId },
    update: {
      lastEngagementAt: new Date(),
      // Simple engagement score bump
      engagementScore: {
        increment: action === 'click' ? 2 : 1,
      },
    },
    create: {
      userId,
      lastEngagementAt: new Date(),
      engagementScore: action === 'click' ? 2 : 1,
    },
  });
}

/**
 * Suppress a user from receiving emails
 */
async function suppressUser(userId: string, reason: string): Promise<void> {
  await prisma.emailPreference.upsert({
    where: { userId },
    update: {
      isSuppressed: true,
      suppressionReason: reason,
      suppressedAt: new Date(),
    },
    create: {
      userId,
      isSuppressed: true,
      suppressionReason: reason,
      suppressedAt: new Date(),
    },
  });

  logger.info({ userId, reason }, 'User suppressed from emails');
}

/**
 * Update campaign analytics based on event
 */
async function updateCampaignAnalytics(campaignId: string, eventType: string): Promise<void> {
  const updateData: Record<string, { increment: number }> = {};

  switch (eventType) {
    case 'email.delivered':
      updateData.deliveredCount = { increment: 1 };
      break;
    case 'email.opened':
      updateData.openedCount = { increment: 1 };
      break;
    case 'email.clicked':
      updateData.clickedCount = { increment: 1 };
      break;
    case 'email.bounced':
      updateData.bouncedCount = { increment: 1 };
      break;
    case 'email.complained':
      updateData.complainedCount = { increment: 1 };
      break;
  }

  if (Object.keys(updateData).length > 0) {
    await prisma.emailCampaign.update({
      where: { id: campaignId },
      data: updateData,
    });
  }
}

/**
 * Unsubscribe user via one-click unsubscribe
 */
export async function unsubscribeUser(
  unsubscribeToken: string
): Promise<{ success: boolean; error?: string }> {
  const preferences = await prisma.emailPreference.findUnique({
    where: { unsubscribeToken },
  });

  if (!preferences) {
    return { success: false, error: 'Invalid unsubscribe token' };
  }

  await prisma.emailPreference.update({
    where: { id: preferences.id },
    data: {
      globalUnsubscribedAt: new Date(),
      marketingOptIn: false,
    },
  });

  // Exit any active email sequences
  await prisma.emailSequenceEnrollment.updateMany({
    where: {
      userId: preferences.userId,
      status: 'ACTIVE',
    },
    data: {
      status: 'EXITED',
      exitedAt: new Date(),
      exitReason: 'user_unsubscribed',
    },
  });

  logger.info({ userId: preferences.userId }, 'User unsubscribed');

  return { success: true };
}

/**
 * Create or get email preferences for a user
 */
export async function ensureEmailPreferences(userId: string): Promise<void> {
  await prisma.emailPreference.upsert({
    where: { userId },
    update: {},
    create: {
      userId,
      marketingOptIn: false,
      categories: {
        weekly_reports: true,
        health_insights: true,
        tips: true,
        features: true,
        promotions: false,
      },
    },
  });
}

export default {
  compileMjml,
  substituteVariables,
  sendEmail,
  sendTemplateEmail,
  handleEmailWebhook,
  unsubscribeUser,
  ensureEmailPreferences,
};
