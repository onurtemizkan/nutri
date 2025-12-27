/**
 * Admin Email Controller
 *
 * Handles admin email management endpoints including:
 * - Email template CRUD
 * - Email campaign management
 * - Email sequence/drip campaign management
 * - Email analytics
 * - Subscriber management
 */

import { Response } from 'express';
import { z } from 'zod';
import prisma from '../config/database';
import { logger } from '../config/logger';
import { AdminAuthenticatedRequest } from '../types';
import { sendMarketing } from '../services/emailService';
import { getResend, EMAIL_CONFIG } from '../config/resend';
import mjml2html from 'mjml';
import { Prisma, SubscriptionTier } from '@prisma/client';

// =============================================================================
// VALIDATION SCHEMAS
// =============================================================================

const createTemplateSchema = z.object({
  name: z.string().min(1).max(100),
  slug: z
    .string()
    .min(1)
    .max(100)
    .regex(/^[a-z0-9-]+$/),
  category: z.enum(['TRANSACTIONAL', 'MARKETING']),
  subject: z.string().min(1).max(200),
  mjmlContent: z.string().min(1),
  variables: z.record(z.unknown()).optional(),
});

const updateTemplateSchema = createTemplateSchema.partial();

const createCampaignSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().optional(),
  templateId: z.string().cuid(),
  scheduledAt: z.string().datetime().optional(),
  segmentCriteria: z.record(z.unknown()).optional(),
  abTestConfig: z.record(z.unknown()).optional(),
});

const updateCampaignSchema = createCampaignSchema.partial();

const createSequenceSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().optional(),
  triggerEvent: z.enum([
    'SIGNUP',
    'FIRST_MEAL',
    'GOAL_ACHIEVED',
    'SUBSCRIPTION_CHANGED',
    'INACTIVITY_7D',
    'INACTIVITY_14D',
    'INACTIVITY_30D',
  ]),
  isActive: z.boolean().optional(),
  steps: z.array(z.record(z.unknown())).optional(),
  enrollmentCriteria: z.record(z.unknown()).optional(),
  exitCriteria: z.record(z.unknown()).optional(),
});

const updateSequenceSchema = createSequenceSchema.partial();

// =============================================================================
// EMAIL TEMPLATE ENDPOINTS
// =============================================================================

/**
 * List all email templates
 * GET /api/admin/email/templates
 */
export async function listTemplates(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const {
    page = '1',
    limit = '20',
    category,
    search,
    isActive,
  } = req.query as Record<string, string>;

  const pageNum = parseInt(page, 10);
  const limitNum = Math.min(parseInt(limit, 10), 100);
  const skip = (pageNum - 1) * limitNum;

  const where: Prisma.EmailTemplateWhereInput = {};

  if (category === 'TRANSACTIONAL' || category === 'MARKETING') {
    where.category = category;
  }

  if (isActive !== undefined) {
    where.isActive = isActive === 'true';
  }

  if (search) {
    where.OR = [
      { name: { contains: search, mode: 'insensitive' } },
      { slug: { contains: search, mode: 'insensitive' } },
      { subject: { contains: search, mode: 'insensitive' } },
    ];
  }

  const [templates, total] = await Promise.all([
    prisma.emailTemplate.findMany({
      where,
      skip,
      take: limitNum,
      orderBy: { updatedAt: 'desc' },
      select: {
        id: true,
        name: true,
        slug: true,
        category: true,
        subject: true,
        isActive: true,
        version: true,
        createdAt: true,
        updatedAt: true,
        _count: {
          select: { campaigns: true },
        },
      },
    }),
    prisma.emailTemplate.count({ where }),
  ]);

  res.json({
    templates,
    pagination: {
      page: pageNum,
      limit: limitNum,
      total,
      pages: Math.ceil(total / limitNum),
    },
  });
}

/**
 * Get single template with full content
 * GET /api/admin/email/templates/:id
 */
export async function getTemplate(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const template = await prisma.emailTemplate.findUnique({
    where: { id },
    include: {
      campaigns: {
        take: 10,
        orderBy: { createdAt: 'desc' },
        select: {
          id: true,
          name: true,
          status: true,
          sentAt: true,
        },
      },
    },
  });

  if (!template) {
    res.status(404).json({ error: 'Template not found' });
    return;
  }

  res.json({ template });
}

/**
 * Create new email template
 * POST /api/admin/email/templates
 */
export async function createTemplate(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const result = createTemplateSchema.safeParse(req.body);
  if (!result.success) {
    res.status(400).json({ error: 'Invalid input', details: result.error.format() });
    return;
  }

  const { name, slug, category, subject, mjmlContent, variables } = result.data;

  // Check for duplicate slug
  const existing = await prisma.emailTemplate.findUnique({
    where: { slug },
  });

  if (existing) {
    res.status(409).json({ error: 'Template with this slug already exists' });
    return;
  }

  // Compile MJML to HTML
  let htmlContent: string;
  let plainTextContent: string;

  try {
    const compiled = mjml2html(mjmlContent, { validationLevel: 'strict' });
    if (compiled.errors.length > 0) {
      res.status(400).json({
        error: 'Invalid MJML',
        details: compiled.errors.map((e) => e.message),
      });
      return;
    }
    htmlContent = compiled.html;
    // Simple HTML to plain text conversion
    plainTextContent = htmlContent
      .replace(/<style[^>]*>.*?<\/style>/gis, '')
      .replace(/<[^>]+>/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  } catch (error) {
    logger.error({ error }, 'Failed to compile MJML');
    res.status(400).json({ error: 'Failed to compile MJML template' });
    return;
  }

  const template = await prisma.emailTemplate.create({
    data: {
      name,
      slug,
      category,
      subject,
      mjmlContent,
      htmlContent,
      plainTextContent,
      variables: variables ? (variables as Prisma.InputJsonValue) : undefined,
      createdByAdminId: req.adminUser?.id,
      updatedByAdminId: req.adminUser?.id,
    },
  });

  logger.info(
    { templateId: template.id, slug, adminId: req.adminUser?.id },
    'Email template created'
  );

  res.status(201).json({ template });
}

/**
 * Update email template
 * PUT /api/admin/email/templates/:id
 */
export async function updateTemplate(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const result = updateTemplateSchema.safeParse(req.body);
  if (!result.success) {
    res.status(400).json({ error: 'Invalid input', details: result.error.format() });
    return;
  }

  const existing = await prisma.emailTemplate.findUnique({
    where: { id },
  });

  if (!existing) {
    res.status(404).json({ error: 'Template not found' });
    return;
  }

  // Save current version to history before updating
  await prisma.emailTemplateVersion.create({
    data: {
      templateId: existing.id,
      version: existing.version,
      subject: existing.subject,
      mjmlContent: existing.mjmlContent,
      htmlContent: existing.htmlContent,
      plainTextContent: existing.plainTextContent,
      variables: existing.variables ? (existing.variables as Prisma.InputJsonValue) : undefined,
      changeNotes: req.body.changeNotes || null,
      createdByAdminId: req.adminUser?.id,
    },
  });

  const updateData: Prisma.EmailTemplateUpdateInput = {
    updatedByAdminId: req.adminUser?.id,
    version: existing.version + 1,
  };

  if (result.data.name) updateData.name = result.data.name;
  if (result.data.slug) updateData.slug = result.data.slug;
  if (result.data.category) updateData.category = result.data.category;
  if (result.data.subject) updateData.subject = result.data.subject;
  if (result.data.variables) updateData.variables = result.data.variables as Prisma.InputJsonValue;

  // If MJML content changed, recompile
  if (result.data.mjmlContent) {
    try {
      const compiled = mjml2html(result.data.mjmlContent, { validationLevel: 'strict' });
      if (compiled.errors.length > 0) {
        res.status(400).json({
          error: 'Invalid MJML',
          details: compiled.errors.map((e) => e.message),
        });
        return;
      }
      updateData.mjmlContent = result.data.mjmlContent;
      updateData.htmlContent = compiled.html;
      updateData.plainTextContent = compiled.html
        .replace(/<style[^>]*>.*?<\/style>/gis, '')
        .replace(/<[^>]+>/g, '')
        .replace(/\s+/g, ' ')
        .trim();
    } catch (error) {
      logger.error({ error }, 'Failed to compile MJML');
      res.status(400).json({ error: 'Failed to compile MJML template' });
      return;
    }
  }

  const template = await prisma.emailTemplate.update({
    where: { id },
    data: updateData,
  });

  logger.info(
    { templateId: template.id, version: template.version, adminId: req.adminUser?.id },
    'Email template updated'
  );

  res.json({ template });
}

/**
 * Delete email template
 * DELETE /api/admin/email/templates/:id
 */
export async function deleteTemplate(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const template = await prisma.emailTemplate.findUnique({
    where: { id },
    include: {
      _count: {
        select: { campaigns: true },
      },
    },
  });

  if (!template) {
    res.status(404).json({ error: 'Template not found' });
    return;
  }

  // Prevent deletion if used by campaigns
  if (template._count.campaigns > 0) {
    res.status(409).json({
      error: 'Cannot delete template with associated campaigns',
      campaignCount: template._count.campaigns,
    });
    return;
  }

  await prisma.emailTemplate.delete({ where: { id } });

  logger.info({ templateId: id, adminId: req.adminUser?.id }, 'Email template deleted');

  res.json({ message: 'Template deleted' });
}

/**
 * Preview template with test data
 * POST /api/admin/email/templates/:id/preview
 */
export async function previewTemplate(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  const { id } = req.params;
  const { variables = {} } = req.body;

  const template = await prisma.emailTemplate.findUnique({
    where: { id },
  });

  if (!template) {
    res.status(404).json({ error: 'Template not found' });
    return;
  }

  // Simple variable substitution for preview
  let previewHtml = template.htmlContent || '';
  let previewSubject = template.subject;

  for (const [key, value] of Object.entries(variables)) {
    const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
    previewHtml = previewHtml.replace(regex, String(value));
    previewSubject = previewSubject.replace(regex, String(value));
  }

  res.json({
    subject: previewSubject,
    html: previewHtml,
    plainText: previewHtml
      .replace(/<style[^>]*>.*?<\/style>/gis, '')
      .replace(/<[^>]+>/g, '')
      .replace(/\s+/g, ' ')
      .trim(),
  });
}

/**
 * Send test email
 * POST /api/admin/email/templates/:id/test
 */
export async function sendTestEmail(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;
  const { email, variables = {} } = req.body;

  if (!email || typeof email !== 'string') {
    res.status(400).json({ error: 'Email address required' });
    return;
  }

  const template = await prisma.emailTemplate.findUnique({
    where: { id },
  });

  if (!template) {
    res.status(404).json({ error: 'Template not found' });
    return;
  }

  // Simple variable substitution
  let html = template.htmlContent || '';
  let subject = `[TEST] ${template.subject}`;

  for (const [key, value] of Object.entries(variables)) {
    const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
    html = html.replace(regex, String(value));
    subject = subject.replace(regex, String(value));
  }

  try {
    const resend = getResend();
    await resend.emails.send({
      from: EMAIL_CONFIG.from.transactional,
      to: email,
      subject,
      html,
    });

    logger.info({ templateId: id, email, adminId: req.adminUser?.id }, 'Test email sent');

    res.json({ message: 'Test email sent successfully' });
  } catch (error) {
    logger.error({ error, templateId: id, email }, 'Failed to send test email');
    res.status(500).json({ error: 'Failed to send test email' });
  }
}

// =============================================================================
// TEMPLATE VERSION ENDPOINTS
// =============================================================================

/**
 * List template versions
 * GET /api/admin/email/templates/:id/versions
 */
export async function listTemplateVersions(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  const { id } = req.params;
  const { page = '1', limit = '20' } = req.query as Record<string, string>;

  const pageNum = parseInt(page, 10);
  const limitNum = Math.min(parseInt(limit, 10), 50);
  const skip = (pageNum - 1) * limitNum;

  // Verify template exists
  const template = await prisma.emailTemplate.findUnique({
    where: { id },
    select: { id: true, version: true },
  });

  if (!template) {
    res.status(404).json({ error: 'Template not found' });
    return;
  }

  const [versions, total] = await Promise.all([
    prisma.emailTemplateVersion.findMany({
      where: { templateId: id },
      skip,
      take: limitNum,
      orderBy: { version: 'desc' },
      select: {
        id: true,
        version: true,
        subject: true,
        changeNotes: true,
        createdByAdminId: true,
        createdAt: true,
      },
    }),
    prisma.emailTemplateVersion.count({ where: { templateId: id } }),
  ]);

  res.json({
    versions,
    currentVersion: template.version,
    pagination: {
      page: pageNum,
      limit: limitNum,
      total,
      pages: Math.ceil(total / limitNum),
    },
  });
}

/**
 * Get specific template version
 * GET /api/admin/email/templates/:id/versions/:version
 */
export async function getTemplateVersion(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  const { id, version } = req.params;
  const versionNum = parseInt(version, 10);

  if (isNaN(versionNum)) {
    res.status(400).json({ error: 'Invalid version number' });
    return;
  }

  const templateVersion = await prisma.emailTemplateVersion.findUnique({
    where: {
      templateId_version: {
        templateId: id,
        version: versionNum,
      },
    },
  });

  if (!templateVersion) {
    res.status(404).json({ error: 'Version not found' });
    return;
  }

  res.json({ version: templateVersion });
}

/**
 * Restore template to specific version
 * POST /api/admin/email/templates/:id/versions/:version/restore
 */
export async function restoreTemplateVersion(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  const { id, version } = req.params;
  const versionNum = parseInt(version, 10);

  if (isNaN(versionNum)) {
    res.status(400).json({ error: 'Invalid version number' });
    return;
  }

  // Get current template
  const current = await prisma.emailTemplate.findUnique({
    where: { id },
  });

  if (!current) {
    res.status(404).json({ error: 'Template not found' });
    return;
  }

  // Get version to restore
  const versionToRestore = await prisma.emailTemplateVersion.findUnique({
    where: {
      templateId_version: {
        templateId: id,
        version: versionNum,
      },
    },
  });

  if (!versionToRestore) {
    res.status(404).json({ error: 'Version not found' });
    return;
  }

  // Save current version to history
  await prisma.emailTemplateVersion.create({
    data: {
      templateId: current.id,
      version: current.version,
      subject: current.subject,
      mjmlContent: current.mjmlContent,
      htmlContent: current.htmlContent,
      plainTextContent: current.plainTextContent,
      variables: current.variables ? (current.variables as Prisma.InputJsonValue) : undefined,
      changeNotes: `Restored from v${versionNum}`,
      createdByAdminId: req.adminUser?.id,
    },
  });

  // Update template with restored version content
  const template = await prisma.emailTemplate.update({
    where: { id },
    data: {
      subject: versionToRestore.subject,
      mjmlContent: versionToRestore.mjmlContent,
      htmlContent: versionToRestore.htmlContent,
      plainTextContent: versionToRestore.plainTextContent,
      variables: versionToRestore.variables
        ? (versionToRestore.variables as Prisma.InputJsonValue)
        : undefined,
      version: current.version + 1,
      updatedByAdminId: req.adminUser?.id,
    },
  });

  logger.info(
    {
      templateId: id,
      restoredFrom: versionNum,
      newVersion: template.version,
      adminId: req.adminUser?.id,
    },
    'Template version restored'
  );

  res.json({ template, message: `Restored from version ${versionNum}` });
}

// =============================================================================
// EMAIL CAMPAIGN ENDPOINTS
// =============================================================================

/**
 * List email campaigns
 * GET /api/admin/email/campaigns
 */
export async function listCampaigns(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { page = '1', limit = '20', status, search } = req.query as Record<string, string>;

  const pageNum = parseInt(page, 10);
  const limitNum = Math.min(parseInt(limit, 10), 100);
  const skip = (pageNum - 1) * limitNum;

  const where: Prisma.EmailCampaignWhereInput = {};

  if (status) {
    where.status = status as Prisma.EnumEmailCampaignStatusFilter<'EmailCampaign'>;
  }

  if (search) {
    where.name = { contains: search, mode: 'insensitive' };
  }

  const [campaigns, total] = await Promise.all([
    prisma.emailCampaign.findMany({
      where,
      skip,
      take: limitNum,
      orderBy: { createdAt: 'desc' },
      include: {
        template: {
          select: {
            id: true,
            name: true,
            slug: true,
          },
        },
      },
    }),
    prisma.emailCampaign.count({ where }),
  ]);

  // Add email statistics
  const campaignsWithStats = await Promise.all(
    campaigns.map(async (campaign) => {
      const stats = await prisma.emailLog.groupBy({
        by: ['status'],
        where: { campaignId: campaign.id },
        _count: true,
      });

      const statsMap = stats.reduce(
        (acc, s) => {
          acc[s.status.toLowerCase()] = s._count;
          return acc;
        },
        {} as Record<string, number>
      );

      return {
        ...campaign,
        stats: {
          sent: statsMap.sent || 0,
          delivered: statsMap.delivered || 0,
          opened: statsMap.opened || 0,
          clicked: statsMap.clicked || 0,
          bounced: statsMap.bounced || 0,
          complained: statsMap.complained || 0,
        },
      };
    })
  );

  res.json({
    campaigns: campaignsWithStats,
    pagination: {
      page: pageNum,
      limit: limitNum,
      total,
      pages: Math.ceil(total / limitNum),
    },
  });
}

/**
 * Get single campaign with full stats
 * GET /api/admin/email/campaigns/:id
 */
export async function getCampaign(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const campaign = await prisma.emailCampaign.findUnique({
    where: { id },
    include: {
      template: true,
    },
  });

  if (!campaign) {
    res.status(404).json({ error: 'Campaign not found' });
    return;
  }

  // Get detailed statistics
  const [statusStats, hourlyStats, recentLogs] = await Promise.all([
    prisma.emailLog.groupBy({
      by: ['status'],
      where: { campaignId: id },
      _count: true,
    }),
    prisma.$queryRaw<{ hour: Date; count: bigint }[]>`
      SELECT DATE_TRUNC('hour', "sentAt") as hour, COUNT(*) as count
      FROM "EmailLog"
      WHERE "campaignId" = ${id} AND "sentAt" IS NOT NULL
      GROUP BY DATE_TRUNC('hour', "sentAt")
      ORDER BY hour DESC
      LIMIT 24
    `,
    prisma.emailLog.findMany({
      where: { campaignId: id },
      orderBy: { createdAt: 'desc' },
      take: 20,
      select: {
        id: true,
        email: true,
        status: true,
        createdAt: true,
        sentAt: true,
        deliveredAt: true,
        openedAt: true,
        clickedAt: true,
        bouncedAt: true,
        providerError: true,
      },
    }),
  ]);

  const stats = statusStats.reduce(
    (acc, s) => {
      acc[s.status.toLowerCase()] = s._count;
      return acc;
    },
    {} as Record<string, number>
  );

  const totalSent = stats.sent || stats.delivered || 0;
  const openRate = totalSent > 0 ? ((stats.opened || 0) / totalSent) * 100 : 0;
  const clickRate = totalSent > 0 ? ((stats.clicked || 0) / totalSent) * 100 : 0;
  const bounceRate = totalSent > 0 ? ((stats.bounced || 0) / totalSent) * 100 : 0;

  res.json({
    campaign,
    stats: {
      ...stats,
      total: Object.values(stats).reduce((a, b) => a + b, 0),
      openRate: openRate.toFixed(2),
      clickRate: clickRate.toFixed(2),
      bounceRate: bounceRate.toFixed(2),
    },
    hourlyStats: hourlyStats.map((h) => ({
      hour: h.hour,
      count: Number(h.count),
    })),
    recentLogs,
  });
}

/**
 * Create new email campaign
 * POST /api/admin/email/campaigns
 */
export async function createCampaign(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const result = createCampaignSchema.safeParse(req.body);
  if (!result.success) {
    res.status(400).json({ error: 'Invalid input', details: result.error.format() });
    return;
  }

  const { name, description, templateId, scheduledAt, segmentCriteria, abTestConfig } = result.data;

  // Verify template exists
  const template = await prisma.emailTemplate.findUnique({
    where: { id: templateId },
  });

  if (!template) {
    res.status(404).json({ error: 'Template not found' });
    return;
  }

  const campaign = await prisma.emailCampaign.create({
    data: {
      name,
      description,
      templateId,
      status: scheduledAt ? 'SCHEDULED' : 'DRAFT',
      scheduledAt: scheduledAt ? new Date(scheduledAt) : null,
      segmentCriteria: (segmentCriteria || {}) as Prisma.InputJsonValue,
      abTestConfig: abTestConfig ? (abTestConfig as Prisma.InputJsonValue) : undefined,
      createdByAdminId: req.adminUser?.id,
    },
    include: {
      template: {
        select: { id: true, name: true, slug: true },
      },
    },
  });

  logger.info(
    { campaignId: campaign.id, name, adminId: req.adminUser?.id },
    'Email campaign created'
  );

  res.status(201).json({ campaign });
}

/**
 * Update email campaign
 * PUT /api/admin/email/campaigns/:id
 */
export async function updateCampaign(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const result = updateCampaignSchema.safeParse(req.body);
  if (!result.success) {
    res.status(400).json({ error: 'Invalid input', details: result.error.format() });
    return;
  }

  const campaign = await prisma.emailCampaign.findUnique({
    where: { id },
  });

  if (!campaign) {
    res.status(404).json({ error: 'Campaign not found' });
    return;
  }

  // Only allow updates to draft or scheduled campaigns
  if (!['DRAFT', 'SCHEDULED'].includes(campaign.status)) {
    res.status(409).json({ error: 'Cannot update campaign in current status' });
    return;
  }

  const updateData: Prisma.EmailCampaignUpdateInput = {};

  if (result.data.name) updateData.name = result.data.name;
  if (result.data.description) updateData.description = result.data.description;
  if (result.data.templateId) updateData.template = { connect: { id: result.data.templateId } };
  if (result.data.segmentCriteria)
    updateData.segmentCriteria = result.data.segmentCriteria as Prisma.InputJsonValue;
  if (result.data.abTestConfig)
    updateData.abTestConfig = result.data.abTestConfig as Prisma.InputJsonValue;

  if (result.data.scheduledAt) {
    updateData.scheduledAt = new Date(result.data.scheduledAt);
    updateData.status = 'SCHEDULED';
  }

  const updated = await prisma.emailCampaign.update({
    where: { id },
    data: updateData,
    include: {
      template: {
        select: { id: true, name: true, slug: true },
      },
    },
  });

  logger.info({ campaignId: id, adminId: req.adminUser?.id }, 'Email campaign updated');

  res.json({ campaign: updated });
}

/**
 * Delete email campaign (draft only)
 * DELETE /api/admin/email/campaigns/:id
 */
export async function deleteCampaign(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const campaign = await prisma.emailCampaign.findUnique({
    where: { id },
  });

  if (!campaign) {
    res.status(404).json({ error: 'Campaign not found' });
    return;
  }

  if (campaign.status !== 'DRAFT') {
    res.status(409).json({ error: 'Only draft campaigns can be deleted' });
    return;
  }

  await prisma.emailCampaign.delete({ where: { id } });

  logger.info({ campaignId: id, adminId: req.adminUser?.id }, 'Email campaign deleted');

  res.json({ message: 'Campaign deleted' });
}

/**
 * Send campaign immediately
 * POST /api/admin/email/campaigns/:id/send
 */
export async function sendCampaignNow(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  const { id } = req.params;

  const campaign = await prisma.emailCampaign.findUnique({
    where: { id },
    include: { template: true },
  });

  if (!campaign) {
    res.status(404).json({ error: 'Campaign not found' });
    return;
  }

  if (!['DRAFT', 'SCHEDULED'].includes(campaign.status)) {
    res.status(409).json({ error: 'Campaign cannot be sent in current status' });
    return;
  }

  // Update status to SENDING
  await prisma.emailCampaign.update({
    where: { id },
    data: { status: 'SENDING', sentAt: new Date() },
  });

  // Get eligible users based on segment criteria
  const segmentCriteria = (campaign.segmentCriteria as Record<string, unknown>) || {};
  const userWhere: Prisma.UserWhereInput = {};

  // Build user query based on segment
  if (segmentCriteria.subscriptionTier) {
    userWhere.subscriptionTier = segmentCriteria.subscriptionTier as SubscriptionTier;
  }

  // Get users who haven't unsubscribed
  const users = await prisma.user.findMany({
    where: {
      ...userWhere,
      emailPreference: {
        OR: [
          { globalUnsubscribedAt: null },
          { globalUnsubscribedAt: { not: null }, marketingOptIn: true },
        ],
      },
    },
    select: { id: true },
    take: 10000, // Limit batch size
  });

  const userIds = users.map((u) => u.id);

  if (userIds.length === 0) {
    await prisma.emailCampaign.update({
      where: { id },
      data: { status: 'SENT', completedAt: new Date(), actualSent: 0 },
    });
    res.json({ message: 'No eligible recipients', recipientCount: 0 });
    return;
  }

  // Update campaign with recipient count estimate
  await prisma.emailCampaign.update({
    where: { id },
    data: { estimatedAudience: userIds.length },
  });

  // Queue the campaign send
  try {
    await sendMarketing(userIds, id);
    logger.info(
      { campaignId: id, recipientCount: userIds.length, adminId: req.adminUser?.id },
      'Campaign send initiated'
    );
    res.json({ message: 'Campaign send initiated', recipientCount: userIds.length });
  } catch (error) {
    logger.error({ error, campaignId: id }, 'Failed to send campaign');
    await prisma.emailCampaign.update({
      where: { id },
      data: { status: 'CANCELLED' },
    });
    res.status(500).json({ error: 'Failed to send campaign' });
  }
}

/**
 * Cancel scheduled campaign
 * POST /api/admin/email/campaigns/:id/cancel
 */
export async function cancelCampaign(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const campaign = await prisma.emailCampaign.findUnique({
    where: { id },
  });

  if (!campaign) {
    res.status(404).json({ error: 'Campaign not found' });
    return;
  }

  if (!['SCHEDULED', 'SENDING'].includes(campaign.status)) {
    res.status(409).json({ error: 'Only scheduled or sending campaigns can be cancelled' });
    return;
  }

  await prisma.emailCampaign.update({
    where: { id },
    data: { status: 'CANCELLED' },
  });

  logger.info({ campaignId: id, adminId: req.adminUser?.id }, 'Campaign cancelled');

  res.json({ message: 'Campaign cancelled' });
}

// =============================================================================
// EMAIL SEQUENCE ENDPOINTS
// =============================================================================

/**
 * List email sequences
 * GET /api/admin/email/sequences
 */
export async function listSequences(_req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const sequences = await prisma.emailSequence.findMany({
    orderBy: { createdAt: 'desc' },
    include: {
      _count: {
        select: { enrollments: true },
      },
    },
  });

  // Add enrollment statistics
  const sequencesWithStats = await Promise.all(
    sequences.map(async (seq) => {
      const stats = await prisma.emailSequenceEnrollment.groupBy({
        by: ['status'],
        where: { sequenceId: seq.id },
        _count: true,
      });

      const statsMap = stats.reduce(
        (acc, s) => {
          acc[s.status.toLowerCase()] = s._count;
          return acc;
        },
        {} as Record<string, number>
      );

      return {
        ...seq,
        enrollmentStats: {
          active: statsMap.active || 0,
          completed: statsMap.completed || 0,
          paused: statsMap.paused || 0,
          exited: statsMap.exited || 0,
        },
      };
    })
  );

  res.json({ sequences: sequencesWithStats });
}

/**
 * Get sequence with steps
 * GET /api/admin/email/sequences/:id
 */
export async function getSequence(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const sequence = await prisma.emailSequence.findUnique({
    where: { id },
  });

  if (!sequence) {
    res.status(404).json({ error: 'Sequence not found' });
    return;
  }

  // Get enrollment stats and recent enrollments
  const [enrollmentStats, recentEnrollments] = await Promise.all([
    prisma.emailSequenceEnrollment.groupBy({
      by: ['status'],
      where: { sequenceId: id },
      _count: true,
    }),
    prisma.emailSequenceEnrollment.findMany({
      where: { sequenceId: id },
      orderBy: { createdAt: 'desc' },
      take: 20,
      select: {
        id: true,
        userId: true,
        currentStep: true,
        status: true,
        startedAt: true,
        completedAt: true,
        exitedAt: true,
        exitReason: true,
      },
    }),
  ]);

  res.json({
    sequence,
    stats: enrollmentStats.reduce(
      (acc, s) => {
        acc[s.status.toLowerCase()] = s._count;
        return acc;
      },
      {} as Record<string, number>
    ),
    recentEnrollments,
  });
}

/**
 * Create email sequence
 * POST /api/admin/email/sequences
 */
export async function createSequence(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const result = createSequenceSchema.safeParse(req.body);
  if (!result.success) {
    res.status(400).json({ error: 'Invalid input', details: result.error.format() });
    return;
  }

  const { name, description, triggerEvent, isActive, steps, enrollmentCriteria, exitCriteria } =
    result.data;

  const sequence = await prisma.emailSequence.create({
    data: {
      name,
      description,
      triggerEvent,
      isActive: isActive ?? false,
      steps: (steps || []) as Prisma.InputJsonValue,
      enrollmentCriteria: enrollmentCriteria
        ? (enrollmentCriteria as Prisma.InputJsonValue)
        : undefined,
      exitCriteria: exitCriteria ? (exitCriteria as Prisma.InputJsonValue) : undefined,
      createdByAdminId: req.adminUser?.id,
    },
  });

  logger.info(
    { sequenceId: sequence.id, name, adminId: req.adminUser?.id },
    'Email sequence created'
  );

  res.status(201).json({ sequence });
}

/**
 * Update email sequence
 * PUT /api/admin/email/sequences/:id
 */
export async function updateSequence(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const result = updateSequenceSchema.safeParse(req.body);
  if (!result.success) {
    res.status(400).json({ error: 'Invalid input', details: result.error.format() });
    return;
  }

  const sequence = await prisma.emailSequence.findUnique({
    where: { id },
  });

  if (!sequence) {
    res.status(404).json({ error: 'Sequence not found' });
    return;
  }

  const updateData: Prisma.EmailSequenceUpdateInput = {};

  if (result.data.name) updateData.name = result.data.name;
  if (result.data.description !== undefined) updateData.description = result.data.description;
  if (result.data.triggerEvent) updateData.triggerEvent = result.data.triggerEvent;
  if (result.data.isActive !== undefined) updateData.isActive = result.data.isActive;
  if (result.data.steps) updateData.steps = result.data.steps as Prisma.InputJsonValue;
  if (result.data.enrollmentCriteria)
    updateData.enrollmentCriteria = result.data.enrollmentCriteria as Prisma.InputJsonValue;
  if (result.data.exitCriteria)
    updateData.exitCriteria = result.data.exitCriteria as Prisma.InputJsonValue;

  const updated = await prisma.emailSequence.update({
    where: { id },
    data: updateData,
  });

  logger.info({ sequenceId: id, adminId: req.adminUser?.id }, 'Email sequence updated');

  res.json({ sequence: updated });
}

/**
 * Delete email sequence
 * DELETE /api/admin/email/sequences/:id
 */
export async function deleteSequence(req: AdminAuthenticatedRequest, res: Response): Promise<void> {
  const { id } = req.params;

  const sequence = await prisma.emailSequence.findUnique({
    where: { id },
    include: {
      _count: {
        select: { enrollments: true },
      },
    },
  });

  if (!sequence) {
    res.status(404).json({ error: 'Sequence not found' });
    return;
  }

  // Check for active enrollments
  const activeEnrollments = await prisma.emailSequenceEnrollment.count({
    where: { sequenceId: id, status: 'ACTIVE' },
  });

  if (activeEnrollments > 0) {
    res.status(409).json({
      error: 'Cannot delete sequence with active enrollments',
      activeEnrollments,
    });
    return;
  }

  await prisma.emailSequence.delete({ where: { id } });

  logger.info({ sequenceId: id, adminId: req.adminUser?.id }, 'Email sequence deleted');

  res.json({ message: 'Sequence deleted' });
}

// =============================================================================
// EMAIL ANALYTICS ENDPOINTS
// =============================================================================

/**
 * Get email analytics overview
 * GET /api/admin/email/analytics
 */
export async function getEmailAnalytics(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  const { days = '30' } = req.query as Record<string, string>;
  const daysNum = parseInt(days, 10);
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - daysNum);

  const [totalEmails, statusBreakdown, dailyStats, topCampaigns, bouncesByType] = await Promise.all(
    [
      // Total emails sent
      prisma.emailLog.count({
        where: { createdAt: { gte: startDate } },
      }),

      // Status breakdown
      prisma.emailLog.groupBy({
        by: ['status'],
        where: { createdAt: { gte: startDate } },
        _count: true,
      }),

      // Daily send volume
      prisma.$queryRaw<{ date: Date; sent: bigint; opened: bigint; clicked: bigint }[]>`
      SELECT
        DATE_TRUNC('day', "createdAt") as date,
        COUNT(*) FILTER (WHERE status IN ('SENT', 'DELIVERED')) as sent,
        COUNT(*) FILTER (WHERE status = 'OPENED') as opened,
        COUNT(*) FILTER (WHERE status = 'CLICKED') as clicked
      FROM "EmailLog"
      WHERE "createdAt" >= ${startDate}
      GROUP BY DATE_TRUNC('day', "createdAt")
      ORDER BY date DESC
      LIMIT ${daysNum}
    `,

      // Top performing campaigns
      prisma.emailCampaign.findMany({
        where: { status: 'SENT', sentAt: { gte: startDate } },
        orderBy: { actualSent: 'desc' },
        take: 10,
        select: {
          id: true,
          name: true,
          sentAt: true,
          actualSent: true,
          deliveredCount: true,
          openedCount: true,
          clickedCount: true,
        },
      }),

      // Bounces by type
      prisma.emailLog.groupBy({
        by: ['bounceType'],
        where: {
          status: 'BOUNCED',
          createdAt: { gte: startDate },
        },
        _count: true,
      }),
    ]
  );

  const stats = statusBreakdown.reduce(
    (acc, s) => {
      acc[s.status.toLowerCase()] = s._count;
      return acc;
    },
    {} as Record<string, number>
  );

  const delivered = stats.delivered || stats.sent || 0;
  const openRate = delivered > 0 ? ((stats.opened || 0) / delivered) * 100 : 0;
  const clickRate = delivered > 0 ? ((stats.clicked || 0) / delivered) * 100 : 0;
  const bounceRate = delivered > 0 ? ((stats.bounced || 0) / delivered) * 100 : 0;
  const complaintRate = delivered > 0 ? ((stats.complained || 0) / delivered) * 100 : 0;

  res.json({
    period: { days: daysNum, startDate: startDate.toISOString() },
    overview: {
      totalEmails,
      delivered,
      opened: stats.opened || 0,
      clicked: stats.clicked || 0,
      bounced: stats.bounced || 0,
      complained: stats.complained || 0,
      openRate: openRate.toFixed(2),
      clickRate: clickRate.toFixed(2),
      bounceRate: bounceRate.toFixed(2),
      complaintRate: complaintRate.toFixed(4),
    },
    dailyStats: dailyStats.map((d) => ({
      date: d.date,
      sent: Number(d.sent),
      opened: Number(d.opened),
      clicked: Number(d.clicked),
    })),
    topCampaigns,
    bouncesByType: bouncesByType.map((b) => ({
      type: b.bounceType || 'unknown',
      count: b._count,
    })),
  });
}

/**
 * Get subscriber statistics
 * GET /api/admin/email/subscribers
 */
export async function getSubscriberStats(
  _req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  const [totalSubscribers, unsubscribed, marketingOptIn, doubleOptInConfirmed] = await Promise.all([
    // Total users with email preferences
    prisma.emailPreference.count(),

    // Globally unsubscribed
    prisma.emailPreference.count({
      where: { globalUnsubscribedAt: { not: null } },
    }),

    // Marketing opt-in
    prisma.emailPreference.count({
      where: { marketingOptIn: true },
    }),

    // Double opt-in confirmed
    prisma.emailPreference.count({
      where: { doubleOptInConfirmedAt: { not: null } },
    }),
  ]);

  res.json({
    totalSubscribers,
    unsubscribed,
    subscribed: totalSubscribers - unsubscribed,
    marketingOptIn,
    doubleOptInConfirmed,
    subscriptionRate:
      totalSubscribers > 0
        ? (((totalSubscribers - unsubscribed) / totalSubscribers) * 100).toFixed(2)
        : '0.00',
  });
}

export default {
  // Templates
  listTemplates,
  getTemplate,
  createTemplate,
  updateTemplate,
  deleteTemplate,
  previewTemplate,
  sendTestEmail,
  // Campaigns
  listCampaigns,
  getCampaign,
  createCampaign,
  updateCampaign,
  deleteCampaign,
  sendCampaignNow,
  cancelCampaign,
  // Sequences
  listSequences,
  getSequence,
  createSequence,
  updateSequence,
  deleteSequence,
  // Analytics
  getEmailAnalytics,
  getSubscriberStats,
};
