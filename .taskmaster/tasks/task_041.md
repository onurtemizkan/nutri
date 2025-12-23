# Task ID: 41

**Title:** Email Marketing System with Admin Content Management

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Implement a comprehensive email marketing and transactional email system with full admin panel content management. Includes rich HTML email editor, template library, campaign scheduling, audience segmentation, automated email sequences (drip campaigns), analytics tracking, and user email preference management. Follow 2025 email deliverability best practices including authentication (DKIM/SPF/DMARC), one-click unsubscribe (RFC 8058), Apple Mail Privacy Protection awareness, and GDPR/CAN-SPAM compliance.

**Details:**

## Technical Architecture

### Email Service Provider Integration

1. **Provider Selection**
   - Primary: Resend (modern API, excellent DX, React Email support)
   - Alternative: SendGrid or Postmark for enterprise needs
   - Separate sending domains: 
     * mail.nutriapp.com (transactional)
     * marketing.nutriapp.com (campaigns)
   - Dedicated IP consideration for high volume (>100k/month)

2. **Authentication Setup**
   - DKIM: Domain key signing for email authenticity
   - SPF: Authorized sender IP records
   - DMARC: Policy for failed authentication handling
   - BIMI: Brand logo in email clients (future enhancement)

3. **Webhook Integration**
   - Bounce handling (hard/soft bounces)
   - Complaint handling (spam reports)
   - Delivery confirmations
   - Open tracking (with privacy caveats)
   - Click tracking
   - Unsubscribe events

### Email Types & Categories

1. **Transactional Emails (always delivered)**
   - Welcome email (after signup)
   - Email verification
   - Password reset
   - Password changed confirmation
   - Subscription confirmation/changes
   - Payment receipts (if applicable)
   - Account security alerts
   - Goal achievement celebrations
   - Weekly/monthly progress reports (user-triggered)

2. **Marketing Emails (opt-in required)**
   - Feature announcements
   - Tips and educational content
   - Recipe suggestions based on goals
   - Health insights digest
   - Re-engagement campaigns
   - Promotional offers
   - Newsletter

3. **Automated Sequences (Drip Campaigns)**
   - Onboarding sequence (Days 1, 3, 7, 14)
   - First meal logged celebration
   - 7-day streak achievement
   - Dormant user re-engagement (7, 14, 30 days inactive)
   - Goal milestone celebrations
   - Subscription renewal reminders

### Backend Architecture

1. **Database Schema (Prisma)**
   - EmailTemplate model: id, name, slug, category, subject, mjmlContent, htmlContent, variables, version, isActive
   - EmailCampaign model: id, name, templateId, segment, scheduledAt, sentAt, status, analytics
   - EmailLog model: id, userId, templateSlug, campaignId, status, sentAt, deliveredAt, openedAt, clickedAt, bouncedAt, unsubscribedAt
   - EmailPreference model: userId, categories (Json), frequency, unsubscribedAt, doubleOptInAt
   - EmailSequence model: id, name, triggerEvent, steps (Json array of delays and templateIds)
   - EmailSequenceEnrollment model: userId, sequenceId, currentStep, startedAt, completedAt

2. **Email Service (server/src/services/emailService.ts)**
   - sendTransactional(userId, templateSlug, variables): Send transactional email
   - sendMarketing(userIds, campaignId): Send marketing campaign
   - sendBatch(emails[]): Batch sending with rate limiting
   - renderTemplate(templateSlug, variables): Render MJML to HTML
   - trackEvent(emailId, event): Handle webhook events
   - validateEmail(email): Check deliverability

3. **Queue System (Bull/BullMQ)**
   - emailQueue: Process email sends with retries
   - campaignQueue: Handle large campaign sends in batches
   - sequenceQueue: Process drip campaign steps
   - Rate limiting: 100 emails/second default
   - Retry strategy: Exponential backoff, max 3 retries

4. **API Endpoints**
   User-facing:
   - GET /api/email/preferences - Get user email preferences
   - PUT /api/email/preferences - Update preferences
   - POST /api/email/unsubscribe - One-click unsubscribe
   - GET /api/email/unsubscribe/:token - Unsubscribe page

   Admin:
   - Full CRUD for templates, campaigns, sequences
   - Analytics endpoints
   - Subscriber management

### Admin Panel Features

1. **Email Template Editor**
   - MJML-based editor (compiles to responsive HTML)
   - React Email components for developer templates
   - Visual drag-and-drop builder option (unlayer.com or similar)
   - Variable system: {{userName}}, {{goalProgress}}, {{weeklyCalories}}, etc.
   - Conditional blocks: {{#if isPremium}}...{{/if}}
   - Dynamic content based on user data
   - Version history with rollback
   - Template categories and organization

2. **Template Preview & Testing**
   - Desktop/mobile/dark mode preview
   - Send test email to admin
   - Spam score check (mail-tester integration)
   - Render with sample user data
   - Preview across email clients (Litmus/Email on Acid integration - optional)

3. **Campaign Management**
   - Create campaign from template
   - Audience segmentation:
     * Activity level (active, dormant, churned)
     * Subscription tier
     * Goal type
     * Signup date range
     * Custom filters (SQL for admins)
   - Schedule send time with timezone handling
   - A/B testing:
     * Subject line variants
     * Content variants
     * Send time variants
     * Statistical significance calculator
   - Estimated audience size preview

4. **Automated Sequences Builder**
   - Visual sequence builder (node-based flow)
   - Trigger events: signup, meal_logged, goal_achieved, subscription_changed, inactivity
   - Steps: Wait (delay), Send Email, Check Condition, Branch
   - Entry conditions and exit conditions
   - Enrollment management (pause, resume, remove users)
   - Performance analytics per step

5. **Analytics Dashboard**
   - Campaign metrics: Sent, Delivered, Opened, Clicked, Bounced, Unsubscribed
   - Open rate with Apple MPP caveat (machine opens flagged)
   - Click heatmaps (which links clicked most)
   - Conversion tracking (email → app action)
   - Time-series trends
   - Comparative analysis (this campaign vs average)
   - Best performing subject lines
   - Optimal send time analysis

6. **Subscriber Management**
   - Subscriber list with search and filters
   - Individual subscriber view: Preference, history, engagement score
   - Bulk import/export (CSV)
   - Suppression list management (bounces, complaints, unsubscribes)
   - Re-engagement targeting
   - List hygiene tools (identify inactive subscribers)

7. **Deliverability Monitoring**
   - Bounce rate tracking
   - Complaint rate tracking (stay below 0.1%)
   - Domain reputation indicators
   - Authentication status (DKIM/SPF/DMARC)
   - Blocklist monitoring integration (optional)

### Mobile App Integration

1. **Email Preferences Screen**
   - Category toggles:
     * Weekly Progress Reports
     * Health Insights
     * Tips & Recipes
     * Feature Updates
     * Promotional Offers
   - Frequency control: Real-time, Daily digest, Weekly digest
   - Unsubscribe all marketing (keep transactional)
   - Update email address

2. **Onboarding Email Opt-in**
   - During signup or shortly after
   - Explain email benefits
   - Pre-select recommended categories
   - GDPR: Explicit consent checkbox for marketing

3. **In-App Email Preview**
   - Show recent emails in notification center
   - Link to web version of email
   - Quick unsubscribe from specific category

### Compliance & Privacy

1. **GDPR Compliance**
   - Explicit opt-in for marketing emails
   - Double opt-in option (confirmation email)
   - Easy preference management
   - Complete unsubscribe option
   - Data export includes email preferences
   - Right to erasure includes email logs

2. **CAN-SPAM Compliance**
   - Physical address in email footer
   - Clear sender identification
   - Honest subject lines
   - Unsubscribe link in every email
   - Honor unsubscribes within 10 days

3. **One-Click Unsubscribe (RFC 8058)**
   - List-Unsubscribe-Post header
   - Immediate processing of unsubscribe
   - No login required
   - Confirmation page (not confirmation email)

4. **Apple Mail Privacy Protection**
   - Detect Apple proxy opens
   - Don't rely solely on open rates
   - Focus on click tracking for engagement
   - Document in analytics with caveat

### Email Design System

1. **Brand Consistency**
   - Nutri color palette in email templates
   - Logo and visual assets
   - Typography matching app design
   - Consistent CTA button styles

2. **Responsive Design**
   - Mobile-first templates
   - Fluid layouts
   - Touch-friendly buttons (44px min)
   - Dark mode support (@media prefers-color-scheme)

3. **Accessibility**
   - Alt text for images
   - Sufficient color contrast
   - Semantic HTML structure
   - Screen reader friendly

### Environment Variables

```env
# Email Provider (Resend)
RESEND_API_KEY=re_xxxxx
RESEND_WEBHOOK_SECRET=whsec_xxxxx
EMAIL_FROM_TRANSACTIONAL=Nutri <hello@mail.nutriapp.com>
EMAIL_FROM_MARKETING=Nutri <updates@marketing.nutriapp.com>
EMAIL_REPLY_TO=support@nutriapp.com

# Domain Configuration
EMAIL_DOMAIN_TRANSACTIONAL=mail.nutriapp.com
EMAIL_DOMAIN_MARKETING=marketing.nutriapp.com

# Rate Limiting
EMAIL_RATE_LIMIT_PER_SECOND=100
EMAIL_BATCH_SIZE=1000

# Optional Integrations
LITMUS_API_KEY=xxxxx  # Email client testing
MAIL_TESTER_API_KEY=xxxxx  # Spam score
```

### Dependencies

- resend (or @sendgrid/mail)
- mjml (email templating)
- react-email (React components for emails)
- bull (job queue - already in stack)
- handlebars or mustache (variable interpolation)
- unlayer-react (optional drag-and-drop builder)

### Security Considerations

- Validate all template variables server-side
- Sanitize user-generated content in emails
- Rate limit email sending per user
- Secure webhook endpoints with signature verification
- Encrypt sensitive tokens in unsubscribe links
- Audit log all template changes

**Test Strategy:**

## Testing Strategy

### Unit Tests
- Template rendering with variables
- MJML to HTML compilation
- Email validation
- Unsubscribe token generation/verification
- Segment filter logic
- Sequence step progression

### Integration Tests
- End-to-end email sending (use test mode/sandbox)
- Webhook event processing
- Campaign scheduling and execution
- Sequence enrollment and progression
- Preference update sync

### E2E Tests
- Email preference screen interactions
- Unsubscribe flow (web page)
- Admin template editor save/preview
- Campaign creation and scheduling

### Deliverability Testing
- SPF/DKIM/DMARC verification
- Spam score testing (mail-tester.com)
- Inbox placement testing (optional Litmus integration)

### Load Testing
- Batch sending performance (10k+ emails)
- Queue processing under load
- Webhook handling at scale

### Manual Testing
- Email rendering across clients (Gmail, Outlook, Apple Mail)
- Dark mode rendering
- Mobile responsiveness
- Accessibility audit

## Subtasks

### 41.1. Design and implement email database schema in Prisma

**Status:** pending  
**Dependencies:** None  

Create Prisma models for EmailTemplate, EmailCampaign, EmailLog, EmailPreference, EmailSequence, and EmailSequenceEnrollment to support the complete email marketing system

**Details:**

Add to server/prisma/schema.prisma:

1. EmailTemplate model:
   - id, name, slug (unique), category (enum: TRANSACTIONAL, MARKETING)
   - subject, mjmlContent, htmlContent, plainTextContent
   - variables (Json array of variable names)
   - version (Int), isActive (Boolean)
   - createdBy, updatedBy (reference to AdminUser)
   - timestamps
   - @@index on [slug], [category, isActive]

2. EmailCampaign model:
   - id, name, description
   - templateId (reference to EmailTemplate)
   - segmentCriteria (Json - targeting rules)
   - scheduledAt, sentAt, completedAt
   - status (enum: DRAFT, SCHEDULED, SENDING, SENT, CANCELLED)
   - abTestConfig (Json - subject variants, content variants)
   - estimatedAudience (Int), actualSent (Int)
   - createdBy (AdminUser reference)
   - timestamps
   - @@index on [status, scheduledAt]

3. EmailLog model:
   - id, userId, email
   - templateSlug, campaignId (optional)
   - status (enum: QUEUED, SENDING, SENT, DELIVERED, OPENED, CLICKED, BOUNCED, COMPLAINED, UNSUBSCRIBED)
   - metadata (Json - provider message ID, error details)
   - sentAt, deliveredAt, openedAt, clickedAt, bouncedAt, unsubscribedAt
   - isAppleProxyOpen (Boolean - for MPP detection)
   - clickData (Json array - URLs clicked)
   - bounceType (String - hard/soft)
   - timestamps
   - @@index on [userId, createdAt], [campaignId, status], [templateSlug]
   - @@unique on [userId, templateSlug, campaignId, sentAt] for deduplication

4. EmailPreference model:
   - id, userId (unique)
   - categories (Json - {weekly_reports: true, health_insights: true, tips: true, features: true, promotions: false})
   - frequency (enum: REALTIME, DAILY_DIGEST, WEEKLY_DIGEST)
   - marketingOptIn (Boolean)
   - doubleOptInConfirmedAt (DateTime)
   - globalUnsubscribedAt (DateTime)
   - unsubscribeToken (String, unique, indexed)
   - timestamps
   - @@index on [userId], [unsubscribeToken]

5. EmailSequence model:
   - id, name, description
   - triggerEvent (enum: SIGNUP, FIRST_MEAL, GOAL_ACHIEVED, SUBSCRIPTION_CHANGED, INACTIVITY_7D, INACTIVITY_14D, INACTIVITY_30D)
   - isActive (Boolean)
   - steps (Json array - [{stepNumber: 1, delayHours: 0, templateId: "...", condition: {...}}])
   - enrollmentCriteria (Json - who can enter this sequence)
   - exitCriteria (Json - when to stop sequence)
   - totalEnrollments (Int), activeEnrollments (Int), completedEnrollments (Int)
   - createdBy (AdminUser reference)
   - timestamps
   - @@index on [triggerEvent, isActive]

6. EmailSequenceEnrollment model:
   - id, userId, sequenceId
   - currentStep (Int - which step they're on)
   - status (enum: ACTIVE, PAUSED, COMPLETED, EXITED)
   - startedAt, pausedAt, completedAt, exitedAt
   - exitReason (String - why they exited early)
   - metadata (Json - tracking data)
   - nextStepScheduledAt (DateTime)
   - timestamps
   - @@index on [userId, status], [sequenceId, status], [nextStepScheduledAt]
   - @@unique on [userId, sequenceId] - user can only be enrolled once

7. Create enums:
   - EmailCategory: TRANSACTIONAL, MARKETING
   - EmailCampaignStatus: DRAFT, SCHEDULED, SENDING, SENT, CANCELLED
   - EmailLogStatus: QUEUED, SENDING, SENT, DELIVERED, OPENED, CLICKED, BOUNCED, COMPLAINED, UNSUBSCRIBED
   - EmailFrequency: REALTIME, DAILY_DIGEST, WEEKLY_DIGEST
   - EmailSequenceTrigger: SIGNUP, FIRST_MEAL, GOAL_ACHIEVED, SUBSCRIPTION_CHANGED, INACTIVITY_7D, INACTIVITY_14D, INACTIVITY_30D
   - EmailSequenceStatus: ACTIVE, PAUSED, COMPLETED, EXITED

8. After schema changes:
   - Run `npm run db:generate`
   - Run `npm run db:push` (dev) or create migration
   - Verify schema in Prisma Studio

### 41.2. Integrate email service provider (Resend) with authentication setup

**Status:** pending  
**Dependencies:** 41.1  

Set up Resend SDK integration with proper API key management, webhook signature verification, and email domain configuration for both transactional and marketing emails

**Details:**

1. Install dependencies in server/:
   ```bash
   npm install resend @types/node
   ```

2. Update server/.env with email configuration:
   ```env
   # Resend Configuration
   RESEND_API_KEY=re_xxxxx
   RESEND_WEBHOOK_SECRET=whsec_xxxxx
   EMAIL_FROM_TRANSACTIONAL=Nutri <hello@mail.nutriapp.com>
   EMAIL_FROM_MARKETING=Nutri <updates@marketing.nutriapp.com>
   EMAIL_REPLY_TO=support@nutriapp.com
   EMAIL_DOMAIN_TRANSACTIONAL=mail.nutriapp.com
   EMAIL_DOMAIN_MARKETING=marketing.nutriapp.com
   ```

3. Update server/src/config/env.ts:
   - Add email configuration to config object
   - Validate required env vars (RESEND_API_KEY, EMAIL_FROM_TRANSACTIONAL)

4. Create server/src/config/resend.ts:
   ```typescript
   import { Resend } from 'resend';
   import { config } from './env';

   export const resend = new Resend(config.email.resendApiKey);

   export const EMAIL_CONFIG = {
     from: {
       transactional: config.email.fromTransactional,
       marketing: config.email.fromMarketing,
     },
     replyTo: config.email.replyTo,
     domains: {
       transactional: config.email.domainTransactional,
       marketing: config.email.domainMarketing,
     },
   };
   ```

5. Domain authentication setup (documentation/manual steps):
   - Add DNS records for DKIM, SPF, DMARC for both domains
   - Verify domains in Resend dashboard
   - Configure return-path for bounce handling
   - Set up BIMI record (optional, future enhancement)

6. Create server/src/utils/emailHelpers.ts:
   - generateUnsubscribeToken(userId): Create secure token
   - verifyUnsubscribeToken(token): Verify and decode token
   - validateEmailDeliverability(email): Basic email validation
   - generateListUnsubscribeHeader(userId, campaignId): RFC 8058 headers

7. Create webhook signature verification utility:
   - verifyResendWebhookSignature(payload, signature, secret): boolean

8. Error handling:
   - Wrap Resend calls with try/catch
   - Log all API errors with correlation IDs
   - Handle rate limiting (429 responses)
   - Implement exponential backoff for retries

### 41.3. Implement backend email service with Bull queue-based sending

**Status:** pending  
**Dependencies:** 41.1, 41.2  

Create a centralized email service that uses Bull/BullMQ for queue-based email sending with retry logic, rate limiting, batch processing, and template rendering

**Details:**

1. Install dependencies in server/:
   ```bash
   npm install bull @types/bull ioredis @types/ioredis mjml handlebars
   ```

2. Update server/.env:
   ```env
   REDIS_URL=redis://localhost:6379
   EMAIL_RATE_LIMIT_PER_SECOND=100
   EMAIL_BATCH_SIZE=1000
   EMAIL_QUEUE_CONCURRENCY=10
   ```

3. Create server/src/queues/emailQueue.ts:
   - Initialize Bull queue connected to Redis
   - Configure retry strategy: exponential backoff, max 3 retries
   - Rate limiting: 100 emails/second (configurable)
   - Job types: TRANSACTIONAL, CAMPAIGN_BATCH, SEQUENCE_STEP
   - Job data interface: { userId?, email, templateSlug, variables, campaignId?, sequenceId? }
   - Process handler: calls emailService.sendEmail()
   - Event handlers: completed, failed, stalled

4. Create server/src/services/emailService.ts:
   Main methods:
   - sendTransactional(userId: string, templateSlug: string, variables: Record<string, any>): Promise<void>
     * Look up user email and preferences
     * Check if unsubscribed (skip if marketing category)
     * Render template with variables
     * Queue email job
     * Create EmailLog entry with status QUEUED
   
   - sendMarketing(userIds: string[], campaignId: string): Promise<void>
     * Filter out unsubscribed users
     * Check email preferences (opt-in, categories)
     * Batch users into chunks (EMAIL_BATCH_SIZE)
     * Queue batch jobs
     * Update EmailCampaign status to SENDING
   
   - sendBatch(emails: Array<{userId, email, templateSlug, variables, campaignId?}>): Promise<void>
     * Send batch via Resend batch API
     * Create EmailLog entries for each
     * Handle partial failures
   
   - renderTemplate(templateSlug: string, variables: Record<string, any>): Promise<{html: string, plainText: string, subject: string}>
     * Fetch EmailTemplate from database
     * Compile MJML to HTML (mjml2html)
     * Interpolate variables with Handlebars
     * Generate plain text version (html-to-text)
     * Return compiled template
   
   - processSequenceStep(enrollmentId: string): Promise<void>
     * Get enrollment and current step
     * Send email for current step
     * Increment step or mark completed
     * Schedule next step if exists
   
   - trackEvent(emailLogId: string, event: EmailLogStatus, metadata?: any): Promise<void>
     * Update EmailLog status and timestamp
     * Handle bounce detection (hard/soft)
     * Process unsubscribe events
     * Detect Apple Mail Privacy Protection opens

5. Template variable interpolation:
   - Use Handlebars for variable substitution: {{userName}}, {{goalProgress}}
   - Conditional blocks: {{#if isPremium}}...{{/if}}
   - Helpers: formatDate, formatNumber, formatCurrency
   - Sanitize user-generated content (escape HTML)

6. Error handling:
   - Catch Resend API errors
   - Log failures with correlation ID
   - Update EmailLog with error details
   - Implement circuit breaker for Resend outages

7. Queue monitoring:
   - Expose queue stats endpoint for admin panel
   - Track: pending jobs, active jobs, failed jobs, completed jobs
   - Alert on high failure rate

8. One-click unsubscribe:
   - Add List-Unsubscribe and List-Unsubscribe-Post headers (RFC 8058)
   - Immediate unsubscribe on POST to unsubscribe endpoint

### 41.4. Implement backend webhook handler for email events

**Status:** pending  
**Dependencies:** 41.1, 41.2, 41.3  

Create webhook endpoint to receive and process email events from Resend (bounces, opens, clicks, unsubscribes, deliveries) with signature verification and proper event logging

**Details:**

1. Create server/src/controllers/emailWebhookController.ts:
   - handleResendWebhook(req, res): Main webhook handler
   - Verify webhook signature using RESEND_WEBHOOK_SECRET
   - Parse event payload
   - Route to appropriate event handler
   - Return 200 OK immediately (process async)
   - Log all webhook events for debugging

2. Event handlers in emailWebhookController:
   - handleDelivered(event): Update EmailLog status to DELIVERED, set deliveredAt
   - handleBounce(event): Update status to BOUNCED, set bouncedAt, bounceType (hard/soft), bounceReason
     * Hard bounce: Add email to suppression list, mark user preference
     * Soft bounce: Increment retry counter
   - handleOpen(event): Update status to OPENED, set openedAt
     * Detect Apple Mail Privacy Protection (check user agent)
     * Set isAppleProxyOpen flag if detected
   - handleClick(event): Update status to CLICKED, set clickedAt
     * Append clicked URL to clickData array
     * Track link performance
   - handleComplaint(event): Update status to COMPLAINED
     * Auto-unsubscribe user from all marketing
     * Flag account for review
   - handleUnsubscribe(event): Update status to UNSUBSCRIBED, set unsubscribedAt
     * Update EmailPreference: set globalUnsubscribedAt
     * Remove from active sequences

3. Create route in server/src/routes/emailWebhookRoutes.ts:
   - POST /api/webhooks/email/resend
   - No authentication (uses signature verification)
   - Rate limiting: 1000 requests/minute
   - Request logging with correlation ID

4. Suppression list management:
   - Track bounced emails in EmailPreference
   - Prevent sending to hard bounced addresses
   - Daily cleanup job for old soft bounces

5. Apple Mail Privacy Protection handling:
   - Detect proxy opens from Apple's IP ranges
   - Set isAppleProxyOpen = true in EmailLog
   - Don't count as genuine engagement
   - Display caveat in analytics dashboard

6. Analytics aggregation:
   - Update EmailCampaign analytics on each event
   - Increment counters: delivered, opened, clicked, bounced, unsubscribed
   - Calculate rates: open rate, click rate, bounce rate
   - Store in campaign metadata for fast access

7. Error handling:
   - Catch database errors
   - Log invalid webhook payloads
   - Handle missing EmailLog entries gracefully
   - Implement idempotency (duplicate webhook prevention)

8. Security:
   - Always verify webhook signature
   - Reject unsigned requests with 401
   - Rate limit webhook endpoint
   - Sanitize all incoming data

9. Monitoring:
   - Track webhook processing latency
   - Alert on high bounce rate (>5%)
   - Alert on high complaint rate (>0.1%)
   - Dashboard for webhook event volume

### 41.5. Create backend API endpoints for user email preferences

**Status:** pending  
**Dependencies:** 41.1, 41.2, 41.3  

Implement user-facing API endpoints for managing email preferences, including opt-in/opt-out, frequency settings, category preferences, and one-click unsubscribe

**Details:**

1. Create Zod schemas in server/src/validation/schemas.ts:
   - emailPreferencesUpdateSchema:
     * categories: object with boolean fields (weekly_reports, health_insights, tips, features, promotions)
     * frequency: enum (REALTIME, DAILY_DIGEST, WEEKLY_DIGEST)
     * marketingOptIn: boolean
   - emailUnsubscribeSchema:
     * token: string (for one-click unsubscribe)
     * categories: optional array of strings (selective unsubscribe)

2. Create server/src/services/emailPreferenceService.ts:
   - getPreferences(userId: string): Get or create user email preferences
   - updatePreferences(userId: string, data: UpdatePreferencesInput): Update preferences
   - doubleOptIn(userId: string): Send double opt-in confirmation email, update doubleOptInConfirmedAt on confirmation
   - unsubscribeAll(userId: string): Set globalUnsubscribedAt, exit all active sequences
   - unsubscribeCategory(userId: string, category: string): Disable specific category
   - unsubscribeByToken(token: string): One-click unsubscribe, verify token, update preferences
   - resubscribe(userId: string): Clear globalUnsubscribedAt, restore previous preferences

3. Create server/src/controllers/emailPreferenceController.ts:
   - getPreferences(req, res): GET /api/email/preferences
     * Requires authentication
     * Returns EmailPreference for current user
     * Creates default preferences if none exist
   
   - updatePreferences(req, res): PUT /api/email/preferences
     * Validate input with Zod schema
     * Update preferences
     * Return updated preferences
   
   - requestDoubleOptIn(req, res): POST /api/email/opt-in
     * Send double opt-in confirmation email
     * Return success message
   
   - confirmDoubleOptIn(req, res): GET /api/email/opt-in/confirm?token=xxx
     * Verify token
     * Update doubleOptInConfirmedAt
     * Redirect to success page
   
   - unsubscribeAll(req, res): POST /api/email/unsubscribe
     * Requires authentication OR valid token
     * Unsubscribe user from all marketing
     * Return success message
   
   - unsubscribePage(req, res): GET /api/email/unsubscribe/:token
     * Display unsubscribe confirmation page (HTML)
     * Show unsubscribe options (all marketing, specific categories)
     * Form submits to POST /api/email/unsubscribe
   
   - oneClickUnsubscribe(req, res): POST /api/email/unsubscribe/one-click
     * RFC 8058 compliant
     * Verify List-Unsubscribe-Post header
     * Process unsubscribe immediately
     * Return 200 OK (no body)

4. Create routes in server/src/routes/emailPreferenceRoutes.ts:
   - GET /api/email/preferences (authenticated)
   - PUT /api/email/preferences (authenticated)
   - POST /api/email/opt-in (authenticated)
   - GET /api/email/opt-in/confirm (public, requires token)
   - POST /api/email/unsubscribe (public with token or authenticated)
   - GET /api/email/unsubscribe/:token (public)
   - POST /api/email/unsubscribe/one-click (public, RFC 8058)

5. Default preferences:
   - On user signup: Create EmailPreference with default settings
   - Default to marketingOptIn = false (GDPR compliant)
   - Transactional emails always enabled (can't be disabled)

6. Security:
   - Unsubscribe tokens: Signed JWT with userId and expiry (7 days)
   - Rate limiting on unsubscribe endpoints (prevent abuse)
   - Validate all inputs with Zod
   - Sanitize user input

7. GDPR compliance:
   - Explicit opt-in for marketing emails
   - Double opt-in option (recommended)
   - Easy unsubscribe mechanism
   - Clear preference management
   - Data export includes email preferences
   - Right to erasure deletes all email logs

8. CAN-SPAM compliance:
   - Physical address in all marketing emails
   - Unsubscribe link in every email
   - Honor unsubscribes within 10 days (immediate in our case)
   - No false/misleading header information

### 41.6. Create backend admin API endpoints for email management

**Status:** pending  
**Dependencies:** 41.1, 41.2, 41.3, 41.5  

Implement admin-only API endpoints for managing email templates, campaigns, sequences, analytics, and subscribers with role-based access control

**Details:**

1. Create Zod schemas in server/src/validation/adminSchemas.ts (if not exists, create it):
   - emailTemplateCreateSchema: name, slug, category, subject, mjmlContent, variables
   - emailTemplateUpdateSchema: partial of create schema
   - emailCampaignCreateSchema: name, templateId, segmentCriteria, scheduledAt, abTestConfig
   - emailCampaignUpdateSchema: partial of create schema
   - emailSequenceCreateSchema: name, triggerEvent, steps, enrollmentCriteria, exitCriteria
   - emailSequenceUpdateSchema: partial of create schema
   - emailAnalyticsQuerySchema: campaignId, startDate, endDate, groupBy

2. Create server/src/services/adminEmailService.ts:
   Template methods:
   - listTemplates(filters, pagination): List all templates with filtering
   - getTemplate(id): Get single template
   - createTemplate(data, adminUserId): Create new template, compile MJML, create version 1
   - updateTemplate(id, data, adminUserId): Update template, increment version, recompile MJML
   - deleteTemplate(id, adminUserId): Soft delete (set isActive=false), audit log
   - duplicateTemplate(id, newName, adminUserId): Copy template as new version
   - testTemplate(id, testEmail, sampleData): Send test email with sample variables
   
   Campaign methods:
   - listCampaigns(filters, pagination): List campaigns
   - getCampaign(id): Get campaign with analytics
   - createCampaign(data, adminUserId): Create campaign in DRAFT status
   - updateCampaign(id, data, adminUserId): Update draft campaign
   - deleteCampaign(id, adminUserId): Delete draft campaign only
   - scheduleCampaign(id, scheduledAt, adminUserId): Set status to SCHEDULED, schedule queue job
   - cancelCampaign(id, adminUserId): Cancel scheduled campaign
   - getAudienceSize(segmentCriteria): Calculate estimated audience size
   - sendTestCampaign(id, testEmails, adminUserId): Send campaign to test emails
   
   Sequence methods:
   - listSequences(filters, pagination): List sequences
   - getSequence(id): Get sequence with analytics
   - createSequence(data, adminUserId): Create sequence
   - updateSequence(id, data, adminUserId): Update sequence (only if no active enrollments)
   - deleteSequence(id, adminUserId): Delete sequence (only if no enrollments)
   - activateSequence(id, adminUserId): Set isActive=true, enable triggers
   - deactivateSequence(id, adminUserId): Set isActive=false, pause new enrollments
   - getEnrollments(sequenceId, filters, pagination): List enrollments
   - pauseEnrollment(enrollmentId, adminUserId): Pause individual enrollment
   - resumeEnrollment(enrollmentId, adminUserId): Resume paused enrollment
   - exitEnrollment(enrollmentId, reason, adminUserId): Exit user from sequence
   
   Analytics methods:
   - getCampaignAnalytics(campaignId): Get detailed campaign stats
   - getTemplatePerformance(templateId): Aggregate stats across all campaigns
   - getOverviewStats(dateRange): High-level metrics (total sent, open rate, click rate, etc.)
   - getEmailLogs(filters, pagination): Search/filter email logs
   - exportAnalytics(query): Export analytics to CSV
   
   Subscriber methods:
   - listSubscribers(filters, pagination): List all users with email preferences
   - getSubscriber(userId): Get detailed subscriber info (preferences, logs, engagement)
   - updateSubscriberPreferences(userId, data, adminUserId): Admin override of preferences
   - suppressEmail(email, reason, adminUserId): Add to suppression list
   - removeSuppress(email, adminUserId): Remove from suppression list
   - importSubscribers(csvData, adminUserId): Bulk import (validate emails, create preferences)
   - exportSubscribers(filters, adminUserId): Export to CSV

3. Create server/src/controllers/adminEmailController.ts:
   Organize endpoints by resource:
   - Template endpoints: GET/POST/PUT/DELETE /api/admin/email/templates
   - Campaign endpoints: GET/POST/PUT/DELETE /api/admin/email/campaigns
   - Sequence endpoints: GET/POST/PUT/DELETE /api/admin/email/sequences
   - Analytics endpoints: GET /api/admin/email/analytics/*
   - Subscriber endpoints: GET/PUT /api/admin/email/subscribers
   - Each endpoint validates input, calls service method, returns response
   - Audit logging for all mutations

4. Add routes to server/src/routes/admin.ts:
   - All under /api/admin/email/*
   - Require adminAuth middleware (SUPER_ADMIN or SUPPORT role)
   - Require adminAudit middleware for all mutations
   - Rate limiting: 1000 requests/hour per admin user

5. RBAC permissions:
   - SUPER_ADMIN: Full access to all endpoints
   - SUPPORT: Read-only access to analytics, can update subscriber preferences
   - ANALYST: Read-only access to analytics only
   - VIEWER: Read-only access to templates and campaigns

6. Audience segmentation logic:
   - Support filters: activityLevel, subscriptionTier, goalType, signupDateRange, lastLoginDateRange
   - SQL query builder for custom filters (admin only, sanitized)
   - Preview audience size before sending
   - Save segment as reusable template

7. A/B testing:
   - Subject line variants: split audience, track performance
   - Content variants: different templates for same campaign
   - Send time optimization: test different send times
   - Statistical significance calculator: determine winner

8. Error handling:
   - Validate all inputs with Zod
   - Return detailed error messages for admin debugging
   - Log all errors with correlation ID
   - Handle partial failures in bulk operations

### 41.7. Build admin panel MJML email template editor with variable support

**Status:** pending  
**Dependencies:** 41.1, 41.2, 41.6  

Create React-based MJML editor in the Next.js admin panel with syntax highlighting, variable insertion, live preview, and template management UI

**Details:**

1. Install dependencies in admin-panel/:
   ```bash
   npm install @monaco-editor/react monaco-editor mjml-browser handlebars html-to-text
   npm install @radix-ui/react-popover @radix-ui/react-switch @radix-ui/react-separator
   ```

2. Create admin-panel/app/(dashboard)/email/templates/page.tsx:
   - List all email templates in table
   - Columns: Name, Category, Subject, Status, Last Updated, Actions
   - Filters: Category (Transactional/Marketing), Status (Active/Inactive)
   - Search by name or slug
   - Pagination (20 per page)
   - Action buttons: Edit, Duplicate, Test, Delete
   - "Create Template" button → navigate to /email/templates/new

3. Create admin-panel/app/(dashboard)/email/templates/new/page.tsx:
   - Template form fields:
     * Name (text input)
     * Slug (auto-generated from name, editable, validates uniqueness)
     * Category (select: Transactional/Marketing)
     * Subject (text input with variable picker)
     * Description (textarea)
   - MJML editor (Monaco Editor with MJML syntax)
   - Variable picker sidebar:
     * Common variables: {{userName}}, {{email}}, {{goalProgress}}, etc.
     * Click to insert at cursor position
     * Group by category (User, Goals, Nutrition, Health, App)
   - Live preview panel (right side):
     * Compile MJML to HTML on change (debounced 500ms)
     * Render HTML in iframe
     * Desktop/Mobile/Dark mode toggle
     * Sample data for variables
   - Error display:
     * Show MJML compilation errors inline
     * Highlight error line in editor
   - Actions:
     * Save Draft (create template with isActive=false)
     * Save & Activate (create template with isActive=true)
     * Send Test Email (modal to enter test email address)
     * Cancel (navigate back)

4. Create admin-panel/app/(dashboard)/email/templates/[id]/edit/page.tsx:
   - Same UI as new template page
   - Load existing template data
   - Version history sidebar:
     * Show all versions of template
     * Preview previous versions
     * Rollback to previous version (creates new version)
   - Additional actions:
     * Duplicate Template
     * Deactivate/Activate
     * Delete (soft delete)

5. Create admin-panel/components/email/MJMLEditor.tsx:
   - Monaco Editor configured for MJML
   - Custom MJML syntax highlighting (if available)
   - Auto-complete for MJML tags
   - Validation on change
   - Error indicators in gutter
   - Variable insertion at cursor

6. Create admin-panel/components/email/EmailPreview.tsx:
   - Iframe for rendering HTML
   - Device preview modes (desktop 600px, mobile 375px)
   - Dark mode preview (CSS media query simulation)
   - Refresh button to force re-render
   - "Open in New Tab" to view full email

7. Create admin-panel/components/email/VariablePicker.tsx:
   - Collapsible sidebar or popover
   - Search/filter variables
   - Grouped by category
   - Click to insert {{variableName}} at cursor
   - Show variable description on hover

8. Create admin-panel/components/email/TemplateTestModal.tsx:
   - Input for test email address
   - Select sample data set (different user scenarios)
   - Send button
   - Loading state
   - Success/error message

9. MJML compilation:
   - Use mjml-browser for client-side compilation (preview)
   - Server-side compilation in backend (for actual sending)
   - Display compilation errors with line numbers
   - Validate required MJML structure (mj-body, mj-section, etc.)

10. Variable system:
   - Define standard variables in constants file
   - Validate variable usage in template
   - Warn about undefined variables
   - Support conditional blocks: {{#if isPremium}}...{{/if}}
   - Support loops: {{#each meals}}...{{/each}}

11. Template library:
   - Provide starter templates (Welcome, Password Reset, Weekly Report, etc.)
   - "Use Template" button to copy starter template
   - Template categories for organization

12. Error handling:
   - Catch MJML compilation errors
   - Display user-friendly error messages
   - Prevent saving invalid templates
   - Validate slug uniqueness before save

### 41.8. Build admin panel template preview, testing, and version management

**Status:** pending  
**Dependencies:** 41.7  

Create UI components for previewing templates across email clients, sending test emails, checking spam scores, and managing template versions with rollback capability

**Details:**

1. Create admin-panel/components/email/TemplatePreviewTabs.tsx:
   - Tab interface:
     * Desktop Preview (600px width)
     * Mobile Preview (375px width)
     * Dark Mode Preview (with prefers-color-scheme: dark)
     * HTML Source (syntax highlighted HTML code)
     * Plain Text (auto-generated from HTML)
   - Each tab renders EmailPreview component with different config
   - Persist selected tab in URL query params

2. Create admin-panel/components/email/EmailClientPreview.tsx:
   - Integration with Litmus or Email on Acid (optional, paid service)
   - If configured:
     * "Preview in Email Clients" button
     * Generate previews for: Gmail, Outlook, Apple Mail, etc.
     * Display screenshots in grid
     * Note: Requires LITMUS_API_KEY or EMAIL_ON_ACID_API_KEY
   - If not configured:
     * Show note about email client testing
     * Link to documentation for setup

3. Create admin-panel/components/email/SpamScoreChecker.tsx:
   - Integration with mail-tester.com or similar
   - "Check Spam Score" button
   - Generate unique email address from mail-tester API
   - Send test email to that address
   - Fetch and display spam score (0-10)
   - Show recommendations:
     * SPF/DKIM/DMARC status
     * Spam trigger words detected
     * Image-to-text ratio
     * Link quality
     * Blacklist status
   - Action items to improve score

4. Create admin-panel/components/email/SendTestEmailForm.tsx:
   - Input fields:
     * Test email address (comma-separated for multiple)
     * Sample data selector (dropdown of user scenarios)
     * Custom variable values (JSON editor for advanced users)
   - "Send Test" button
   - Loading state with progress indicator
   - Success message with links to email logs
   - Error handling with detailed messages
   - Rate limiting: max 10 test emails per minute

5. Create admin-panel/components/email/VersionHistory.tsx:
   - Timeline view of template versions
   - Each version shows:
     * Version number
     * Created by (admin user)
     * Created at (timestamp)
     * Change summary (if provided)
     * Preview button
   - Compare versions:
     * Side-by-side diff of MJML content
     * Highlight changes (additions in green, deletions in red)
   - Rollback action:
     * "Restore this version" button
     * Creates new version (doesn't delete newer versions)
     * Confirmation modal before rollback
   - Export version as JSON for backup

6. Create admin-panel/components/email/TemplateDiffViewer.tsx:
   - Use react-diff-viewer or similar library
   - Side-by-side diff of two template versions
   - Syntax highlighting for MJML
   - Line-by-line comparison
   - Collapsible unchanged sections

7. Update backend API in server/src/controllers/adminEmailController.ts:
   - POST /api/admin/email/templates/:id/test
     * Input: testEmails[], sampleDataKey
     * Render template with sample data
     * Send via Resend
     * Return email log IDs
   
   - POST /api/admin/email/templates/:id/spam-score
     * Integration with mail-tester.com API
     * Send test email to mail-tester
     * Poll for results
     * Return spam score and recommendations
   
   - GET /api/admin/email/templates/:id/versions
     * Return all versions of template
     * Include diff metadata
   
   - POST /api/admin/email/templates/:id/rollback
     * Input: targetVersion
     * Create new version with content from targetVersion
     * Audit log the rollback action
   
   - GET /api/admin/email/templates/:id/diff
     * Input: version1, version2
     * Return diff data for comparison

8. Sample data presets:
   - Create admin-panel/lib/emailSampleData.ts
   - Define user scenarios:
     * New user (just signed up)
     * Active user (7-day streak)
     * Premium user (all features)
     * Inactive user (14 days no activity)
     * Goal achieved user (reached target weight)
   - Each scenario has complete variable values
   - Used for preview and test emails

9. Error handling:
   - Catch Litmus/Email on Acid API errors
   - Handle mail-tester API failures gracefully
   - Validate test email addresses
   - Rate limit test email sends
   - Display clear error messages to admin

10. Performance:
   - Debounce live preview rendering
   - Cache MJML compilation results
   - Lazy load email client previews
   - Optimize version history queries (paginate if >50 versions)

### 41.9. Build admin panel campaign creation with audience segmentation

**Status:** pending  
**Dependencies:** 41.6, 41.7  

Create UI for creating and managing email campaigns with advanced audience segmentation, A/B testing configuration, scheduling, and estimated audience preview

**Details:**

1. Create admin-panel/app/(dashboard)/email/campaigns/page.tsx:
   - Campaign list table:
     * Columns: Name, Template, Status, Scheduled At, Audience Size, Sent, Opened, Clicked, Actions
     * Status badges: Draft, Scheduled, Sending, Sent, Cancelled
     * Filters: Status, Date range, Template
     * Search by name
     * Pagination (20 per page)
   - Action buttons: View, Edit (draft only), Cancel (scheduled only), Duplicate, Analytics
   - "Create Campaign" button → /email/campaigns/new
   - Campaign stats summary cards:
     * Total campaigns
     * Active campaigns (scheduled/sending)
     * Total emails sent (last 30 days)
     * Average open rate

2. Create admin-panel/app/(dashboard)/email/campaigns/new/page.tsx:
   - Multi-step wizard:
     Step 1: Campaign Details
     - Name (text input)
     - Description (textarea)
     - Template selector (dropdown with preview)
     - Campaign type (One-time, Recurring - future enhancement)
     
     Step 2: Audience Segmentation
     - Segment builder UI (see details below)
     - Estimated audience size (live update)
     - Preview audience (show sample users)
     - Save segment as template (optional)
     
     Step 3: A/B Testing (optional)
     - Enable A/B testing toggle
     - Test type: Subject line, Content, Send time
     - Variant configuration:
       * Variant A (default)
       * Variant B (if subject: alternate subject, if content: alternate template)
       * Split percentage (50/50, 70/30, etc.)
     - Sample size for test (percentage of audience)
     - Winner criteria: Open rate, Click rate, Manual
     - Auto-send winner: If enabled, send winning variant to remaining audience after X hours
     
     Step 4: Schedule
     - Send immediately (radio button)
     - Schedule for later (radio button)
       * Date picker
       * Time picker
       * Timezone selector (user's timezone, recipient's timezone, UTC)
     - Throttling: Send X emails per hour (optional, for large campaigns)
     
     Step 5: Review & Confirm
     - Summary of all settings
     - Estimated audience size
     - Template preview
     - "Send Test Email" button
     - "Save as Draft" or "Schedule Campaign" buttons

3. Create admin-panel/components/email/SegmentBuilder.tsx:
   - Visual query builder for audience segmentation
   - Add condition button (+)
   - Each condition has:
     * Field selector (dropdown): Activity Level, Subscription Tier, Goal Type, Signup Date, Last Login, Goal Progress, Total Meals Logged, etc.
     * Operator selector: is, is not, greater than, less than, between, contains, etc.
     * Value input: varies by field type (text, number, date, select)
   - Logical operators: AND/OR between conditions
   - Condition groups (nested AND/OR)
   - Example conditions:
     * Activity Level is "active"
     * Subscription Tier is "PRO"
     * Signup Date is between "2024-01-01" and "2024-12-31"
     * Total Meals Logged > 50
     * Last Login < 7 days ago
   - Live preview:
     * Show estimated audience size
     * "Preview Users" button → show sample users matching criteria
   - Save as segment template:
     * Name the segment
     * Reuse in future campaigns

4. Create admin-panel/components/email/AudiencePreview.tsx:
   - Modal with table of sample users (10-20 users)
   - Columns: Name, Email, Signup Date, Subscription Tier, Last Login
   - "Export Full List" button (CSV download)
   - Refresh button to re-fetch sample

5. Create admin-panel/components/email/ABTestConfig.tsx:
   - Subject line A/B test:
     * Input for Variant A subject
     * Input for Variant B subject
     * Split slider (adjust percentage)
   - Content A/B test:
     * Template selector for Variant A
     * Template selector for Variant B
     * Preview both variants side-by-side
   - Send time A/B test:
     * Time picker for Variant A
     * Time picker for Variant B
   - Winner selection:
     * Auto-select based on metric (open rate, click rate)
     * Manual selection by admin
     * Minimum sample size (e.g., 1000 emails)
     * Minimum time before declaring winner (e.g., 24 hours)

6. Create admin-panel/components/email/CampaignScheduler.tsx:
   - Date picker (react-day-picker or similar)
   - Time picker (hour/minute selector)
   - Timezone selector (Intl.DateTimeFormat)
   - "Send immediately" checkbox
   - Throttling options:
     * Max emails per hour
     * Spread over X hours
   - Confirmation modal:
     * "You're about to send to X users"
     * Show send time in user's timezone
     * Final confirmation button

7. Update backend API in server/src/controllers/adminEmailController.ts:
   - POST /api/admin/email/campaigns (create campaign)
   - PUT /api/admin/email/campaigns/:id (update draft campaign)
   - POST /api/admin/email/campaigns/:id/schedule (schedule campaign)
   - POST /api/admin/email/campaigns/:id/cancel (cancel scheduled campaign)
   - GET /api/admin/email/campaigns/:id/audience-preview (get sample users matching segment)
   - POST /api/admin/email/campaigns/:id/test (send test campaign)
   - POST /api/admin/email/segments (save segment template)
   - GET /api/admin/email/segments (list saved segments)

8. Segment query builder backend:
   - Parse segment criteria JSON
   - Build dynamic Prisma query
   - Apply filters, joins, aggregations
   - Count matching users (for audience size)
   - Fetch sample users (for preview)
   - Validate segment criteria (prevent SQL injection)

9. Campaign scheduling:
   - Create Bull queue job for scheduled campaigns
   - Job data: campaignId, scheduledAt
   - On job execution:
     * Fetch campaign and audience
     * Send emails in batches
     * Update campaign status (SENDING → SENT)
     * Track progress

10. Error handling:
   - Validate segment criteria (no empty conditions)
   - Validate A/B test config (must have 2 variants)
   - Check scheduled time is in future
   - Prevent scheduling campaign with 0 audience
   - Handle timezone conversion errors
   - Display clear error messages

### 41.10. Build admin panel automated email sequence builder (drip campaigns)

**Status:** pending  
**Dependencies:** 41.6, 41.7  

Create visual sequence builder UI with drag-and-drop flow editor, trigger configuration, step delays, conditional branching, and enrollment management

**Details:**

1. Install dependencies in admin-panel/:
   ```bash
   npm install reactflow @xyflow/react
   npm install dagre  # for auto-layout
   ```

2. Create admin-panel/app/(dashboard)/email/sequences/page.tsx:
   - Sequence list table:
     * Columns: Name, Trigger Event, Status, Active Enrollments, Completed, Actions
     * Status badges: Active, Inactive
     * Filters: Trigger Event, Status
     * Search by name
   - Action buttons: View, Edit, Activate/Deactivate, Duplicate, Analytics
   - "Create Sequence" button → /email/sequences/new
   - Sequence stats summary:
     * Total sequences
     * Active sequences
     * Total enrollments (all time)
     * Active enrollments (current)

3. Create admin-panel/app/(dashboard)/email/sequences/new/page.tsx:
   - Multi-step wizard:
     Step 1: Sequence Details
     - Name (text input)
     - Description (textarea)
     - Trigger event selector (dropdown):
       * User signup
       * First meal logged
       * Goal achieved
       * Subscription changed (free → pro)
       * Inactivity 7 days
       * Inactivity 14 days
       * Inactivity 30 days
     - Enrollment criteria (optional filters):
       * Who can enter this sequence?
       * E.g., Only free users, Only users with goal type "weight_loss"
     - Exit criteria (optional):
       * When to automatically exit sequence?
       * E.g., User upgrades to Pro, User achieves goal
     
     Step 2: Build Sequence Flow
     - Visual flow builder (ReactFlow)
     - Node types:
       * Start node (trigger event)
       * Wait node (delay for X hours/days)
       * Email node (send specific template)
       * Condition node (if/else branching)
       * End node (complete sequence)
     - Drag-and-drop interface:
       * Sidebar with node types
       * Drag onto canvas
       * Connect nodes with edges
     - Node configuration:
       * Wait node: duration (hours/days/weeks)
       * Email node: template selector, variable values
       * Condition node: criteria (e.g., "has logged meal in last 3 days?")
     - Auto-layout button (arrange nodes neatly)
     - Validation:
       * All nodes must be connected
       * No orphan nodes
       * Must end with End node
     
     Step 3: Review & Activate
     - Sequence summary
     - Flow visualization (read-only)
     - Estimated duration (time from start to end)
     - "Save as Draft" or "Activate Sequence" buttons

4. Create admin-panel/components/email/SequenceFlowBuilder.tsx:
   - ReactFlow canvas
   - Custom node components:
     * StartNode: Display trigger event, green color
     * WaitNode: Display delay duration, gray color
     * EmailNode: Display template name and subject, blue color
     * ConditionNode: Display condition logic, yellow diamond shape
     * EndNode: Display "Complete", red color
   - Node toolbar:
     * Edit (open config modal)
     * Delete
     * Duplicate
   - Edge types:
     * Default (solid line)
     * Conditional (dashed line with label: "Yes"/"No")
   - Controls:
     * Zoom in/out
     * Fit view
     * Auto-layout (dagre algorithm)
   - Validation on save:
     * Check all nodes connected
     * Verify no infinite loops
     * Ensure at least one Email node exists

5. Create admin-panel/components/email/SequenceNodeConfig.tsx:
   - Modal for configuring each node type
   - WaitNode config:
     * Duration input (number)
     * Unit selector (hours, days, weeks)
   - EmailNode config:
     * Template selector (dropdown with preview)
     * Override variables (optional, JSON editor)
   - ConditionNode config:
     * Condition builder (similar to segment builder)
     * Fields: Meals logged, Goal progress, Subscription tier, etc.
     * Operators: >, <, =, between, etc.
     * Yes/No branches

6. Create admin-panel/app/(dashboard)/email/sequences/[id]/enrollments/page.tsx:
   - Enrollments list table:
     * Columns: User, Status, Current Step, Started At, Next Email, Actions
     * Status badges: Active, Paused, Completed, Exited
     * Filters: Status, Date range
     * Search by user email/name
   - Action buttons: View Details, Pause, Resume, Exit
   - Bulk actions: Pause all, Resume all, Export

7. Create admin-panel/components/email/EnrollmentTimeline.tsx:
   - Timeline view of user's progress through sequence
   - Each step shows:
     * Step name (Wait, Email sent, Condition checked)
     * Timestamp
     * Status (Completed, Pending, Skipped)
   - Email steps show:
     * Email sent status (Sent, Delivered, Opened, Clicked)
     * Link to email log
   - Current step highlighted
   - Next scheduled email with countdown

8. Update backend API in server/src/controllers/adminEmailController.ts:
   - POST /api/admin/email/sequences (create sequence)
   - PUT /api/admin/email/sequences/:id (update sequence)
   - POST /api/admin/email/sequences/:id/activate (activate sequence)
   - POST /api/admin/email/sequences/:id/deactivate (deactivate sequence)
   - GET /api/admin/email/sequences/:id/enrollments (list enrollments)
   - POST /api/admin/email/sequences/:id/enrollments/:enrollmentId/pause (pause enrollment)
   - POST /api/admin/email/sequences/:id/enrollments/:enrollmentId/resume (resume enrollment)
   - POST /api/admin/email/sequences/:id/enrollments/:enrollmentId/exit (exit enrollment)
   - DELETE /api/admin/email/sequences/:id (delete sequence, only if no enrollments)

9. Sequence execution (backend):
   - Create Bull queue: sequenceQueue
   - On trigger event (e.g., user signup):
     * Check if user meets enrollment criteria
     * Create EmailSequenceEnrollment (status: ACTIVE, currentStep: 0)
     * Schedule first step (queue job with delay)
   - On sequence step job:
     * Get enrollment and sequence
     * Execute current step (send email, check condition)
     * If Email node: send email, create EmailLog
     * If Condition node: evaluate, determine next path
     * If Wait node: schedule next step after delay
     * Increment currentStep
     * If End node: mark enrollment as COMPLETED
   - Cron job to check for stuck enrollments:
     * Find enrollments with nextStepScheduledAt in past
     * Retry or exit with error

10. Error handling:
   - Validate sequence flow (no loops, all nodes connected)
   - Handle email send failures (retry or exit sequence)
   - Prevent activating sequence with errors
   - Log all sequence execution errors
   - Display clear error messages in admin panel

### 41.11. Build admin panel email analytics dashboard

**Status:** pending  
**Dependencies:** 41.6, 41.9, 41.10  

Create comprehensive analytics dashboard with campaign performance metrics, email log search, trend charts, and export capabilities with Apple Mail Privacy Protection awareness

**Details:**

1. Create admin-panel/app/(dashboard)/email/analytics/page.tsx:
   - Overview metrics cards (top of page):
     * Total Emails Sent (last 30 days)
     * Overall Open Rate (with Apple MPP caveat)
     * Overall Click Rate
     * Bounce Rate
     * Unsubscribe Rate
     * Active Campaigns
   - Date range selector: Last 7 days, Last 30 days, Last 90 days, Custom range
   - Filter by: Campaign, Template, Email category (Transactional/Marketing)
   - Charts section (see below for details)
   - Top performing campaigns table:
     * Columns: Campaign Name, Sent, Delivered, Opened, Clicked, CTR
     * Sort by any column
     * Click to view detailed analytics
   - Bottom performing campaigns (for optimization)

2. Create admin-panel/components/email/EmailVolumeChart.tsx:
   - Line chart: Emails sent over time
   - Series: Total sent, Delivered, Bounced
   - X-axis: Date
   - Y-axis: Email count
   - Stacked area chart option
   - Tooltip with detailed stats
   - Export chart as PNG/CSV

3. Create admin-panel/components/email/EngagementChart.tsx:
   - Multi-line chart: Open rate, Click rate, Unsubscribe rate over time
   - X-axis: Date
   - Y-axis: Percentage (0-100%)
   - Apple Mail Privacy Protection indicator:
     * Show percentage of opens from Apple proxy
     * Display caveat: "X% of opens may be automated by Apple Mail Privacy Protection"
   - Toggle to show/hide Apple proxy opens
   - Compare mode: Compare two time periods side-by-side

4. Create admin-panel/components/email/CampaignFunnelChart.tsx:
   - Funnel visualization for a campaign:
     * Sent → Delivered → Opened → Clicked
     * Show count and percentage at each stage
   - Highlight drop-off points
   - Identify bottlenecks (e.g., high bounce rate, low open rate)

5. Create admin-panel/components/email/EmailHeatmap.tsx:
   - Heatmap: Best send times analysis
   - X-axis: Hour of day (0-23)
   - Y-axis: Day of week (Mon-Sun)
   - Color intensity: Open rate or Click rate
   - Identify optimal send times
   - Recommendation: "Best time to send: Tuesday at 10 AM"

6. Create admin-panel/app/(dashboard)/email/analytics/campaigns/[id]/page.tsx:
   - Detailed campaign analytics page
   - Campaign info section:
     * Name, Template, Sent date, Audience size
   - Key metrics cards:
     * Sent, Delivered, Opened, Clicked, Bounced, Unsubscribed
     * Open rate, Click-through rate, Bounce rate, Unsubscribe rate
   - Email lifecycle funnel (sent → delivered → opened → clicked)
   - Click heatmap:
     * Show which links in email were clicked most
     * Overlay click counts on email preview
   - Time-series chart: Opens and clicks over time (first 48 hours)
   - Top clicked links table:
     * Columns: Link URL, Clicks, Unique Clicks, CTR
   - Bounces breakdown:
     * Hard bounces vs. Soft bounces
     * Bounce reasons (table)
   - Unsubscribe reasons (if collected)
   - A/B test results (if applicable):
     * Variant A vs. Variant B performance
     * Winner declared at
     * Statistical significance
   - Export campaign analytics (CSV, PDF)

7. Create admin-panel/components/email/ClickHeatmap.tsx:
   - Overlay click data on email HTML preview
   - Use absolute positioning to show click counts on links
   - Color-code links by click volume (green = most clicks, red = least)
   - Tooltip on hover: Show exact click count and CTR
   - Toggle to show/hide heatmap

8. Create admin-panel/app/(dashboard)/email/logs/page.tsx:
   - Email log search and filter interface
   - Filters:
     * Date range
     * Status (Sent, Delivered, Opened, Clicked, Bounced, Unsubscribed)
     * User email/name
     * Campaign
     * Template
     * Email category (Transactional/Marketing)
   - Search bar: Free text search (email, subject)
   - Logs table:
     * Columns: User, Email, Template, Campaign, Status, Sent At, Opened At, Clicked At, Actions
     * Expandable row: Show full email details (metadata, bounce reason, clicks)
   - Pagination (50 per page)
   - Export logs (CSV) with filters applied
   - Bulk actions: Resend (for bounced emails), View email content

9. Create admin-panel/components/email/EmailLogDetails.tsx:
   - Modal or drawer with detailed log info
   - Sections:
     * Email metadata (Message ID, From, To, Subject, Sent At)
     * Delivery timeline (Sent → Delivered → Opened → Clicked)
     * Bounce info (if bounced): Type, Reason, Provider response
     * Click data (if clicked): URLs clicked, Timestamps
     * Apple MPP detection (if opened): Flag if proxy open
   - "View Email" button: Open email HTML in new tab
   - "Resend Email" button (for failed/bounced emails)

10. Update backend API in server/src/controllers/adminEmailController.ts:
   - GET /api/admin/email/analytics/overview (date range, filters → overview stats)
   - GET /api/admin/email/analytics/trends (date range, metric → time-series data)
   - GET /api/admin/email/analytics/campaigns/:id (detailed campaign analytics)
   - GET /api/admin/email/analytics/click-heatmap/:campaignId (click data for heatmap)
   - GET /api/admin/email/analytics/send-time-heatmap (best send times analysis)
   - GET /api/admin/email/logs (filters, pagination → email logs)
   - GET /api/admin/email/logs/:id (detailed log info)
   - POST /api/admin/email/analytics/export (filters → CSV/PDF export)

11. Analytics calculations:
   - Open rate: (Opened / Delivered) * 100
     * Caveat: Exclude or flag Apple proxy opens
   - Click-through rate: (Clicked / Delivered) * 100
   - Bounce rate: (Bounced / Sent) * 100
   - Unsubscribe rate: (Unsubscribed / Delivered) * 100
   - Conversion rate: (Conversions / Clicked) * 100 (if tracking conversions)

12. Apple Mail Privacy Protection handling:
   - Flag opens from Apple proxy IPs
   - Display MPP percentage in analytics
   - Show caveat in open rate metrics
   - Provide "Adjusted Open Rate" (excluding MPP)
   - Recommend focusing on click rate for engagement

13. Performance optimizations:
   - Cache analytics queries (5-minute TTL)
   - Pre-aggregate campaign stats in EmailCampaign metadata
   - Use database indexes for log queries
   - Paginate large result sets
   - Export large datasets as background job (send download link via email)

### 41.12. Build admin panel subscriber management and list hygiene

**Status:** pending  
**Dependencies:** 41.6, 41.11  

Create UI for managing email subscribers, viewing engagement scores, importing/exporting lists, managing suppression lists, and performing list hygiene operations

**Details:**

1. Create admin-panel/app/(dashboard)/email/subscribers/page.tsx:
   - Subscribers list table:
     * Columns: User, Email, Preferences, Engagement Score, Last Email Opened, Status, Actions
     * Status indicators: Active, Unsubscribed, Bounced, Complained
     * Filters:
       - Status (Active, Unsubscribed, Bounced, Complained)
       - Subscription tier (Free, Pro)
       - Engagement level (High, Medium, Low, Inactive)
       - Preference categories (subscribed to X category)
       - Signup date range
       - Last activity date range
     * Search: Email, Name, User ID
   - Pagination (50 per page)
   - Action buttons: View Details, Edit Preferences, Suppress, Unsuppress
   - Bulk actions:
     * Export selected (CSV)
     * Suppress selected
     * Update preferences (bulk)
   - "Import Subscribers" button → open import modal
   - Subscriber stats cards:
     * Total subscribers
     * Active (opted-in)
     * Unsubscribed
     * Suppressed (bounced/complained)

2. Create admin-panel/components/email/SubscriberDetails.tsx:
   - Modal or side panel with detailed subscriber info
   - Sections:
     * User info: Name, Email, Signup date, Subscription tier
     * Email preferences:
       - Categories (checkboxes for each category)
       - Frequency (dropdown)
       - Marketing opt-in status
       - Double opt-in confirmed date
     * Engagement metrics:
       - Total emails received
       - Total opened (percentage)
       - Total clicked (percentage)
       - Last email opened (date)
       - Engagement score (0-100, calculated)
     * Email history:
       - Recent emails sent (last 10)
       - Table: Template, Campaign, Sent At, Status (Delivered/Opened/Clicked)
       - Click to view email log details
     * Suppression status:
       - Is suppressed? (Yes/No)
       - Reason (Hard bounce, Soft bounce, Complaint, Manual)
       - Suppressed at (date)
       - "Remove from suppression" button
   - Edit preferences:
     * Update any preference
     * Save button
     * Audit log action
   - Actions:
     * Send test email
     * View full email logs
     * Suppress/Unsuppress
     * Export subscriber data (JSON)

3. Create admin-panel/components/email/EngagementScoreCalculator.tsx:
   - Calculate engagement score (0-100) based on:
     * Open rate (40 points max): (Opened / Sent) * 40
     * Click rate (40 points max): (Clicked / Sent) * 40
     * Recency (20 points max):
       - Last opened < 7 days ago: 20 points
       - Last opened 7-30 days ago: 10 points
       - Last opened > 30 days ago: 0 points
   - Display score with color indicator:
     * 80-100: Green (Highly engaged)
     * 50-79: Yellow (Moderately engaged)
     * 20-49: Orange (Low engagement)
     * 0-19: Red (Inactive)
   - Show score breakdown tooltip

4. Create admin-panel/components/email/ImportSubscribersModal.tsx:
   - CSV upload interface
   - Expected CSV format:
     * Required columns: email, name
     * Optional columns: goalCalories, goalProtein, marketingOptIn, categories
   - Upload steps:
     1. Upload CSV file
     2. Preview data (show first 10 rows)
     3. Map columns (if headers don't match exactly)
     4. Validate emails (check format, duplicates)
     5. Confirm import
   - Import options:
     * Skip duplicates
     * Update existing users
     * Send welcome email to new subscribers
   - Progress indicator during import
   - Results summary:
     * X subscribers imported
     * Y skipped (duplicates)
     * Z errors (invalid emails)
   - Download error log (CSV) if any errors

5. Create admin-panel/components/email/ExportSubscribersModal.tsx:
   - Export options:
     * Format: CSV, JSON
     * Filters: Apply current filters or select custom
     * Include fields:
       - User info (email, name, signup date)
       - Preferences (categories, frequency)
       - Engagement metrics (open rate, click rate, score)
       - Email history (optional)
   - Export button
   - Download link (generated server-side)
   - For large exports (>10k subscribers):
     * Process as background job
     * Send download link via email when ready

6. Create admin-panel/app/(dashboard)/email/suppression/page.tsx:
   - Suppression list management
   - Tabs:
     * Hard Bounces
     * Soft Bounces
     * Complaints (spam reports)
     * Manual Suppressions
   - Each tab shows table:
     * Columns: Email, Reason, Suppressed At, Actions
   - Filters: Date range, Reason
   - Search by email
   - Action buttons: View Details, Remove from Suppression
   - Bulk actions:
     * Remove selected from suppression
     * Export suppression list (CSV)
   - Suppression stats:
     * Total suppressed
     * Hard bounces count
     * Soft bounces count
     * Complaints count
     * Manual suppressions count

7. Create admin-panel/components/email/ListHygieneTools.tsx:
   - List hygiene operations panel
   - Tools:
     * Identify inactive subscribers:
       - Define: No email opened in last X days (configurable)
       - Show count of inactive subscribers
       - Actions: Suppress, Re-engagement campaign, Export
     * Remove duplicate emails:
       - Find users with duplicate emails
       - Show count and list
       - Action: Merge or delete duplicates
     * Validate email addresses:
       - Check for invalid email formats
       - Check for disposable email domains
       - Show count and list
       - Action: Suppress invalid emails
     * Clean soft bounces:
       - Remove soft bounces older than X days (default 30)
       - Give them another chance
   - Schedule automated list hygiene:
     * Cron job to run list hygiene weekly
     * Send report to admin email

8. Update backend API in server/src/controllers/adminEmailController.ts:
   - GET /api/admin/email/subscribers (filters, pagination → subscribers list)
   - GET /api/admin/email/subscribers/:userId (detailed subscriber info)
   - PUT /api/admin/email/subscribers/:userId/preferences (update preferences)
   - POST /api/admin/email/subscribers/:userId/suppress (add to suppression list)
   - POST /api/admin/email/subscribers/:userId/unsuppress (remove from suppression)
   - POST /api/admin/email/subscribers/import (CSV upload → import subscribers)
   - POST /api/admin/email/subscribers/export (filters → CSV/JSON export)
   - GET /api/admin/email/suppression (filters, pagination → suppression list)
   - DELETE /api/admin/email/suppression/:email (remove from suppression)
   - POST /api/admin/email/hygiene/inactive (threshold → identify inactive subscribers)
   - POST /api/admin/email/hygiene/duplicates (→ find and merge/delete duplicates)
   - POST /api/admin/email/hygiene/validate (→ validate email addresses)
   - POST /api/admin/email/hygiene/soft-bounces (→ clean old soft bounces)

9. Engagement score calculation (backend):
   - Calculate on-demand or cache in EmailPreference
   - Recalculate daily via cron job
   - Use for segmentation (e.g., send re-engagement campaign to low-engagement users)

10. List hygiene automations:
   - Daily cron job:
     * Suppress hard bounces automatically
     * Flag users with >3 soft bounces
     * Identify inactive users (no opens in 90 days)
   - Weekly cron job:
     * Clean soft bounces older than 30 days
     * Validate new email addresses
     * Generate list hygiene report
   - Monthly cron job:
     * Re-engagement campaign for inactive users
     * Remove suppression for soft bounces (give second chance)

11. Security and compliance:
   - Audit log all subscriber preference updates
   - Require admin authentication for all operations
   - RBAC: SUPER_ADMIN can suppress/unsuppress, SUPPORT can view only
   - Validate all imported data
   - Prevent suppression of transactional emails

12. Performance:
   - Index EmailPreference by email, status
   - Cache engagement scores
   - Paginate large subscriber lists
   - Process large imports as background jobs
   - Export large lists as background jobs

### 41.13. Build mobile app email preferences screen

**Status:** pending  
**Dependencies:** 41.5  

Create React Native screens for users to manage their email preferences, including category toggles, frequency settings, and unsubscribe options, integrated with backend email preference API

**Details:**

1. Create app/email-preferences.tsx:
   - Screen title: "Email Preferences"
   - Description text: "Choose which emails you'd like to receive from Nutri"
   - Fetch user's EmailPreference on mount (GET /api/email/preferences)
   - Loading state while fetching
   - Error handling with retry button

2. Email category toggles:
   - Section: "Email Categories"
   - Each category as toggle switch:
     * Weekly Progress Reports
       - Description: "Your weekly nutrition and health summary"
       - Icon: ChartBarIcon
     * Health Insights
       - Description: "Personalized insights based on your data"
       - Icon: LightbulbIcon
     * Tips & Recipes
       - Description: "Nutrition tips and healthy recipe ideas"
       - Icon: BookOpenIcon
     * Feature Updates
       - Description: "New features and app improvements"
       - Icon: BellIcon
     * Promotional Offers
       - Description: "Special offers and promotions"
       - Icon: TagIcon
   - Each toggle updates local state immediately
   - Debounced save to backend (500ms after last change)

3. Email frequency setting:
   - Section: "Email Frequency"
   - Radio buttons or picker:
     * Real-time (as events happen)
     * Daily Digest (once per day)
     * Weekly Digest (once per week)
   - Selected option highlighted
   - Save to backend on change

4. Marketing opt-in:
   - Section: "Marketing Emails"
   - Toggle switch: "Receive marketing emails"
   - Description: "Get updates about new features, tips, and offers"
   - Disabled if user has globally unsubscribed
   - Note: "Transactional emails (like password resets) will always be sent"

5. Global unsubscribe:
   - Section: "Unsubscribe"
   - Danger zone (red background)
   - "Unsubscribe from all marketing emails" button
   - Confirmation alert:
     * Title: "Unsubscribe from all?"
     * Message: "You'll no longer receive marketing emails. Transactional emails (like password resets) will still be sent."
     * Buttons: Cancel, Confirm
   - On confirm: Call POST /api/email/unsubscribe
   - Show success message: "You've been unsubscribed. You can resubscribe anytime."

6. Resubscribe option:
   - If user is globally unsubscribed:
     * Show message: "You're currently unsubscribed from all marketing emails."
     * "Resubscribe" button
     * On tap: Clear globalUnsubscribedAt, restore previous preferences
     * Show success message: "You've been resubscribed. Update your preferences below."

7. Update email address:
   - Section: "Email Address"
   - Display current email (read-only or editable)
   - "Change Email" button → navigate to profile settings
   - Note: "Email updates will be sent to this address"

8. Save button:
   - Sticky button at bottom: "Save Preferences"
   - Enabled only if preferences have changed
   - Loading state during save
   - Success message: "Preferences saved!"
   - Error handling: Display error message, retry option

9. Create lib/api/emailPreferences.ts:
   - getPreferences(): GET /api/email/preferences
   - updatePreferences(data): PUT /api/email/preferences
   - unsubscribeAll(): POST /api/email/unsubscribe
   - resubscribe(): POST /api/email/resubscribe (or updatePreferences with globalUnsubscribedAt = null)
   - All methods use authenticated API client

10. Add navigation to email preferences:
   - From app/(tabs)/profile.tsx:
     * Add "Email Preferences" row in Settings section
     * Icon: EnvelopeIcon
     * On tap: router.push('/email-preferences')

11. Optional: Onboarding email opt-in:
   - During signup flow (app/auth/signup.tsx):
     * After creating account, show modal:
       - Title: "Stay updated with Nutri"
       - Description: "Get personalized health insights, nutrition tips, and progress reports delivered to your inbox."
       - Checkboxes for pre-selected categories (Weekly Reports, Health Insights)
       - "Get Started" button (opt-in)
       - "Skip" button (opt-out)
     * On "Get Started": Create EmailPreference with marketingOptIn=true
     * On "Skip": Create EmailPreference with marketingOptIn=false
   - GDPR compliant: Explicit consent, no pre-checked boxes (unless user actively taps "Get Started")

12. Styling:
   - Use consistent design system (lib/theme/colors.ts)
   - Toggle switches: iOS-style or Material-style
   - Sections separated with borders or spacing
   - Danger zone (unsubscribe) with red/orange background
   - Icons for each category (Lucide icons or SF Symbols)
   - Responsive padding and margins

13. Accessibility:
   - accessibilityLabel for all toggles and buttons
   - accessibilityHint for non-obvious actions
   - Screen reader announcements for save success/error
   - Sufficient color contrast
   - Touch targets ≥44px

14. Analytics (optional):
   - Track events:
     * Email preferences viewed
     * Category toggled (on/off)
     * Frequency changed
     * Unsubscribed
     * Resubscribed
   - Use existing analytics service

### 41.14. Implement compliance features: GDPR double opt-in, one-click unsubscribe (RFC 8058), and CAN-SPAM

**Status:** pending  
**Dependencies:** 41.1, 41.2, 41.3, 41.5  

Implement all compliance features including GDPR-compliant double opt-in flow, RFC 8058 one-click unsubscribe headers, CAN-SPAM footer requirements, and privacy policy integration

**Details:**

1. GDPR Double Opt-In Implementation:
   - Update backend service (server/src/services/emailPreferenceService.ts):
     * initiateDoubleOptIn(userId: string): 
       - Generate secure token (JWT with 7-day expiry)
       - Send confirmation email with confirmation link
       - Email template: "Please confirm your email subscription"
       - Link format: https://nutriapp.com/api/email/opt-in/confirm?token=xxx
     * confirmDoubleOptIn(token: string):
       - Verify token
       - Update EmailPreference: set doubleOptInConfirmedAt = now()
       - Send welcome email
       - Return success message
   - Create double opt-in email template:
     * Subject: "Please confirm your email subscription to Nutri"
     * Body: 
       - "Thanks for subscribing to Nutri emails!"
       - "Please confirm your subscription by clicking the button below."
       - Confirmation button (CTA)
       - "If you didn't subscribe, you can safely ignore this email."
     * Template type: TRANSACTIONAL (always delivered)
   - API endpoints:
     * POST /api/email/opt-in (authenticated) → sends confirmation email
     * GET /api/email/opt-in/confirm?token=xxx (public) → confirms opt-in

2. One-Click Unsubscribe (RFC 8058):
   - Update emailService.ts to include List-Unsubscribe headers:
     * List-Unsubscribe: <https://nutriapp.com/api/email/unsubscribe/{{token}}>, <mailto:unsubscribe@nutriapp.com?subject=unsubscribe>
     * List-Unsubscribe-Post: List-Unsubscribe=One-Click
   - Unsubscribe token generation:
     * generateUnsubscribeToken(userId, campaignId):
       - JWT with userId, campaignId, expiry (90 days)
       - Signed with EMAIL_UNSUBSCRIBE_SECRET
   - Backend endpoint:
     * POST /api/email/unsubscribe/one-click
       - No authentication required
       - Parse List-Unsubscribe-Post header
       - Verify token
       - Immediately unsubscribe user (set globalUnsubscribedAt)
       - Exit user from all active sequences
       - Return 200 OK (no body, per RFC 8058)
       - Log action in EmailLog
   - GET /api/email/unsubscribe/:token
     * Display unsubscribe confirmation page (HTML)
     * Show message: "You've been unsubscribed from Nutri marketing emails."
     * Options:
       - "Unsubscribe from all marketing" (default, already done)
       - "Unsubscribe from specific categories" (show checkboxes)
       - "Update email preferences" (link to mobile app or web preferences page)
     * No login required
     * Track unsubscribe in EmailLog

3. CAN-SPAM Compliance:
   - Email footer template (add to all marketing emails):
     * Physical address: 
       - "Nutri Inc., [Your Company Address], [City, State, ZIP]"
       - Use env var: EMAIL_PHYSICAL_ADDRESS
     * Sender identification:
       - "This email was sent to {{email}} by Nutri."
     * Unsubscribe link:
       - "Unsubscribe from marketing emails: {{unsubscribeUrl}}"
       - "Update your email preferences: {{preferencesUrl}}"
     * Disclaimer:
       - "We respect your privacy. View our Privacy Policy: {{privacyPolicyUrl}}"
   - Update email templates:
     * Include footer in all MJML templates
     * Use mj-section for footer
     * Style: Gray background, small font, centered text
   - Honest subject lines:
     * Validate subject lines in admin panel
     * Warn if subject contains misleading words ("Free", "Urgent", "Act now")
     * Prevent "RE:" or "FWD:" in subject (unless genuine reply/forward)
   - Honor unsubscribes:
     * Process unsubscribes immediately (already implemented)
     * Do not send marketing emails to unsubscribed users
     * Check unsubscribe status before queuing emails

4. Privacy Policy Integration:
   - Update app/privacy.tsx (mobile app):
     * Add section: "Email Communications"
     * Explain:
       - What emails we send (transactional vs. marketing)
       - How to opt-in/opt-out
       - How we use email data (open/click tracking)
       - Third-party email service provider (Resend)
       - Data retention policy (email logs kept for X days)
     * Link to email preferences screen
   - Email footer includes link to privacy policy:
     * https://nutriapp.com/privacy (web version)
     * Or app deep link: nutriapp://privacy

5. Data Retention Policy:
   - Define retention periods:
     * EmailLog: Keep for 90 days, then delete (GDPR right to erasure)
     * EmailPreference: Keep as long as user account exists
     * EmailCampaign: Keep indefinitely for analytics (anonymize after 1 year)
     * EmailTemplate: Keep indefinitely (version history)
   - Implement data deletion cron job:
     * Daily job: Delete EmailLog entries older than 90 days
     * On user deletion: Delete all EmailLog, EmailPreference, EmailSequenceEnrollment
   - GDPR data export:
     * Include email preferences and email history in user data export
     * Format: JSON or CSV

6. GDPR Right to Erasure:
   - On user account deletion:
     * Delete EmailPreference
     * Delete EmailLog
     * Delete EmailSequenceEnrollment
     * Anonymize EmailCampaign analytics (remove userId, keep aggregated stats)
   - Provide user data export:
     * Include email preferences, email history, unsubscribe status
     * Downloadable from profile settings

7. Consent Management:
   - Track consent:
     * EmailPreference.doubleOptInConfirmedAt: When user confirmed opt-in
     * EmailPreference.marketingOptIn: Explicit consent for marketing
     * Audit log: Track all preference changes
   - Granular consent:
     * Category-level opt-in (Weekly Reports, Tips, Promotions, etc.)
     * User can opt-in/out of specific categories
   - Re-consent flow (if needed):
     * If email laws change, trigger re-consent campaign
     * Send email: "We've updated our email policy. Please confirm your subscription."
     * Link to double opt-in confirmation

8. Apple Mail Privacy Protection Handling:
   - Detect Apple proxy opens:
     * Check user agent for "AppleMailProxy"
     * Check IP address against known Apple proxy IPs
     * Set EmailLog.isAppleProxyOpen = true
   - Analytics adjustments:
     * Display open rate with caveat: "May include automated opens"
     * Provide "Adjusted Open Rate" (excluding Apple proxy opens)
     * Focus on click rate for engagement metrics
   - Inform admins:
     * Dashboard notice: "Apple Mail Privacy Protection may inflate open rates"
     * Link to documentation explaining MPP

9. Unsubscribe Confirmation Page:
   - Create server/src/views/unsubscribe.html (or use template engine):
     * HTML page shown after one-click unsubscribe
     * Message: "You've been unsubscribed from Nutri marketing emails."
     * Options:
       - "Resubscribe" button (POST /api/email/resubscribe with token)
       - "Update preferences" link (to mobile app or web preferences)
     * Footer with privacy policy link
   - Serve via Express:
     * app.get('/email/unsubscribe/:token', renderUnsubscribePage)

10. Compliance Monitoring:
   - Admin dashboard alerts:
     * High unsubscribe rate (>2% per campaign)
     * High complaint rate (>0.1%)
     * High bounce rate (>5%)
     * Low double opt-in confirmation rate (<50%)
   - Weekly compliance report:
     * Unsubscribe rate trend
     * Complaint rate trend
     * Bounce rate trend
     * Email sent to unsubscribed users (should be 0)
   - Audit log:
     * All unsubscribe events
     * All preference changes
     * All admin actions (suppress, unsuppress, etc.)

11. Email Authentication Monitoring:
   - Monitor DKIM/SPF/DMARC status:
     * Check domain authentication in Resend dashboard
     * Alert if authentication fails
   - Monitor domain reputation:
     * Check blacklist status (use MXToolbox API or similar)
     * Alert if domain is blacklisted
   - Monitor deliverability:
     * Track bounce rate, complaint rate
     * Alert if rates exceed thresholds

12. Testing Compliance:
   - Test double opt-in flow:
     * User opts in → receives confirmation email → confirms → doubleOptInConfirmedAt set
   - Test one-click unsubscribe:
     * Email sent with List-Unsubscribe headers → user clicks unsubscribe → immediately unsubscribed
   - Test CAN-SPAM footer:
     * All marketing emails include footer with address, unsubscribe link, privacy policy link
   - Test unsubscribe honor:
     * Unsubscribed user should not receive marketing emails
     * Verify in campaign sending logic
   - Test data deletion:
     * User deletes account → all email data deleted/anonymized
   - Test GDPR data export:
     * User requests data export → receives email preferences and history
