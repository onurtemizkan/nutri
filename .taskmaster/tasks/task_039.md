# Task ID: 39

**Title:** Build Admin Panel for User and Subscription Management

**Status:** in-progress

**Dependencies:** 38

**Priority:** high

**Description:** Build a secure, internal admin panel for managing users, subscriptions, webhook events, and analytics. Essential for customer support, subscription troubleshooting, GDPR compliance, and business analytics. Phased approach starting with MVP features critical for subscription launch.

**Details:**

## Overview
Build a production-ready admin panel for the Nutri app to handle customer support, subscription management, compliance requirements, and business analytics. The admin panel is essential for operating a subscription-based app - while Apple handles billing, customer experience and support require internal tooling.

## Why This Is Needed (Not Optional)

1. **Customer Support**: Users will report "I paid but don't have access" - need instant lookup
2. **Subscription Troubleshooting**: Debug webhook failures, verify transaction status
3. **GDPR Compliance**: Legal requirement for data export/deletion tooling
4. **Business Analytics**: MRR, churn, conversion - not available at this granularity in App Store Connect
5. **Operational Efficiency**: Feature flags, food database management, ML monitoring

## Technology Stack (2025 Best Practices)

### Recommended: Next.js 14+ Admin App
- **Framework**: Next.js 14+ with App Router (React Server Components)
- **UI Library**: shadcn/ui + Tailwind CSS (consistent with React Native web styling)
- **Charts/Dashboards**: Tremor (built on shadcn/ui) or Recharts
- **Tables**: TanStack Table v8 with server-side pagination
- **Forms**: React Hook Form + Zod (same as backend validation)
- **Authentication**: NextAuth.js v5 with separate admin user table
- **Deployment**: Vercel (or same infrastructure as backend)

### Alternative Options Considered:
- **Retool/Appsmith**: Faster to build but vendor lock-in, less customizable
- **React Admin**: Mature but opinionated, less modern DX
- **Refine**: Good option, similar to our recommendation

### Why Next.js:
- Same React skills as mobile app
- Server Components for secure data fetching
- API routes can proxy to main backend
- Excellent DX with hot reload
- Easy deployment

## Architecture

```
admin-panel/                    # Separate Next.js app
├── app/
│   ├── (auth)/
│   │   ├── login/page.tsx     # Admin login
│   │   └── layout.tsx
│   ├── (dashboard)/
│   │   ├── layout.tsx         # Dashboard layout with sidebar
│   │   ├── page.tsx           # Overview dashboard
│   │   ├── users/
│   │   │   ├── page.tsx       # User list with search/filter
│   │   │   └── [id]/page.tsx  # User detail view
│   │   ├── subscriptions/
│   │   │   ├── page.tsx       # Subscription list
│   │   │   └── [id]/page.tsx  # Subscription detail
│   │   ├── webhooks/
│   │   │   └── page.tsx       # Webhook event logs
│   │   ├── analytics/
│   │   │   └── page.tsx       # Subscription analytics
│   │   └── settings/
│   │       ├── feature-flags/page.tsx
│   │       └── team/page.tsx  # Admin user management
│   └── api/                   # Proxy to main backend
├── components/
│   ├── ui/                    # shadcn/ui components
│   ├── dashboard/             # Dashboard-specific components
│   └── data-table/            # Reusable table components
├── lib/
│   ├── api.ts                 # Backend API client
│   ├── auth.ts                # NextAuth configuration
│   └── utils.ts
└── middleware.ts              # Auth middleware
```

## Phase 1: MVP (Launch Critical) - 2-3 weeks

### 1.1 Admin Authentication & Security
- Separate AdminUser table in database (NOT shared with app users)
- Email/password with mandatory MFA (TOTP)
- Role-based access: SUPER_ADMIN, SUPPORT, VIEWER
- Session-based auth with secure cookies
- Audit logging for ALL admin actions
- IP allowlisting option for production
- Rate limiting on login endpoint

### 1.2 User Management
- **User Search**: By email, ID, name (instant search with debounce)
- **User List**: Paginated, sortable, filterable table
- **User Detail View**:
  - Profile information (email, name, created date)
  - Subscription status (tier, expires, trial info)
  - Recent activity (meals logged, health metrics)
  - Account actions: Reset password link, disable account
- **GDPR Actions**:
  - Export user data (JSON download)
  - Delete user account (with confirmation, cascades to all data)

### 1.3 Subscription Management
- **Subscription List**: All subscriptions with status filter
- **Subscription Detail**:
  - Current status, product, expiration
  - Transaction history (all originalTransactionIds)
  - Webhook events related to this subscription
  - Auto-renew status
- **Manual Actions** (SUPER_ADMIN only):
  - Grant Pro access (specify duration, reason logged)
  - Extend subscription (specify days, reason logged)
  - Revoke access (with reason)
- **Subscription Lookup**: By originalTransactionId for support tickets

### 1.4 Webhook Event Viewer
- **Event List**: All App Store webhook events, newest first
- **Filters**: By notification type, date range, subscription ID
- **Event Detail**: Full JSON payload, processing status, errors
- **Retry Failed**: Button to reprocess failed webhooks
- **Search**: By originalTransactionId for debugging

### 1.5 Basic Analytics Dashboard
- **Subscription Metrics** (real-time):
  - Total active subscribers (by tier)
  - MRR (Monthly Recurring Revenue)
  - New subscriptions today/this week/this month
  - Cancellations/churn today/this week/this month
- **Trial Metrics**:
  - Active trials
  - Trial conversion rate (7-day, 30-day)
- **Charts**:
  - Subscribers over time (line chart)
  - Revenue over time (line chart)
  - Subscription status distribution (pie chart)

## Phase 2: Growth Features - 2-3 weeks

### 2.1 Advanced Analytics
- Cohort retention analysis
- LTV (Lifetime Value) calculation
- Churn prediction indicators
- Geographic distribution
- Revenue by product (monthly vs yearly)
- Refund rate monitoring with alerts

### 2.2 Feature Flags Management
- Create/edit/delete feature flags
- Target by: user ID, subscription tier, percentage rollout
- Flag types: boolean, string, number, JSON
- Instant propagation to app (webhook or polling)
- Audit log of flag changes

### 2.3 Food Database Management
- View/search food database entries
- Edit nutritional information
- Add new foods
- Flag/review user-submitted entries
- Bulk import from CSV

### 2.4 Push Notification Management
- Send push notifications to segments
- Segments: all users, Pro users, trial users, inactive users
- Schedule notifications
- View delivery stats

## Phase 3: Scale Features - 2+ weeks

### 3.1 Team Management
- Invite admin users
- Role assignment (SUPER_ADMIN, SUPPORT, VIEWER, ANALYST)
- Permission matrix by role
- Activity log per admin user
- Disable/remove admin users

### 3.2 A/B Testing Dashboard
- View active experiments
- Create new experiments (paywall variants, onboarding flows)
- View results with statistical significance
- Declare winners and roll out

### 3.3 ML Model Monitoring
- Model performance metrics over time
- Inference latency tracking
- Error rate monitoring
- Model version management
- A/B test model versions

### 3.4 Advanced Security
- Login anomaly detection
- Suspicious activity alerts
- API key management for integrations
- Webhook secret rotation

## Database Schema Additions

```prisma
// Add to server/prisma/schema.prisma

model AdminUser {
  id            String   @id @default(cuid())
  email         String   @unique
  passwordHash  String
  name          String
  role          AdminRole @default(SUPPORT)
  mfaSecret     String?  // TOTP secret
  mfaEnabled    Boolean  @default(false)
  isActive      Boolean  @default(true)
  lastLoginAt   DateTime?
  lastLoginIp   String?
  createdAt     DateTime @default(now())
  updatedAt     DateTime @updatedAt
  
  auditLogs     AdminAuditLog[]
  
  @@index([email])
}

enum AdminRole {
  SUPER_ADMIN  // Full access, can manage other admins
  SUPPORT      // User/subscription management, no settings
  ANALYST      // Read-only analytics access
  VIEWER       // Read-only all access
}

model AdminAuditLog {
  id          String   @id @default(cuid())
  adminUserId String
  adminUser   AdminUser @relation(fields: [adminUserId], references: [id])
  action      String   // e.g., "USER_LOOKUP", "SUBSCRIPTION_GRANT", "USER_DELETE"
  targetType  String?  // e.g., "User", "Subscription"
  targetId    String?  // ID of affected record
  details     Json?    // Additional context
  ipAddress   String
  userAgent   String?
  createdAt   DateTime @default(now())
  
  @@index([adminUserId])
  @@index([action])
  @@index([targetType, targetId])
  @@index([createdAt])
}

model FeatureFlag {
  id          String   @id @default(cuid())
  key         String   @unique  // e.g., "new_paywall_design"
  name        String
  description String?
  type        FeatureFlagType @default(BOOLEAN)
  value       Json     // Default value
  isEnabled   Boolean  @default(false)
  targeting   Json?    // Rules for user targeting
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
  
  @@index([key])
}

enum FeatureFlagType {
  BOOLEAN
  STRING
  NUMBER
  JSON
}
```

## Backend API Additions

Create admin-specific endpoints in server/src/routes/admin/:

```typescript
// Admin authentication
POST   /api/admin/auth/login
POST   /api/admin/auth/logout
POST   /api/admin/auth/mfa/setup
POST   /api/admin/auth/mfa/verify
GET    /api/admin/auth/me

// User management
GET    /api/admin/users              // List with pagination, search, filters
GET    /api/admin/users/:id          // User detail with subscription
POST   /api/admin/users/:id/export   // GDPR data export
DELETE /api/admin/users/:id          // GDPR deletion
POST   /api/admin/users/:id/disable  // Disable account

// Subscription management
GET    /api/admin/subscriptions      // List with filters
GET    /api/admin/subscriptions/:id  // Detail with events
POST   /api/admin/subscriptions/:id/grant   // Manual grant
POST   /api/admin/subscriptions/:id/extend  // Extend duration
POST   /api/admin/subscriptions/:id/revoke  // Revoke access
GET    /api/admin/subscriptions/lookup?txn=XXX  // By transaction ID

// Webhook events
GET    /api/admin/webhooks           // List with filters
GET    /api/admin/webhooks/:id       // Event detail
POST   /api/admin/webhooks/:id/retry // Retry processing

// Analytics
GET    /api/admin/analytics/subscriptions  // Subscription metrics
GET    /api/admin/analytics/revenue        // Revenue metrics
GET    /api/admin/analytics/trials         // Trial metrics

// Feature flags
GET    /api/admin/feature-flags
POST   /api/admin/feature-flags
PUT    /api/admin/feature-flags/:id
DELETE /api/admin/feature-flags/:id

// Audit logs
GET    /api/admin/audit-logs         // List with filters
```

## Security Requirements (Non-Negotiable)

1. **Authentication**
   - Separate admin user table (NEVER share with app users)
   - Mandatory MFA for all admin accounts
   - Session expiration: 8 hours inactive, 24 hours max
   - Secure cookie flags: HttpOnly, Secure, SameSite=Strict

2. **Authorization**
   - RBAC enforced on every endpoint
   - Principle of least privilege
   - SUPER_ADMIN required for: user deletion, subscription grants, admin management

3. **Audit Logging**
   - Log ALL admin actions with: who, what, when, IP, user agent
   - Immutable audit trail (no deletes/updates)
   - Retention: 2 years minimum

4. **Network Security**
   - Separate subdomain: admin.nutri.app
   - HTTPS only (HSTS enabled)
   - Consider IP allowlisting for production
   - Rate limiting on all endpoints

5. **Data Protection**
   - PII access logged
   - Data exports encrypted
   - No PII in URL parameters
   - Mask sensitive data in logs

## Files to Create

### Admin Panel (New Next.js App)
```
admin-panel/
├── package.json
├── next.config.js
├── tailwind.config.js
├── tsconfig.json
├── .env.local
├── middleware.ts
├── app/
│   ├── layout.tsx
│   ├── (auth)/login/page.tsx
│   ├── (dashboard)/
│   │   ├── layout.tsx
│   │   ├── page.tsx                 # Dashboard overview
│   │   ├── users/page.tsx
│   │   ├── users/[id]/page.tsx
│   │   ├── subscriptions/page.tsx
│   │   ├── subscriptions/[id]/page.tsx
│   │   ├── webhooks/page.tsx
│   │   └── analytics/page.tsx
├── components/
│   ├── ui/                          # shadcn/ui
│   ├── layout/sidebar.tsx
│   ├── layout/header.tsx
│   ├── users/user-table.tsx
│   ├── users/user-detail.tsx
│   ├── subscriptions/subscription-table.tsx
│   ├── subscriptions/grant-modal.tsx
│   ├── webhooks/event-table.tsx
│   ├── analytics/metrics-cards.tsx
│   └── analytics/charts.tsx
└── lib/
    ├── api.ts
    ├── auth.ts
    └── utils.ts
```

### Backend Additions
```
server/src/
├── controllers/adminController.ts
├── services/adminService.ts
├── services/adminAuthService.ts
├── services/adminAnalyticsService.ts
├── routes/admin.ts
├── middleware/adminAuth.ts
├── middleware/adminAudit.ts
└── validation/adminSchemas.ts
```

## Acceptance Criteria

### Phase 1 (MVP)
1. Admin can log in with MFA
2. Admin can search and view user details
3. Admin can view subscription status for any user
4. Admin can manually grant/extend/revoke subscriptions
5. Admin can view webhook event history
6. Admin can export user data (GDPR)
7. Admin can delete user account (GDPR)
8. All actions are audit logged
9. Role-based access control enforced
10. Dashboard shows key subscription metrics

### Phase 2
11. Feature flags can be managed via UI
12. Advanced analytics dashboards functional
13. Food database can be edited
14. Push notifications can be sent

### Phase 3
15. Multiple admin users with role management
16. A/B testing dashboard functional
17. ML model monitoring integrated

**Test Strategy:**

## Testing Strategy

### Unit Tests
1. **Admin Authentication**
   - Password hashing and verification
   - MFA token generation and validation
   - Session management
   - Role-based permission checks

2. **Admin Services**
   - User search and filtering logic
   - Subscription grant/extend/revoke logic
   - Analytics calculations (MRR, churn rate)
   - Data export generation

3. **Audit Logging**
   - All actions create audit log entries
   - Correct data captured (IP, user agent, details)

### Integration Tests
1. **Admin API Endpoints**
   - Authentication flow with MFA
   - CRUD operations for all resources
   - Permission enforcement by role
   - Pagination and filtering

2. **Subscription Management**
   - Manual grant creates correct database records
   - Extension updates expiration correctly
   - Revocation removes entitlements

3. **Webhook Retry**
   - Failed webhooks can be reprocessed
   - Correct handling of retry results

### E2E Tests
1. **Admin Login Flow**
   - Login with valid credentials + MFA
   - Session persistence
   - Logout clears session

2. **User Management Flow**
   - Search for user by email
   - View user details
   - Export user data
   - Delete user (with cascading)

3. **Subscription Flow**
   - Find user with subscription issue
   - View webhook history
   - Grant extended access
   - Verify audit log created

### Security Tests
1. **Authentication**
   - Brute force protection (rate limiting)
   - Invalid MFA rejected
   - Session fixation prevention

2. **Authorization**
   - VIEWER cannot perform write operations
   - SUPPORT cannot access admin management
   - Only SUPER_ADMIN can delete users

3. **Audit Trail**
   - Cannot delete or modify audit logs
   - All sensitive actions logged

### Manual Testing Checklist
- [ ] Admin login with MFA
- [ ] User search by email
- [ ] User detail view shows subscription
- [ ] Grant Pro access to free user
- [ ] Extend subscription by 30 days
- [ ] View webhook event history
- [ ] Retry failed webhook
- [ ] Export user data
- [ ] Delete user account
- [ ] Verify audit log entries
- [ ] Test role-based restrictions
- [ ] Dashboard metrics accurate
- [ ] Charts render correctly

## Subtasks

### 39.1. Initialize Next.js 14+ admin app with App Router and shadcn/ui

**Status:** done  
**Dependencies:** None  

Create a new Next.js 14+ application in admin-panel/ directory with TypeScript, Tailwind CSS, App Router, and shadcn/ui component library. Configure project structure following the architecture specified in task details.

**Details:**

1. Run `npx create-next-app@latest admin-panel` with TypeScript, Tailwind, App Router
2. Install shadcn/ui: `npx shadcn-ui@latest init`
3. Install dependencies: `npm install @tanstack/react-table tremor recharts react-hook-form zod axios next-auth@beta`
4. Create folder structure: app/(auth), app/(dashboard), components/ui, components/layout, lib/
5. Configure tailwind.config.js to match React Native web styling tokens from lib/theme/colors.ts
6. Create .env.local with NEXTAUTH_SECRET, NEXTAUTH_URL, API_URL (pointing to http://localhost:3000)
7. Set up tsconfig.json with strict mode (zero 'any' types policy)
8. Create lib/api.ts with axios client configured to proxy to backend API
9. Add middleware.ts placeholder for auth protection
10. Verify build: `npm run build` succeeds

### 39.2. Extend Prisma schema with AdminUser and AdminAuditLog models

**Status:** done  
**Dependencies:** None  

Add AdminUser, AdminAuditLog, and FeatureFlag models to server/prisma/schema.prisma with proper indexes, relations, and enum types. Follow existing schema conventions.

**Details:**

1. Add AdminRole enum (SUPER_ADMIN, SUPPORT, ANALYST, VIEWER)
2. Add FeatureFlagType enum (BOOLEAN, STRING, NUMBER, JSON)
3. Create AdminUser model with fields: id (cuid), email (unique), passwordHash, name, role, mfaSecret, mfaEnabled, isActive, lastLoginAt, lastLoginIp, createdAt, updatedAt
4. Add indexes: @@index([email]) on AdminUser
5. Create AdminAuditLog model with fields: id (cuid), adminUserId, action, targetType, targetId, details (Json), ipAddress, userAgent, createdAt
6. Add relation: AdminUser.auditLogs (one-to-many)
7. Add indexes on AdminAuditLog: @@index([adminUserId]), @@index([action]), @@index([targetType, targetId]), @@index([createdAt])
8. Create FeatureFlag model with fields: id (cuid), key (unique), name, description, type, value (Json), isEnabled, targeting (Json), createdAt, updatedAt
9. Add @@index([key]) on FeatureFlag
10. Run `npm run db:generate` to generate Prisma client
11. Run `npm run db:push` (dev) or create migration with `npm run db:migrate`

### 39.3. Implement admin authentication backend with MFA (TOTP) support

**Status:** done  
**Dependencies:** 39.2  

Create admin authentication service with bcrypt password hashing, JWT session tokens, and TOTP-based MFA. Build endpoints for login, logout, MFA setup, and MFA verification.

**Details:**

1. Install dependencies in server/: `npm install speakeasy qrcode @types/speakeasy @types/qrcode`
2. Create server/src/services/adminAuthService.ts with functions: loginAdmin(email, password), verifyMFA(adminUserId, token), setupMFA(adminUserId), generateSessionToken(adminUser)
3. Use bcryptjs (already in deps) for password verification
4. Use speakeasy.generateSecret() for MFA setup, return QR code data URL using qrcode library
5. Session tokens: JWT with payload { adminUserId, role, sessionId } signed with JWT_SECRET, 8-hour expiration
6. Create server/src/controllers/adminAuthController.ts with handlers: login (POST), logout (POST), setupMFA (POST), verifyMFA (POST), getMe (GET)
7. Create server/src/validation/adminSchemas.ts with Zod schemas: adminLoginSchema, adminMFASetupSchema, adminMFAVerifySchema
8. Add constants to server/src/config/constants.ts: ADMIN_SESSION_EXPIRY = '8h', ADMIN_SESSION_MAX = '24h'
9. Store session metadata (sessionId, expiresAt) in AdminUser or separate AdminSession table
10. Return { token, requiresMFA, qrCode? } from login endpoint

### 39.4. Configure NextAuth.js v5 with credentials provider for admin authentication

**Status:** done  
**Dependencies:** 39.1, 39.3  

Set up NextAuth.js v5 in admin panel with custom credentials provider that calls backend admin auth API, handles MFA flow, and manages admin sessions with proper security.

**Details:**

1. Create admin-panel/lib/auth.ts with NextAuth configuration
2. Configure credentials provider to call backend POST /api/admin/auth/login
3. Handle MFA flow: if requiresMFA=true in response, store pendingMfaToken in session and redirect to MFA page
4. On MFA verification, call POST /api/admin/auth/mfa/verify and complete sign-in
5. Store admin JWT token in session callbacks (jwt, session)
6. Configure session strategy: 'jwt', maxAge: 8 hours
7. Set cookies: { secure: true, httpOnly: true, sameSite: 'strict' }
8. Create admin-panel/middleware.ts to protect /dashboard routes with NextAuth middleware
9. Implement role-based access control in middleware (check session.user.role)
10. Create admin-panel/app/api/auth/[...nextauth]/route.ts with NextAuth handler
11. Add NEXTAUTH_SECRET to .env.local (generate with `openssl rand -base64 32`)
12. Create types in admin-panel/lib/types.ts for AdminUser (id, email, name, role)

### 39.5. Build admin login UI with MFA flow and session management

**Status:** done  
**Dependencies:** 39.4  

Create login page with email/password form, MFA verification page, and session management UI using shadcn/ui components and react-hook-form with Zod validation.

**Details:**

1. Create admin-panel/app/(auth)/login/page.tsx with email/password form
2. Use shadcn/ui components: Card, Input, Button, Label from `components/ui/`
3. Install react-hook-form and integrate with Zod: `@hookform/resolvers/zod`
4. Create login form schema matching backend adminLoginSchema
5. On submit, call signIn('credentials', { email, password })
6. Handle MFA required response: redirect to /auth/mfa page
7. Create admin-panel/app/(auth)/mfa/page.tsx for MFA token input (6-digit code)
8. Display QR code if setupMFA=true (first-time setup)
9. Use speakeasy-compatible TOTP input (6 digits, numeric only)
10. Show error states with shadcn/ui Alert component (invalid credentials, MFA failed)
11. Create admin-panel/app/(auth)/layout.tsx with centered auth card design
12. Add loading states with shadcn/ui Spinner during API calls
13. Redirect to /dashboard on successful authentication

### 39.6. Create RBAC middleware and audit logging middleware for admin API

**Status:** done  
**Dependencies:** 39.3  

Implement role-based access control middleware and comprehensive audit logging middleware for all admin endpoints. Every admin action must be logged with who, what, when, IP, and user agent.

**Details:**

1. Create server/src/middleware/adminAuth.ts with requireAdmin(roles?: AdminRole[]) middleware
2. Verify JWT token from Authorization header, decode and validate
3. Load AdminUser from database, check isActive=true
4. If roles specified, verify adminUser.role is in allowed roles
5. Attach req.adminUser to request object (extend Express Request type)
6. Return 401 if token invalid/expired, 403 if role not allowed
7. Create server/src/middleware/adminAudit.ts with auditLog(action: string) middleware factory
8. Capture: adminUserId, action, targetType (from route), targetId (from req.params), details (from req.body), ipAddress (req.ip), userAgent (req.headers['user-agent'])
9. Insert AdminAuditLog record asynchronously (don't block response)
10. Use res.on('finish') to log after response sent
11. Create server/src/types/express.d.ts to extend Express Request with adminUser property
12. Add audit logging to ALL admin endpoints (chain auditLog middleware after requireAdmin)

### 39.7. Build user search and list backend API with pagination and filters

**Status:** done  
**Dependencies:** 39.6  

Create GET /api/admin/users endpoint with search (email, name), pagination, sorting, and filtering (subscription status, account status). Return user list with subscription info.

**Details:**

1. Create server/src/controllers/adminUserController.ts with listUsers handler
2. Create server/src/services/adminUserService.ts with getUserList(params) function
3. Query parameters: search (string), page (number), limit (number, max 100, default 20), sortBy (createdAt, email, name), sortOrder (asc, desc), status (active, disabled), subscriptionStatus (active, trial, expired, none)
4. Use Prisma where clause with OR for search: { OR: [{ email: { contains: search, mode: 'insensitive' } }, { name: { contains: search } }] }
5. Include subscription data with include: { subscriptions: { where: { status: 'ACTIVE' }, take: 1 } }
6. Calculate total count for pagination metadata
7. Return { users: [], pagination: { page, limit, total, totalPages } }
8. Create Zod schema in server/src/validation/adminSchemas.ts: listUsersQuerySchema
9. Apply RBAC: requireAdmin() (all roles can list users)
10. Apply audit logging: auditLog('USER_LIST')
11. Add route in server/src/routes/admin.ts: GET /api/admin/users

### 39.8. Build user detail API with GDPR export and delete endpoints

**Status:** done  
**Dependencies:** 39.6  

Create GET /api/admin/users/:id for detailed user view, POST /api/admin/users/:id/export for GDPR data export (JSON), and DELETE /api/admin/users/:id for GDPR-compliant account deletion with cascades.

**Details:**

1. Create getUserDetail(userId) in adminUserService: return user with all relations (subscriptions, meals, healthMetrics, activities count)
2. GET /api/admin/users/:id handler in adminUserController, verify user exists (404 if not)
3. Create exportUserData(userId) in adminUserService: fetch ALL user data (profile, meals, health metrics, activities, subscriptions, webhook events)
4. Return as downloadable JSON with structure: { user: {}, meals: [], healthMetrics: [], activities: [], subscriptions: [] }
5. POST /api/admin/users/:id/export handler: set Content-Type: application/json, Content-Disposition: attachment; filename=user-{userId}-export.json
6. Create deleteUserAccount(userId, adminUserId, reason) in adminUserService
7. Validation: require SUPER_ADMIN role for deletion (403 for other roles)
8. Use Prisma transaction to delete in order: AppStoreWebhookEvent, Subscription, Activity, HealthMetric, Meal, User (cascade)
9. Log deletion in AdminAuditLog with reason in details field
10. DELETE /api/admin/users/:id handler: require { reason } in request body (Zod schema)
11. Add routes: GET /users/:id, POST /users/:id/export, DELETE /users/:id
12. Apply RBAC: requireAdmin(['SUPER_ADMIN']) for DELETE, requireAdmin() for GET/export

### 39.9. Build user management UI with search, list, and detail views

**Status:** pending  
**Dependencies:** 39.1, 39.7  

Create admin panel pages for user search, paginated user list with TanStack Table, and detailed user view. Implement instant search with debouncing and client-side filtering.

**Details:**

1. Create admin-panel/app/(dashboard)/users/page.tsx for user list
2. Implement search input with debounced onChange (300ms delay using useDebouncedValue hook)
3. Use TanStack Table v8 with server-side pagination, sorting
4. Define columns: email, name, subscription status (badge), created date, actions (View button)
5. Fetch data from GET /api/admin/users with useQuery (use @tanstack/react-query)
6. Add filters dropdown: subscription status (active, trial, expired, none), account status (active, disabled)
7. Create admin-panel/components/users/user-table.tsx reusable component
8. Implement pagination controls (Previous, Next, page numbers) using shadcn/ui Pagination
9. Create admin-panel/app/(dashboard)/users/[id]/page.tsx for user detail view
10. Display user profile card: email, name, created date, current weight, goal weight
11. Show subscription status card: tier, expiration, trial info, auto-renew status
12. Show recent activity stats: meals logged (last 7 days), health metrics synced (last 7 days)
13. Add action buttons: Export Data, Delete Account (with role check)
14. Use shadcn/ui Card, Badge, Tabs components for layout

### 39.10. Implement GDPR export and delete UI with confirmation modals

**Status:** done  
**Dependencies:** 39.8, 39.9  

Create UI components for GDPR data export (download JSON) and account deletion with multi-step confirmation flow, reason input, and role-based access control.

**Details:**

1. Create admin-panel/components/users/export-data-button.tsx
2. On click, call POST /api/admin/users/:id/export, trigger browser download of JSON file
3. Show loading spinner during export (can take 5-10 seconds for large datasets)
4. Display success toast notification using shadcn/ui Toast
5. Create admin-panel/components/users/delete-user-modal.tsx with shadcn/ui AlertDialog
6. Require SUPER_ADMIN role to show Delete Account button (check session.user.role)
7. First confirmation: "Are you sure you want to delete this account? This cannot be undone."
8. Second step: require typing user's email to confirm (text input must match exactly)
9. Third step: require deletion reason (textarea, minimum 10 characters)
10. On confirm, call DELETE /api/admin/users/:id with { reason } body
11. Show loading state during deletion
12. On success, redirect to user list with success toast
13. On error, display error message in modal (e.g., 403 Forbidden if not SUPER_ADMIN)

### 39.11. Build subscription management backend API with manual operations

**Status:** done  
**Dependencies:** 39.6  

Create backend APIs for subscription list, detail view, and manual operations (grant Pro access, extend subscription, revoke access). All operations require SUPER_ADMIN role and are audit logged.

**Details:**

1. Create server/src/controllers/adminSubscriptionController.ts and server/src/services/adminSubscriptionService.ts
2. GET /api/admin/subscriptions: list all subscriptions with filters (status, productId, userId), pagination, include user info
3. GET /api/admin/subscriptions/:id: return subscription with full transaction history, related webhook events
4. GET /api/admin/subscriptions/lookup?txn={originalTransactionId}: lookup by Apple transaction ID for support tickets
5. POST /api/admin/subscriptions/:id/grant: manually grant Pro access (require { duration, reason } in body)
6. Implementation: create or update Subscription record with status=ACTIVE, expiresAt=now + duration, source=MANUAL_GRANT
7. POST /api/admin/subscriptions/:id/extend: extend expiration (require { days, reason })
8. Implementation: update expiresAt = expiresAt + days, log in metadata
9. POST /api/admin/subscriptions/:id/revoke: revoke access (require { reason })
10. Implementation: set status=CANCELLED, expiresAt=now
11. All operations require requireAdmin(['SUPER_ADMIN']) middleware
12. All operations use auditLog with action (SUBSCRIPTION_GRANT, SUBSCRIPTION_EXTEND, SUBSCRIPTION_REVOKE)
13. Create Zod schemas: grantSubscriptionSchema, extendSubscriptionSchema, revokeSubscriptionSchema
14. Add routes in server/src/routes/admin.ts

### 39.12. Build subscription management UI with manual operation modals

**Status:** done  
**Dependencies:** 39.1, 39.11  

Create subscription list page, detail view, and modals for manual grant/extend/revoke operations. Include originalTransactionId search for support workflows.

**Details:**

1. Create admin-panel/app/(dashboard)/subscriptions/page.tsx with TanStack Table
2. Columns: user email, product (Pro Monthly/Yearly), status (badge: active=green, expired=red, trial=blue), expires at, actions
3. Add search by originalTransactionId input (for support tickets like "customer says they paid but no access")
4. Create admin-panel/app/(dashboard)/subscriptions/[id]/page.tsx for detail view
5. Show subscription info card: user, product, status, created/expires dates, auto-renew status
6. Show transaction history table: all originalTransactionIds, purchase dates, amounts (from webhook metadata)
7. Show related webhook events table: notification type, received at, processing status
8. Create admin-panel/components/subscriptions/grant-modal.tsx for manual grant
9. Require SUPER_ADMIN role to show manual operation buttons
10. Grant modal fields: duration (select: 7 days, 30 days, 90 days, 1 year), reason (textarea, required)
11. Create extend-modal.tsx with days input (number, 1-365) and reason
12. Create revoke-modal.tsx with reason input only
13. All modals use shadcn/ui Dialog component with confirmation step
14. On success, refetch subscription data and show toast notification

### 39.13. Build webhook event viewer backend API with filtering and retry

**Status:** done  
**Dependencies:** 39.6  

Create API endpoints for listing App Store webhook events with filtering (notification type, date range, subscription ID), event detail view, and retry failed webhook processing.

**Details:**

1. Assumption: AppStoreWebhookEvent model exists from Task 38 (add to schema if missing)
2. Create server/src/controllers/adminWebhookController.ts and server/src/services/adminWebhookService.ts
3. GET /api/admin/webhooks: list webhook events with pagination (newest first)
4. Query params: notificationType (SUBSCRIBED, DID_RENEW, etc.), status (success, failed), startDate, endDate, subscriptionId, originalTransactionId
5. Use Prisma where clause with filters, order by createdAt DESC
6. Include related subscription and user data
7. GET /api/admin/webhooks/:id: return full webhook event with complete JSON payload, processing status, error message if failed
8. POST /api/admin/webhooks/:id/retry: retry processing failed webhook
9. Implementation: call webhook processing service (from Task 38) with stored payload, update status based on result
10. Require SUPER_ADMIN role for retry endpoint
11. Apply audit logging: auditLog('WEBHOOK_RETRY')
12. Create Zod schemas: listWebhooksQuerySchema
13. Add routes in server/src/routes/admin.ts

### 39.14. Build webhook event viewer UI with filtering and retry functionality

**Status:** pending  
**Dependencies:** 39.1, 39.13  

Create webhook event list page with advanced filtering, event detail view showing full JSON payload, and retry button for failed events with real-time status updates.

**Details:**

1. Create admin-panel/app/(dashboard)/webhooks/page.tsx with TanStack Table
2. Columns: notification type, subscription ID (link to subscription detail), received at, status (badge: success=green, failed=red, pending=yellow), actions
3. Add filter controls: notification type dropdown (all types from Task 38), status dropdown (success, failed, pending), date range picker (shadcn/ui Calendar)
4. Implement client-side filtering with URL params (persist filters on page reload)
5. Click row to expand and show full JSON payload using shadcn/ui Collapsible or Accordion
6. Use JSON syntax highlighting library (react-json-view or similar) for payload display
7. For failed events, show error message prominently
8. Add Retry button for failed events (require SUPER_ADMIN role)
9. On retry click, call POST /api/admin/webhooks/:id/retry
10. Show loading spinner on retry button during processing
11. Poll event status after retry (useQuery with refetchInterval) or use optimistic update
12. Display success/failure toast after retry completes
13. Add search input for originalTransactionId (quick lookup from support tickets)

### 39.15. Build analytics calculations backend for MRR, churn, and trial conversion

**Status:** done  
**Dependencies:** 39.6  

Create backend API endpoint and service layer for calculating key subscription analytics: MRR, churn rate, trial conversion rate, new subscriptions, and active subscribers by tier.

**Details:**

1. Create server/src/services/adminAnalyticsService.ts with analytics calculation functions
2. getSubscriptionMetrics(): calculate active subscribers by tier (Pro Monthly, Pro Yearly, Trial)
3. MRR calculation: sum of (monthly subscription price) for active subscriptions + (yearly price / 12) for yearly subscriptions
4. Use Prisma aggregation: Subscription.count({ where: { status: 'ACTIVE', productId: 'pro_monthly' } })
5. getNewSubscriptions(period): count subscriptions created in period (today, this week, this month)
6. Use Prisma where: { createdAt: { gte: startDate } }
7. getChurnMetrics(period): calculate cancellations in period, churn rate = cancellations / active_at_start
8. Use Subscription status transitions (need to track cancelledAt timestamp - add to schema if missing)
9. getTrialMetrics(): active trials count, trial conversion rate = (trials converted to paid) / (trials started in last 30 days)
10. Query: Subscription.count({ where: { status: 'ACTIVE', AND: [{ createdAt: { lte: 30 days ago } }, { originalTransactionId: { not: null } }] } })
11. Create GET /api/admin/analytics/overview endpoint in adminAnalyticsController
12. Return JSON: { mrr, activeSubscribers: { total, proMonthly, proYearly, trial }, newSubscriptions: { today, week, month }, churn: { rate, count }, trials: { active, conversionRate } }
13. Add route in server/src/routes/admin.ts
14. Apply RBAC: requireAdmin() (all roles can view analytics)

### 39.16. Build analytics dashboard UI with Tremor charts and metric cards

**Status:** pending  
**Dependencies:** 39.1, 39.15  

Create dashboard overview page with real-time subscription metrics, MRR/revenue charts, subscriber count over time, and trial conversion metrics using Tremor chart library.

**Details:**

1. Create admin-panel/app/(dashboard)/page.tsx for dashboard overview
2. Fetch analytics data from GET /api/admin/analytics/overview using useQuery
3. Create metric cards grid (4 cards across) using Tremor Card component:
   - Total Active Subscribers (with count by tier in subtitle)
   - Monthly Recurring Revenue (MRR in USD, format with currency)
   - New Subscriptions This Month (with week/today in subtitle)
   - Churn Rate (percentage with count in subtitle)
4. Use Tremor Metric and Text components for card content
5. Add color coding: green for positive metrics, red for churn
6. Create admin-panel/components/analytics/metrics-cards.tsx reusable component
7. Fetch time-series data for charts: GET /api/admin/analytics/subscribers-over-time and GET /api/admin/analytics/revenue-over-time (create these endpoints)
8. Create Subscribers Over Time line chart using Tremor LineChart component (last 30 days, daily data points)
9. Create Revenue Over Time area chart using Tremor AreaChart (last 12 months, monthly MRR)
10. Create Subscription Status pie chart using Tremor DonutChart (active, trial, expired distribution)
11. Add Trial Conversion Funnel using Tremor BarList (trials started → active → converted)
12. Implement auto-refresh every 60 seconds using useQuery refetchInterval
13. Add date range selector for charts (last 7 days, 30 days, 90 days, 1 year)

### 39.17. Implement security hardening: rate limiting, session security, IP allowlist

**Status:** done  
**Dependencies:** 39.6  

Add production-grade security hardening to admin panel: strict rate limiting on admin endpoints, secure session configuration, optional IP allowlisting, and enhanced security headers.

**Details:**

1. Create server/src/middleware/adminRateLimiter.ts using express-rate-limit
2. Admin login endpoint: 5 requests per 15 minutes per IP (more strict than app endpoints)
3. Admin API endpoints: 100 requests per 15 minutes per admin user (track by adminUserId from JWT)
4. Use Redis store for rate limiting if available (fallback to memory store)
5. Return 429 Too Many Requests with Retry-After header
6. Create server/src/middleware/ipAllowlist.ts for optional IP restriction
7. Read ADMIN_IP_ALLOWLIST from environment (comma-separated IPs), skip if not set
8. Check req.ip against allowlist, return 403 if not allowed
9. Log blocked IPs in AdminAuditLog with action='IP_BLOCKED'
10. Update helmet configuration in server/src/middleware/security.ts for admin routes:
    - Stricter CSP: no inline scripts, only same-origin
    - X-Frame-Options: DENY (prevent iframe embedding)
    - X-Content-Type-Options: nosniff
11. Configure CORS for admin API: only allow admin-panel origin (admin.nutri.app or localhost:3001)
12. Add security headers to Next.js admin panel in next.config.js
13. Apply adminRateLimiter and ipAllowlist middleware to all /api/admin/* routes
14. Document in .env.example: ADMIN_IP_ALLOWLIST (optional)

### 39.18. Write comprehensive test suite and deployment configuration

**Status:** done  
**Dependencies:** 39.2, 39.3, 39.6, 39.7, 39.8, 39.11, 39.13, 39.15, 39.17  

Create unit tests, integration tests, E2E tests for admin panel MVP. Set up deployment configuration for Next.js app (Vercel) and document deployment process with environment variables.

**Details:**

1. Backend unit tests in server/src/__tests__/admin/:
   - adminAuthService.test.ts: password verification, MFA setup, session tokens
   - adminUserService.test.ts: user list, search, export, delete
   - adminSubscriptionService.test.ts: grant, extend, revoke
   - adminAnalyticsService.test.ts: MRR, churn, trial conversion calculations
2. Backend integration tests in server/src/__tests__/admin/integration/:
   - adminAuth.integration.test.ts: login flow, MFA flow, session management
   - adminUsers.integration.test.ts: GET /users, GET /users/:id, DELETE /users/:id
   - adminSubscriptions.integration.test.ts: manual operations
   - Test RBAC enforcement (403 for unauthorized roles)
   - Test audit logging (verify records created)
3. Frontend tests in admin-panel/:
   - Install @testing-library/react, @testing-library/jest-dom
   - Test components: user-table.test.tsx, grant-modal.test.tsx, metrics-cards.test.tsx
   - Test hooks: useResponsive (if created), useDebounce
   - Mock API calls with MSW (Mock Service Worker)
4. Create admin-panel/vercel.json for Vercel deployment configuration
5. Set environment variables in Vercel dashboard: NEXTAUTH_SECRET, NEXTAUTH_URL, API_URL (backend URL)
6. Create deployment guide in admin-panel/README.md:
   - Prerequisites (Node.js 18+, npm)
   - Environment variables required
   - Build command: npm run build
   - Start command: npm start
   - Database migrations (run before deployment)
7. Create seed script for AdminUser in server/prisma/seed.ts (create first SUPER_ADMIN account)
8. Document admin user creation: npm run seed:admin (create initial admin with email/password)
9. Add npm scripts to admin-panel/package.json: test, test:watch, test:coverage
10. Achieve > 80% coverage for critical paths (auth, RBAC, audit logging)
